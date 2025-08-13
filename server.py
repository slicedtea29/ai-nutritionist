# server.py
import os, json, traceback, datetime as dt
from functools import wraps
from flask import Flask, request, jsonify, session, abort
from flask_cors import CORS
from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, create_engine, select
from sqlalchemy.orm import declarative_base, relationship, Session
from openai import OpenAI

# ----- Config -----
APP_PASSWORD = os.environ.get("APP_PASSWORD")
SECRET_KEY   = os.environ.get("SECRET_KEY", os.urandom(32))
OPENAI_KEY   = os.environ.get("OPENAI_API_KEY")
PROJECT_ID   = os.environ.get("OPENAI_PROJECT")           # optional
ALLOWED_ORIGIN = os.environ.get("ALLOWED_ORIGIN", "*")
DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///app.db")

if not APP_PASSWORD:  raise SystemExit("ERROR: APP_PASSWORD not set.")
if not OPENAI_KEY:    raise SystemExit("ERROR: OPENAI_API_KEY not set.")

app = Flask(__name__)
app.secret_key = SECRET_KEY
CORS(app, resources={r"/*": {"origins": ALLOWED_ORIGIN}}, supports_credentials=True)

client = OpenAI(project=PROJECT_ID) if PROJECT_ID else OpenAI()
MODEL = "gpt-4o-mini"

# ----- DB -----
Base = declarative_base()
engine = create_engine(
    DATABASE_URL,
    connect_args=({"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}),
    future=True,
)

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    email = Column(String(255), unique=True, nullable=True)
    created_at = Column(DateTime, default=dt.datetime.utcnow)
    preferences = relationship("Preference", uselist=False, back_populates="user", cascade="all,delete-orphan")
    conversations = relationship("Conversation", back_populates="user", cascade="all,delete-orphan")

class Preference(Base):
    __tablename__ = "preferences"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    data = Column(Text, default="{}")
    user = relationship("User", back_populates="preferences")

class Conversation(Base):
    __tablename__ = "conversations"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    title = Column(String(200), default="New Conversation")
    created_at = Column(DateTime, default=dt.datetime.utcnow)
    updated_at = Column(DateTime, default=dt.datetime.utcnow)
    user = relationship("User", back_populates="conversations")
    messages = relationship("Message", back_populates="conversation", cascade="all,delete-orphan", order_by="Message.id")

class Message(Base):
    __tablename__ = "messages"
    id = Column(Integer, primary_key=True)
    conversation_id = Column(Integer, ForeignKey("conversations.id"), nullable=False)
    role = Column(String(20))     # "user" | "assistant"
    content = Column(Text)
    created_at = Column(DateTime, default=dt.datetime.utcnow)
    conversation = relationship("Conversation", back_populates="messages")

Base.metadata.create_all(engine)

# ----- Auth helpers -----
def require_auth(fn):
    @wraps(fn)
    def _wrap(*args, **kwargs):
        if session.get("authed") is True and session.get("user_id"):
            return fn(*args, **kwargs)
        if request.method == "OPTIONS":
            return ("", 204)
        return abort(401)
    return _wrap

def get_user(db: Session):
    uid = session.get("user_id")
    return db.get(User, uid) if uid else None

# ----- OpenAI -----
def chat_complete(messages, max_tokens=500, temperature=0.4):
    resp = client.chat.completions.create(
        model=MODEL, messages=messages, temperature=temperature, max_tokens=max_tokens
    )
    return resp.choices[0].message.content.strip()

# ----- Routes -----
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"ok": True})

@app.route("/login", methods=["POST", "OPTIONS"])
def login():
    if request.method == "OPTIONS": return ("", 204)
    data = request.get_json(force=True) or {}
    pwd = (data.get("password") or "").strip()
    if pwd != APP_PASSWORD:
        return jsonify({"ok": False, "error": "Bad password"}), 401
    with Session(engine) as db:
        u = db.execute(select(User).limit(1)).scalar_one_or_none()
        if not u:
            u = User(); db.add(u); db.commit()
        session["authed"] = True
        session["user_id"] = u.id
    return jsonify({"ok": True})

@app.route("/logout", methods=["POST", "OPTIONS"])
@require_auth
def logout():
    if request.method == "OPTIONS": return ("", 204)
    session.clear()
    return jsonify({"ok": True})

@app.route("/me", methods=["GET"])
@require_auth
def me():
    with Session(engine) as db:
        u = get_user(db)
        if not u: return abort(401)
        prefs = u.preferences.data if u.preferences else "{}"
        return jsonify({"id": u.id, "preferences": json.loads(prefs)})

@app.route("/preferences", methods=["GET", "POST", "OPTIONS"])
@require_auth
def preferences():
    if request.method == "OPTIONS": return ("", 204)
    with Session(engine) as db:
        u = get_user(db)
        if not u: return abort(401)
        if request.method == "GET":
            data = u.preferences.data if u.preferences else "{}"
            return jsonify({"data": json.loads(data)})
        body = request.get_json(force=True) or {}
        data = json.dumps(body.get("data") or {})
        if not u.preferences:
            u.preferences = Preference(data=data)
        else:
            u.preferences.data = data
        db.add(u); db.commit()
        return jsonify({"ok": True})

@app.route("/conversations", methods=["GET", "POST", "OPTIONS"])
@require_auth
def conversations():
    if request.method == "OPTIONS": return ("", 204)
    with Session(engine) as db:
        u = get_user(db)
        if not u: return abort(401)
        if request.method == "GET":
            rows = db.execute(
                select(Conversation).where(Conversation.user_id == u.id).order_by(Conversation.updated_at.desc())
            ).scalars().all()
            return jsonify([{"id": c.id, "title": c.title, "updated_at": c.updated_at.isoformat()} for c in rows])
        title = (request.get_json(force=True) or {}).get("title") or "New Conversation"
        c = Conversation(user_id=u.id, title=title)
        db.add(c); db.commit()
        return jsonify({"id": c.id, "title": c.title})

@app.route("/conversations/<int:cid>/messages", methods=["GET", "POST", "OPTIONS"])
@require_auth
def conv_messages(cid: int):
    if request.method == "OPTIONS": return ("", 204)
    with Session(engine) as db:
        u = get_user(db)
        if not u: return abort(401)
        c = db.get(Conversation, cid)
        if not c or c.user_id != u.id: return abort(404)

        if request.method == "GET":
            msgs = [{"id": m.id, "role": m.role, "content": m.content, "created_at": m.created_at.isoformat()} for m in c.messages]
            return jsonify(msgs)

        body = request.get_json(force=True) or {}
        text = (body.get("message") or "").strip()
        if not text:
            return jsonify({"error": "message required"}), 400

        m_user = Message(conversation_id=cid, role="user", content=text)
        db.add(m_user); db.flush()

        prefs_json = json.loads(c.user.preferences.data) if c.user and c.user.preferences else {}
        last_msgs = [{"role": m.role, "content": m.content} for m in c.messages[-12:]] + [{"role": "user", "content": text}]
        sys = "You are an expert AI nutrition coach. Be concise and practical."
        if prefs_json: sys += f" User preferences: {prefs_json}"

        reply = chat_complete([{"role":"system","content":sys}] + last_msgs, max_tokens=500)

        m_ai = Message(conversation_id=cid, role="assistant", content=reply)
        db.add(m_ai)
        c.updated_at = dt.datetime.utcnow()
        db.commit()
        return jsonify({"reply": reply, "message_id": m_ai.id})

# Legacy endpoint used by your current UI
@app.route("/chat", methods=["POST", "OPTIONS"])
@require_auth
def chat_legacy():
    if request.method == "OPTIONS": return ("", 204)
    data = request.get_json(force=True) or {}
    text = (data.get("message") or "").strip()
    with Session(engine) as db:
        u = get_user(db)
        if not u: return abort(401)
        # FIX: safely get the latest conversation (or None)
        c = db.execute(
            select(Conversation)
            .where(Conversation.user_id == u.id)
            .order_by(Conversation.updated_at.desc())
        ).scalars().first()
        if not c:
            c = Conversation(user_id=u.id, title="Coach"); db.add(c); db.commit()
        # append + reply
        m_user = Message(conversation_id=c.id, role="user", content=text)
        db.add(m_user); db.flush()
        prefs_json = json.loads(u.preferences.data) if u.preferences else {}
        ctx = [{"role": m.role, "content": m.content} for m in c.messages[-12:]] + [{"role":"user","content":text}]
        sys = "You are an expert AI nutrition coach. Be concise and practical."
        if prefs_json: sys += f" User preferences: {prefs_json}"
        reply = chat_complete([{"role":"system","content":sys}] + ctx, max_tokens=500)
        m_ai = Message(conversation_id=c.id, role="assistant", content=reply)
        db.add(m_ai); c.updated_at = dt.datetime.utcnow(); db.commit()
        return jsonify({"reply": reply})

@app.route("/plan", methods=["POST", "OPTIONS"])
@require_auth
def plan():
    if request.method == "OPTIONS": return ("", 204)
    data = request.get_json(force=True) or {}
    P, F, C = data.get("protein"), data.get("fat"), data.get("carbs")
    prefs = data.get("prefs", "")
    try:
        raw = chat_complete(
            [{"role":"system","content":"Return ONLY strict JSON, no commentary."},
             {"role":"user","content":
              ("Create a simple 1-day meal plan (Breakfast, Lunch, Dinner, Snack) that roughly hits "
               f"{P}g protein, {F}g fat, {C}g carbs. Output JSON array with 4 meals; each item must have: "
               "title, kcal (int), macros {P,C,F}, items (3-6 ingredients). Keep prep simple and cost-aware. "
               f"Preferences: {prefs}")}],
            max_tokens=700, temperature=0.2
        )
        try: meals_in = json.loads(raw)
        except: meals_in = []
        if not isinstance(meals_in, list) or not meals_in:
            meals_in = [
                {"title":"High-Protein Breakfast Bowl","kcal":520,"macros":{"P":45,"C":55,"F":15},
                 "items":["4 eggs (2 whole, 2 whites)","Oats 60g","Blueberries","Almonds"]},
                {"title":"Simple Steak & Rice","kcal":680,"macros":{"P":55,"C":65,"F":20},
                 "items":["Sirloin 7oz","Jasmine rice 200g cooked","Broccoli","Olive oil"]},
                {"title":"Greek Yogurt Parfait","kcal":380,"macros":{"P":30,"C":40,"F":10},
                 "items":["Greek yogurt 250g","Honey 1 tsp","Granola 30g","Strawberries"]},
                {"title":"Evening Snack Wrap","kcal":360,"macros":{"P":25,"C":35,"F":10},
                 "items":["Whole-wheat wrap","Turkey 4oz","Spring mix","Greek yogurt sauce"]},
            ]
        out = []
        for i, m in enumerate(meals_in, 1):
            out.append({
                "id": i,
                "title": m.get("title", f"Meal #{i}"),
                "kcal": int(m.get("kcal", 400)),
                "macros": {
                    "P": int(m.get("macros", {}).get("P", 25)),
                    "C": int(m.get("macros", {}).get("C", 40)),
                    "F": int(m.get("macros", {}).get("F", 12)),
                },
                "items": (m.get("items") or [])[:8],
            })
        return jsonify({"meals": out})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"{type(e).__name__}: {str(e)}"}), 500

@app.route("/analyze_meal", methods=["POST", "OPTIONS"])
@require_auth
def analyze_meal():
    if request.method == "OPTIONS": return ("", 204)
    data = request.get_json(force=True) or {}
    text = data.get("text", "")
    macros = data.get("macros", {})
    try:
        advice = chat_complete(
            [{"role": "system", "content": "Be concise, bullet-y."},
             {"role": "user", "content":
              ("Analyze this meal for approximate calories and macros. "
               "Then give 2–3 quick swap ideas to better match targets. "
               f"Targets: {macros}. Meal: {text}. Keep it under 120 words.")}],
            max_tokens=220
        )
        return jsonify({"advice": advice})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"{type(e).__name__}: {str(e)}"}), 500

@app.route("/estimate_macros", methods=["POST", "OPTIONS"])
@require_auth
def estimate_macros():
    if request.method == "OPTIONS": return ("", 204)
    data = request.get_json(force=True) or {}
    text = data.get("text", "")
    try:
        summary = chat_complete(
            [{"role":"system","content":"Keep it short."},
             {"role":"user","content": f"Estimate calories/macros: {text}. Output: Calories ~xxxx | P xx | C xx | F xx."}],
            max_tokens=180
        )
        return jsonify({"summary": summary})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"{type(e).__name__}: {str(e)}"}), 500

@app.route("/suggest_swaps", methods=["POST", "OPTIONS"])
@require_auth
def suggest_swaps():
    if request.method == "OPTIONS": return ("", 204)
    data = request.get_json(force=True) or {}
    text = data.get("text", "")
    macros = data.get("macros", {})
    try:
        swaps = chat_complete(
            [{"role":"system","content":"Actionable and brief."},
             {"role":"user","content": f"Suggest 3–5 swaps to match {macros}. Meal: {text}. Format: Swap -> Reason (P/C/F impact)."}],
            max_tokens=220
        )
        return jsonify({"swaps": swaps})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"{type(e).__name__}: {str(e)}"}), 500

if __name__ == "__main__":
    # production: gunicorn server:app
    app.run(host="0.0.0.0", port=5000, debug=False)
