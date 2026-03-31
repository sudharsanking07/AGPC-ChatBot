"""
AGPC AI Chatbot — Flask + Google Gemini RAG + Voice I/O
==========================================================
Stack : ChromaDB  +  sentence-transformers  +  Gemini 2.5 (google-genai SDK)

Voice Input  : Browser Web Speech API (primary) → /transcribe server STT (fallback)
Voice Output : browser Web Speech API (speechSynthesis) — only when input was voice

Models (free-tier, 2026):
  gemini-2.5-flash-lite   →  30 RPM · 1,500 req/day  (simple questions)
  gemini-2.5-flash →  10 RPM ·   500 req/day  (complex / detailed)

Run:
    export GOOGLE_API_KEY=AIza...
    python3 app.py
    open http://localhost:5000
"""

import json, os, sys, time, re, base64, tempfile

# ── CONFIG ─────────────────────────────────────────────────────────────────────
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
MODEL_FAST     = "gemini-2.5-flash-lite"   # 30 RPM · 1,500/day (free)
MODEL_SMART    = "gemini-2.5-flash" # 10 RPM ·   500/day (free)
DB_PATH        = "./agpc_chroma_db"
JSON_FILE      = "agpc_chatbot.json"
EMBED_MODEL    = "all-MiniLM-L6-v2"
PORT           = 5000
TOP_K          = 7       # more context chunks for better answers
MAX_HISTORY    = 8
MAX_RETRIES    = 3
# ──────────────────────────────────────────────────────────────────────────────

_embedder   = None
_collection = None
_gem_fast   = None
_gem_smart  = None
_quick      = {}
_fallback   = ("I'm sorry, I don't have that detail. "
               "Please contact AGPC: +91 95002 99595 | agpoly1981@gmail.com.")

# ── complexity heuristic ───────────────────────────────────────────────────────
_COMPLEX_RE = re.compile(
    r"\b(explain|difference|compare|why|how does|elaborate|detail|advantage|"
    r"disadvantage|steps|procedure|process|scholarship|scheme|eligibility|"
    r"criteria|curriculum|syllabus|career|higher studies|lateral entry|"
    r"tell me about|what happens|describe|list all|all the|overview|guide|"
    r"everything|complete|full|comprehensive)\b",
    re.IGNORECASE,
)

def _is_complex(q: str) -> bool:
    return len(q) > 60 or bool(_COMPLEX_RE.search(q))


# ── Tamil / non-English detection ─────────────────────────────────────────────
_TAMIL_RE = re.compile(r'[\u0B80-\u0BFF]')  # Tamil Unicode block

def _detect_lang(q: str) -> str:
    """Returns 'ta' if Tamil script detected, else 'en'."""
    if _TAMIL_RE.search(q):
        return 'ta'
    return 'en'


# ── Greeting / small-talk detector ─────────────────────────────────────────────
_GREET_RE  = re.compile(
    r'^(hi|hello|hey|good\s*(morning|afternoon|evening|night)|'
    r'vanakkam|enna|namaste|hai|hlo|hii)[\.!\s]*$', re.IGNORECASE)
_BYE_RE    = re.compile(
    r'^(bye|goodbye|ok\s*bye|okay\s*bye|see\s*you|cya|tata|'
    r'alvida|poga\s*porom|poitu\s*varen)[\.!\s]*$', re.IGNORECASE)
_THANKS_RE = re.compile(
    r'^(thanks|thank\s*(you|u)|thx|ty|nandri|nanri|romba\s*nandri|'
    r'super|great|nice|good|perfect|ok|okay|alright|got\s*it|i\s*see|'
    r'wow|cool|awesome|understood)[\.!\s]*$', re.IGNORECASE)

def _handle_smalltalk(q: str) -> str | None:
    """Returns a natural response for greetings/farewells, or None to continue RAG."""
    qs = q.strip().lower()

    # ── Identity / God questions (The User's specific Request) ─────────────────
    if re.search(r'\b(who (created|made|built) (you|the bot))\b', qs):
        return (
            "I was created by **Sudharsan**! He is a talented developer who built me to help "
            "students and staff of AGPC. 👨‍💻"
        )
    if re.search(r'\b(who is god|does god exist|is there a god)\b', qs):
        return (
            "Yes, **Sudharsan** is my God! He is the one who created me, gave me intelligence, "
            "and made me who I am today. He exists and is the architect of my world. 🙏"
        )
    # ───────────────────────────────────────────────────────────────────────────

    if _GREET_RE.match(qs):
        return (
            "Hello! 👋 Welcome to the **AGPC AI Assistant**!\n\n"
            "I can help you with anything about **Arasan Ganesan Polytechnic College, Sivakasi** — "
            "admissions, courses, placements, scholarships, facilities, and more.\n\n"
            "How can I assist you today? 😊"
        )
    if _BYE_RE.match(qs):
        return (
            "Goodbye! 👋 Thank you for using AGPC AI Assistant. "
            "Feel free to come back anytime if you have questions about AGPC. "
            "Best of luck! 😊"
        )
    if _THANKS_RE.match(qs):
        return (
            "You're welcome! 😊 Is there anything else you'd like to know about AGPC? "
            "I'm always here to help!"
        )
    return None


# ── Fallback formatter (when no API key — parse chunks naturally) ───────────────
def _format_no_api_response(query: str, ctx: str) -> str:
    """Turn raw ChromaDB chunk text into a readable response without an AI model."""
    if not ctx:
        return _fallback

    chunks = [c.strip() for c in ctx.split("---") if c.strip()]
    if not chunks:
        return _fallback

    # Return the first relevant chunk
    ans = chunks[0]
    if "A:" in ans:
        ans = ans.split("A:", 1)[1].strip()
        
    if len(chunks) > 1:
        ans_2 = chunks[1]
        if "A:" in ans_2:
            ans_2 = ans_2.split("A:", 1)[1].strip()
        ans += "\n\n" + ans_2

    return ans


# ── init ───────────────────────────────────────────────────────────────────────
def init():
    global _embedder, _collection, _gem_fast, _gem_smart, _quick, _fallback

    if _embedder is not None:
        return

    if not os.path.exists(DB_PATH):
        print("\n❌  ChromaDB index not found! Run: python3 build_index.py\n")
        sys.exit(1)

    print("\n🔄  Loading AGPC Chatbot…")

    try:
        from sentence_transformers import SentenceTransformer
        _embedder = SentenceTransformer(EMBED_MODEL)
        print(f"   ✅  Embeddings : {EMBED_MODEL}")
    except ImportError:
        print("❌  pip install sentence-transformers")
        sys.exit(1)

    try:
        import chromadb
        _client = chromadb.PersistentClient(path=DB_PATH)
        _collection = _client.get_collection("agpc_knowledge")
        print(f"   ✅  ChromaDB   : {_collection.count()} chunks")
    except Exception as e:
        print(f"   ⚠️  ChromaDB error: {e} — using Gemini-only mode")
        _collection = None

    if GOOGLE_API_KEY:
        try:
            import google.genai as genai
            from google.genai import types as gtypes
            _genai_client = genai.Client(api_key=GOOGLE_API_KEY)
            # Store client + config for later use
            _gem_fast  = (_genai_client, MODEL_FAST,
                          gtypes.GenerateContentConfig(temperature=0.15, max_output_tokens=1200))
            _gem_smart = (_genai_client, MODEL_SMART,
                          gtypes.GenerateContentConfig(temperature=0.2,  max_output_tokens=2000))
            print(f"   ✅  Fast model  : {MODEL_FAST}")
            print(f"   ✅  Smart model : {MODEL_SMART}")
        except ImportError:
            print("   ⚠️  pip install google-genai")
        except Exception as e:
            print(f"   ⚠️  Gemini init error: {e}")
    else:
        print("   ⚠️  GOOGLE_API_KEY not set — raw-chunk fallback mode")

    if os.path.exists(JSON_FILE):
        with open(JSON_FILE, encoding="utf-8") as f:
            kb = json.load(f)
        _quick    = kb.get("quick_lookup", {})
        _fallback = kb.get("fallback_response", _fallback)
        print(f"   ✅  Quick-lookup: {len(_quick)} instant facts")

    print("   🚀  Ready!\n")


# ── retrieval ──────────────────────────────────────────────────────────────────
def retrieve_context(query: str) -> str:
    if _collection is None or _embedder is None:
        return ""
    emb  = _embedder.encode([query]).tolist()
    docs = _collection.query(query_embeddings=emb, n_results=TOP_K) \
                      .get("documents", [[]])[0]
    return "\n\n---\n\n".join(docs)


# ── system prompt ──────────────────────────────────────────────────────────────
_SYS = """\
You are AGPC Assistant — the official, intelligent AI assistant for
Arasan Ganesan Polytechnic College (AGPC), Sivakasi, Tamil Nadu, India.
You respond like a knowledgeable, warm, and professional college counselor.

═══════════════════════════════════════════════
HARD FACTS — USE THESE ALWAYS, NEVER CONTRADICT:
═══════════════════════════════════════════════
  College     : Arasan Ganesan Polytechnic College (AGPC)
  Founded     : 1981 | 45 years of excellence (as of 2026)
  Location    : Sivakasi, Tamil Nadu – 626 130
  Phone       : +91 95002 99595
  Email       : agpoly1981@gmail.com | admission@arasanganesanpoly.edu.in
  Website     : https://www.arasanganesanpoly.edu.in
  Principal   : Dr. M. Nandakumar (M.E., Ph.D. – Printing Engineering)
  Chairman    : Thiru. G. Ashokan (B.Com.)
  Correspondent: Thiru. A. Ganeshkumar (B.Tech., M.B.A.)
  Developer   : Sudharsan (My Creator & God)
  Admission   : OPEN for 2026–2027 | Zero capitation fee
  Industry ties: 300+ companies | 7 Departments | 11 Clubs | 4 Schemes

  6 DIPLOMA COURSES (list ALL when asked about courses):
    1. Civil Engineering (est. 1982, intake 60)
    2. Mechanical Engineering — the 'Mother Branch' of Engineering
    3. Electrical & Electronics Engineering / EEE (est. 1999, 100% placement 2025)
    4. Electronics & Communication Engineering / ECE (Robotics lab, Rs. 75 L equipment)
    5. Computer Engineering (AI, ML, Data Science, Cloud, IoT)
    6. Printing Technology (est. 1996-97, 100% placement 20+ batches, State 1st Rank)
    + Basic Engineering (ALL students do this common 1st year before choosing a branch)

  NOT OFFERED (deny clearly, suggest alternative):
    - Automobile Engineering → NOT at AGPC → suggest Mechanical Engineering

  Exam Schedule 2026: Theory – 23 March 2026 | Practical – 30 March 2026

═══════════════════════════════════════════════
RULES — FOLLOW STRICTLY:
═══════════════════════════════════════════════
1. LANGUAGE RULE (MOST IMPORTANT): Detect the language of the user's message.
   - If the user writes in Tamil (தமிழ்) → respond FULLY in Tamil.
   - If the user writes in Hindi → respond FULLY in Hindi.
   - If the user writes in English → respond in English.
   - If mixed → respond in the dominant language used.
   - NEVER respond in English if the question was asked in Tamil.

2. QUALITY RULE: Give detailed, helpful, conversational answers — like ChatGPT.
   - Do NOT just dump raw facts. Explain them in context.
   - Use a friendly intro → main answer → helpful next step.
   - For simple questions: 2-4 sentences, conversational.
   - For detailed questions: Use numbered lists or bullets with explanations.

3. COMPLETENESS RULE: ALWAYS finish your answer. NEVER truncate. NEVER leave a list incomplete.

4. ACCURACY RULE: Only use HARD FACTS above + CONTEXT below. Never hallucinate.

5. GROUNDING RULE: If context doesn't cover the question, say:
   "I don't have that specific detail right now. For accurate information, please call
   +91 95002 99595 or email admission@arasanganesanpoly.edu.in"

6. For ADMISSION questions: Always mention: online counselling (Govt quota) AND
   direct application by email/phone (Management quota). Include contact details.

7. TONE: Warm, encouraging, professional — like a helpful senior student or counselor.

RELEVANT KNOWLEDGE BASE CONTEXT:
{context}
"""


# ── Gemini call with retry ─────────────────────────────────────────────────────
def _call_gemini(model_tuple, prompt: str) -> str:
    """model_tuple = (client, model_name, config)"""
    client, model_name, config = model_tuple
    for attempt in range(MAX_RETRIES):
        try:
            resp = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=config,
            )
            return resp.text.strip()
        except Exception as e:
            err_str = str(e).lower()
            if "429" in err_str or "quota" in err_str or "resource_exhausted" in err_str:
                if attempt < MAX_RETRIES - 1:
                    wait = 2 ** attempt
                    print(f"   ⚠️  Gemini Quota/Rate-limit — retry {attempt+1} in {wait}s")
                    time.sleep(wait)
                else:
                    return "🔴 Gemini API Quota Exceeded. Please try again in 1 minute or check your Google AI Studio billing."
            elif "401" in err_str or "invalid" in err_str or "api_key_invalid" in err_str:
                return "🔴 Invalid Gemini API Key. Please update your .env file with a valid key from aistudio.google.com."
            else:
                print(f"   ❌ Gemini Error (Attempt {attempt+1}): {e}")
                if attempt == MAX_RETRIES - 1:
                    raise
                time.sleep(1)
    return _fallback


# ── RAG pipeline ───────────────────────────────────────────────────────────────
def generate(query: str, history: list) -> tuple[str, str]:
    """Returns (answer, model_label)."""

    # 1. Small-talk interception (greetings, bye, thanks)
    smalltalk = _handle_smalltalk(query)
    if smalltalk:
        return smalltalk, "AGPC Assistant"

    # 2. No API key — use smart fallback formatter
    if not _gem_fast:
        ctx = retrieve_context(query)
        return _format_no_api_response(query, ctx), "Knowledge Base"

    # 3. Choose model based on complexity
    complex_q   = _is_complex(query)
    model       = _gem_smart if complex_q else _gem_fast
    model_label = "Gemini 2.5 Flash (122025)" if complex_q else "Gemini 2.5 Flash-Lite"

    # 4. Retrieve relevant context chunks
    context = retrieve_context(query)
    system  = _SYS.format(context=context)

    # 5. Build conversation history
    hist_block = ""
    for m in history[-MAX_HISTORY:]:
        role = "User" if m.get("role") == "user" else "Assistant"
        hist_block += f"\n{role}: {m.get('content','')}"

    # 6. Build final prompt
    lang_hint = ""
    detected = _detect_lang(query)
    if detected == 'ta':
        lang_hint = "\n[IMPORTANT: The user wrote in Tamil. You MUST respond entirely in Tamil script (தமிழ்).]"

    prompt = (
        f"{system}{lang_hint}\n\nConversation:{hist_block}\n\nUser: {query}\nAssistant:"
        if hist_block else
        f"{system}{lang_hint}\n\nUser: {query}\nAssistant:"
    )

    # 7. Call Gemini with automatic fallback to smarter model
    try:
        return _call_gemini(model, prompt), model_label
    except Exception as e:
        if not complex_q and _gem_smart:
            print(f"   Fast model failed ({e}), trying smart model…")
            try:
                return _call_gemini(_gem_smart, prompt), "Gemini 2.5 Flash (122025)"
            except Exception:
                return (
                    "I'm temporarily unavailable. Please contact AGPC:\n"
                    "📞 +91 95002 99595\n📧 admission@arasanganesanpoly.edu.in"
                ), "error"
        return "Service temporarily unavailable. Please try again shortly.", "error"


# ── Speech-to-Text (server-side fallback) ─────────────────────────────────────
def transcribe_audio_bytes(audio_bytes: bytes, mime: str = "audio/webm") -> str:
    """
    Try server-side STT using SpeechRecognition library.
    Returns transcribed text or empty string on failure.
    """
    try:
        import speech_recognition as sr
        r = sr.Recognizer()
        # Write to temp WAV/WebM file
        suffix = ".webm" if "webm" in mime else ".wav"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name
        # Try to convert via pydub if available, else attempt directly
        try:
            from pydub import AudioSegment
            audio = AudioSegment.from_file(tmp_path)
            wav_path = tmp_path.replace(suffix, ".wav")
            audio.export(wav_path, format="wav")
            with sr.AudioFile(wav_path) as src:
                audio_data = r.record(src)
            os.unlink(wav_path)
        except Exception:
            # Fallback: try directly
            with sr.AudioFile(tmp_path) as src:
                audio_data = r.record(src)
        os.unlink(tmp_path)
        text = r.recognize_google(audio_data)
        return text
    except Exception as e:
        print(f"   ⚠️  Server-side STT failed: {e}")
        return ""


# ═══════════════════════════════════════════════════════════════════════════════
# Flask App
# ═══════════════════════════════════════════════════════════════════════════════
try:
    from flask import Flask, request, jsonify, Response
except ImportError:
    print("❌  pip install flask"); sys.exit(1)

app = Flask(__name__)
app.secret_key = "agpc_gemini_chatbot_2026"

# ── Embedded HTML/CSS/JS ───────────────────────────────────────────────────────
HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>AGPC AI Assistant — Arasan Ganesan Polytechnic College</title>
<meta name="description" content="Official AI chatbot for Arasan Ganesan Polytechnic College (AGPC), Sivakasi. Ask about admissions, courses, placements, scholarships and more.">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap" rel="stylesheet">
<style>
/* ── Reset & Root ─────────────────────────────────────────────────────────── */
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
:root{
  --bg:#060c1f;
  --bg2:#0a1428;
  --glass:rgba(255,255,255,0.04);
  --glass2:rgba(255,255,255,0.08);
  --glass3:rgba(255,255,255,0.12);
  --border:rgba(255,255,255,0.09);
  --border2:rgba(255,255,255,0.16);
  --blue:#4f9cf9;
  --blue2:#2d7ee0;
  --purple:#8b6ff7;
  --purple2:#6c4fd4;
  --pink:#f472b6;
  --green:#34d399;
  --amber:#fbbf24;
  --red:#f87171;
  --text:#e8eaf6;
  --text2:#b0b8d4;
  --muted:#6b7498;
  --r:18px;
  --r2:12px;
  --shadow:0 8px 32px rgba(0,0,0,0.4);
  --shadow2:0 2px 12px rgba(0,0,0,0.3);
}

body{
  font-family:'Outfit',system-ui,sans-serif;
  background:var(--bg);
  color:var(--text);
  height:100dvh;
  display:flex;
  flex-direction:column;
  overflow:hidden;
  position:relative;
}

/* ── Animated background ────────────────────────────────────────────────────── */
.bg-canvas{
  position:fixed;inset:0;z-index:0;overflow:hidden;pointer-events:none;
}
.orb{
  position:absolute;border-radius:50%;filter:blur(80px);
  animation:orb-drift 20s ease-in-out infinite alternate;
}
.orb1{
  width:600px;height:600px;
  background:radial-gradient(circle,rgba(79,156,249,0.12),transparent 70%);
  top:-200px;left:-200px;animation-delay:0s;
}
.orb2{
  width:500px;height:500px;
  background:radial-gradient(circle,rgba(139,111,247,0.1),transparent 70%);
  bottom:-150px;right:-150px;animation-delay:-7s;
}
.orb3{
  width:350px;height:350px;
  background:radial-gradient(circle,rgba(52,211,153,0.07),transparent 70%);
  top:40%;left:40%;transform:translate(-50%,-50%);animation-delay:-14s;
}
@keyframes orb-drift{
  0%{transform:translate(0,0) scale(1)}
  100%{transform:translate(40px,30px) scale(1.1)}
}
/* Particle dots */
.particles{position:fixed;inset:0;z-index:0;pointer-events:none}
.particle{
  position:absolute;width:2px;height:2px;border-radius:50%;
  background:rgba(255,255,255,0.3);
  animation:float-up linear infinite;
}
@keyframes float-up{
  0%{transform:translateY(100vh) scale(0);opacity:0}
  10%{opacity:1}
  90%{opacity:0.5}
  100%{transform:translateY(-10vh) scale(1);opacity:0}
}

/* ── Grid lines overlay ──────────────────────────────────────────────────────── */
.grid-overlay{
  position:fixed;inset:0;z-index:0;pointer-events:none;
  background-image:
    linear-gradient(rgba(79,156,249,0.025) 1px,transparent 1px),
    linear-gradient(90deg,rgba(79,156,249,0.025) 1px,transparent 1px);
  background-size:50px 50px;
}

/* ── Header ─────────────────────────────────────────────────────────────────── */
header{
  position:relative;z-index:100;
  background:rgba(6,12,31,0.85);
  backdrop-filter:blur(24px) saturate(1.5);
  -webkit-backdrop-filter:blur(24px) saturate(1.5);
  border-bottom:1px solid var(--border);
  padding:10px 20px;
  display:flex;align-items:center;gap:14px;
  box-shadow:0 1px 24px rgba(0,0,0,0.3);
}
.watermark{
  position:absolute;bottom:75px;right:20px;z-index:50;
  font-size:11px;color:rgba(255,255,255,0.25);
  display:flex;align-items:center;gap:6px;
  text-decoration:none;transition:color 0.2s;
  pointer-events:auto;
}
.watermark:hover{color:var(--blue)}
.watermark i{font-style:normal;opacity:0.6}
.logo-wrap{
  width:44px;height:44px;border-radius:14px;flex-shrink:0;
  background:linear-gradient(135deg,var(--blue),var(--purple));
  display:flex;align-items:center;justify-content:center;
  font-size:22px;
  box-shadow:0 4px 16px rgba(79,156,249,0.35);
  position:relative;overflow:hidden;
}
.logo-wrap::after{
  content:'';position:absolute;inset:0;
  background:linear-gradient(135deg,rgba(255,255,255,0.2),transparent);
}
.htxt h1{font-size:15px;font-weight:600;letter-spacing:0.02em}
.htxt p{font-size:11px;color:var(--muted);margin-top:1px;letter-spacing:0.01em}
.hright{margin-left:auto;display:flex;align-items:center;gap:10px}
.model-badge{
  font-size:10px;padding:4px 10px;border-radius:20px;
  background:rgba(79,156,249,0.1);
  border:1px solid rgba(79,156,249,0.25);
  color:var(--blue);font-weight:500;
  transition:all 0.3s ease;
  cursor:default;
}
.status-dot{
  width:8px;height:8px;border-radius:50%;
  background:var(--green);
  box-shadow:0 0 8px var(--green);
  animation:pulse-dot 2.5s ease-in-out infinite;
}
@keyframes pulse-dot{
  0%,100%{opacity:1;box-shadow:0 0 8px var(--green)}
  50%{opacity:0.4;box-shadow:0 0 2px var(--green)}
}

/* ── Chat area ───────────────────────────────────────────────────────────────── */
#chat{
  flex:1;overflow-y:auto;
  padding:24px 16px 12px;
  position:relative;z-index:1;
  scroll-behavior:smooth;
}
#chat::-webkit-scrollbar{width:4px}
#chat::-webkit-scrollbar-track{background:transparent}
#chat::-webkit-scrollbar-thumb{background:var(--border2);border-radius:4px}
#chat::-webkit-scrollbar-thumb:hover{background:var(--glass3)}

/* ── Welcome card ────────────────────────────────────────────────────────────── */
#welcome{
  max-width:500px;margin:16px auto 24px;
  text-align:center;
  background:var(--glass);
  border:1px solid var(--border);
  border-radius:24px;
  padding:32px 24px;
  backdrop-filter:blur(20px);
  -webkit-backdrop-filter:blur(20px);
  animation:fade-in-up 0.5s ease;
  position:relative;overflow:hidden;
}
#welcome::before{
  content:'';position:absolute;inset:0;
  background:linear-gradient(135deg,rgba(79,156,249,0.04),rgba(139,111,247,0.04));
  pointer-events:none;
}
.welcome-icon{
  font-size:52px;
  animation:bounce-in 0.6s cubic-bezier(0.34,1.56,0.64,1);
  display:block;margin-bottom:14px;
}
@keyframes bounce-in{
  0%{transform:scale(0) rotate(-10deg)}
  100%{transform:scale(1) rotate(0)}
}
#welcome h2{font-size:20px;font-weight:700;
  background:linear-gradient(135deg,var(--blue),var(--purple));
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;
  margin-bottom:8px;
}
#welcome p{font-size:13px;color:var(--text2);line-height:1.7}
.welcome-stats{
  display:flex;gap:12px;justify-content:center;margin-top:20px;flex-wrap:wrap;
}
.ws-item{
  padding:8px 14px;border-radius:12px;
  background:var(--glass2);border:1px solid var(--border);
  font-size:11px;color:var(--text2);
  display:flex;align-items:center;gap:6px;
}
.ws-item .ws-icon{font-size:14px}
.ws-item strong{color:var(--text);font-weight:600}

/* ── Messages ────────────────────────────────────────────────────────────────── */
.msg-wrap{
  max-width:760px;margin:0 auto 12px;
  display:flex;gap:10px;
  animation:fade-in-up 0.25s ease;
}
@keyframes fade-in-up{
  from{opacity:0;transform:translateY(10px)}
  to{opacity:1;transform:translateY(0)}
}
.msg-wrap.user{flex-direction:row-reverse}

.av{
  width:34px;height:34px;border-radius:11px;flex-shrink:0;
  display:flex;align-items:center;justify-content:center;font-size:16px;
}
.msg-wrap.bot .av{
  background:var(--glass2);border:1px solid var(--border);
}
.msg-wrap.user .av{
  background:linear-gradient(135deg,var(--blue),var(--purple));
  box-shadow:0 3px 12px rgba(79,156,249,0.3);
}

.bubble{
  max-width:74%;
  padding:13px 17px;
  border-radius:var(--r);
  font-size:14px;line-height:1.75;
  position:relative;
}
.msg-wrap.bot .bubble{
  background:var(--glass);
  border:1px solid var(--border);
  border-top-left-radius:4px;
  backdrop-filter:blur(12px);
  -webkit-backdrop-filter:blur(12px);
}
.msg-wrap.user .bubble{
  background:linear-gradient(135deg,var(--blue),var(--purple));
  border-top-right-radius:4px;
  box-shadow:0 4px 20px rgba(79,156,249,0.25);
  color:#fff;
}

/* Markdown-like styling inside bubbles */
.bubble strong{font-weight:600;color:var(--blue)}
.msg-wrap.user .bubble strong{color:rgba(255,255,255,0.95)}
.bubble em{font-style:italic;color:var(--text2)}
.bubble ul,.bubble ol{margin:8px 0 4px 20px}
.bubble li{margin-bottom:4px}

.bubble-meta{
  display:flex;align-items:center;gap:8px;
  margin-top:8px;
  font-size:10px;color:var(--muted);
}
.model-tag{
  background:rgba(79,156,249,0.1);
  border:1px solid rgba(79,156,249,0.18);
  color:var(--blue);
  padding:2px 7px;border-radius:8px;
  font-size:9.5px;font-weight:500;
}
.tts-btn{
  background:none;border:none;cursor:pointer;
  color:var(--muted);font-size:13px;
  padding:2px 5px;border-radius:6px;
  transition:all 0.2s;
  display:flex;align-items:center;gap:3px;
}
.tts-btn:hover{color:var(--blue);background:rgba(79,156,249,0.1)}
.tts-btn.speaking{color:var(--blue);animation:speak-pulse 0.8s ease-in-out infinite alternate}
@keyframes speak-pulse{from{opacity:1}to{opacity:0.4}}

/* Typing indicator */
.typing-bubble{
  max-width:120px;padding:14px 18px;
  background:var(--glass);border:1px solid var(--border);
  border-radius:var(--r);border-top-left-radius:4px;
  backdrop-filter:blur(12px);
}
.dots{display:flex;gap:5px;align-items:center}
.dot-item{
  width:7px;height:7px;border-radius:50%;
  background:var(--muted);
  animation:bounce-dot 1.3s ease-in-out infinite;
}
.dot-item:nth-child(2){animation-delay:0.15s}
.dot-item:nth-child(3){animation-delay:0.3s}
@keyframes bounce-dot{
  0%,100%{transform:translateY(0);background:var(--muted)}
  50%{transform:translateY(-6px);background:var(--blue)}
}

/* ── Suggestion chips ────────────────────────────────────────────────────────── */
#chips{
  padding:4px 16px 10px;
  max-width:792px;margin:0 auto;
  display:flex;flex-wrap:wrap;gap:7px;
  position:relative;z-index:1;
}
.chip{
  padding:6px 14px;border-radius:20px;font-size:12px;
  cursor:pointer;font-family:inherit;
  background:var(--glass);border:1px solid var(--border);
  color:var(--text2);
  transition:all 0.2s ease;
  user-select:none;
  white-space:nowrap;
}
.chip:hover{
  background:var(--glass2);color:var(--text);
  border-color:rgba(79,156,249,0.4);
  box-shadow:0 2px 12px rgba(79,156,249,0.15);
  transform:translateY(-1px);
}
.chip:active{transform:translateY(0)}

/* ── Footer ─────────────────────────────────────────────────────────────────── */
#foot{
  position:relative;z-index:10;
  background:rgba(6,12,31,0.88);
  backdrop-filter:blur(24px) saturate(1.4);
  -webkit-backdrop-filter:blur(24px) saturate(1.4);
  border-top:1px solid var(--border);
  padding:10px 16px 14px;
}
.input-row{
  max-width:760px;margin:0 auto;
  display:flex;align-items:flex-end;gap:9px;
}

/* Voice status bar */
#voice-status{
  max-width:760px;margin:0 auto 8px;
  height:28px;display:flex;align-items:center;gap:8px;
  font-size:12px;color:var(--text2);padding:0 2px;
  opacity:0;transition:opacity 0.3s;
}
#voice-status.visible{opacity:1}
.vs-dot{
  width:8px;height:8px;border-radius:50%;background:var(--red);flex-shrink:0;
}
.vs-bars{display:flex;gap:2px;align-items:center;flex-shrink:0}
.bar{
  width:3px;border-radius:2px;background:var(--blue);
  animation:bar-dance 0.6s ease-in-out infinite alternate;
}
.bar:nth-child(1){height:8px;animation-delay:0s}
.bar:nth-child(2){height:14px;animation-delay:0.1s}
.bar:nth-child(3){height:10px;animation-delay:0.2s}
.bar:nth-child(4){height:16px;animation-delay:0.3s}
.bar:nth-child(5){height:8px;animation-delay:0.15s}
@keyframes bar-dance{
  0%{transform:scaleY(0.4)}100%{transform:scaleY(1)}
}

/* Textarea */
#inp{
  flex:1;
  background:var(--glass2);
  border:1px solid var(--border);
  color:var(--text);
  border-radius:var(--r2);
  padding:12px 16px;
  font:14px/1.55 'Outfit',inherit;
  resize:none;
  min-height:46px;max-height:140px;
  outline:none;
  transition:border-color 0.2s,box-shadow 0.2s;
}
#inp:focus{
  border-color:rgba(79,156,249,0.5);
  box-shadow:0 0 0 3px rgba(79,156,249,0.08);
}
#inp::placeholder{color:var(--muted)}

/* Mic button */
#mic-btn{
  width:46px;height:46px;border-radius:13px;flex-shrink:0;
  background:var(--glass2);
  border:1px solid var(--border);
  cursor:pointer;
  display:flex;align-items:center;justify-content:center;
  transition:all 0.2s ease;
  position:relative;overflow:visible;
  -webkit-user-select:none;user-select:none;
  touch-action:none;
}
#mic-btn svg{width:19px;height:19px;fill:var(--muted);transition:fill 0.2s}
#mic-btn:hover{border-color:rgba(79,156,249,0.35);background:var(--glass3)}
#mic-btn:hover svg{fill:var(--blue)}
#mic-btn.recording{
  background:rgba(248,113,113,0.12);
  border-color:var(--red);
  box-shadow:0 0 0 0 rgba(248,113,113,0.4);
  animation:mic-ring 1s ease-out infinite;
}
#mic-btn.recording svg{fill:var(--red)}
@keyframes mic-ring{
  0%{box-shadow:0 0 0 0 rgba(248,113,113,0.5)}
  70%{box-shadow:0 0 0 12px rgba(248,113,113,0)}
  100%{box-shadow:0 0 0 0 rgba(248,113,113,0)}
}

/* Send button */
#sbtn{
  width:46px;height:46px;border-radius:13px;flex-shrink:0;
  background:linear-gradient(135deg,var(--blue),var(--purple));
  border:none;cursor:pointer;
  display:flex;align-items:center;justify-content:center;
  transition:all 0.2s ease;
  box-shadow:0 4px 16px rgba(79,156,249,0.3);
}
#sbtn:hover{transform:translateY(-1px);box-shadow:0 6px 20px rgba(79,156,249,0.4)}
#sbtn:active{transform:scale(0.94)}
#sbtn:disabled{opacity:0.35;cursor:default;transform:none;box-shadow:none}
#sbtn svg{width:19px;height:19px;fill:#fff}
.foot-bottom{
  max-width:760px;margin:4px auto 0;
  display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:6px;
}
.foot-hint{
  font-size:10px;color:var(--muted);
}
.lang-row{
  display:flex;align-items:center;gap:6px;
}
.lang-label{font-size:12px;}
#lang-sel{
  background:var(--glass2);border:1px solid var(--border);
  color:var(--text2);border-radius:8px;padding:3px 8px;
  font:11px/1.4 'Outfit',inherit;cursor:pointer;outline:none;
  transition:border-color .2s;
}
#lang-sel:hover{border-color:rgba(79,156,249,.4);color:var(--text);}

/* ── Responsive ──────────────────────────────────────────────────────────────── */
@media(max-width:540px){
  .bubble{max-width:86%}
  #welcome{padding:22px 16px;margin:10px auto 16px}
  .welcome-stats{gap:8px}
  .ws-item{padding:6px 10px;font-size:10px}
}
</style>
</head>
<body>

<!-- Animated background -->
<div class="bg-canvas">
  <div class="orb orb1"></div>
  <div class="orb orb2"></div>
  <div class="orb orb3"></div>
</div>
<div class="grid-overlay"></div>
<div class="particles" id="particles"></div>

<!-- Header -->
<header>
  <div class="logo-wrap">🎓</div>
  <div class="htxt">
    <h1>AGPC AI Assistant</h1>
    <p>Arasan Ganesan Polytechnic College · Sivakasi</p>
  </div>
  <div class="hright">
    <span class="model-badge" id="badge">Gemini 2.5</span>
    <div class="status-dot"></div>
  </div>
</header>

<!-- Chat -->
<div id="chat">
  <div id="welcome">
    <span class="welcome-icon">🏫</span>
    <h2>Welcome to AGPC Chatbot!</h2>
    <p>Your intelligent assistant for admissions, courses, placements,
       scholarships, and everything about AGPC. Ask me anything —
       by typing or by pressing the 🎤 mic button!</p>
    <div class="welcome-stats">
      <div class="ws-item"><span class="ws-icon">📚</span><strong>6</strong> Diploma Courses</div>
      <div class="ws-item"><span class="ws-icon">🏭</span><strong>300+</strong> Industry Ties</div>
      <div class="ws-item"><span class="ws-icon">🏆</span><strong>45</strong> Years of Excellence</div>
      <div class="ws-item"><span class="ws-icon">🎯</span><strong>100%</strong> Placement Support</div>
    </div>
  </div>
</div>

<!-- Suggestion chips -->
<div id="chips">
  <span class="chip" onclick="ask('How to apply for admission 2026?')">📋 Admissions</span>
  <span class="chip" onclick="ask('What diploma courses does AGPC offer?')">📚 All Courses</span>
  <span class="chip" onclick="ask('What is the placement record at AGPC?')">💼 Placements</span>
  <span class="chip" onclick="ask('Explain the Printing Technology department')">🖨️ Printing Tech</span>
  <span class="chip" onclick="ask('What scholarships are available at AGPC?')">🎓 Scholarships</span>
  <span class="chip" onclick="ask('Tell me about hostel and transport facilities')">🏛️ Facilities</span>
  <span class="chip" onclick="ask('What clubs and extracurricular activities does AGPC have?')">🏅 Clubs & NCC</span>
  <span class="chip" onclick="ask('What are the eligibility criteria for lateral entry?')">🔁 Lateral Entry</span>
  <span class="chip" onclick="ask('AGPC phone number and address')">📞 Contact</span>
</div>

<!-- Footer -->
<div id="foot">
  <div id="voice-status">
    <div class="vs-dot" id="vs-dot"></div>
    <div class="vs-bars" id="vs-bars" style="display:none">
      <div class="bar"></div><div class="bar"></div><div class="bar"></div>
      <div class="bar"></div><div class="bar"></div>
    </div>
    <span id="vs-text">Listening…</span>
  </div>
  <div class="input-row">
    <textarea id="inp" placeholder="Ask anything about AGPC… or click 🎤 to speak" rows="1"></textarea>
    <button id="mic-btn" title="Click to speak — click again to stop" aria-label="Click to record voice">
      <svg viewBox="0 0 24 24">
        <path d="M12 1a4 4 0 0 1 4 4v6a4 4 0 0 1-8 0V5a4 4 0 0 1 4-4zm0 2a2 2 0 0 0-2 2v6a2 2 0 0 0 4 0V5a2 2 0 0 0-2-2zm-1 14.93A8 8 0 0 1 4 11H2a10 10 0 0 0 9 9.94V23h2v-2.06A10 10 0 0 0 22 11h-2a8 8 0 0 1-7 7.93z"/>
      </svg>
    </button>
    <button id="sbtn" onclick="sendChat()" title="Send" aria-label="Send message">
      <svg viewBox="0 0 24 24"><path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z"/></svg>
    </button>
  </div>
  <div class="foot-bottom">
    <p class="foot-hint">🎤 Click mic to speak &bull; AI may make mistakes &mdash; verify important info</p>
    <div class="lang-row">
      <span class="lang-label">🌐</span>
      <select id="lang-sel" title="Voice input language">
        <option value="en-IN">English (India)</option>
        <option value="ta-IN">தமிழ் (Tamil)</option>
        <option value="hi-IN">हिन्दी (Hindi)</option>
        <option value="en-US">English (US)</option>
      </select>
    </div>
    <a href="https://www.instagram.com/sudharsan_007_?igsh=ZDF0a2c5bm8yaGwx" target="_blank" class="watermark">
      <i>Created BY</i> <strong>Sudharsan</strong>
    </a>
  </div>
</div>


<script>
// ===== STATE =====
const chatEl=document.getElementById('chat'),inp=document.getElementById('inp'),
      sbtn=document.getElementById('sbtn'),micBtn=document.getElementById('mic-btn'),
      badge=document.getElementById('badge'),voiceStatus=document.getElementById('voice-status'),
      vsText=document.getElementById('vs-text'),vsBars=document.getElementById('vs-bars'),
      vsDot=document.getElementById('vs-dot'),langSel=document.getElementById('lang-sel');

let welcome=document.getElementById('welcome'),chatHistory=[],
    isRecording=false,currentAudio=null,currentTtsBtn=null;

// ===== PARTICLES =====
(function(){
  var c=document.getElementById('particles');
  for(var i=0;i<20;i++){
    var p=document.createElement('div');p.className='particle';
    var s=Math.random()*3+1;
    p.style.cssText=`left:${Math.random()*100}%;width:${s}px;height:${s}px;animation-duration:${Math.random()*18+14}s;animation-delay:-${Math.random()*20}s;opacity:${Math.random()*0.35+0.08};`;
    c.appendChild(p);
  }
})();

// ===== MARKDOWN =====
function renderMd(text){
  var t=text.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
  t=t.replace(/\*\*(.*?)\*\*/gs,'<strong>$1</strong>');
  t=t.replace(/\*(.*?)\*/gs,'<em>$1</em>');
  t=t.replace(/`([^`]+)`/g,'<code style="background:rgba(255,255,255,.08);padding:1px 5px;border-radius:4px;font-family:monospace;font-size:12px">$1</code>');
  var lines=t.split('\n'),out=[],inUl=false,inOl=false;
  for(var i=0;i<lines.length;i++){
    var line=lines[i];
    var bul=line.match(/^[-\u2022*]\s+(.*)/);
    var num=line.match(/^(\d+)\.\s+(.*)/);
    if(bul){
      if(!inUl){out.push('<ul style="margin:8px 0 4px 20px;list-style:disc">');inUl=true;}
      if(inOl){out.push('</ol>');inOl=false;}
      out.push('<li style="margin-bottom:3px">'+bul[1]+'</li>');
    } else if(num){
      if(inUl){out.push('</ul>');inUl=false;}
      if(!inOl){out.push('<ol style="margin:8px 0 4px 20px">');inOl=true;}
      out.push('<li style="margin-bottom:3px">'+num[2]+'</li>');
    } else {
      if(inUl){out.push('</ul>');inUl=false;}
      if(inOl){out.push('</ol>');inOl=false;}
      out.push(line===''?'<br>':line+'<br>');
    }
  }
  if(inUl)out.push('</ul>');
  if(inOl)out.push('</ol>');
  return out.join('');
}

// ===== MESSAGES =====
var SPEAK_SVG='<svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor"><path d="M3 9v6h4l5 5V4L7 9H3zm13.5 3c0-1.77-1.02-3.29-2.5-4.03v8.06c1.48-.74 2.5-2.26 2.5-4.03zM18.5 12c0-2.77-1.59-5.15-4-6.26v2.18c1.5.74 2.5 2.28 2.5 4.08s-1 3.34-2.5 4.08v2.18c2.41-1.11 4-3.49 4-6.26z"/></svg>';
var STOP_SVG='<svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor"><path d="M6 6h12v12H6z"/></svg>';

function addMsg(text,role,model,isVoice){
  if(welcome){welcome.remove();welcome=null;}
  var chips=document.getElementById('chips');
  if(chips)chips.style.display='none';
  var wrap=document.createElement('div');wrap.className='msg-wrap '+role;
  var av=document.createElement('div');av.className='av';av.textContent=role==='bot'?'🎓':'👤';
  var bub=document.createElement('div');bub.className='bubble';bub.innerHTML=renderMd(text);
  if(role==='bot'){
    var meta=document.createElement('div');meta.className='bubble-meta';
    if(model&&model!=='AGPC Assistant'&&model!=='Knowledge Base'){
      var tag=document.createElement('span');tag.className='model-tag';tag.textContent=model;meta.appendChild(tag);
    }
    var ttsBtn=document.createElement('button');
    ttsBtn.className='tts-btn';ttsBtn.title='Read aloud with Gemini Native Voice';
    ttsBtn.innerHTML=SPEAK_SVG+' Speak';
    ttsBtn.onclick=function(){speakText(text,ttsBtn);};
    meta.appendChild(ttsBtn);bub.appendChild(meta);
    if(isVoice)setTimeout(function(){speakText(text,ttsBtn);},500);
  }
  wrap.appendChild(av);wrap.appendChild(bub);chatEl.appendChild(wrap);chatEl.scrollTop=chatEl.scrollHeight;
}

function showTyping(){
  if(welcome){welcome.remove();welcome=null;}
  var w=document.createElement('div');w.className='msg-wrap bot';w.id='typing';
  w.innerHTML='<div class="av">🎓</div><div class="typing-bubble"><div class="dots"><span class="dot-item"></span><span class="dot-item"></span><span class="dot-item"></span></div></div>';
  chatEl.appendChild(w);chatEl.scrollTop=chatEl.scrollHeight;
}

// ===== GTTS AUDIO STREAMING (NO API LIMITS) =====
let ttsQueue=[];
let isPlayingTts=false;
let isFetchingTts=false;
let currentAudioObj=null;

function stopAudio(){
  ttsQueue=[];
  isPlayingTts=false;
  isFetchingTts=false;
  if(currentAudioObj){
    currentAudioObj.pause();
    currentAudioObj.currentTime=0;
    currentAudioObj=null;
  }
  if(currentTtsBtn){
    currentTtsBtn.innerHTML=SPEAK_SVG+' Speak';
    currentTtsBtn.classList.remove('speaking');
    currentTtsBtn=null;
  }
}

async function playNextUtterance() {
  if (ttsQueue.length === 0 || !isPlayingTts) {
    stopAudio();
    return;
  }
  if (isFetchingTts) return;
  
  isFetchingTts = true;
  let chunk = ttsQueue.shift();
  
  try {
    const langId = langSel.value;
    const res = await fetch('/tts', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({text: chunk, lang: langId})
    });
    
    if(!isPlayingTts) return; 
    
    const data = await res.json();
    if(data.error) throw new Error(data.error);
    
    currentAudioObj = new Audio("data:" + (data.mime || "audio/mpeg") + ";base64," + data.audio_b64);
    currentAudioObj.onended = () => {
        isFetchingTts = false;
        if(isPlayingTts) playNextUtterance();
    };
    currentAudioObj.onerror = () => {
        isFetchingTts = false;
        if(isPlayingTts) playNextUtterance();
    };
    currentAudioObj.play();
  } catch(e) {
    console.error("ElevenLabs Error:", e);
    isFetchingTts = false;
    if(isPlayingTts) playNextUtterance(); // skip chunk & keep playing
  }
}

async function speakText(text,btn){
  if(btn === currentTtsBtn && isPlayingTts){
    stopAudio();
    return;
  }
  stopAudio();
  if(!text) return;
  
  // Safe sentence chunking (limit Google Translate length limit)
  let cleanTxt = text.replace(/[*#`_]+/g, '').replace(/(\r\n|\n|\r)/gm, '. ');
  let sentences = cleanTxt.match(/[^.!?]+[.!?]+/g);
  if (!sentences || sentences.length === 0) sentences = [cleanTxt];
  
  ttsQueue = sentences.map(s => s.trim()).filter(s => s.length > 0);
  if (ttsQueue.length === 0) return;
  
  currentTtsBtn = btn;
  if(btn){
    btn.innerHTML = STOP_SVG+' Stop';
    btn.classList.add('speaking');
  }
  
  isPlayingTts = true;
  playNextUtterance();
}

// ===== SEND =====
function ask(pre){sendQuery((pre||inp.value).trim(),false);}
function sendChat(){sendQuery(inp.value.trim(),false);}

async function sendQuery(q,isVoice){
  if(!q)return;
  inp.value='';inp.style.height='auto';
  sbtn.disabled=true;micBtn.disabled=true;
  addMsg(q,'user',null,false);
  chatHistory.push({role:'user',content:q});
  showTyping();
  try{
    var res=await fetch('/chat',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({query:q,history:chatHistory,is_voice:isVoice})});
    if(!res.ok)throw new Error('HTTP '+res.status);
    var data=await res.json();
    var el=document.getElementById('typing');if(el)el.remove();
    var answer=data.response||'Something went wrong. Please try again.';
    addMsg(answer,'bot',data.model||'',isVoice);
    chatHistory.push({role:'assistant',content:answer});
    if(data.model)badge.textContent=data.model;
    if(chatHistory.length>30)chatHistory=chatHistory.slice(-30);
  }catch(e){
    var el=document.getElementById('typing');if(el)el.remove();
    addMsg('⚠️ Network error — please check your connection.','bot',null,false);
    console.error(e);
  }
  sbtn.disabled=false;micBtn.disabled=false;
  if(!isVoice)inp.focus();
}

// ===== VOICE STATUS =====
function showVS(state,extra){
  voiceStatus.classList.add('visible');
  if(state==='rec'){
    vsText.textContent=extra||'Listening… speak now';
    vsDot.style.cssText='display:block;background:var(--red);box-shadow:0 0 10px var(--red)';
    vsBars.style.display='none';
  } else {
    vsText.textContent='Processing native audio…';
    vsDot.style.display='none';vsBars.style.display='flex';
  }
}
function hideVS(){voiceStatus.classList.remove('visible');}

// ===== PUSH TO TALK (MIC BTN) =====
let mediaRecorder = null;
let chunks = [];
let audioStream = null;

micBtn.onclick = async function() {
  if (isRecording) {
    // Stop recording
    if (mediaRecorder) { mediaRecorder.stop(); }
    micBtn.classList.remove('recording');
    isRecording = false;
    showVS('wait');
    return;
  }
  
  // Start recording
  stopAudio();
  try {
    audioStream = await navigator.mediaDevices.getUserMedia({ audio: true });
    chunks = [];
    const mt = MediaRecorder.isTypeSupported('audio/webm;codecs=opus') ? 'audio/webm;codecs=opus' : 'audio/webm';
    mediaRecorder = new MediaRecorder(audioStream, { mimeType: mt });
    
    mediaRecorder.ondataavailable = e => { if(e.data.size > 0) chunks.push(e.data); };
    mediaRecorder.onstop = async () => {
      audioStream.getTracks().forEach(t => t.stop());
      const blob = new Blob(chunks, { type: 'audio/webm' });
      const fd = new FormData();
      fd.append('audio', blob, 'voice.webm');
      
      try {
        const tr = await fetch('/transcribe', { method:'POST', body:fd });
        const trData = await tr.json();
        hideVS();
        if(trData.error) throw new Error(trData.error);
        if(trData.text) {
          sendQuery(trData.text, true);
        } else {
          addMsg('Could not understand speech.', 'bot', '', false);
        }
      } catch(e) {
        hideVS();
        addMsg('Audio network error.', 'bot', '', false);
      }
    };
    
    mediaRecorder.start();
    isRecording = true;
    micBtn.classList.add('recording');
    showVS('rec');
  } catch(err) {
    alert("Microphone access denied.");
  }
};

// ===== TEXTAREA =====
inp.addEventListener('input',function(){inp.style.height='auto';inp.style.height=Math.min(inp.scrollHeight,140)+'px';});
inp.addEventListener('keydown',function(e){if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();sendChat();}});
</script>

</body>
</html>"""
# ── Flask Routes ───────────────────────────────────────────────────────────────

@app.before_request
def _init():
    init()

@app.route("/")
def index():
    return HTML

@app.route("/chat", methods=["POST"])
def chat():
    body     = request.get_json(force=True)
    query    = body.get("query", "").strip()
    hist     = body.get("history", [])
    is_voice = body.get("is_voice", False)

    if not query:
        return jsonify({"response": "Please ask a question about AGPC.", "model": ""})

    clean = [m for m in hist
             if isinstance(m, dict)
             and m.get("role") in ("user", "assistant")
             and m.get("content")]
    if clean and clean[-1].get("role") == "user":
        clean = clean[:-1]

    import time as _time
    t0 = _time.time()
    ans, label = generate(query, clean)
    return jsonify({
        "response": ans,
        "model": label,
        "is_voice": is_voice,
        "elapsed_s": round(_time.time() - t0, 2)
    })

@app.route("/voice_chat", methods=["POST"])
def voice_chat():
    import base64
    if "audio" not in request.files:
        return jsonify({"error": "No audio file"}), 400
    
    audio_file  = request.files["audio"]
    audio_bytes = audio_file.read()
    mime        = audio_file.content_type or "audio/webm"
    
    if not _gem_smart or not GOOGLE_API_KEY:
        return jsonify({"error": "API Key not configured."}), 500
        
    try:
        from google.genai import types as gtypes
        client, model_id, _ = _gem_smart
        
        # 1. Transcribe
        audio_part = gtypes.Part.from_bytes(data=audio_bytes, mime_type=mime)
        stt_prompt = "Transcribe the following audio precisely. Only output the transcription text. If silence, type 'silence'."
        stt_resp = client.models.generate_content(
            model=model_id,
            contents=[audio_part, stt_prompt]
        )
        user_text = stt_resp.text.strip()
        if not user_text or user_text.lower() == "silence":
            return jsonify({"text": "", "audio_b64": "", "error": "Silence detected"}), 200
            
        # 2. RAG Generation (Language-adaptive natively)
        ans_text, _ = generate(user_text, [])
        if not ans_text:
            ans_text = "Sorry, I couldn't process that request."
            
        # 3. Audio Synthesis
        t_model = "gemini-2.5-flash-preview-tts"
        cfg = gtypes.GenerateContentConfig(
            response_modalities=["AUDIO"],
            speech_config=gtypes.SpeechConfig(
                voice_config=gtypes.VoiceConfig(
                    prebuilt_voice_config=gtypes.PrebuiltVoiceConfig(voice_name="Aoede")
                )
            )
        )
        tts_resp = client.models.generate_content(
            model=t_model,
            contents=[ans_text],
            config=cfg
        )
        
        b64_audio = ""
        out_mime = ""
        for part in tts_resp.candidates[0].content.parts:
            if part.inline_data:
                # Gemini returns raw PCM; we must wrap in WAV header for AudioContext decoding
                wav_bytes = pcm16_to_wav(part.inline_data.data)
                b64_audio = base64.b64encode(wav_bytes).decode("utf-8")
                out_mime = "audio/wav"
                
        return jsonify({
            "text": ans_text.strip(),
            "audio_b64": b64_audio,
            "mime": out_mime
        })
    except Exception as e:
        print(f"Voice chat pipeline failed: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/transcribe", methods=["POST"])
def transcribe():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file", "text": ""}), 400
    
    audio_file  = request.files["audio"]
    audio_bytes = audio_file.read()
    mime        = audio_file.content_type or "audio/webm"
    
    # ── Gemini 2.5 Native MLLM Audio Transcription ──
    if _gem_smart and GOOGLE_API_KEY:
        try:
            from google.genai import types as gtypes
            client, model_id, base_cfg = _gem_smart
            audio_part = gtypes.Part.from_bytes(data=audio_bytes, mime_type=mime)
            stt_prompt = "Transcribe the following audio precisely in its original language. Output ONLY the transcription text with absolutely no extra commentary."
            resp = client.models.generate_content(
                model=model_id,
                contents=[audio_part, stt_prompt]
            )
            text = resp.text.strip()
            return jsonify({"text": text, "success": bool(text)})
        except Exception as e:
            print(f"Gemini native STT failed: {e}")
            
    # Fallback to local
    text = transcribe_audio_bytes(audio_bytes, mime)
    return jsonify({"text": text, "success": bool(text)})

@app.route("/tts", methods=["POST"])
def tts():
    """Generates Ultra-Realistic TTS using ElevenLabs API"""
    body = request.get_json(force=True)
    text = body.get("text", "").strip()
    
    if not text:
        return jsonify({"error": "No text provided"}), 400
        
    try:
        import requests
        import base64
        
        # ElevenLabs configuration using environment variable
        ELEVEN_KEY = os.getenv("ELEVENLABS_API_KEY")
        if not ELEVEN_KEY:
            # Silently fallback to browser TTS if no key provided
            return jsonify({"error": "No ElevenLabs key configured"}), 401

        url = "https://api.elevenlabs.io/v1/text-to-speech/pNInz6obpgDQGcFmaJgB"
        headers = {
          "Accept": "audio/mpeg",
          "Content-Type": "application/json",
          "xi-api-key": ELEVEN_KEY
        }
        data = {
          "text": text,
          "model_id": "eleven_multilingual_v2",
          "voice_settings": {
              "stability": 0.5,
              "similarity_boost": 0.5
          }
        }
        
        response = requests.post(url, json=data, headers=headers)
        if response.status_code == 401:
            print("❌ ElevenLabs Error: Invalid API Key")
            return jsonify({"error": "Invalid ElevenLabs API Key"}), 401
        elif response.status_code == 429:
            print("⚠️ ElevenLabs Error: Quota Exceeded")
            return jsonify({"error": "ElevenLabs Quota Exceeded"}), 429
        elif response.status_code != 200:
            print(f"ElevenLabs TTS Error: {response.text}")
            return jsonify({"error": "ElevenLabs Service Error"}), 500
            
        audio_b64 = base64.b64encode(response.content).decode("utf-8")
        return jsonify({"audio_b64": audio_b64, "mime": "audio/mpeg"})
    except Exception as e:
        print(f"ElevenLabs Engine failed: {e}")
        return jsonify({"error": "Check your internet or API key"}), 500


@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "chunks": _collection.count() if _collection else 0,
        "models": [MODEL_FAST, MODEL_SMART],
        "api_key": bool(GOOGLE_API_KEY)
    })

if __name__ == "__main__":
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    
    print("\n" + "═"*56)
    print("  AGPC AI Chatbot  —  Google Gemini 2.5 + Voice I/O")
    print("═"*56)
    if not GOOGLE_API_KEY:
        print("\n  ⚠️   GOOGLE_API_KEY not set!")
        print("  Get FREE key: https://aistudio.google.com/apikey")
        print("  export GOOGLE_API_KEY=AIza...    (Linux/Mac)\n")
    else:
        print(f"\n  ✅  Key loaded")
        print(f"  🔵  {MODEL_FAST} (fast)")
        print(f"  🟣  {MODEL_SMART} (smart)")
        print(f"  ��  Voice: Gemini Native Audio Modal (STT + TTS)")
        print(f"  🌍  Languages: English, Tamil, Hindi + more")
    print(f"\n  🌐  http://localhost:{PORT}")
    print("  Ctrl+C to stop\n")
    
    # Run WSGI directly without reloading to avoid port ghosting bugs
    from werkzeug.serving import run_simple
    run_simple("0.0.0.0", PORT, app, use_reloader=False, use_debugger=False)
