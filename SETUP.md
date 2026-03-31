# AGPC AI Chatbot — Setup Guide (Google Gemini Edition)
# Stack: ChromaDB + sentence-transformers + Gemini 2.5 (all FREE)

## Gemini Free Tier Reality (March 2026)

| Model | RPM | Req/Day | Best for |
|---|---|---|---|
| gemini-2.5-flash-lite | 15 | 1,000 | Simple factual Q&A (used by default) |
| gemini-2.5-flash | 10 | 250 | Detailed explanations, complex questions |

The chatbot automatically routes:
- Short / factual → Flash-Lite (saves your 250/day Flash quota)
- Long / complex  → Flash (smarter answers)

> ⚠️ Gemini 2.0 Flash was **deprecated March 3, 2026**. This project uses 2.5 models only.

---

## Step 1 — Get Your Free Google API Key (2 minutes)

1. Go to https://aistudio.google.com/apikey
2. Sign in with your Google account (no credit card needed)
3. Click **"Create API key"**
4. Copy the key (starts with `AIza...`)

---

## Step 2 — Folder Structure

```
agpc_chatbot/
├── agpc_chatbot.json        ← Your main KB (intents + FAQ)
├── agpc_scraped_v2.json     ← Detailed scraped data
├── build_index.py           ← Run ONCE to build the vector index
├── app.py                   ← The chatbot web app (Gemini powered)
└── requirements.txt
```

---

## Step 3 — Install Dependencies

```bash
pip install -r requirements.txt --break-system-packages
```

Installs:
- `flask` — web server
- `sentence-transformers` — local embeddings (no API, no cost)
- `chromadb` — local vector database
- `google-generativeai` — Gemini API SDK

---

## Step 4 — Build the Knowledge Index (Run Once)

```bash
python3 build_index.py
```

Output:
```
🔨  AGPC Chatbot — Building Knowledge Index
📊  Total chunks to index: 120+
✅  Index built! Stored in ./agpc_chroma_db/
```

Only re-run if you update your JSON data files.

---

## Step 5 — Set API Key and Start

**Linux / Mac:**
```bash
export GOOGLE_API_KEY=AIza_your_key_here
python3 app.py
```

**Windows CMD:**
```cmd
set GOOGLE_API_KEY=AIza_your_key_here
python app.py
```

**Windows PowerShell:**
```powershell
$env:GOOGLE_API_KEY="AIza_your_key_here"
python app.py
```

---

## Step 6 — Open the Chatbot

Go to **http://localhost:5000**

You'll see a dark glassmorphism chat UI with quick-action chips.
The model badge in the top-right shows which Gemini model answered.

---

## How Smart Routing Works

Every question is automatically classified:

```
"What is the phone number?"       → Flash-Lite (fast, 1000/day)
"Explain the Printing Technology department" → Flash (smart, 250/day)
"How to apply for admission?"     → Flash (eligibility = complex)
"Is hostel available?"            → Flash-Lite (short/factual)
```

If Flash-Lite hits its rate limit, the app automatically falls back to Flash.
If Flash also hits limits, it retries with exponential back-off (1s → 2s → 4s).

---

## Managing Your Daily Quota

1000 req/day (Flash-Lite) is very generous for a college chatbot.
A realistic busy day might be 50-200 student queries.

To stretch your quota further:
- Keep the chatbot running only during college hours
- Add response caching for repeated common questions (optional upgrade)

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `ModuleNotFoundError` | `pip install -r requirements.txt --break-system-packages` |
| `ChromaDB index not found` | Run `python3 build_index.py` first |
| `429 Resource Exhausted` | Quota hit — app retries automatically; wait 1 minute |
| `Invalid API key` | Check your `GOOGLE_API_KEY` env variable |
| Slow first start | Normal — embedding model loads (~5s) then stays fast |
| Port busy | Change `PORT = 5000` in `app.py` |

---

## Architecture

```
User Question
    │
    ▼
sentence-transformers (local, free)   ← converts question to vector
    │
    ▼
ChromaDB (local, free)                ← finds top-5 relevant knowledge chunks
    │
    ▼
Smart Router                          ← Flash-Lite or Flash based on complexity
    │
    ▼
Google Gemini 2.5 (free API)          ← reads chunks, generates natural answer
    │
    ▼
Flask Web UI                          ← streams answer to user
```

---

## Files Summary

| File | Purpose | Run? |
|------|---------|------|
| `build_index.py` | Index your JSON into ChromaDB | Once |
| `app.py` | Flask chatbot app (Gemini powered) | Every time |
| `requirements.txt` | Python dependencies | Once |
| `agpc_chatbot.json` | Main knowledge base | Don't edit |
| `agpc_scraped_v2.json` | Detailed scraped data | Don't edit |
| `agpc_chroma_db/` | Auto-created vector DB folder | Don't touch |
