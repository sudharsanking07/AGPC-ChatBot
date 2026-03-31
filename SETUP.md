# 🚀 AGPC AI Chatbot — Quick Setup Guide

Welcome! This chatbot is designed for **Arasan Ganesan Polytechnic College (AGPC)**, Sivakasi. It uses **Google Gemini 2.5** and a local **ChromaDB** knowledge base to provide expert answers about admissions, courses, and more.

---

## ⚡ Option 1 — The One-Click Launcher (Recommended)

This project includes a **Secure Automated Launcher** that handles everything: creating a virtual environment, installing dependencies, building the knowledge index, and prompting you for API keys.

1.  **Clone the Repo**:
    ```bash
    git clone https://github.com/sudharsanking07/AGPC-ChatBot.git
    cd AGPC-ChatBot
    ```

2.  **Run the Launcher**:
    ```bash
    # Linux / Mac
    python3 launcher.py

    # Windows (CMD)
    python launcher.py
    ```

3.  **Follow the Prompts**:
    *   It will automatically install AI libraries.
    *   It will ask for your **Gemini API Key** (starts with `AIza...`). Get one for free at [aistudio.google.com/apikey](https://aistudio.google.com/apikey).
    *   Once setup is complete, it will open the chatbot in your web browser at **http://localhost:5000**.

---

## 🛠️ Option 2 — Manual Installation (Step-by-Step)

If you prefer to set things up manually, follow these steps:

### 1. Get Your Free Google API Key
1. Go to [aistudio.google.com/apikey](https://aistudio.google.com/apikey).
2. Create an API key and copy it.

### 2. Install Dependencies
It's recommended to use a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

pip install -r requirements.txt
```

### 3. Build the Knowledge Index (Run ONCE)
This converts the JSON data into a searchable vector database:
```bash
python3 build_index.py
```

### 4. Start the Server
```bash
export GOOGLE_API_KEY=your_key_here
python3 app.py
```

---

## 🎙️ Voice & Multilingual Features

*   **Multi-language**: Type or speak in **Tamil (தமிழ்)** or **Hindi**—the bot will detect it and reply in the same language.
*   **Voice Control**: Click the **Mic** button to speak. It uses the browser's native Speech-to-Text for speed.
*   **Realistic Voice (Optional)**: If you provide an **ElevenLabs API Key** in the launcher or `.env`, the bot will gain a premium human-like voice. Otherwise, it uses the standard browser voice.

---

## 📂 File Structure

| File | Purpose |
|:---|:---|
| `launcher.py` | **Start here!** Automates setup and execution. |
| `app.py` | The main Flask web application and AI engine. |
| `build_index.py` | Processes college data into the ChromaDB vector database. |
| `agpc_chatbot.json` | The core "Source of Truth" knowledge base. |
| `requirements.txt` | List of AI and web libraries required. |

---

## 🧭 Troubleshooting

| Issue | Solution |
|:---|:---|
| **Port 5000 Busy** | The launcher will try to "kill" the old process for you. If it fails, restart the terminal. |
| **Quota Hit (429)** | You are on the free tier. Wait 1 minute and try again. |
| **Invalid API Key** | Ensure you copied the full key from AI Studio without extra spaces. |
| **Slow Startup** | The first time you run it, it downloads a small Embedding model (~80MB). Subsequent starts are instant. |

---

**Need Help?**
Contact AGPC Admission Cell: **+91 95002 99595** or email **admission@arasanganesanpoly.edu.in**.
