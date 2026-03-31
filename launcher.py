#!/usr/bin/env python3
import os
import subprocess
import webbrowser
import time
import sys
import signal
import socket

# 1. Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VENV_DIR = os.path.join(BASE_DIR, "venv")
VENV_PYTHON = os.path.join(VENV_DIR, "bin", "python3")
VENV_PIP    = os.path.join(VENV_DIR, "bin", "pip")
APP_PY      = os.path.join(BASE_DIR, "app.py")
BUILD_PY    = os.path.join(BASE_DIR, "build_index.py")
REQ_FILE    = os.path.join(BASE_DIR, "requirements.txt")
DB_PATH     = os.path.join(BASE_DIR, "agpc_chroma_db")
PORT        = 5000

def check_port(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

def kill_on_port(port):
    try:
        # Use fuser to find and kill process on port
        subprocess.run(["fuser", "-k", f"{port}/tcp"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass

def setup_environment():
    """Automated One-Time Setup: venv -> pip install -> build index"""
    
    # Check if venv exists
    if not os.path.exists(VENV_DIR):
        print("🛠️  First-time setup: Creating Python environment (venv)...")
        try:
            subprocess.run([sys.executable, "-m", "venv", "venv"], cwd=BASE_DIR, check=True)
            print("   ✅ Environment created.")
        except Exception as e:
            print(f"❌  Error creating venv: {e}")
            sys.exit(1)

    # Check for requirements
    if os.path.exists(REQ_FILE):
        # We use a sentinel file inside the venv to track if installation is complete
        sentinel = os.path.join(VENV_DIR, ".setup_done")
        if not os.path.exists(sentinel):
            print("📦  Installing required AI libraries (this may take 1-2 minutes)...")
            try:
                # Upgrade pip first
                subprocess.run([VENV_PYTHON, "-m", "pip", "install", "--upgrade", "pip"], check=True, stdout=subprocess.DEVNULL)
                # Install requirements
                subprocess.run([VENV_PYTHON, "-m", "pip", "install", "-r", REQ_FILE], check=True)
                # Mark as setup complete
                with open(sentinel, "w") as f: f.write("done")
                print("   ✅ AI Libraries installed.")
            except Exception as e:
                print(f"❌  Error installing dependencies: {e}")
                sys.exit(1)
    
    # Check if knowledge base is indexed
    if not os.path.exists(DB_PATH):
        if os.path.exists(BUILD_PY):
            print("🔮  First-time run: Building Knowledge Index (ChromaDB)...")
            try:
                subprocess.run([VENV_PYTHON, BUILD_PY], cwd=BASE_DIR, check=True)
                print("   ✅ Knowledge Index ready.")
            except Exception as e:
                print(f"⚠️  Error building index: {e}")
                print("   (The app may still work but with limited facts)")
    else:
        print("✅  Knowledge system found.")

def check_api_keys():
    """Check for .env file and prompt for keys if missing"""
    env_path = os.path.join(BASE_DIR, ".env")
    keys = {"GOOGLE_API_KEY": "", "ELEVENLABS_API_KEY": ""}
    
    if os.path.exists(env_path):
        with open(env_path, "r") as f:
            for line in f:
                if "=" in line:
                    k, v = line.strip().split("=", 1)
                    if k in keys: keys[k] = v.strip('"').strip("'")

    needs_save = False
    
    # 1. Gemini Key (Required)
    if not keys["GOOGLE_API_KEY"]:
        print("\n🔑  Gemini API Key Required!")
        print("   Get one for FREE at: https://aistudio.google.com/apikey")
        val = input("   Enter Gemini API Key: ").strip()
        if val:
            keys["GOOGLE_API_KEY"] = val
            needs_save = True
        else:
            print("❌  Gemini API Key is required to run the chatbot.")
            sys.exit(1)

    # 2. ElevenLabs Key (Optional)
    if not keys["ELEVENLABS_API_KEY"]:
        print("\n🎙️  ElevenLabs API Key (Optional for Ultra-Realistic Voice)")
        print("   Get one at: https://elevenlabs.io (or press Enter to skip)")
        val = input("   Enter ElevenLabs Key (Skip if unsure): ").strip()
        if val:
            keys["ELEVENLABS_API_KEY"] = val
            needs_save = True

    if needs_save:
        with open(env_path, "w") as f:
            for k, v in keys.items():
                if v: f.write(f"{k}={v}\n")
        print("   ✅ API Keys saved to .env")

    # Load into environment
    os.environ.update({k: v for k, v in keys.items() if v})

print("\n" + "═"*60)
print("  AGPC AI Chatbot  —  Secure Automated Launcher")
print("═"*60 + "\n")

# 2. Automated setup
setup_environment()
check_api_keys()

# 3. Port Cleanup (Ensure fresh start)
if check_port(PORT):
    print(f"⚠️  Port {PORT} is already in use. Cleaning up...")
    kill_on_port(PORT)
    time.sleep(1)

# 4. Starting the Server
print("🎓  Initializing AI Knowledge Base & BERT Models...")
try:
    server_process = subprocess.Popen(
        [VENV_PYTHON, APP_PY],
        cwd=BASE_DIR,
        preexec_fn=os.setsid # Allow group-killing
    )
except Exception as e:
    print(f"❌  Critical Error: Could not start app.py\n    Details: {e}")
    sys.exit(1)

# 5. Wait for server (Polling instead of fixed sleep)
print("⌛  Starting AI engine...")
start_time = time.time()
max_wait = 60 # Increased wait for slow embeddings loading
is_ready = False

while time.time() - start_time < max_wait:
    if check_port(PORT):
        print(f"✅  Server is ready (started in {int(time.time() - start_time)}s)!")
        is_ready = True
        break
    # Check if the process died early
    if server_process.poll() is not None:
        print("❌  AI server crashed during startup.")
        sys.exit(1)
    time.sleep(1)

if not is_ready:
    print("❌  Server failed to respond within 60 seconds.")
    print("   Please check for errors in the terminal above.")
    sys.exit(1)

# 6. Open Browser
url = f"http://localhost:{PORT}"
print(f"🌐  Opening Chatbot UI at {url}...")
try:
    time.sleep(1.5)
    webbrowser.open(url)
except Exception:
    print(f"⚠️  Could not open browser automatically. Visit {url} manually.")

print("\n" + "═"*60)
print("   🚀  AGPC AI CHATBOT IS NOW ACTIVE!")
print("   💡  Keep this terminal open while using the chat.")
print("   💡  Press Ctrl+C to shut down safely.")
print("═"*60 + "\n")

try:
    # Keep the launcher script alive to keep the server alive
    server_process.wait()
except KeyboardInterrupt:
    print("\n🛑  Shutting down AI system safely...")
    try:
        os.killpg(os.getpgid(server_process.pid), signal.SIGTERM)
    except Exception:
        pass
    print("👋  Goodbye!")
    time.sleep(1)

