# Ollama-GUI-Voice-Assistant
Local Voice Assistant with Whisper ASR &amp; Ollama LLM — A cross-platform Python desktop app with real-time microphone input, volume meter, and model selectors. Records speech, transcribes it locally using Whisper (supports faster-whisper for GPU acceleration), and sends the text to an Ollama large language model for instant responses.
# 🎙️ Local Voice Assistant (Whisper + Ollama)

A local desktop voice assistant built with Python, integrating **Whisper** for speech-to-text transcription and **Ollama** for local large language model responses.  
**Runs fully offline** — no cloud dependencies, ensuring privacy.

This project provides two versions:
1. **Chinese UI Version**  
   - UI in Chinese  
   - Whisper fixed with `language="zh"`  
   - Supports **Chinese speech input only**
2. **English UI Version (Multi-language)**  
   - UI in English  
   - Whisper **does not specify language** (auto-detects)  
   - Supports mixed or multi-language speech input

---

##  Features

- **Real-time microphone recording** with device selector
-**Volume meter (VU meter)** to display live input levels
-**Local Whisper ASR**
  - Chinese UI version: fixed to Chinese
  - English UI version: automatic language detection
- **Local Ollama LLM integration** (e.g., `llama3:8b`, `qwen2.5:7b`)
-**Tkinter GUI** — no command line needed
-**Fully offline** — no audio or text is sent to external servers

---

## 📦 Requirements

- Python **3.10+**
- Conda or venv virtual environment (recommended)
- **Ollama** installed and running locally
- **Whisper models** (auto-downloaded by `faster-whisper` or `openai-whisper`)
- **FFmpeg** (required for `openai-whisper`)

**Installation**
Follow these steps even if you are new to Python.
We will install Python, create a safe working environment, and run the assistant.

1️⃣ Install Python (if you don’t have it yet)
--------------------------------------------
We recommend using Miniconda because it makes Python environments easier to manage.

- Download Miniconda from: https://docs.conda.io/en/latest/miniconda.html
- Choose the version for your operating system (Windows / macOS / Linux)
- During installation, make sure to check "Add Miniconda to PATH"

2️⃣ Download this project
-------------------------
If you have Git installed:
    git clone https://github.com/<your-username>/<repo-name>.git
    cd <repo-name>

If you don’t have Git:
1. Click the green "Code" button on the GitHub page
2. Select "Download ZIP"
3. Extract the ZIP file to a folder (for example, Desktop)

3️⃣ Open a terminal in the project folder
-----------------------------------------
Windows:
    - Open the folder where you extracted the project
    - Right-click an empty space → "Open in Terminal" (or "Open PowerShell here")

macOS / Linux:
    - Open Terminal
    - Use 'cd' to go to the project folder, e.g.:
      cd ~/Desktop/<repo-name>

4️⃣ Create and activate a Python environment
--------------------------------------------
This keeps your project’s Python setup separate from other projects.

    conda create -n voiceassistant python=3.10 -y
    conda activate voiceassistant

5️⃣ Install required Python packages
------------------------------------
    pip install -U pip
    pip install faster-whisper openai-whisper sounddevice soundfile numpy scipy requests tqdm rich python-dotenv

6️⃣ Install FFmpeg
------------------
Whisper needs FFmpeg to process audio.

Windows (recommended way):
    winget install Gyan.FFmpeg

macOS:
    brew install ffmpeg

Linux (Debian/Ubuntu):
    sudo apt install ffmpeg

7️⃣ Install and prepare Ollama
------------------------------
Ollama runs the local AI model.

1. Download Ollama: https://ollama.ai/download
2. Start Ollama:
       Press WinR+R, enter cmd and run
       In the comming window enter **ollama run (modelname,eg:phi3)** 

8️⃣ Run the program
-------------------
Chinese UI version (Chinese input only):
    python ollama_gui_voice_fixed.py

English UI version (multi-language input):
    python ollama_gui_voice_fixed_en_autolang.py

9️⃣ **Basic Checks**
-------------------
· You can check whether Ollama is running and connected correctly by **Test Connection** button after **Step8**

· You can check whether your microphone is running by the **Input Volume(RMS)** bar. If it is not empty after you started recording, your voice is recorded.

    You can also test whether your voice is recorded by going to the **Recordings** file and opening the **.wav** documents one by one
    
· Other processing error will be directly shown in the UI

✅ You are ready! Select a microphone, start recording, and enjoy your local voice assistant




