
import tkinter as tk
from tkinter import ttk, messagebox
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
import os
import threading
import requests
import time
import pathlib

# ---- Optional: use faster-whisper if installed, else fall back to openai-whisper ----
USE_FASTER = False
try:
    from faster_whisper import WhisperModel
    USE_FASTER = True
except Exception:
    try:
        import whisper  # type: ignore
    except Exception:
        whisper = None

APP_TITLE = "心理语音助手（修正版）"
DEFAULT_SAMPLE_RATE = 16000
DEFAULT_CHANNELS = 1
DEFAULT_DTYPE = "int16"   # ensures stable meter math
DEFAULT_SAVE = "input.wav"

# If an ffmpeg.exe is bundled next to the script, add it to PATH (good for openai-whisper fallback)
_here = os.path.dirname(__file__)
_ffmpeg = os.path.join(_here, "ffmpeg.exe")
if os.path.exists(_ffmpeg):
    os.environ["PATH"] = os.path.dirname(_ffmpeg) + os.pathsep + os.environ.get("PATH", "")

def list_input_devices():
    devices = []
    for i, d in enumerate(sd.query_devices()):
        if d.get("max_input_channels", 0) > 0:
            # show API-host as well to avoid duplicates
            host = sd.query_hostapis()[d["hostapi"]]["name"] if "hostapi" in d else ""
            name = f"{d['name']} ({host})"
            devices.append((i, name))
    return devices

def check_ollama_model_ready(model: str, base_url="http://localhost:11434", timeout_s=15):
    """
    Query Ollama for available models via /api/list and /api/tags.
    Accepts exact match or prefix (e.g., 'llama3' matches 'llama3:8b').
    """
    deadline = time.time() + timeout_s
    tried_endpoints = ["/api/list", "/api/tags"]
    seen_models = set()

    def _names_from_resp(js):
        names = []
        if isinstance(js, dict) and "models" in js:
            for m in js["models"]:
                # Try common fields
                for k in ("name", "model"):
                    if k in m and isinstance(m[k], str):
                        names.append(m[k])
        return names

    while time.time() < deadline:
        for ep in tried_endpoints:
            try:
                r = requests.get(base_url + ep, timeout=2)
                if r.ok:
                    for n in _names_from_resp(r.json()):
                        seen_models.add(n)
                    # direct or prefix match accepted
                    if any(n == model or n.startswith(model + ":") for n in seen_models):
                        return True, sorted(seen_models)
            except Exception:
                pass
        time.sleep(1.0)

    return False, sorted(seen_models)

class VoiceApp:
    def __init__(self, master: tk.Tk):
        self.master = master
        master.title(APP_TITLE)
        master.geometry("720x560")

        # --- Whisper (ASR) model selector ---
        tk.Label(master, text="Whisper 语音识别模型（本地）").pack(anchor="w", padx=12, pady=(10, 2))
        self.asr_model_var = tk.StringVar(value="small")  # tiny/base/small/medium/large-v3
        self.asr_model_cb = ttk.Combobox(master, textvariable=self.asr_model_var, state="readonly",
                                         values=["tiny","base","small","medium","large-v2","large-v3"])
        self.asr_model_cb.pack(fill="x", padx=12)

        # --- Ollama (LLM) model selector ---
        row = tk.Frame(master); row.pack(fill="x", padx=12, pady=8)
        tk.Label(row, text="Ollama 大模型名称").pack(side="left")
        self.ollama_model_var = tk.StringVar(value="llama3:8b")
        self.ollama_entry = ttk.Entry(row, textvariable=self.ollama_model_var)
        self.ollama_entry.pack(side="left", fill="x", expand=True, padx=8)

        self.test_ollama_btn = ttk.Button(row, text="测试连接", command=self._probe_ollama)
        self.test_ollama_btn.pack(side="left")

        # --- Input device selector ---
        tk.Label(master, text="选择麦克风设备").pack(anchor="w", padx=12)
        self.device_map = {}  # name -> index
        devs = list_input_devices()
        for idx, name in devs:
            self.device_map[name] = idx
        device_names = list(self.device_map.keys()) or ["<未检测到输入设备>"]
        self.device_var = tk.StringVar(value=device_names[0])
        self.device_cb = ttk.Combobox(master, textvariable=self.device_var, state="readonly", values=device_names)
        self.device_cb.pack(fill="x", padx=12, pady=(0,8))

        # --- VU meter ---
        tk.Label(master, text="输入音量（RMS）").pack(anchor="w", padx=12)
        self.vu = ttk.Progressbar(master, orient="horizontal", mode="determinate", maximum=100)
        self.vu.pack(fill="x", padx=12, pady=(0,8))

        # --- Status box ---
        tk.Label(master, text="状态/日志").pack(anchor="w", padx=12)
        self.log = tk.Text(master, height=8)
        self.log.pack(fill="x", padx=12, pady=(0,8))

        # --- Buttons ---
        btn = tk.Frame(master); btn.pack(pady=6)
        self.start_btn = ttk.Button(btn, text="🔴 开始录音", command=self.start_record)
        self.start_btn.grid(row=0, column=0, padx=6)
        self.stop_btn  = ttk.Button(btn, text="⏹ 停止录音", command=self.stop_record, state="disabled")
        self.stop_btn.grid(row=0, column=1, padx=6)

        # --- Transcript ---
        tk.Label(master, text="转写/回答").pack(anchor="w", padx=12)
        self.out = tk.Text(master, height=12)
        self.out.pack(fill="both", expand=True, padx=12, pady=(0,12))

        # runtime
        self.recording = False
        self._chunks = []   # collected int16 blocks
        self.max_retries = 5

    # ----------------------- UI helpers -----------------------
    def log_write(self, s: str):
        self.log.insert(tk.END, s.rstrip() + "\n")
        self.log.see(tk.END)

    # ----------------------- Audio ----------------------------
    def start_record(self):
        if "<未检测到输入设备>" in self.device_var.get():
            messagebox.showerror("错误", "未检测到可用的输入设备。")
            return
        self.recording = True
        self._chunks.clear()
        self.start_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        t = threading.Thread(target=self._record_loop, daemon=True)
        t.start()
        self.log_write("[INFO] 开始录音...")

    def stop_record(self):
        self.recording = False
        self.start_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
        self.log_write("[INFO] 停止录音，保存音频并开始转写...")

    def _record_loop(self):
        fs = DEFAULT_SAMPLE_RATE
        dev_index = self.device_map.get(self.device_var.get(), None)

        def _cb(indata, frames, time_info, status):
            if status:
                self.log_write(f"[AUDIO] {status}")
            # indata is int16
            audio = indata[:, 0].copy()
            # meter (RMS in dBFS mapped to 0..100)
            rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
            vu = int(min(100, max(0, (rms / 32768.0) * 100)))
            # schedule UI update
            self.master.after(0, lambda: self.vu.configure(value=vu))
            self._chunks.append(audio)

        with sd.InputStream(samplerate=fs, channels=1, dtype=DEFAULT_DTYPE,
                            device=dev_index, callback=_cb):
            while self.recording:
                sd.sleep(50)

        # save WAV
        if self._chunks:
            audio = np.concatenate(self._chunks).astype(np.int16)
            write(DEFAULT_SAVE, DEFAULT_SAMPLE_RATE, audio)
            self.master.after(0, lambda: self.log_write(f"[INFO] 已保存音频 -> {os.path.abspath(DEFAULT_SAVE)}"))
            # next: transcribe + ask LLM
            threading.Thread(target=self._transcribe_and_chat, args=(DEFAULT_SAVE,), daemon=True).start()
        else:
            self.master.after(0, lambda: self.log_write("[WARN] 没有采集到任何音频块。"))

    # ----------------------- ASR + LLM ------------------------
    def _load_asr(self, model_name: str):
        if USE_FASTER:
            # auto device: try CUDA, else CPU int8
            device = "cuda"
            compute_type = "float16"
            try:
                m = WhisperModel(model_name, device=device, compute_type=compute_type)
            except Exception:
                m = WhisperModel(model_name, device="cpu", compute_type="int8")
            return ("faster", m)
        else:
            if whisper is None:
                raise RuntimeError("未安装 faster-whisper 或 openai-whisper，无法转写。")
            m = whisper.load_model(model_name)
            return ("openai", m)

    def _run_asr(self, model, audio_path: str, language="zh"):
        if USE_FASTER:
            segments, info = model.transcribe(audio_path, language=language, beam_size=1)
            return "".join(seg.text for seg in segments).strip()
        else:
            res = model.transcribe(audio_path, language=language, fp16=False)
            return res.get("text", "").strip()

    def _transcribe_and_chat(self, wav_path: str):
        # 1) ASR
        asr_model_name = self.asr_model_var.get()
        self.master.after(0, lambda: self.log_write(f"[INFO] 加载 ASR 模型：{asr_model_name}"))
        try:
            kind, asr_model = self._load_asr(asr_model_name)
            self.master.after(0, lambda: self.log_write(f"[INFO] ASR 引擎：{kind}"))
            text = self._run_asr(asr_model, wav_path, language="zh")
        except Exception as e:
            self.master.after(0, lambda: self.log_write(f"[ERROR] 转写失败：{e}"))
            return

        self.master.after(0, lambda: (self.out.delete("1.0", tk.END), self.out.insert(tk.END, text + "\n")))

        # 2) Ollama chat
        ollama_model = self.ollama_model_var.get().strip()
        ok, names = check_ollama_model_ready(ollama_model)
        if not ok:
            self.master.after(0, lambda: self.log_write(f"[WARN] 未检测到 Ollama 模型 `{ollama_model}`。本机已加载：{names}"))
            return

        payload = {
            "model": ollama_model,
            "prompt": text,
            "stream": False
        }
        for i in range(self.max_retries):
            try:
                r = requests.post("http://localhost:11434/api/generate", json=payload,
                                  headers={"Content-Type": "application/json"}, timeout=60)
                if r.status_code == 503:
                    self.master.after(0, lambda i=i: self.log_write(f"[WARN] Ollama 忙碌，重试 {i+1}/5..."))
                    time.sleep(2.5)
                    continue
                r.raise_for_status()
                ans = r.json().get("response", "").strip()
                self.master.after(0, lambda: self.out.insert(tk.END, f"\n🤖 回答：{ans}\n"))
                break
            except Exception as e:
                self.master.after(0, lambda i=i, e=e: self.log_write(f"[ERROR] 请求失败（{i+1}/5）：{e}"))
        else:
            self.master.after(0, lambda: self.log_write("[ERROR] 未能获取 Ollama 响应。"))

    # ----------------------- Actions --------------------------
    def _probe_ollama(self):
        model = self.ollama_model_var.get().strip()
        ok, names = check_ollama_model_ready(model, timeout_s=5)
        if ok:
            messagebox.showinfo("Ollama 连接正常", f"找到了模型：{model}\n\n可用模型：\n- " + "\n- ".join(names))
        else:
            messagebox.showwarning("Ollama 未就绪", "未找到目标模型：{}\n本机当前可用：\n- {}".format(model, "\n- ".join(names) if names else "(空)"))
            self.log_write("提示：先在终端运行 `ollama run {}` 或 `ollama pull {}`。".format(model, model))

if __name__ == "__main__":
    root = tk.Tk()
    app = VoiceApp(root)
    root.mainloop()
