"""
Live Translator — any language -> Vietnamese, shown in a floating overlay box.

Pipeline:
  system audio (what's playing, e.g. a YouTube video)
      -> captured in short chunks (loopback)
      -> transcribed by whisper.cpp (via pywhispercpp)
      -> translated to Vietnamese (via deep-translator / Google Translate)
      -> shown live in an always-on-top Tkinter box

Run:  python live_translate.py
Stop: close the overlay window, or Ctrl+C in the terminal.

See README.md for one-time setup (model download, audio device notes).
"""

import queue
import threading
import time
import tkinter as tk

import numpy as np
import soundcard as sc
from deep_translator import GoogleTranslator
from pywhispercpp.model import Model

# ---------------------------------------------------------------------------
# Config — tweak these
# ---------------------------------------------------------------------------
WHISPER_MODEL = "tiny"       # tiny / base / small ... bigger = more accurate, slower
CHUNK_SECONDS = 4            # how much audio to transcribe at a time
SAMPLE_RATE = 16000          # whisper wants 16kHz mono
TARGET_LANG = "vi"           # Vietnamese

# ---------------------------------------------------------------------------
# Shared state between threads
# ---------------------------------------------------------------------------
audio_queue: "queue.Queue[np.ndarray]" = queue.Queue()
latest_text = {"original": "", "translated": "", "status": "starting..."}
stop_flag = threading.Event()


def pick_loopback_speaker():
    """Grab the default speaker's loopback microphone (records what's playing)."""
    default_speaker = sc.default_speaker()
    for mic in sc.all_microphones(include_loopback=True):
        if default_speaker.name in mic.name:
            return mic
    # Fallback: some platforms expose loopback differently
    return sc.get_microphone(default_speaker.name, include_loopback=True)


def audio_capture_thread():
    """Continuously records system audio in CHUNK_SECONDS blocks and queues them."""
    try:
        mic = pick_loopback_speaker()
    except Exception as e:
        latest_text["status"] = f"audio error: {e}"
        return

    latest_text["status"] = f"listening via: {mic.name}"
    frames_per_chunk = int(SAMPLE_RATE * CHUNK_SECONDS)

    with mic.recorder(samplerate=SAMPLE_RATE, channels=1) as recorder:
        while not stop_flag.is_set():
            data = recorder.record(numframes=frames_per_chunk)
            mono = data[:, 0].astype(np.float32)
            # skip near-silent chunks so we don't waste time transcribing dead air
            if np.abs(mono).mean() > 0.001:
                audio_queue.put(mono)


def transcribe_translate_thread():
    """Pulls audio chunks, runs whisper.cpp, translates result, updates latest_text."""
    latest_text["status"] = "loading whisper model..."
    model = Model(WHISPER_MODEL, print_realtime=False, print_progress=False)
    translator = GoogleTranslator(source="auto", target=TARGET_LANG)
    latest_text["status"] = "ready"

    while not stop_flag.is_set():
        try:
            chunk = audio_queue.get(timeout=0.5)
        except queue.Empty:
            continue

        segments = model.transcribe(chunk)
        text = " ".join(seg.text.strip() for seg in segments).strip()
        if not text:
            continue

        latest_text["original"] = text
        try:
            latest_text["translated"] = translator.translate(text)
        except Exception as e:
            latest_text["translated"] = f"[translation error: {e}]"


# ---------------------------------------------------------------------------
# Overlay GUI
# ---------------------------------------------------------------------------
def run_overlay():
    root = tk.Tk()
    root.title("Live Translate -> Vietnamese")
    root.attributes("-topmost", True)
    root.configure(bg="#111111")
    root.geometry("700x180+100+100")

    status_label = tk.Label(
        root, text="", fg="#888888", bg="#111111", font=("Segoe UI", 9), anchor="w"
    )
    status_label.pack(fill="x", padx=10, pady=(6, 0))

    original_label = tk.Label(
        root, text="", fg="#aaaaaa", bg="#111111",
        font=("Segoe UI", 11), wraplength=670, justify="left", anchor="w",
    )
    original_label.pack(fill="x", padx=10, pady=(6, 0))

    translated_label = tk.Label(
        root, text="", fg="#ffffff", bg="#111111",
        font=("Segoe UI", 16, "bold"), wraplength=670, justify="left", anchor="w",
    )
    translated_label.pack(fill="x", padx=10, pady=(6, 10))

    def refresh():
        status_label.config(text=latest_text["status"])
        original_label.config(text=latest_text["original"])
        translated_label.config(text=latest_text["translated"])
        root.after(200, refresh)

    def on_close():
        stop_flag.set()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    refresh()
    root.mainloop()


if __name__ == "__main__":
    t1 = threading.Thread(target=audio_capture_thread, daemon=True)
    t2 = threading.Thread(target=transcribe_translate_thread, daemon=True)
    t1.start()
    t2.start()
    try:
        run_overlay()
    except KeyboardInterrupt:
        stop_flag.set()
        time.sleep(0.5)