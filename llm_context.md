Codebase Context: Real-Time Multilingual Translator & Caption OverlayProject Architecture OverviewThis application is a real-time speech translation system that captures system or loopback audio, performs incremental automatic speech recognition (ASR) via OpenAI Whisper, translates text into localized Vietnamese, and renders floating subtitle overlay graphics.Multi-Threaded & Event-Driven Engine TopologyThe system carefully orchestrates separate asynchronous execution layers to ensure UI rendering frames do not dropped and live audio processing remains unblocked:OS Main UI GUI Thread (PySide6 / Qt): Renders the translucent desktop caption window and handles display ticks.Dedicated Asyncio Worker Thread: Manages a non-blocking asynchronous multi-stage pipeline using bounded asyncio.Queue primitives to ensure real-time latency.Hardware Capture Audio Thread: Continuously polls and streams WASAPI loopback channels into thread-safe rolling history rings.Component Module Directory1. main.py (System Bootstrapper & Lifecycle Controller)Responsibility: Houses global logging setup, reads configuration constants, spins up the independent background asyncio loop, and coordinates graceful shutdowns when exit traps fire.Key Components:run_asyncio_background_loop: Runs a loop target continuously inside a dedicated daemon thread.handle_graceful_shutdown: Closes network pipes, hardware tasks, and joins worker contexts cleanly.2. audio.py (Audio Capture & Stream Preprocessing)Responsibility: Opens hardware audio inputs via loopback context, normalizes PCM floats to 16kHz mono data, applies frame-by-frame Voice Activity Detection (VAD) via Silero VAD, and pushes blocks forward.Key Structures:AudioChunk: Dataclass holding 1D float32 arrays, durations, timestamps, and VAD states.RingBuffer: A thread-safe, preallocated fixed-capacity circular numpy array implementing $O(1)$ updates without runtime reallocations.StreamingVAD: State machine confirming speech start/end blocks via moving window confirmation indices.3. asr.py (Sliding Window Whisper Engine)Responsibility: Pulls audio from the rolling ring history and runs local incremental sliding-window decoding.Key Algorithmic Flow:Employs a Local Agreement Streaming Policy: Extracts word-level token sequences and matches them with preceding histories via a Longest Common Subsequence (LCS) dynamic programming alignment matrix (_lcs_matched_pairs).Divides output transcripts into a confirmed immutable prefix (stable_text) and mutable streaming updates (live_text).Key Structures:WordToken: Positional timestamps and confidence values per token.Transcript: Data package containing stable prefixes, live text suffixes, and detected language IDs.4. translation.py (Asynchronous Worker Pool & Normalization)Responsibility: Normalizes transcription variants and handles network localization safely.Key Components:GoogleRPCBackend: Uses standard library networks with strict HTTP timeouts to drop third-party dependencies._LRUTTLCache: Bounded Least-Recently-Used hot cache with per-entry Time-To-Live parameters to protect system memory footprints.Translator: High-level coordinator managing the _TranslationJob queues across multiple concurrently execution backend workers.5. pipeline.py (The Async Dataflow Brain)Responsibility: Links individual modules using an independent four-stage asynchronous message layout:$$\text{AudioCapture} \rightarrow \text{ASR Worker} \rightarrow \text{Translation Worker} \rightarrow \text{Render Worker}$$Key Components:CaptionPipeline: Controls inter-task boundaries._enqueue_dropping_oldest: When queues are saturated, old pending blocks are discarded to avoid blocking upstream stages or letting latency float unbounded.6. overlay.py (Translucent UI Window Graphic)Responsibility: Draggable frameless overlay rendering subtitles with responsive fade animations and DPI-aware scaling.Key Components:wrap_caption: Wraps strings using whitespace word boundaries to preserve terms, capping output blocks at a maximum of two lines.OverlayWindow: Uses thread-safe Qt Signal channels (caption_updated) to transfer textual payload mutations directly into the UI draw sequence.System Configuration Settings ReferenceConstants NamespaceVariable NameDefault ValueDescriptionmain.pyAUDIO_CHUNK_DURATION_SECONDS0.5Interval pacing for raw audio window groupings.AUDIO_USE_VADTrueFlags whether streaming speech boundaries filter input arrays.WHISPER_MODEL_NAME"base"Quantized network target configuration used for decoding.pipeline.pyASR_QUEUE_MAXSIZE50Bound limit on pending transcription elements.STATS_LOG_INTERVAL_SECONDS10.0Metrics collection pacing index for latency profiling.audio.pyTARGET_SAMPLE_RATE16000Fixed input sample frequency expected by standard models.RING_BUFFER_HISTORY_SECONDS30.0Preallocated maximum trace length for audio buffer retention.translation.pyCACHE_MAX_SIZE500Size barrier threshold for the LRU cache store.TRANSLATION_WORKER_COUNT3Parallel concurrency slots assigned to network requests.TARGET_LANGUAGE"vi"Strict translation localization target language.overlay.pyMAX_VISIBLE_LINES2Ceiling count for visible subtitle text tracks.MIN_UPDATE_INTERVAL_MS60Coalesces rapid updates to minimize screen flickering.Complete Core Code ListingsPythonimport asyncio
import logging
import sys
import threading
from typing import Optional

from PySide6.QtWidgets import QApplication
from dotenv import load_dotenv

from audio import AudioCapture
from asr import WhisperEngine
from translation import Translator
from overlay import OverlayWindow
from pipeline import CaptionPipeline

AUDIO_CHUNK_DURATION_SECONDS: float = 0.5
AUDIO_USE_VAD: bool = True
AUDIO_VAD_THRESHOLD: float = 0.4

WHISPER_MODEL_NAME: str = "base"
WHISPER_DEVICE: Optional[str] = None  # None -> auto-select CUDA if available, else CPU

OVERLAY_CLEAR_TIMEOUT_SECONDS: float = 4.0
SHUTDOWN_JOIN_TIMEOUT_SECONDS: float = 4.0

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("realtime_translator.main")

def run_asyncio_background_loop(loop: asyncio.AbstractEventLoop) -> None:
    asyncio.set_event_loop(loop)
    try:
        loop.run_forever()
    finally:
        try:
            pending_tasks = asyncio.all_tasks(loop)
            if pending_tasks:
                loop.run_until_complete(asyncio.gather(*pending_tasks, return_exceptions=True))
            loop.run_until_complete(loop.shutdown_asyncgens())
        except Exception as shutdown_err:
            logger.error(f"Error encountered during background event loop teardown: {shutdown_err}")
        loop.close()
        logger.info("Isolated background asyncio event loop closed safely.")
Pythonimport asyncio
import logging
import math
import threading
import time
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, List, Optional

import numpy as np
import soundcard as sc
import torch
from scipy.signal import resample_poly

logger = logging.getLogger("realtime_translator.audio")

TARGET_SAMPLE_RATE: int = 16000
RING_BUFFER_HISTORY_SECONDS: float = 30.0
CAPTURE_BLOCK_DURATION_SECONDS: float = 0.05
VAD_FRAME_DURATION_SECONDS: float = 0.03
VAD_DEFAULT_THRESHOLD: float = 0.5
VAD_SPEECH_CONFIRM_FRAMES: int = 2
VAD_SILENCE_CONFIRM_FRAMES: int = 5

@dataclass
class AudioChunk:
    data: np.ndarray
    sample_rate: int
    timestamp: float
    duration: float
    is_speech: bool

class RingBuffer:
    def __init__(self, capacity_seconds: float, sample_rate: int = TARGET_SAMPLE_RATE) -> None:
        self.sample_rate: int = sample_rate
        self.capacity: int = max(1, int(capacity_seconds * sample_rate))
        self._buffer: np.ndarray = np.zeros(self.capacity, dtype=np.float32)
        self._write_pos: int = 0
        self._filled: int = 0
        self._lock: threading.Lock = threading.Lock()

    def append(self, samples: np.ndarray) -> None:
        if samples.size == 0:
            return
        with self._lock:
            n = samples.shape[0]
            if n >= self.capacity:
                self._buffer[:] = samples[-self.capacity:]
                self._write_pos = 0
                self._filled = self.capacity
                return
            end_space = self.capacity - self._write_pos
            if n <= end_space:
                self._buffer[self._write_pos:self._write_pos + n] = samples
            else:
                first_part = end_space
                second_part = n - end_space
                self._buffer[self._write_pos:] = samples[:first_part]
                self._buffer[:second_part] = samples[first_part:]
            self._write_pos = (self._write_pos + n) % self.capacity
            self._filled = min(self.capacity, self._filled + n)

    def get_last(self, seconds: float) -> np.ndarray:
        with self._lock:
            n = min(self._filled, int(seconds * self.sample_rate))
            if n <= 0:
                return np.zeros(0, dtype=np.float32)
            start = (self._write_pos - n) % self.capacity
            if start + n <= self.capacity:
                return self._buffer[start:start + n].copy()
            first_part = self.capacity - start
            out = np.empty(n, dtype=np.float32)
            out[:first_part] = self._buffer[start:]
            out[first_part:] = self._buffer[:n - first_part]
            return out
Pythonimport logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import whisper
from audio import RingBuffer, TARGET_SAMPLE_RATE

logger = logging.getLogger("realtime_translator.asr")

WINDOW_SECONDS: float = 2.5
OVERLAP_SECONDS: float = 0.5
STEP_SECONDS: float = 0.5
UNSTABLE_TAIL_SECONDS: float = OVERLAP_SECONDS
MIN_DECODE_SECONDS: float = 1.0
SILENCE_FLUSH_SECONDS: float = 1.2
MAX_PROMPT_CHARS: int = 200

@dataclass
class WordToken:
    text: str
    start: float
    end: float
    probability: float

@dataclass
class Transcript:
    text: str
    language: str
    is_final: bool
    timestamp: float
    confidence: float
    stable_text: str = ""
    live_text: str = ""

def _normalize_word(word: str) -> str:
    return word.strip().lower().strip(".,!?;:\"'")

def _lcs_matched_pairs(seq_a: List[WordToken], seq_b: List[WordToken]) -> List[Tuple[int, int]]:
    n, m = len(seq_a), len(seq_b)
    if n == 0 or m == 0:
        return []
    keys_a = [_normalize_word(w.text) for w in seq_a]
    keys_b = [_normalize_word(w.text) for w in seq_b]
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n - 1, -1, -1):
        for j in range(m - 1, -1, -1):
            if keys_a[i] == keys_b[j]:
                dp[i][j] = dp[i + 1][j + 1] + 1
            else:
                dp[i][j] = max(dp[i + 1][j], dp[i][j + 1])
    pairs: List[Tuple[int, int]] = []
    i = j = 0
    while i < n and j < m:
        if keys_a[i] == keys_b[j]:
            pairs.append((i, j))
            i += 1; j += 1
        elif dp[i + 1][j] >= dp[i][j + 1]:
            i += 1
        else:
            j += 1
    return pairs
Pythonimport asyncio
import logging
import re
import time
import urllib.parse
import urllib.request
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, Optional, Protocol, Tuple
import orjson

logger = logging.getLogger("realtime_translator.translation")

CACHE_MAX_SIZE: int = 500
CACHE_TTL_SECONDS: float = 300.0
TRANSLATION_WORKER_COUNT: int = 3
TRANSLATION_QUEUE_MAX_SIZE: int = 200

@dataclass
class Translation:
    original_text: str
    translated_text: str
    source_lang: str
    target_lang: str
    timestamp: float
    cached: bool

class GoogleRPCBackend:
    def __init__(self, timeout: float = 3.0) -> None:
        self.timeout = timeout
        self.url = "https://translate.googleapis.com/translate_a/single"
        self.user_agent = "Mozilla/5.0..."

    def translate_text(self, text: str, source_lang: str, target_lang: str) -> str:
        if not text.strip(): return ""
        params = {"client": "gtx", "sl": source_lang, "tl": target_lang, "dt": "t", "q": text}
        encoded_query = urllib.parse.urlencode(params)
        request = urllib.request.Request(f"{self.url}?{encoded_query}")
        request.add_header("User-Agent", self.user_agent)
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as r:
                parsed_json = orjson.loads(r.read())
                if not parsed_json or not parsed_json[0]: return text
                return "".join([s[0] for s in parsed_json[0] if s and s[0]])
        except Exception as e:
            logger.error(f"HTTP transport failure: {e}")
            return text
Pythonimport asyncio
import logging
import time
from dataclasses import dataclass
from typing import List, Optional

from audio import AudioCapture, AudioChunk
from asr import WhisperEngine, Transcript
from translation import Translator
from overlay import OverlayWindow

ASR_QUEUE_MAXSIZE: int = 50
RENDER_QUEUE_MAXSIZE: int = 50

@dataclass
class _TimedTranscript:
    transcript: Transcript
    asr_latency_ms: float

@dataclass
class _RenderJob:
    text: str
    is_final: bool
    timestamp: float
    asr_latency_ms: float = 0.0
    translation_latency_ms: float = 0.0
Pythonimport logging
from typing import List, Optional
from PySide6.QtCore import QEasingCurve, QPoint, QPropertyAnimation, Qt, QTimer, Signal, Slot
from PySide6.QtGui import QAction, QColor, QContextMenuEvent, QFont, QMouseEvent, QScreen
from PySide6.QtWidgets import QApplication, QGraphicsDropShadowEffect, QLabel, QMenu, QVBoxLayout, QWidget

MAX_VISIBLE_LINES: int = 2
MAX_LINE_CHARS: int = 42

def wrap_caption(text: str, max_chars: int = MAX_LINE_CHARS, max_lines: int = MAX_VISIBLE_LINES) -> str:
    words = text.split()
    if not words: return ""
    lines: List[str] = []
    current_line = ""
    for word in words:
        candidate = f"{current_line} {word}".strip()
        if len(candidate) <= max_chars or not current_line:
            current_line = candidate
        else:
            lines.append(current_line)
            current_line = word
    if current_line: lines.append(current_line)
    if len(lines) > max_lines: lines = lines[-max_lines:]
    return "\n".join(lines)