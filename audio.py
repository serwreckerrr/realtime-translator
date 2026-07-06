import asyncio
import logging
import math
import queue
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

# --------------------------------------------------------------------------
# Configuration constants
# --------------------------------------------------------------------------
TARGET_SAMPLE_RATE: int = 16000
RING_BUFFER_HISTORY_SECONDS: float = 30.0          
CAPTURE_BLOCK_DURATION_SECONDS: float = 0.2        
VAD_FRAME_DURATION_SECONDS: float = 0.03            
VAD_DEFAULT_THRESHOLD: float = 0.5
VAD_SPEECH_CONFIRM_FRAMES: int = 2                  
VAD_SILENCE_CONFIRM_FRAMES: int = 5                 
DEVICE_RECOVERY_RETRY_DELAY_SECONDS: float = 1.5
DEVICE_RECOVERY_MAX_BACKOFF_SECONDS: float = 8.0
DEVICE_POLL_INTERVAL_SECONDS: float = 2.0           
RESAMPLE_MAX_DENOMINATOR: int = 1000 
VAD_FRAME_SAMPLES = 512               


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

    def clear(self) -> None:
        with self._lock:
            self._write_pos = 0
            self._filled = 0

    @property
    def filled_seconds(self) -> float:
        with self._lock:
            return self._filled / self.sample_rate


class StreamingVAD:
    def __init__(
        self,
        sample_rate: int = TARGET_SAMPLE_RATE,
        threshold: float = VAD_DEFAULT_THRESHOLD,
        speech_confirm_frames: int = VAD_SPEECH_CONFIRM_FRAMES,
        silence_confirm_frames: int = VAD_SILENCE_CONFIRM_FRAMES,
    ) -> None:
        self.sample_rate: int = sample_rate
        self.threshold: float = threshold
        self.speech_confirm_frames: int = speech_confirm_frames
        self.silence_confirm_frames: int = silence_confirm_frames

        self._model: Any = None
        self._available: bool = False
        self._consecutive_speech: int = 0
        self._consecutive_silence: int = 0
        self._is_speaking: bool = False
        self._silence_start_time: Optional[float] = None
        self._speech_start_time: Optional[float] = None

        self._load_model()

    def _load_model(self) -> None:
        try:
            from silero_vad import load_silero_vad
            self._model = load_silero_vad()

            # print("Silero VAD model loaded via Torch Hub:")
            # print(type(self._model))
            # print(repr(self._model))

            self._available = True
            logger.info("Silero VAD local module loaded successfully for streaming VAD.")
        except ImportError:
            logger.info("Silero VAD library not found locally. Loading fallback via Torch Hub...")
            try:
                model, _utils = torch.hub.load(
                    repo_or_dir="snakers4/silero-vad",
                    model="silero_vad",
                    force_reload=False,
                    trust_repo=True,  # type: ignore
                )
                self._model = model

                self._available = True
                logger.info("Silero VAD fallback via Torch Hub successfully initialized.")
            except Exception as hub_err:
                logger.error(f"Failed to initialize Silero VAD: {hub_err}")
                self._available = False

    def process_frame(self, frame: np.ndarray) -> "VADState":
        speech_prob = 0.0

        # Debug the input to Silero VAD
        # print("=" * 50)
        # print("shape      :", frame.shape)
        # print("dtype      :", frame.dtype)
        # print("min        :", frame.min())
        # print("max        :", frame.max())
        # print("mean abs   :", np.mean(np.abs(frame)))
        # print("sample_rate:", self.sample_rate)

        if self._available and self._model is not None and frame.size > 0:
            try:
                with torch.no_grad():
                    # tensor_frame = torch.from_numpy(frame)
                    # speech_prob = float(self._model(tensor_frame, self.sample_rate).item())
                    
                    tensor_frame = torch.from_numpy(frame).unsqueeze(0)
                    out = self._model(tensor_frame, self.sample_rate)
                    # print(out)
                    speech_prob = float(out.item())

            # except Exception as infer_err:
            #     logger.debug(f"Streaming VAD frame inference slip: {infer_err}")
            #     speech_prob = 1.0 if self._is_speaking else 0.0
            except Exception as infer_err:
                print(infer_err)

        is_frame_speech = speech_prob >= self.threshold
        started = False
        ended = False
        now = time.time()

        if is_frame_speech:
            self._consecutive_speech += 1
            self._consecutive_silence = 0
            if not self._is_speaking and self._consecutive_speech >= self.speech_confirm_frames:
                self._is_speaking = True
                self._speech_start_time = now
                self._silence_start_time = None
                started = True
        else:
            self._consecutive_silence += 1
            self._consecutive_speech = 0
            if self._silence_start_time is None:
                self._silence_start_time = now
            if self._is_speaking and self._consecutive_silence >= self.silence_confirm_frames:
                self._is_speaking = False
                ended = True

        silence_duration = (now - self._silence_start_time) if self._silence_start_time else 0.0

        return VADState(
            is_speech=self._is_speaking or is_frame_speech,
            speech_started=started,
            speech_ended=ended,
            silence_duration=silence_duration,
            speech_probability=speech_prob,
        )

    def reset(self) -> None:
        self._consecutive_speech = 0
        self._consecutive_silence = 0
        self._is_speaking = False
        self._silence_start_time = None
        self._speech_start_time = None


@dataclass
class VADState:
    is_speech: bool
    speech_started: bool
    speech_ended: bool
    silence_duration: float
    speech_probability: float


def resample_audio(audio_data: np.ndarray, orig_sr: int, target_sr: int = TARGET_SAMPLE_RATE) -> np.ndarray:
    if len(audio_data) == 0:
        return np.zeros(0, dtype=np.float32)
    if orig_sr == target_sr:
        return audio_data.astype(np.float32)

    ratio = Fraction(target_sr, orig_sr).limit_denominator(RESAMPLE_MAX_DENOMINATOR)
    up, down = ratio.numerator, ratio.denominator
    resampled = resample_poly(audio_data.astype(np.float32), up, down)
    return resampled.astype(np.float32)


class AudioCapture:
    def __init__(self, chunk_duration: float = 0.5, use_vad: bool = True, vad_threshold: float = 0.5) -> None:
        self.chunk_duration: float = chunk_duration
        self.use_vad: bool = use_vad
        self.vad_threshold: float = vad_threshold

        self._queue: asyncio.Queue[AudioChunk] = asyncio.Queue(maxsize=10)
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._running: bool = False
        self.ring_buffer: RingBuffer = RingBuffer(RING_BUFFER_HISTORY_SECONDS, TARGET_SAMPLE_RATE)

    def start(self, loop: asyncio.AbstractEventLoop) -> None:
        if self._running:
            logger.warning("AudioCapture is already running.")
            return
        self._loop = loop
        self._running = True
        self._thread = threading.Thread(target=self._capture_supervisor_loop, name="WASAPICaptureThread", daemon=True)
        self._thread.start()
        logger.info("AudioCapture background thread started successfully.")

    def stop(self) -> None:
        if not self._running:
            return
        logger.info("Stopping AudioCapture loop...")
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None
        self.ring_buffer.clear()
        logger.info("AudioCapture loop stopped safely.")

    async def get_chunk(self) -> AudioChunk:
        return await self._queue.get()
    
    def _enqueue_chunk(self, chunk: AudioChunk) -> None:
        try:
            self._queue.put_nowait(chunk)
        except asyncio.QueueFull:
            try:
                self._queue.get_nowait()      
            except asyncio.QueueEmpty:
                pass
            try:
                self._queue.put_nowait(chunk)
            except asyncio.QueueFull:
                pass

    def _capture_supervisor_loop(self) -> None:
        backoff = DEVICE_RECOVERY_RETRY_DELAY_SECONDS
        while self._running:
            try:
                self._run_capture_session()
                backoff = DEVICE_RECOVERY_RETRY_DELAY_SECONDS
            except Exception as session_err:
                logger.error(f"WASAPI capture session terminated unexpectedly: {session_err}", exc_info=True)

            if not self._running:
                break
            logger.warning(f"Attempting WASAPI reconnection in {backoff:.1f}s...")
            time.sleep(backoff)
            backoff = min(DEVICE_RECOVERY_MAX_BACKOFF_SECONDS, backoff * 2)

        logger.info("WASAPI capture supervisor loop exited.")

    def _resolve_loopback_mic(self) -> Any:
        default_speaker = sc.default_speaker()
        logger.info(f"Primary Windows Output Device: {default_speaker.name}")
        mic = None
        try:
            mic = sc.get_microphone(id=default_speaker.name, include_loopback=True)
        except Exception as lookup_err:
            logger.warning(f"Direct loopback lookup failed ({lookup_err}). Querying available devices...")
            for potential_mic in sc.all_microphones(include_loopback=True):
                if default_speaker.name in potential_mic.name or "loopback" in potential_mic.name.lower():
                    mic = potential_mic
                    break
            if not mic:
                mics = sc.all_microphones(include_loopback=True)
                mic = mics[0] if mics else sc.default_microphone()

        logger.info(f"Selected WASAPI Loopback Interface: {mic.name}")
        return mic, default_speaker.name

    def _run_capture_session(self) -> None:
        mic, bound_speaker_name = self._resolve_loopback_mic()

        native_sr: int = 48000
        recorder_ctx = None
        for rate in (48000, 44100, 96000, 16000):
            try:
                recorder_ctx = mic.recorder(samplerate=rate)
                native_sr = rate
                break
            except Exception:
                continue

        if not recorder_ctx:
            native_sr = getattr(mic, "default_samplerate", 48000)
            recorder_ctx = mic.recorder(samplerate=native_sr)

        num_channels: int = 2
        try:
            if hasattr(mic, "channels"):
                num_channels = len(mic.channels) if isinstance(mic.channels, list) else int(mic.channels)
        except Exception:
            pass

        num_frames: int = int(native_sr * CAPTURE_BLOCK_DURATION_SECONDS)
        
        # Intermediate bounded queue to bridge capture loop with processing workers
        raw_audio_queue: queue.Queue = queue.Queue(maxsize=100)
        processing_running = [True]

        def background_processing_worker() -> None:
            """Background processing worker handling compute-heavy resampling, downmixing, and VAD."""
            streaming_vad: Optional[StreamingVAD] = StreamingVAD(
                sample_rate=TARGET_SAMPLE_RATE, threshold=self.vad_threshold
            ) if self.use_vad else None
            
            #512 is required for Silero VAD, as it processes audio in 512-sample frames
            vad_frame_samples = VAD_FRAME_SAMPLES
            accumulated_audio: List[np.ndarray] = []
            accumulated_frames = 0
            target_frames = int(native_sr * self.chunk_duration)
            vad_carry = np.zeros(0, dtype=np.float32)

            while processing_running[0] or not raw_audio_queue.empty():
                try:
                    data_block = raw_audio_queue.get(timeout=0.1)
                except queue.Empty:
                    continue

                accumulated_audio.append(data_block)
                accumulated_frames += len(data_block)

                if accumulated_frames < target_frames:
                    raw_audio_queue.task_done()
                    continue

                full_block = np.concatenate(accumulated_audio, axis=0)
                accumulated_audio = []
                accumulated_frames = 0

                if full_block.ndim > 1 and full_block.shape[1] > 1:
                    mono_audio = np.mean(full_block, axis=1)
                else:
                    mono_audio = full_block.flatten()

                audio_16k = resample_audio(mono_audio, native_sr, TARGET_SAMPLE_RATE)
                if len(audio_16k) > 0:
                    peak = np.max(np.abs(audio_16k))
                    if peak > 1.0:
                        audio_16k = audio_16k / peak

                is_speech = True
                if streaming_vad is not None:
                    vad_carry = np.concatenate([vad_carry, audio_16k]) if vad_carry.size else audio_16k.copy()
                    any_speech_in_chunk = False
                    while vad_carry.shape[0] >= vad_frame_samples:
                        frame = vad_carry[:vad_frame_samples]
                        vad_carry = vad_carry[vad_frame_samples:]
                        state = streaming_vad.process_frame(frame)
                        if state.speech_started:
                            logger.debug("Streaming VAD: speech start event detected.")
                        if state.speech_ended:
                            logger.debug("Streaming VAD: speech end event detected.")
                        any_speech_in_chunk = any_speech_in_chunk or state.is_speech
                    is_speech = any_speech_in_chunk

                chunk = AudioChunk(
                    data=audio_16k,
                    sample_rate=TARGET_SAMPLE_RATE,
                    timestamp=time.time(),
                    duration=len(audio_16k) / TARGET_SAMPLE_RATE,
                    is_speech=is_speech,
                )

                # Track queue size metrics asynchronously
                logger.debug(f"[Queue Metrics] Internal Async Queue Size: {self._queue.qsize()}")

                if self._loop and self._loop.is_running():
                    self._loop.call_soon_threadsafe(self._enqueue_chunk, chunk)
                
                raw_audio_queue.task_done()

        proc_thread = threading.Thread(target=background_processing_worker, name="AudioProcessingWorker", daemon=True)
        proc_thread.start()
        last_device_poll = time.time()

        try:
            with recorder_ctx as recorder:
                logger.info(f"WASAPI streaming started at native rate: {native_sr} Hz, {num_channels} channels.")
                while self._running:
                    now = time.time()
                    if now - last_device_poll >= DEVICE_POLL_INTERVAL_SECONDS:
                        last_device_poll = now
                        try:
                            current_default = sc.default_speaker().name
                            if current_default != bound_speaker_name:
                                logger.warning(
                                    f"Default output device changed ({bound_speaker_name} -> {current_default}). "
                                    "Reinitializing capture session."
                                )
                                break
                        except Exception as poll_err:
                            logger.warning(f"Default device polling failed, assuming device loss: {poll_err}")
                            break

                    callback_start_time = time.time()
                    try:
                        data = recorder.record(numframes=num_frames)
                    except Exception as record_err:
                        logger.error(f"WASAPI hardware read failure, triggering reconnect: {record_err}")
                        raise
                    
                    # Log audio callback execution speed without introducing resource overhead
                    logger.debug(f"[Audio Callback] Execution latency: {(time.time() - callback_start_time) * 1000.0:.2f}ms")

                    if data is None or len(data) == 0:
                        data = np.zeros((num_frames, num_channels), dtype=np.float32)

                    try:
                        raw_audio_queue.put_nowait(data)
                    except queue.Full:
                        logger.warning("Raw hardware capture buffer saturated. Evicting oldest frame data.")
                        try:
                            raw_audio_queue.get_nowait()
                            raw_audio_queue.put_nowait(data)
                        except queue.Empty:
                            pass
        finally:
            processing_running[0] = False
            proc_thread.join(timeout=2.0)

        logger.info("WASAPI capture session ended (loop stopped).")