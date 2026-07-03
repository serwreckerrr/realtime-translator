import asyncio
import logging
import time
from dataclasses import dataclass
from typing import List, Optional

from audio import AudioCapture, AudioChunk
from asr import WhisperEngine, Transcript
from translation import Translator
from overlay import OverlayWindow

logger = logging.getLogger("realtime_translator.pipeline")

# --------------------------------------------------------------------------
# Configuration constants
# --------------------------------------------------------------------------
ASR_QUEUE_MAXSIZE: int = 50
RENDER_QUEUE_MAXSIZE: int = 50
QUEUE_WAIT_TIMEOUT_SECONDS: float = 0.5
DEFAULT_CLEAR_TIMEOUT_SECONDS: float = 4.0
SILENCE_WATCHDOG_INTERVAL_SECONDS: float = 0.5
STATS_LOG_INTERVAL_SECONDS: float = 10.0
STAGE_ERROR_BACKOFF_SECONDS: float = 0.1


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


class CaptionPipeline:
    def __init__(
        self,
        audio_capture: AudioCapture,
        whisper_engine: WhisperEngine,
        translator: Translator,
        overlay_window: OverlayWindow,
        clear_timeout: float = DEFAULT_CLEAR_TIMEOUT_SECONDS
    ) -> None:
        self.audio_capture: AudioCapture = audio_capture
        self.whisper_engine: WhisperEngine = whisper_engine
        self.translator: Translator = translator
        self.overlay_window: OverlayWindow = overlay_window
        self.clear_timeout: float = clear_timeout

        self._asr_queue: "asyncio.Queue[_TimedTranscript]" = asyncio.Queue(maxsize=ASR_QUEUE_MAXSIZE)
        self._render_queue: "asyncio.Queue[_RenderJob]" = asyncio.Queue(maxsize=RENDER_QUEUE_MAXSIZE)

        self._tasks: List[asyncio.Task] = []
        self._running: bool = False
        self._loop: Optional[asyncio.AbstractEventLoop] = None

        self._last_text_time: float = time.time()
        self._last_visible_text: str = ""

        self._asr_latencies_ms: List[float] = []
        self._translation_latencies_ms: List[float] = []

    def start(self, loop: asyncio.AbstractEventLoop) -> None:
        if self._running:
            logger.warning("CaptionPipeline is already operational.")
            return

        self._running = True
        self._loop = loop
        self._last_text_time = time.time()

        self.audio_capture.start(loop)

        # Initialize the translator worker pool immediately at application startup
        loop.create_task(self.translator.ensure_workers_started())

        self._tasks = [
            loop.create_task(self._asr_worker(), name="ASRWorker"),
            loop.create_task(self._translation_worker(), name="TranslationWorker"),
            loop.create_task(self._render_worker(), name="RenderWorker"),
            loop.create_task(self._silence_watchdog(), name="SilenceWatchdog"),
            loop.create_task(self._stats_reporter(), name="StatsReporter"),
        ]
        logger.info("CaptionPipeline workers successfully provisioned and running.")

    def stop(self) -> None:
        if not self._running:
            return

        logger.info("Deactivating CaptionPipeline pipelines...")
        self._running = False
        self.audio_capture.stop()

        for task in self._tasks:
            task.cancel()
        self._tasks = []

        if self._loop is not None and self._loop.is_running():
            self._loop.create_task(self.translator.shutdown())

        logger.info("CaptionPipeline shutdown sequence finalized cleanly.")

    async def _asr_worker(self) -> None:
        logger.info("ASR worker stage entering service loop.")
        while self._running:
            try:
                chunk: AudioChunk = await asyncio.wait_for(
                    self.audio_capture.get_chunk(), timeout=QUEUE_WAIT_TIMEOUT_SECONDS
                )
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

            start_time = time.time()
            try:
                transcript: Transcript = await asyncio.to_thread(
                    self.whisper_engine.process_chunk, chunk.data, chunk.is_speech
                )
            except asyncio.CancelledError:
                break
            except Exception as asr_err:
                logger.error(f"ASR worker stage failure: {asr_err}", exc_info=True)
                await asyncio.sleep(STAGE_ERROR_BACKOFF_SECONDS)
                continue

            asr_latency_ms = (time.time() - start_time) * 1000.0
            self._asr_latencies_ms.append(asr_latency_ms)

            logger.info(
                "[ASR] %.0f ms | speech=%s | final=%s | text='%s'",
                asr_latency_ms,
                chunk.is_speech,
                transcript.is_final,
                transcript.text[:50],
            )

            if not transcript.text.strip() and not transcript.is_final:
                continue

            # Add debug logs for queue depths across the processing layers
            logger.debug(f"[Queue Depth Logs] ASR Queue Depth: {self._asr_queue.qsize()} | Render Queue Depth: {self._render_queue.qsize()}")

            envelope = _TimedTranscript(transcript=transcript, asr_latency_ms=asr_latency_ms)
            self._enqueue_dropping_oldest(self._asr_queue, envelope, "ASR")

        logger.info("ASR worker stage exited.")

    async def _translation_worker(self) -> None:
        logger.info("Translation worker stage entering service loop.")
        while self._running:
            try:
                envelope: _TimedTranscript = await asyncio.wait_for(
                    self._asr_queue.get(), timeout=QUEUE_WAIT_TIMEOUT_SECONDS
                )
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

            transcript = envelope.transcript
            if not transcript.is_final:
                continue

            source_text = (transcript.stable_text or transcript.text).strip()
            if not source_text:
                continue

            start_time = time.time()
            try:
                translation = await self.translator.translate(
                    source_text,
                    transcript.language,
                )
                translated_text = translation.translated_text
            except asyncio.CancelledError:
                break
            except Exception as translation_err:
                logger.error(f"Translation worker stage failure: {translation_err}", exc_info=True)
                await asyncio.sleep(STAGE_ERROR_BACKOFF_SECONDS)
                continue

            translation_latency_ms = (time.time() - start_time) * 1000.0
            self._translation_latencies_ms.append(translation_latency_ms)

            logger.info(
                "[Translation] %.0f ms | text='%s'",
                translation_latency_ms,
                translated_text[:50],
            )

            render_job = _RenderJob(
                text=translated_text,
                is_final=transcript.is_final,
                timestamp=time.time(),
                asr_latency_ms=envelope.asr_latency_ms,
                translation_latency_ms=translation_latency_ms,
            )
            self._enqueue_dropping_oldest(self._render_queue, render_job, "Render")

        logger.info("Translation worker stage exited.")

    async def _render_worker(self) -> None:
        logger.info("Render worker stage entering service loop.")
        while self._running:
            try:
                job: _RenderJob = await asyncio.wait_for(
                    self._render_queue.get(), timeout=QUEUE_WAIT_TIMEOUT_SECONDS
                )
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

            text = job.text.strip()
            logger.info(
                "[Render] final=%s | text='%s'",
                job.is_final,
                text[:50],
            )

            if text:
                self._last_text_time = time.time()
                self._last_visible_text = text
                self.overlay_window.caption_updated.emit(text)

            total_latency_ms = job.asr_latency_ms + job.translation_latency_ms
            logger.debug(
                f"Render pipeline latency: asr={job.asr_latency_ms:.1f}ms "
                f"translation={job.translation_latency_ms:.1f}ms total={total_latency_ms:.1f}ms "
                f"final={job.is_final}"
            )

        logger.info("Render worker stage exited.")

    async def _silence_watchdog(self) -> None:
        while self._running:
            try:
                await asyncio.sleep(SILENCE_WATCHDOG_INTERVAL_SECONDS)
            except asyncio.CancelledError:
                break

            if self._last_visible_text and (time.time() - self._last_text_time > self.clear_timeout):
                logger.debug("Silence watchdog boundary reached. Clearing visible overlay tracks.")
                self.overlay_window.caption_updated.emit("")
                self._last_visible_text = ""
                self.whisper_engine.flush_pipeline()
                self.translator.reset_incremental_state()

    async def _stats_reporter(self) -> None:
        while self._running:
            try:
                await asyncio.sleep(STATS_LOG_INTERVAL_SECONDS)
            except asyncio.CancelledError:
                break

            if self._asr_latencies_ms or self._translation_latencies_ms:
                avg_asr = sum(self._asr_latencies_ms) / len(self._asr_latencies_ms) if self._asr_latencies_ms else 0.0
                avg_translation = (
                    sum(self._translation_latencies_ms) / len(self._translation_latencies_ms)
                    if self._translation_latencies_ms else 0.0
                )
                logger.info(
                    "[Queues] capture=%d asr=%d render=%d | avg_asr=%.0fms avg_translation=%.0fms",
                    self.audio_capture._queue.qsize(),
                    self._asr_queue.qsize(),
                    self._render_queue.qsize(),
                    avg_asr,
                    avg_translation,
                )

            self._asr_latencies_ms = []
            self._translation_latencies_ms = []

    @staticmethod
    def _enqueue_dropping_oldest(queue: "asyncio.Queue", item: object, stage_name: str) -> None:
        try:
            queue.put_nowait(item)
        except asyncio.QueueFull:
            try:
                queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
            logger.warning(f"{stage_name} queue saturated; dropped oldest pending item to stay real-time.")
            try:
                queue.put_nowait(item)
            except asyncio.QueueFull:
                logger.error(f"{stage_name} queue still full after eviction; dropping newest item too.")