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
    sequence_id: int = 0


@dataclass
class _RenderJob:
    text: str
    is_final: bool
    timestamp: float
    asr_latency_ms: float = 0.0
    translation_latency_ms: float = 0.0
    sequence_id: int = 0


class CaptionPipeline:
    def __init__(
        self,
        audio_capture: AudioCapture,
        whisper_engine: WhisperEngine,
        translator: Translator,
        overlay_window: OverlayWindow,
        clear_timeout: float = DEFAULT_CLEAR_TIMEOUT_SECONDS,
    ) -> None:
        self.audio_capture = audio_capture
        self.whisper_engine = whisper_engine
        self.translator = translator
        self.overlay_window = overlay_window
        self._clear_timeout = clear_timeout

        self._asr_queue: asyncio.Queue[_TimedTranscript] = asyncio.Queue(maxsize=ASR_QUEUE_MAXSIZE)
        self._render_queue: asyncio.Queue[_RenderJob] = asyncio.Queue(maxsize=RENDER_QUEUE_MAXSIZE)

        self._running: bool = False
        self._loop: Optional[asyncio.AbstractEventLoop] = None

        self._asr_task: Optional[asyncio.Task] = None
        self._translation_task: Optional[asyncio.Task] = None
        self._render_task: Optional[asyncio.Task] = None
        self._watchdog_task: Optional[asyncio.Task] = None
        self._stats_task: Optional[asyncio.Task] = None

        self._last_caption_time: float = 0.0
        
        # State tracker to prevent redundant API calls on identical partial transcripts
        self._last_translated_source_text: str = ""

        # Monotonically increasing counter to prevent out-of-order stale updates
        self._transcript_counter: int = 0
        self._max_rendered_sequence_id: int = 0

        # Metrics for real-time tracking
        self._asr_latencies_ms: List[float] = []
        self._translation_latencies_ms: List[float] = []

    def start(self, loop: asyncio.AbstractEventLoop) -> None:
        if self._running:
            return
        self._running = True
        self._loop = loop

        logger.info("CaptionPipeline workers successfully provisioned and running.")
        self.audio_capture.start(loop)
        self.translator.start(loop)

        self._asr_task = self._loop.create_task(self._asr_worker())
        self._translation_task = self._loop.create_task(self._translation_worker())
        self._render_task = self._loop.create_task(self._render_worker())
        self._watchdog_task = self._loop.create_task(self._silence_watchdog())
        self._stats_task = self._loop.create_task(self._stats_reporter())

    def stop(self) -> None:
        if not self._running:
            return
        self._running = False
        logger.info("CaptionPipeline workers instructed to shut down.")

        self.audio_capture.stop()
        self.translator.stop()

        tasks = [
            self._asr_task,
            self._translation_task,
            self._render_task,
            self._watchdog_task,
            self._stats_task,
        ]
        for task in tasks:
            if task and not task.done():
                task.cancel()

    async def _asr_worker(self) -> None:
        logger.info("ASR worker stage entering service loop.")
        while self._running:
            try:
                try:
                    # get_chunk() is already an async coroutine (no timeout arg) —
                    # await it directly and bound the wait with asyncio.wait_for.
                    chunk: AudioChunk = await asyncio.wait_for(
                        self.audio_capture.get_chunk(), timeout=QUEUE_WAIT_TIMEOUT_SECONDS
                    )
                except asyncio.TimeoutError:
                    continue
                
                if chunk is None:
                    continue

                start_time = time.time()
                
                transcript = await asyncio.to_thread(
                    self.whisper_engine.process_chunk, chunk.data, chunk.is_speech
                )
                
                asr_latency_ms = (time.time() - start_time) * 1000.0
                self._asr_latencies_ms.append(asr_latency_ms)
                
                logger.info(
                    "[ASR] %.0f ms | speech=%s | final=%s | text='%s'",
                    asr_latency_ms,
                    chunk.is_speech,
                    transcript.is_final,
                    transcript.text[:50],
                )

                self._transcript_counter += 1
                timed_transcript = _TimedTranscript(
                    transcript=transcript, 
                    asr_latency_ms=asr_latency_ms,
                    sequence_id=self._transcript_counter
                )
                self._enqueue_dropping_oldest(self._asr_queue, timed_transcript, "ASR")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"ASR worker failure: {e}", exc_info=True)
                await asyncio.sleep(STAGE_ERROR_BACKOFF_SECONDS)

    async def _translation_worker(self) -> None:
        logger.info("Translation worker stage entering service loop.")
        while self._running:
            try:
                try:
                    timed_transcript = await asyncio.wait_for(
                        self._asr_queue.get(), timeout=QUEUE_WAIT_TIMEOUT_SECONDS
                    )
                except asyncio.TimeoutError:
                    continue

                # Array to record all items pulled from the queue in this batch for precise bookkeeping
                items_to_commit = [timed_transcript]
                chosen_transcript = timed_transcript

                # Safe Queue Draining: If the current item is a partial transcript, look ahead 
                # to catch up to real-time. Stop draining immediately upon hitting a final 
                # transcript to preserve ordering and protect subsequent utterances.
                if not chosen_transcript.transcript.is_final:
                    while not self._asr_queue.empty():
                        try:
                            next_item = self._asr_queue.get_nowait()
                            items_to_commit.append(next_item)
                            
                            if next_item.transcript.is_final:
                                chosen_transcript = next_item
                                break
                            else:
                                chosen_transcript = next_item
                        except asyncio.QueueEmpty:
                            break

                # Wrap item processing and final task validation inside a try...finally block 
                # to guarantee that task_done() is executed exactly once per retrieved queue item.
                try:
                    transcript = chosen_transcript.transcript

                    # Skip processing if transcript text is empty
                    if not transcript.text or not transcript.text.strip():
                        continue

                    # Duplicate suppression: Avoid retranslating identical text back-to-back
                    if not transcript.is_final and transcript.text == self._last_translated_source_text:
                        continue

                    start_time = time.time()
                    
                    # Core translation step
                    translation = await self.translator.translate(transcript.text, transcript.language)
                    
                    translation_latency_ms = (time.time() - start_time) * 1000.0
                    self._translation_latencies_ms.append(translation_latency_ms)

                    # Update source text cache tracker
                    self._last_translated_source_text = transcript.text
                    
                    # Reset the suppression cache edge when a final sentence is locked in
                    if transcript.is_final:
                        self._last_translated_source_text = ""

                    render_job = _RenderJob(
                        text=translation.translated_text,
                        is_final=transcript.is_final,
                        timestamp=time.time(),
                        asr_latency_ms=chosen_transcript.asr_latency_ms,
                        translation_latency_ms=translation_latency_ms,
                        sequence_id=chosen_transcript.sequence_id
                    )
                    
                    self._enqueue_dropping_oldest(self._render_queue, render_job, "Translation")

                finally:
                    # Explicit queue accounting: 1 task_done() per successful queue extraction
                    for item in items_to_commit:
                        self._asr_queue.task_done()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Translation worker failure: {e}", exc_info=True)
                await asyncio.sleep(STAGE_ERROR_BACKOFF_SECONDS)

    async def _render_worker(self) -> None:
        logger.info("Render worker stage entering service loop.")
        while self._running:
            try:
                try:
                    job = await asyncio.wait_for(
                        self._render_queue.get(), timeout=QUEUE_WAIT_TIMEOUT_SECONDS
                    )
                except asyncio.TimeoutError:
                    continue
                
                try:
                    # Prevent stale out-of-order translations from reaching the UI overlay
                    if job.sequence_id >= self._max_rendered_sequence_id:
                        self._max_rendered_sequence_id = job.sequence_id
                        if job.text and job.text.strip():
                            self.overlay_window.update_caption(job.text, is_final=job.is_final)
                            self._last_caption_time = time.time()
                finally:
                    self._render_queue.task_done()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Render worker failure: {e}", exc_info=True)
                await asyncio.sleep(STAGE_ERROR_BACKOFF_SECONDS)

    async def _silence_watchdog(self) -> None:
        """Clears the overlay if no new captions have arrived within the timeout period."""
        while self._running:
            try:
                await asyncio.sleep(SILENCE_WATCHDOG_INTERVAL_SECONDS)
                if self._last_caption_time > 0:
                    idle_time = time.time() - self._last_caption_time
                    if idle_time >= self._clear_timeout:
                        self.overlay_window.clear_caption()
                        self._last_caption_time = 0.0
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Watchdog failure: {e}", exc_info=True)

    async def _stats_reporter(self) -> None:
        while self._running:
            try:
                await asyncio.sleep(STATS_LOG_INTERVAL_SECONDS)
                
                if self._asr_latencies_ms or self._translation_latencies_ms:
                    avg_asr = sum(self._asr_latencies_ms) / len(self._asr_latencies_ms) if self._asr_latencies_ms else 0.0
                    avg_translation = (
                        sum(self._translation_latencies_ms) / len(self._translation_latencies_ms)
                        if self._translation_latencies_ms else 0.0
                    )
                    logger.info(
                        "[Queues] capture=%d asr=%d render=%d | avg_asr=%.0fms avg_translation=%.0fms",
                        self.audio_capture._queue.qsize() if hasattr(self.audio_capture, '_queue') else 0,
                        self._asr_queue.qsize(),
                        self._render_queue.qsize(),
                        avg_asr,
                        avg_translation,
                    )

                self._asr_latencies_ms = []
                self._translation_latencies_ms = []

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Stats reporter failure: {e}", exc_info=True)

    @staticmethod
    def _enqueue_dropping_oldest(queue: asyncio.Queue, item: object, stage_name: str) -> None:
        """Maintains low latency by evicting stale queue entries if components fall behind."""
        try:
            queue.put_nowait(item)
        except asyncio.QueueFull:
            try:
                queue.get_nowait()
                queue.task_done()
            except asyncio.QueueEmpty:
                pass
            logger.warning(f"{stage_name} queue saturated; dropped oldest pending item to stay real-time.")
            try:
                queue.put_nowait(item)
            except asyncio.QueueFull:
                pass