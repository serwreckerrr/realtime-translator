import asyncio
import logging
import os
import re
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, Optional, Protocol, Tuple, List

import orjson
from dotenv import load_dotenv
from google import genai
from google.genai import types

# Load environment variables early to acquire configuration contexts
load_dotenv()

logger = logging.getLogger("realtime_translator.translation")

# --------------------------------------------------------------------------
# Configuration constants
# --------------------------------------------------------------------------
CACHE_MAX_SIZE: int = 500
CACHE_TTL_SECONDS: float = 300.0
TRANSLATION_WORKER_COUNT: int = 3
TRANSLATION_QUEUE_MAX_SIZE: int = 200
JOB_RESULT_TIMEOUT_SECONDS: float = 8.0
BACKEND_HTTP_TIMEOUT_SECONDS: float = 3.0
TARGET_LANGUAGE: str = "vi"


@dataclass
class Translation:
    original_text: str
    translated_text: str
    source_lang: str
    target_lang: str
    timestamp: float
    cached: bool


class TranslationBackend(Protocol):
    """Protocol defining the structural contract for isolated translation backends."""
    def translate_text(self, text: str, source_lang: str, target_lang: str) -> str:
        ...


class ConfigurableTranslationBackend:
    """Production-grade configurable translation backend interacting with the Google Gemini API.
    
    Eliminates external scraping risks and utilizes low-latency deterministic model configurations.
    """
    def __init__(self) -> None:
        self.api_key = os.getenv("GEMINI_API_KEY")
        self.client: Optional[genai.Client] = None
        
        if not self.api_key:
            logger.error(
                "CRITICAL STARTUP ERROR: GEMINI_API_KEY is not configured in the environment or .env file. "
                "The pipeline will continue running but will pass source text unchanged."
            )
        else:
            try:
                # Initialize exactly one single reusable Gemini client structure
                self.client = genai.Client(api_key=self.api_key)
                logger.info("Google Gemini translation infrastructure established successfully.")
            except Exception as init_err:
                logger.error(f"Failed to bootstrap Google Gemini SDK Client: {init_err}", exc_info=True)

    def translate_text(self, text: str, source_lang: str, target_lang: str) -> str:
        if not text or not text.strip():
            return ""

        if not self.client:
            return text

        # Construct a strict deterministic translation blueprint via system instructions
        system_instruction = (
            f"Translate English into Vietnamese.\n"
            f"Preserve names.\n"
            f"Preserve numbers.\n"
            f"Preserve punctuation.\n"
            f"Preserve line order.\n"
            f"Do not explain.\n"
            f"Do not add quotation marks.\n"
            f"Output only the translated text."
        )

        try:
            response = self.client.models.generate_content(
                model="gemini-3.1-flash-lite",
                contents=text,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=0.0,
                    top_p=1.0,
                    top_k=1,
                )
            )
            
            if response and response.text:
                return response.text.strip()
            
            logger.warning("Empty output sequence received from Gemini API. Returning source payload text.")
            return text

        except Exception as api_err:
            # Gracefully catch network timeouts, rate limits, quota limits, and authentication errors
            logger.error(f"Gemini execution layer exception captured during translation request: {api_err}")
            return text


@dataclass
class _TranslationJob:
    text: str
    source_lang: str
    future: asyncio.Future = field(default_factory=asyncio.Future)


class Translator:
    """Manages system-level translation loops, underlying thread pools, and cache validation hooks."""
    def __init__(self, backend: TranslationBackend) -> None:
        self._backend = backend
        self._cache: OrderedDict[str, Translation] = OrderedDict()
        self._cache_lock = threading.Lock()
        
        self._queue: asyncio.Queue[_TranslationJob] = asyncio.Queue(maxsize=TRANSLATION_QUEUE_MAX_SIZE)
        self._worker_tasks: List[asyncio.Task] = []
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        
        self._whitespace_pattern = re.compile(r"\s+")
        self._duplicate_punctuation_pattern = re.compile(r"([.,!?])\1+")

    def start(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop
        for i in range(TRANSLATION_WORKER_COUNT):
            task = loop.create_task(self._worker_loop(i))
            self._worker_tasks.append(task)
        logger.info(f"Initialized translator execution space with {TRANSLATION_WORKER_COUNT} async loops.")

    def stop(self) -> None:
        for task in self._worker_tasks:
            task.cancel()
        self._worker_tasks.clear()
        logger.info("Translation async loop workers torn down successfully.")

    async def translate(self, text: str, source_lang: str) -> Translation:
        cleaned_text = self._normalize_text(text)
        if not cleaned_text:
            return self._empty_translation(text, source_lang)

        key = self._cache_key(source_lang, cleaned_text)
        with self._cache_lock:
            if key in self._cache:
                translation = self._cache[key]
                self._cache.move_to_end(key)
                return Translation(
                    original_text=text,
                    translated_text=translation.translated_text,
                    source_lang=source_lang,
                    target_lang=TARGET_LANGUAGE,
                    timestamp=time.time(),
                    cached=True
                )

        job = _TranslationJob(text=cleaned_text, source_lang=source_lang)
        try:
            self._queue.put_nowait(job)
        except asyncio.QueueFull:
            logger.warning("Translation job queue structural saturation. Evicting incoming translation workload.")
            return self._empty_translation(text, source_lang)

        try:
            translated_text = await asyncio.wait_for(job.future, timeout=JOB_RESULT_TIMEOUT_SECONDS)
            
            with self._cache_lock:
                if len(self._cache) >= CACHE_MAX_SIZE:
                    self._cache.popitem(last=False)
                new_trans = Translation(
                    original_text=cleaned_text,
                    translated_text=translated_text,
                    source_lang=source_lang,
                    target_lang=TARGET_LANGUAGE,
                    timestamp=time.time(),
                    cached=False
                )
                self._cache[key] = new_trans
            
            return Translation(
                original_text=text,
                translated_text=self._cleanup_punctuation(translated_text),
                source_lang=source_lang,
                target_lang=TARGET_LANGUAGE,
                timestamp=time.time(),
                cached=False
            )
        except asyncio.TimeoutError:
            logger.warning(f"Translation pipeline stage dropped due to timeout restriction bounds (> {JOB_RESULT_TIMEOUT_SECONDS}s).")
            return self._empty_translation(text, source_lang)
        except Exception as exc:
            logger.error(f"Uncaught failure inside async translation interface context: {exc}")
            return self._empty_translation(text, source_lang)

    async def translate_incremental(self, text: str, source_lang: str) -> Translation:
        return await self.translate(text, source_lang)

    async def _worker_loop(self, worker_id: int) -> None:
        while True:
            try:
                job = await self._queue.get()
            except asyncio.CancelledError:
                break

            try:
                loop = asyncio.get_running_loop()
                translated = await loop.run_in_executor(
                    None,
                    self._backend.translate_text,
                    job.text,
                    job.source_lang,
                    TARGET_LANGUAGE
                )
                if not job.future.done():
                    job.future.set_result(translated)
            except asyncio.CancelledError:
                if not job.future.done():
                    job.future.cancel()
                break
            except Exception as job_err:
                logger.error(f"Translator worker {worker_id} job failure: {job_err}", exc_info=True)
                if not job.future.done():
                    job.future.set_result(job.text)
            finally:
                self._queue.task_done()

    def _cache_key(self, source_lang: str, cleaned_source: str) -> str:
        return f"{source_lang}:{TARGET_LANGUAGE}:{cleaned_source.lower()}"

    def _empty_translation(self, text: str, source_lang: str) -> Translation:
        return Translation(
            original_text=text, translated_text="", source_lang=source_lang,
            target_lang=TARGET_LANGUAGE, timestamp=time.time(), cached=False,
        )

    def _normalize_text(self, text: str) -> str:
        if not text:
            return ""
        text = self._whitespace_pattern.sub(" ", text)
        return text.strip()

    def _cleanup_punctuation(self, text: str) -> str:
        if not text:
            return ""
        text = self._duplicate_punctuation_pattern.sub(r"\1", text)
        text = text.replace(" .", ".").replace(" ,", ",").replace(" ?", "?").replace(" !", "!")
        return text.strip()