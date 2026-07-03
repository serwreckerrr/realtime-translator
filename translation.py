import asyncio
import logging
import re
import time
import urllib.parse
import urllib.request
import urllib.error
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, Optional, Protocol, Tuple

import orjson

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
    """Production-grade configurable translation backend interacting with a standard REST API.
    
    Eliminates reliance on unofficial scrapers and catches HTTP 429 responses to prevent 
    futile retries during server-side rate-limiting.
    """

    def __init__(
        self, 
        endpoint_url: str = "https://api.cognitive.microsofttranslator.com/translate", 
        api_key: Optional[str] = None, 
        timeout: float = BACKEND_HTTP_TIMEOUT_SECONDS
    ) -> None:
        self.endpoint_url: str = endpoint_url
        self.api_key: Optional[str] = api_key
        self.timeout: float = timeout
        self.user_agent: str = (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )

    def translate_text(self, text: str, source_lang: str, target_lang: str) -> str:
        if not text.strip():
            return ""

        params = {
            "api-version": "3.0",
            "from": source_lang,
            "to": target_lang
        }
        encoded_query = urllib.parse.urlencode(params)
        full_url = f"{self.endpoint_url}?{encoded_query}"
        body = orjson.dumps([{"Text": text}])

        request = urllib.request.Request(full_url, data=body, method="POST")
        request.add_header("User-Agent", self.user_agent)
        request.add_header("Content-Type", "application/json")
        if self.api_key:
            request.add_header("Ocp-Apim-Subscription-Key", self.api_key)

        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                raw_data = response.read()
                parsed_json = orjson.loads(raw_data)
                if parsed_json and isinstance(parsed_json, list) and "translations" in parsed_json[0]:
                    return parsed_json[0]["translations"][0]["text"]
                return text
        except urllib.error.HTTPError as http_err:
            if http_err.code == 429:
                logger.error("HTTP 429 Too Many Requests: Server-side blocking active. Skipping retries immediately.")
                return text
            logger.error(f"Translation backend HTTP error {http_err.code}: {http_err.reason}")
            return text
        except Exception as net_err:
            logger.error(f"Translation backend network transport failure: {net_err}")
            return text


class _LRUTTLCache:
    def __init__(self, max_size: int = CACHE_MAX_SIZE, ttl_seconds: float = CACHE_TTL_SECONDS) -> None:
        self.max_size: int = max_size
        self.ttl_seconds: float = ttl_seconds
        self._store: "OrderedDict[str, Tuple[str, float]]" = OrderedDict()

    def get(self, key: str) -> Optional[str]:
        entry = self._store.get(key)
        if entry is None:
            return None
        value, expires_at = entry
        if time.time() >= expires_at:
            del self._store[key]
            return None
        self._store.move_to_end(key)
        return value

    def set(self, key: str, value: str) -> None:
        self._store[key] = (value, time.time() + self.ttl_seconds)
        self._store.move_to_end(key)
        while len(self._store) > self.max_size:
            self._store.popitem(last=False)

    def clear(self) -> None:
        self._store.clear()


@dataclass
class _TranslationJob:
    text: str
    source_lang: str
    target_lang: str
    future: "asyncio.Future[str]" = field(compare=False)


class Translator:
    def __init__(self, backend: Optional[TranslationBackend] = None) -> None:
        self.backend: TranslationBackend = backend if backend is not None else ConfigurableTranslationBackend()
        self._cache: _LRUTTLCache = _LRUTTLCache(CACHE_MAX_SIZE, CACHE_TTL_SECONDS)
        self._queue: "asyncio.Queue[_TranslationJob]" = asyncio.Queue(maxsize=TRANSLATION_QUEUE_MAX_SIZE)
        self._workers: list = []
        self._workers_started: bool = False
        self._start_lock: Optional[asyncio.Lock] = None

        self._last_stable_source: str = ""
        self._last_stable_translation: str = ""
        self._whitespace_pattern: re.Pattern = re.compile(r"\s+")
        self._duplicate_punctuation_pattern: re.Pattern = re.compile(r"([.,!?:\-_])\1+")

    async def ensure_workers_started(self) -> None:
        """Public initialization endpoint to spin up the worker pool during application startup."""
        await self._ensure_workers_started()

    async def translate(self, text: str, source_lang: str) -> Translation:
        cleaned_source = self._normalize_text(text)
        if not cleaned_source:
            return self._empty_translation(text, source_lang)

        if source_lang.lower() == TARGET_LANGUAGE:
            return Translation(
                original_text=text, translated_text=cleaned_source, source_lang=source_lang,
                target_lang=TARGET_LANGUAGE, timestamp=time.time(), cached=False,
            )

        cache_key = self._cache_key(source_lang, cleaned_source)
        cached_value = self._cache.get(cache_key)
        if cached_value is not None:
            return Translation(
                original_text=text, translated_text=cached_value, source_lang=source_lang,
                target_lang=TARGET_LANGUAGE, timestamp=time.time(), cached=True,
            )

        try:
            raw = await self._submit_job(cleaned_source, source_lang, TARGET_LANGUAGE)
            final_translation = self._cleanup_punctuation(raw)
            if final_translation:
                self._cache.set(cache_key, final_translation)
            return Translation(
                original_text=text, translated_text=final_translation, source_lang=source_lang,
                target_lang=TARGET_LANGUAGE, timestamp=time.time(), cached=False,
            )
        except Exception as ex:
            logger.critical(f"Unexpected fatal error inside Translator routine pipeline: {ex}", exc_info=True)
            return Translation(
                original_text=text, translated_text=cleaned_source, source_lang=source_lang,
                target_lang=TARGET_LANGUAGE, timestamp=time.time(), cached=False,
            )

    async def translate_incremental(self, stable_text: str, live_text: str, source_lang: str) -> Translation:
        combined_source = (stable_text + " " + live_text).strip() if live_text else stable_text

        if source_lang.lower() == TARGET_LANGUAGE:
            return Translation(
                original_text=combined_source, translated_text=self._normalize_text(combined_source),
                source_lang=source_lang, target_lang=TARGET_LANGUAGE, timestamp=time.time(), cached=False,
            )

        translated_stable = await self._translate_stable_incremental(stable_text, source_lang)
        translated_live = ""
        if live_text.strip():
            cleaned_live = self._normalize_text(live_text)
            cache_key = self._cache_key(source_lang, cleaned_live)
            cached_value = self._cache.get(cache_key)
            if cached_value is not None:
                translated_live = cached_value
            else:
                raw_live = await self._submit_job(cleaned_live, source_lang, TARGET_LANGUAGE)
                translated_live = self._cleanup_punctuation(raw_live)
                if translated_live:
                    self._cache.set(cache_key, translated_live)

        merged = (translated_stable + " " + translated_live).strip() if translated_live else translated_stable
        return Translation(
            original_text=combined_source, translated_text=merged, source_lang=source_lang,
            target_lang=TARGET_LANGUAGE, timestamp=time.time(), cached=False,
        )

    def reset_incremental_state(self) -> None:
        self._last_stable_source = ""
        self._last_stable_translation = ""

    def clear_cache(self) -> None:
        self._cache.clear()
        logger.info("Translation hot cache cleared successfully.")

    async def shutdown(self) -> None:
        for task in self._workers:
            task.cancel()
        if self._workers:
            await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers = []
        self._workers_started = False
        logger.info("Translator worker pool shut down cleanly.")

    async def _translate_stable_incremental(self, stable_text: str, source_lang: str) -> str:
        if not stable_text:
            return self._last_stable_translation if self._last_stable_source == "" else ""
        if stable_text == self._last_stable_source:
            return self._last_stable_translation

        if self._last_stable_source and stable_text.startswith(self._last_stable_source):
            new_portion = stable_text[len(self._last_stable_source):].strip()
            if not new_portion:
                return self._last_stable_translation
            try:
                raw_new = await self._submit_job(new_portion, source_lang, TARGET_LANGUAGE)
                translated_new = self._cleanup_punctuation(raw_new)
            except Exception as ex:
                logger.error(f"Incremental stable-segment translation failed, falling back: {ex}")
                translated_new = new_portion
            combined = (self._last_stable_translation + " " + translated_new).strip()
        else:
            try:
                raw_full = await self._submit_job(stable_text, source_lang, TARGET_LANGUAGE)
                combined = self._cleanup_punctuation(raw_full)
            except Exception as ex:
                logger.error(f"Full stable-segment translation failed, falling back: {ex}")
                combined = stable_text

        self._last_stable_source = stable_text
        self._last_stable_translation = combined
        return combined

    async def _submit_job(self, text: str, source_lang: str, target_lang: str) -> str:
        await self._ensure_workers_started()
        loop = asyncio.get_running_loop()
        job_future: "asyncio.Future[str]" = loop.create_future()
        job = _TranslationJob(text=text, source_lang=source_lang, target_lang=target_lang, future=job_future)

        await self._queue.put(job)
        try:
            return await asyncio.wait_for(job_future, timeout=JOB_RESULT_TIMEOUT_SECONDS)
        except asyncio.TimeoutError:
            logger.warning("Translation job timed out waiting for a worker; returning source text unchanged.")
            return text

    async def _ensure_workers_started(self) -> None:
        if self._workers_started:
            return
        if self._start_lock is None:
            self._start_lock = asyncio.Lock()

        async with self._start_lock:
            if self._workers_started:
                return
            for worker_id in range(TRANSLATION_WORKER_COUNT):
                task = asyncio.create_task(self._worker_loop(worker_id), name=f"TranslatorWorker-{worker_id}")
                self._workers.append(task)
            self._workers_started = True
            logger.info(f"Translator worker pool started with {TRANSLATION_WORKER_COUNT} workers.")

    async def _worker_loop(self, worker_id: int) -> None:
        logger.debug(f"Translator worker {worker_id} entering service loop.")
        while True:
            try:
                job = await self._queue.get()
            except asyncio.CancelledError:
                break

            try:
                start_time = time.time()
                raw_result = await asyncio.to_thread(
                    self.backend.translate_text, job.text, job.source_lang, job.target_lang
                )
                latency_ms = (time.time() - start_time) * 1000.0
                logger.debug(f"Translator worker {worker_id} completed job in {latency_ms:.1f}ms.")
                if not job.future.done():
                    job.future.set_result(raw_result)
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
        if len(text) > 1:
            text = text[0].upper() + text[1:]
        elif len(text) == 1:
            text = text.upper()
        return text.strip()