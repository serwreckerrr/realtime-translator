import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import whisper

from audio import RingBuffer, TARGET_SAMPLE_RATE

logger = logging.getLogger("realtime_translator.asr")

# --------------------------------------------------------------------------
# Configuration constants
# --------------------------------------------------------------------------
WINDOW_SECONDS: float = 5.0               
OVERLAP_SECONDS: float = 2.0              
STEP_SECONDS: float = 0.8                 
UNSTABLE_TAIL_SECONDS: float = OVERLAP_SECONDS          
MIN_DECODE_SECONDS: float = 1.0             
SILENCE_FLUSH_SECONDS: float = 0.6          # Increased from 1.2 to optimize end-of-speech precision
MAX_UTTERANCE_SECONDS: float = 8            # Hard limit on utterance length to prevent runaway memory usage
MAX_PROMPT_CHARS: int = 200                 
LANGUAGE_LOCK_MIN_CONFIDENCE: float = 0.65   
MIN_TOKENS_FOR_LANG_LOCK: int = 4          
MAX_ALIGNMENT_HISTORY_TOKENS: int = 150     # Bounding cap on word tokens to eliminate memory latency growth


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
            i += 1
            j += 1
        elif dp[i + 1][j] >= dp[i][j + 1]:
            i += 1
        else:
            j += 1

    return pairs


class WhisperEngine:
    def __init__(self, model_name: str = "base", device: Optional[str] = None) -> None:
        if device is None:
            self.device: str = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        logger.info(f"Loading OpenAI Whisper model '{model_name}' on device: {self.device}")
        self.model: whisper.Whisper = whisper.load_model(model_name, device=self.device)
        logger.info("Whisper model loaded successfully.")

        self.sample_rate: int = TARGET_SAMPLE_RATE
        self.ring_buffer: RingBuffer = RingBuffer(WINDOW_SECONDS, self.sample_rate)

        self._min_decode_samples: int = int(MIN_DECODE_SECONDS * self.sample_rate)
        self._total_samples_seen: int = 0
        self._samples_since_last_decode: int = 0
        self._step_samples: int = int(STEP_SECONDS * TARGET_SAMPLE_RATE)

        self._committed_until_time: float = 0.0
        self._silence_seconds_accum: float = 0.0

        self._confirmed_word_tokens: List[WordToken] = []
        self._previous_hypothesis: List[WordToken] = []

        self.detected_language: str = "en"
        self._language_locked: bool = False
        self._lang_voting_history: List[str] = []

    def reset_stream_context(self) -> None:
        self._total_samples_seen = 0
        self._samples_since_last_decode = 0
        self._committed_until_time = 0.0
        self._silence_seconds_accum = 0.0
        self._confirmed_word_tokens = []
        self._previous_hypothesis = []
        self._lang_voting_history.clear()
        self.ring_buffer.clear()
        logger.debug("WhisperEngine stream context reset completely.")

    def process_chunk(self, chunk_data: np.ndarray, is_speech: bool) -> Transcript:
        chunk_samples = int(chunk_data.shape[0]) if chunk_data is not None else 0
        chunk_duration = chunk_samples / self.sample_rate if chunk_samples else 0.0

        if chunk_samples > 0:
            self.ring_buffer.append(chunk_data)
            self._total_samples_seen += chunk_samples
            self._samples_since_last_decode += chunk_samples

        if is_speech:
            self._silence_seconds_accum += 0.0
        else:
            self._silence_seconds_accum += chunk_duration

        ends_with_sentence_boundary = False
        if self._confirmed_word_tokens:
            stable_text = self._render_words([w.text for w in self._confirmed_word_tokens]).strip()
            if stable_text.endswith(('.', '?', '...')):
                ends_with_sentence_boundary = True

        has_pending_audio = self.ring_buffer.filled_seconds > 0.0 or bool(self._confirmed_word_tokens)
        should_flush = has_pending_audio and (
            self._silence_seconds_accum >= SILENCE_FLUSH_SECONDS 
            or ends_with_sentence_boundary
        )

        decode_due = (is_speech and self._samples_since_last_decode >= self._step_samples)
        buffer_ready = self.ring_buffer.filled_seconds * self.sample_rate >= self._min_decode_samples

        if not ((decode_due or should_flush) and buffer_ready):
            return Transcript(
                text="", language=self.detected_language, is_final=False,
                timestamp=time.time(), confidence=0.0,
            )

        try:
            return self._decode_and_stabilize(should_flush)
        except Exception as e:
            logger.error(f"Execution failure during Whisper engine inference decoding: {e}", exc_info=True)
            return Transcript(
                text="", language=self.detected_language, is_final=False,
                timestamp=time.time(), confidence=0.0,
            )

    def flush_pipeline(self) -> None:
        self.ring_buffer.clear()
        self._confirmed_word_tokens = []
        self._previous_hypothesis = []
        self._committed_until_time = self._total_samples_seen / self.sample_rate
        self._samples_since_last_decode = 0
        self._silence_seconds_accum = 0.0
        logger.debug("WhisperEngine stream state flushed; ready for a new phrase.")

    def _decode_and_stabilize(self, should_flush: bool, ring_history: Any = None) -> Transcript:
        active_buffer = ring_history if ring_history is not None else self.ring_buffer
        window_end_absolute_seconds = self._total_samples_seen / TARGET_SAMPLE_RATE

        # FIX: Remove '+ OVERLAP_SECONDS'. uncommitted_duration already naturally includes 
        # the UNSTABLE_TAIL_SECONDS (2.0s) from the previous iteration along with the new 
        # STEP_SECONDS (0.8s), providing the exact sliding context needed.
        uncommitted_duration = window_end_absolute_seconds - self._committed_until_time
        # required_duration = uncommitted_duration + OVERLAP_SECONDS
        required_duration = uncommitted_duration
        decode_duration = max(MIN_DECODE_SECONDS, min(required_duration, WINDOW_SECONDS))

        window_audio = active_buffer.get_last(decode_duration)
        self._samples_since_last_decode = 0

        actual_window_len_seconds = len(window_audio) / self.sample_rate
        window_start_absolute_seconds = window_end_absolute_seconds - actual_window_len_seconds

        with torch.inference_mode():
            inference_start = time.time()
            result: Dict[str, Any] = self.model.transcribe(
                window_audio,
                language=self.detected_language if self._language_locked else None,
                task="transcribe",
                temperature=0.0,
                word_timestamps=True,
                condition_on_previous_text=False,
                initial_prompt=self._build_prompt(),
                fp16=(self.device == "cuda" and torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 7),
                beam_size=1,        
                best_of=1,
                verbose=None,
            )
            # Added debug logging for Whisper inference execution speed
            logger.debug(f"[ASR] Whisper inference time: {(time.time() - inference_start) * 1000.0:.2f}ms")

        detected_language = result.get("language")
        current_hyp = self._extract_word_tokens(result, window_start_absolute_seconds)
        avg_probability = float(np.mean([w.probability for w in current_hyp])) if current_hyp else 0.0

        if detected_language and not self._language_locked:
            self._lang_voting_history.append(detected_language)
            if len(self._lang_voting_history) > 3:
                self._lang_voting_history.pop(0)

            if (len(current_hyp) >= MIN_TOKENS_FOR_LANG_LOCK and 
                avg_probability >= LANGUAGE_LOCK_MIN_CONFIDENCE and 
                self._lang_voting_history.count(detected_language) >= 2):
                
                self.detected_language = detected_language
                self._language_locked = True
                logger.info(f"Language context successfully locked to: '{self.detected_language}'")
        elif detected_language and not self._language_locked:
            self.detected_language = detected_language

        self._merge_hypothesis(current_hyp, window_end_absolute_seconds, should_flush)

        stable_text = self._render_words([w.text for w in self._confirmed_word_tokens])
        live_tokens = [w for w in current_hyp if w.end > self._committed_until_time]
        live_text = self._render_words([w.text for w in live_tokens])

        if should_flush:
            if live_tokens:
                self._confirmed_word_tokens.extend(live_tokens)
                stable_text = self._render_words([w.text for w in self._confirmed_word_tokens])
            final_text = stable_text
            
            # Prevent empty finalized transcripts from entering the pipeline
            if not final_text.strip():
                self.flush_pipeline()
                return Transcript(
                    text="", language=self.detected_language, is_final=False,
                    timestamp=time.time(), confidence=0.0,
                    stable_text="", live_text=""
                )

            logger.debug(f"[ASR] Finalized transcript generation event: '{final_text}'")
            self.flush_pipeline()
            return Transcript(
                text=final_text, language=self.detected_language, is_final=True,
                timestamp=time.time(), confidence=avg_probability,
                stable_text=final_text, live_text="",
            )

        self._previous_hypothesis = current_hyp
        combined_text = (stable_text + " " + live_text).strip() if live_text else stable_text

        return Transcript(
            text=combined_text, language=self.detected_language, is_final=False,
            timestamp=time.time(), confidence=avg_probability,
            stable_text=stable_text, live_text=live_text,
        )

    def _merge_hypothesis(
        self, current_hyp: List[WordToken], window_end_abs: float, should_flush: bool
    ) -> None:
        if should_flush or not current_hyp:
            return

        open_curr = [w for w in current_hyp if w.end > self._committed_until_time]
        if not open_curr:
            return

        time_cutoff = window_end_abs - UNSTABLE_TAIL_SECONDS
        time_confirm_idx = -1
        for idx, w in enumerate(open_curr):
            if w.end <= time_cutoff:
                time_confirm_idx = idx
            else:
                break

        if not self._previous_hypothesis:
            agree_confirm_idx = time_confirm_idx  
        else:
            open_prev = [w for w in self._previous_hypothesis if w.end > self._committed_until_time]
            matches = _lcs_matched_pairs(open_prev, open_curr)
            agree_confirm_idx = matches[-1][1] if matches else -1

        final_confirm_idx = min(agree_confirm_idx, time_confirm_idx)
        
        if final_confirm_idx < 0:
            uncommitted_age = window_end_abs - self._committed_until_time
            if uncommitted_age >= (WINDOW_SECONDS - 1.0) and time_confirm_idx >= 0:
                logger.warning("LCS synchronization stalled due to textual drift; deploying safety time-based fallback.")
                final_confirm_idx = time_confirm_idx
            else:
                return

        newly_confirmed = open_curr[: final_confirm_idx + 1]
        if not newly_confirmed:
            return

        self._confirmed_word_tokens.extend(newly_confirmed)

        # BOUNDED STATE HISTORY: Prevent historical token lists from growing indefinitely
        if len(self._confirmed_word_tokens) > MAX_ALIGNMENT_HISTORY_TOKENS:
            self._confirmed_word_tokens = self._confirmed_word_tokens[-MAX_ALIGNMENT_HISTORY_TOKENS:]

        self._committed_until_time = newly_confirmed[-1].end

    def _extract_word_tokens(self, result: Dict[str, Any], window_start_abs: float) -> List[WordToken]:
        tokens: List[WordToken] = []
        hallucination_blacklist = {"thank you", "thanks for watching", "subtitles by", "amara.org"}

        for segment in result.get("segments", []):
            if segment.get("temperature", 0.0) > 0.5 and segment.get("avg_logprob", 0.0) < -1.0:
                continue
            for word_info in segment.get("words", []):
                text = word_info.get("word", "")
                if not text or not text.strip():
                    continue
                normalized = _normalize_word(text)
                if normalized in hallucination_blacklist:
                    continue

                tokens.append(
                    WordToken(
                        text=text,
                        start=window_start_abs + float(word_info.get("start", 0.0)),
                        end=window_start_abs + float(word_info.get("end", 0.0)),
                        probability=float(word_info.get("probability", 0.0)),
                    )
                )
        tokens.sort(key=lambda w: w.start)
        return tokens

    def _render_words(self, words: List[str]) -> str:
        return "".join(words).strip()

    def _build_prompt(self) -> Optional[str]:
        if not self._confirmed_word_tokens:
            return None
        tail_text = self._render_words([w.text for w in self._confirmed_word_tokens])
        if not tail_text:
            return None
        if len(tail_text) <= MAX_PROMPT_CHARS:
            return tail_text
        truncated = tail_text[-MAX_PROMPT_CHARS:]
        first_space_idx = truncated.find(" ")
        if first_space_idx != -1:
            return truncated[first_space_idx + 1:]
        return truncated