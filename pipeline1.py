import asyncio
import logging
from urllib import response
from google import genai
from google.genai import types

from audio1 import AudioCapture
from overlay1 import OverlayWindow

logger = logging.getLogger("realtime_translator.pipeline")

class CaptionPipeline:
    """Manages the bidirectional WebSocket connection with Gemini Live."""
    def __init__(self, audio_capture: AudioCapture, overlay_window: OverlayWindow):
        self.audio = audio_capture
        self.overlay = overlay_window
        self.client = genai.Client() # Uses GEMINI_API_KEY from env
        self._running = False
        self._task = None

    def start(self, loop: asyncio.AbstractEventLoop) -> None:
        if self._running: return
        self._running = True
        self.audio.start(loop)
        self._task = loop.create_task(self._live_session_loop())
        logger.info("Pipeline started.")

    def stop(self) -> None:
        self._running = False
        self.audio.stop()
        if self._task:
            self._task.cancel()

    async def _live_session_loop(self):
        # Dedicated speech-to-speech translation model — no system_instruction,
        # tools, or function calling supported in this mode.
        model = "gemini-3.5-live-translate-preview"

        config = types.LiveConnectConfig(
            response_modalities=[types.Modality.AUDIO],
            input_audio_transcription=types.AudioTranscriptionConfig(),
            output_audio_transcription=types.AudioTranscriptionConfig(),
            translation_config=types.TranslationConfig(
                target_language_code="vi",
                echo_target_language=False,
            ),
        )

        while self._running:
            try:
                logger.info(f"Connecting to {model} Live API...")
                async with self.client.aio.live.connect(model=model, config=config) as session:
                    logger.info("Connected to Gemini Live.")
                    
                    send_task = asyncio.create_task(self._send_audio(session))
                    recv_task = asyncio.create_task(self._receive_text(session))
                    
                    # Wait until one of the tasks finishes or errors
                    done, pending = await asyncio.wait(
                        [send_task, recv_task], 
                        return_when=asyncio.FIRST_COMPLETED
                    )
                    
                    for task in pending:
                        task.cancel()
                        
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Gemini Live connection dropped: {e}. Reconnecting in 2s...")
                await asyncio.sleep(2)

    async def _send_audio(self, session):
        """Pulls PCM bytes from the capture queue and streams them to Gemini."""
        mime_type = f"audio/pcm;rate={self.audio.sample_rate}"
        while self._running:
            pcm_bytes = await self.audio.queue.get()
            await session.send(input={"data": pcm_bytes, "mime_type": mime_type})
            logger.info(f"Sending {len(pcm_bytes)} bytes")

    async def _receive_text(self, session):
        """Listens for incoming translated text from Gemini and pushes it to the UI."""
        async for response in session.receive():
            print(response) # Keep for debugging
            server_content = response.server_content
            if not server_content:
                continue
                
            # Capture the transcription streaming layout from Live Connect
            if server_content.output_transcription:
                translated_text = server_content.output_transcription.text
                if translated_text:
                    # Forward text to PySide6 UI thread safely via Signal
                    self.overlay.caption_updated.emit(translated_text.strip())