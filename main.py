import asyncio
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

# --------------------------------------------------------------------------
# Configuration constants (no magic numbers below this point)
# --------------------------------------------------------------------------
AUDIO_CHUNK_DURATION_SECONDS: float = 0.5
AUDIO_USE_VAD: bool = True
AUDIO_VAD_THRESHOLD: float = 0.4

# Tối ưu cho máy yếu: Sử dụng mẫu "tiny" để tăng tốc độ suy luận gấp nhiều lần trên CPU
WHISPER_MODEL_NAME: str = "base"
WHISPER_DEVICE: Optional[str] = None  # None -> auto-select CUDA if available, else CPU

OVERLAY_CLEAR_TIMEOUT_SECONDS: float = 4.0
SHUTDOWN_JOIN_TIMEOUT_SECONDS: float = 4.0

# Configure structured system logging across all execution layers
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("realtime_translator.main")


def run_asyncio_background_loop(loop: asyncio.AbstractEventLoop) -> None:
    """Target function for execution inside the dedicated asyncio worker thread."""
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


def build_pipeline_components() -> tuple[AudioCapture, WhisperEngine, Translator, OverlayWindow]:
    """Constructs the core dependency-injected components, failing fast with a clear error."""
    try:
        audio_capture = AudioCapture(
            chunk_duration=AUDIO_CHUNK_DURATION_SECONDS,
            use_vad=AUDIO_USE_VAD,
            vad_threshold=AUDIO_VAD_THRESHOLD,
        )
    except Exception as audio_err:
        logger.critical(f"Failed to initialize AudioCapture: {audio_err}", exc_info=True)
        raise

    try:
        whisper_engine = WhisperEngine(model_name=WHISPER_MODEL_NAME, device=WHISPER_DEVICE)
    except Exception as whisper_err:
        logger.critical(f"Failed to initialize WhisperEngine: {whisper_err}", exc_info=True)
        raise

    try:
        translator = Translator(backend=None)
    except Exception as translator_err:
        logger.critical(f"Failed to initialize Translator: {translator_err}", exc_info=True)
        raise

    try:
        overlay_window = OverlayWindow()
    except Exception as overlay_err:
        logger.critical(f"Failed to initialize OverlayWindow: {overlay_err}", exc_info=True)
        raise

    return audio_capture, whisper_engine, translator, overlay_window


def main() -> None:
    """The absolute bootstrap entry point for the Real-Time Multilingual Translator application."""
    logger.info("Initializing application startup sequence...")

    # TỐI ƯU HÓA CPU MÁY YẾU: Giới hạn số thread PyTorch để tránh tranh chấp tài nguyên (CPU Thrashing)
    import torch
    torch.set_num_threads(2)

    load_dotenv()

    # 1. Instantiate Core Application UI Thread Frame
    app = QApplication(sys.argv)
    app.setApplicationName("Real-Time Multilingual Translator")

    # 2. Wire Production Dependencies via Clear Injections
    try:
        audio_capture, whisper_engine, translator, overlay_window = build_pipeline_components()
    except Exception:
        logger.critical("Application startup aborted due to component initialization failure.")
        sys.exit(1)

    # 3. Assemble Core Architectural Brain Coordinator Pipeline
    pipeline = CaptionPipeline(
        audio_capture=audio_capture,
        whisper_engine=whisper_engine,
        translator=translator,
        overlay_window=overlay_window,
        clear_timeout=OVERLAY_CLEAR_TIMEOUT_SECONDS
    )

    # 4. Spin Up Dedicated Non-Blocking Asyncio Event Infrastructure
    asyncio_loop = asyncio.new_event_loop()
    background_thread = threading.Thread(
        target=run_asyncio_background_loop,
        args=(asyncio_loop,),
        name="AsyncioPipelineThread",
        daemon=True
    )
    background_thread.start()
    logger.info("Asyncio background execution thread spawned successfully.")

    # 5. Bootstrap Real-Time Data Pipeline Pipelines Within Safe Threads
    asyncio_loop.call_soon_threadsafe(pipeline.start, asyncio_loop)

    # 6. Coordinate Clean System Exit Traps
    def handle_graceful_shutdown() -> None:
        """Safely winds down hardware threads, network buffers, and memory tasks on app quit."""
        logger.info("Received termination signal from UI thread. Initiating clean exit sequence...")

        asyncio_loop.call_soon_threadsafe(pipeline.stop)
        asyncio_loop.call_soon_threadsafe(asyncio_loop.stop)

        background_thread.join(timeout=SHUTDOWN_JOIN_TIMEOUT_SECONDS)
        logger.info("All engine sub-components torn down. Exiting process space.")

    app.aboutToQuit.connect(handle_graceful_shutdown)

    # 7. Render Viewport Overlay and Engage Main OS GUI Thread Loop
    overlay_window.show()
    logger.info("Application interface mapped. Processing Window event ticks...")

    sys.exit(app.exec())


if __name__ == "__main__":
    main()