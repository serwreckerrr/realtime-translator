import asyncio
import logging
import os

import sys
# Force Windows COM initialization to behave nicely with Qt and Asyncio
if sys.platform == 'win32':
    import ctypes
    # COINIT_APARTMENTTHREADED = 0x2
    # This prevents conflicting initializations from crashing the Qt context
    ctypes.windll.ole32.CoInitializeEx(None, 0x2)

import threading

import soundcard as sc
from PySide6.QtWidgets import QApplication
from dotenv import load_dotenv

from audio1 import AudioCapture
from overlay1 import OverlayWindow
from pipeline1 import CaptionPipeline

# Clean PySide6/Soundcard MTA thread initialization
try:
    _ = sc.default_speaker()
except Exception:
    pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("realtime_translator.main")

def run_asyncio_background_loop(loop: asyncio.AbstractEventLoop) -> None:
    """Runs the asyncio loop in a dedicated background thread."""
    asyncio.set_event_loop(loop)
    try:
        loop.run_forever()
    finally:
        pending_tasks = asyncio.all_tasks(loop)
        if pending_tasks:
            loop.run_until_complete(asyncio.gather(*pending_tasks, return_exceptions=True))
        loop.close()
        logger.info("Background event loop closed.")

def main():
    load_dotenv()
    
    if not os.getenv("GEMINI_API_KEY"):
        logger.error("GEMINI_API_KEY is missing from environment!")
        sys.exit(1)

    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(True)

    # 1. Initialize Components
    overlay_window = OverlayWindow()
    audio_capture = AudioCapture(sample_rate=16000, chunk_duration_ms=100)
    pipeline = CaptionPipeline(audio_capture, overlay_window)

    # 2. Start Background Asyncio Thread
    asyncio_loop = asyncio.new_event_loop()
    background_thread = threading.Thread(
        target=run_asyncio_background_loop,
        args=(asyncio_loop,),
        daemon=True
    )
    background_thread.start()

    # 3. Start Pipeline
    asyncio_loop.call_soon_threadsafe(pipeline.start, asyncio_loop)

    # 4. Handle Shutdown
    def handle_shutdown():
        logger.info("Shutting down...")
        asyncio_loop.call_soon_threadsafe(pipeline.stop)
        asyncio_loop.call_soon_threadsafe(asyncio_loop.stop)
        background_thread.join(timeout=3.0)

    app.aboutToQuit.connect(handle_shutdown)
    
    overlay_window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()