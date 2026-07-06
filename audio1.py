import asyncio
import logging
import threading
import numpy as np
import soundcard as sc

logger = logging.getLogger("realtime_translator.audio")

class AudioCapture:
    """Captures Windows Loopback audio and queues raw PCM bytes for the Live API."""
    def __init__(self, sample_rate: int = 16000, chunk_duration_ms: int = 100):
        self.sample_rate = sample_rate
        self.chunk_size = int(sample_rate * (chunk_duration_ms / 1000.0))
        self.queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=100)
        
        self._running = False
        self._thread = None
        self._loop = None

    def start(self, loop: asyncio.AbstractEventLoop) -> None:
        if self._running: return
        self._running = True
        self._loop = loop
        self._thread = threading.Thread(target=self._record_loop, daemon=True)
        self._thread.start()
        logger.info("Audio capture started.")

    def stop(self) -> None:
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        logger.info("Audio capture stopped.")

    def _record_loop(self) -> None:
        try:
            speaker = sc.default_speaker()
            mics = sc.all_microphones(include_loopback=True)
            # Find loopback for default speaker
            loopback_mic = next((m for m in mics if m.isloopback and speaker.name in m.name), mics[0])
            #print all devices for debugging
            print("Speaker:", speaker.name)
            for m in mics:
                print(m.name, m.isloopback)
                        
            with loopback_mic.recorder(samplerate=self.sample_rate, channels=1) as mic:
                while self._running:
                    # record() returns float32 array
                    data = mic.record(numframes=self.chunk_size)
                    logger.info(f"Recorded {data.shape}, max={abs(data).max()}")
                    
                    # Convert float32 [-1.0, 1.0] to int16 PCM
                    pcm_data = (data[:, 0] * 32767).astype(np.int16).tobytes()
                    
                    # Push bytes to the asyncio queue
                    if self._loop and self._running:
                        try:
                            self._loop.call_soon_threadsafe(self.queue.put_nowait, pcm_data)
                        except asyncio.QueueFull:
                            pass # Drop frame if network is lagging heavily
                            
        except Exception as e:
            logger.error(f"Hardware audio capture failed: {e}")
            self._running = False