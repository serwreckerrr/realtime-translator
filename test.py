import asyncio
import logging
import sys
import time
import traceback
import numpy as np

# Import the isolated components from your existing files
from audio import AudioCapture, AudioChunk
from asr import WhisperEngine

# Configure logging to prevent external dependency debug floods while keeping warnings visible
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
# Silence verbose internal debug logs from the main modules unless an issue occurs
logging.getLogger("realtime_translator.audio").setLevel(logging.WARNING)
logging.getLogger("realtime_translator.asr").setLevel(logging.WARNING)


async def main():
    print("=" * 60)
    print("INITIALIZING ASR ISOLATION TEST PIPELINE")
    print("=" * 60)
    
    # 1. Initialize the required processing engines
    try:
        print("[1/3] Initializing AudioCapture (WASAPI Loopback)...")
        audio_capture = AudioCapture(chunk_duration=0.5, use_vad=True, vad_threshold=0.5)
        
        print("[2/3] Initializing WhisperEngine (Model: 'base')...")
        whisper_engine = WhisperEngine(model_name="base")
    except Exception as init_err:
        print(f"\n[CRITICAL] Initialization failed: {init_err}", file=sys.stderr)
        traceback.print_exc()
        return

    # 2. Start audio capture background thread
    loop = asyncio.get_running_loop()
    print("[3/3] Launching WASAPI capture supervisor thread...")
    audio_capture.start(loop)

    # Performance tracking metrics
    total_processed_chunks = 0
    speech_chunks = 0
    silence_chunks = 0
    
    # Track metrics specifically for active Whisper inference actions
    total_inference_time = 0.0
    inference_calls_count = 0

    last_stats_time = time.time()
    print("\n>>> Pipeline successfully bound! Play system audio/speak now. <<<")
    print(">>> Press Ctrl+C to stop the test session safely. <<<\n")

    try:
        while True:
            try:
                # 3. Continuously pull incoming audio chunks from the queue
                chunk: AudioChunk = await audio_capture.get_chunk()
                current_time = time.time()
                
                total_processed_chunks += 1
                if chunk.is_speech:
                    speech_chunks += 1
                else:
                    silence_chunks += 1

                # Calculate Chunk Root Mean Square (RMS) for audio amplitude sizing
                if chunk.data.size > 0:
                    rms = float(np.sqrt(np.mean(chunk.data ** 2)))
                else:
                    rms = 0.0

                # Compute buffer transfer delay metrics
                queue_delay = current_time - chunk.timestamp

                # 4. Profile and execute the ASR Whisper decoding matrix
                start_inference = time.time()
                transcript = whisper_engine.process_chunk(chunk.data, chunk.is_speech)
                inference_latency = time.time() - start_inference

                # Track true execution times (ignoring instant bypass frames where VAD skips ASR)
                if inference_latency > 0.002:  
                    total_inference_time += inference_latency
                    inference_calls_count += 1

                # 5. Print out the results if transcription text is detected
                if transcript and transcript.text.strip():
                    print("=" * 60)
                    print(f"Speech: {chunk.is_speech}")
                    print(f"Final : {transcript.is_final}")
                    print(f"Lang  : {transcript.language}")
                    print(f"Conf  : {transcript.confidence:.2f}")
                    print(f"Text  : {transcript.text}")
                    print("-" * 40)
                    print(f"  [Diag] Chunk Duration  : {chunk.duration:.2f}s")
                    print(f"  [Diag] Sample Rate      : {chunk.sample_rate} Hz")
                    print(f"  [Diag] Chunk RMS        : {rms:.5f}")
                    print(f"  [Diag] Queue Delay      : {queue_delay * 1000.0:.2f} ms")
                    print(f"  [Diag] Inference Time   : {inference_latency * 1000.0:.2f} ms")
                    print(f"  [Diag] Running Counts   : Total={total_processed_chunks} (Speech={speech_chunks}, Silence={silence_chunks})")
                    print("=" * 60)

                # 6. Every 10 seconds print overall system statistics
                if current_time - last_stats_time >= 10.0:
                    avg_asr_ms = (
                        (total_inference_time / inference_calls_count) * 1000.0 
                        if inference_calls_count > 0 else 0.0
                    )
                    print("\n========== STATS ==========")
                    print(f"Chunks processed : {total_processed_chunks}")
                    print(f"Speech chunks    : {speech_chunks}")
                    print(f"Silence chunks   : {silence_chunks}")
                    print(f"Average ASR time : {avg_asr_ms:.1f} ms")
                    print(f"Current language : {whisper_engine.detected_language}")
                    print("===========================\n")
                    last_stats_time = current_time

            except Exception as pipeline_err:
                # Catch hardware/inference drops without crashing the loop execution
                print("\n[ERROR] Exception handled inside core pipeline processing loop:", file=sys.stderr)
                traceback.print_exc()
                print("-" * 60 + "\n")

    except KeyboardInterrupt:
        print("\n[System] Termination signal caught. Exiting pipeline loop execution...")
    finally:
        print("[System] Releasing capture streams and stopping background threads...")
        audio_capture.stop()
        print("[System] Standalone test pipeline shutdown completed successfully.")


if __name__ == "__main__":
    # Run the async loop supervisor
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
    