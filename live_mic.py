"""
Live version of the IGARD-Net pipeline for real hardware or system sound cards.
Low-latency audio I/O streaming with real-time speech preservation telemetry.
"""

import sys
import numpy as np
import sounddevice as sd
from pipeline import IgardNetPipeline

SAMPLE_RATE = 16000
BLOCK_SIZE = 256  # 256 samples @ 16 kHz = 16.0 ms latency

PRIMARY_MIC_DEVICE_INDEX = None    # e.g. 1 (or None for system default)
REFERENCE_MIC_DEVICE_INDEX = None  # e.g. 2 (or None to synthesize/share)
OUTPUT_DEVICE_INDEX = None         # e.g. 3 (or None for system default)


def run():
    print("=== IGARD-Net Low-Latency Live Mic Stream ===")
    print(f"Sample Rate: {SAMPLE_RATE} Hz, Block Size: {BLOCK_SIZE} ({BLOCK_SIZE/SAMPLE_RATE*1000:.1f} ms)")
    
    pipeline = IgardNetPipeline(sample_rate=SAMPLE_RATE)

    try:
        if PRIMARY_MIC_DEVICE_INDEX is None:
            print("Using default input device for primary audio.")
            in_dev = sd.default.device[0]
        else:
            in_dev = PRIMARY_MIC_DEVICE_INDEX

        out_dev = OUTPUT_DEVICE_INDEX if OUTPUT_DEVICE_INDEX is not None else sd.default.device[1]

        primary_stream = sd.InputStream(
            device=in_dev, channels=1,
            samplerate=SAMPLE_RATE, blocksize=BLOCK_SIZE, dtype="float32",
        )
        
        output_stream = sd.OutputStream(
            device=out_dev, channels=1,
            samplerate=SAMPLE_RATE, blocksize=BLOCK_SIZE, dtype="float32",
        )

        if REFERENCE_MIC_DEVICE_INDEX is not None:
            reference_stream = sd.InputStream(
                device=REFERENCE_MIC_DEVICE_INDEX, channels=1,
                samplerate=SAMPLE_RATE, blocksize=BLOCK_SIZE, dtype="float32",
            )
            has_ref = True
        else:
            has_ref = False
            reference_stream = None

    except Exception as e:
        print(f"Failed to open audio device: {e}")
        print("Run `python -m sounddevice` to check available audio devices.")
        return

    print("Live stream active. Press Ctrl+C to stop.")
    
    def cleanup_streams():
        primary_stream.close()
        output_stream.close()
        if reference_stream:
            reference_stream.close()

    try:
        primary_stream.start()
        output_stream.start()
        if has_ref:
            reference_stream.start()

        rng = np.random.default_rng(42)

        while True:
            primary_block, _ = primary_stream.read(BLOCK_SIZE)
            p_data = primary_block[:, 0].astype(np.float64)

            if has_ref:
                ref_block, _ = reference_stream.read(BLOCK_SIZE)
                r_data = ref_block[:, 0].astype(np.float64)
            else:
                # Synthesize low-level sensor noise when secondary hardware mic is absent
                r_data = rng.normal(0, 0.002, BLOCK_SIZE)

            result = pipeline.process_block(p_data, r_data)

            out_block = np.clip(result["audio"], -1.0, 1.0).astype(np.float32)
            output_stream.write(out_block.reshape(-1, 1))

            print(
                f"\rSpeech: {result.get('speech_prob', 0.0)*100:3.0f}% | "
                f"Noise: {result['noise_label']:<10} ({result['confidence']*100:2.0f}%) | "
                f"Status: {result.get('stage_status', '')[:40]:<40}",
                end="",
                flush=True,
            )

    except KeyboardInterrupt:
        print("\nStopping live mic stream.")
    finally:
        cleanup_streams()


if __name__ == "__main__":
    run()
