"""
Live version of the pipeline for real hardware: 2 USB sound cards, each
carrying one mic (primary + reference), output to a speaker/headphone.

STATUS: written against the sounddevice API and reviewed carefully, but
NOT YET TESTED on real hardware (the mics hadn't arrived at time of
writing). Test this file the moment your 2 USB sound cards are plugged
in -- start with `python3 -m sounddevice` (see below) before running
this script.

HOW TO FIND YOUR DEVICE INDICES ONCE HARDWARE IS CONNECTED:
    python3 -m sounddevice
This prints a numbered list of every audio device the Pi/laptop can see.
Find your two USB sound cards' input indices and your speaker's output
index, then set them below.
"""

import numpy as np
import sounddevice as sd
from pipeline import IgardNetPipeline

SAMPLE_RATE = 16000
BLOCK_SIZE = 512

# ---- EDIT THESE ONCE YOU'VE RUN `python3 -m sounddevice` ----
PRIMARY_MIC_DEVICE_INDEX = None    # e.g. 1
REFERENCE_MIC_DEVICE_INDEX = None  # e.g. 2
OUTPUT_DEVICE_INDEX = None         # e.g. 3 (or None for system default)
# ---------------------------------------------------------------

pipeline = IgardNetPipeline(sample_rate=SAMPLE_RATE)


def run():
    if PRIMARY_MIC_DEVICE_INDEX is None or REFERENCE_MIC_DEVICE_INDEX is None:
        print("Set PRIMARY_MIC_DEVICE_INDEX and REFERENCE_MIC_DEVICE_INDEX at the")
        print("top of this file first. Run `python3 -m sounddevice` to list devices.")
        return

    primary_stream = sd.InputStream(
        device=PRIMARY_MIC_DEVICE_INDEX, channels=1,
        samplerate=SAMPLE_RATE, blocksize=BLOCK_SIZE, dtype="float32",
    )
    reference_stream = sd.InputStream(
        device=REFERENCE_MIC_DEVICE_INDEX, channels=1,
        samplerate=SAMPLE_RATE, blocksize=BLOCK_SIZE, dtype="float32",
    )
    output_stream = sd.OutputStream(
        device=OUTPUT_DEVICE_INDEX, channels=1,
        samplerate=SAMPLE_RATE, blocksize=BLOCK_SIZE, dtype="float32",
    )

    print("Starting live IGARD-Net pipeline. Press Ctrl+C to stop.")
    with primary_stream, reference_stream, output_stream:
        try:
            while True:
                primary_block, _ = primary_stream.read(BLOCK_SIZE)
                reference_block, _ = reference_stream.read(BLOCK_SIZE)

                result = pipeline.process_block(
                    primary_block[:, 0].astype(np.float64),
                    reference_block[:, 0].astype(np.float64),
                )

                out_block = np.clip(result["audio"], -1.0, 1.0).astype(np.float32)
                output_stream.write(out_block.reshape(-1, 1))

                # This is exactly the telemetry your web dashboard should
                # display live: noise type, confidence, whether the
                # cleanup stage ran, and the current filter step size.
                print(
                    f"\rlabel={result['noise_label']:<12} "
                    f"conf={result['confidence']:.2f} "
                    f"cleanup={'ON ' if result['used_cleanup_stage'] else 'OFF'} "
                    f"step={result['nlms_step_size']:.3f}",
                    end="",
                )
        except KeyboardInterrupt:
            print("\nStopped.")


if __name__ == "__main__":
    run()
