"""
Low-Latency Audio Streaming Endpoint for Live Browser Audition.
Yields raw 16-bit PCM audio chunks (16 kHz, mono) to browser clients.
"""

import time
from typing import Generator
from flask import Response
from audio.audio_output import AudioOutputSink


def audio_stream_generator(output_sink: AudioOutputSink, chunk_interval_s: float = 0.032) -> Generator[bytes, None, None]:
    """Generates continuous stream of PCM16 audio blocks."""
    while True:
        chunk = output_sink.get_latest_pcm()
        if chunk:
            yield chunk
        time.sleep(chunk_interval_s)


def create_audio_stream_response(output_sink: AudioOutputSink) -> Response:
    return Response(
        audio_stream_generator(output_sink),
        mimetype="audio/x-raw",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
            "X-Audio-Sample-Rate": "16000",
            "X-Audio-Channels": "1",
            "X-Audio-Bit-Depth": "16",
        }
    )
