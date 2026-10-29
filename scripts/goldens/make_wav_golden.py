"""Writes the WAV golden for tests/wav.test.ts using only the stdlib `wave` module (an independent encoder).

Samples are clamped to [-1, 1], scaled by 32767 and truncated toward zero with int(), matching DataView.setInt16.
All inputs are exactly representable in float32 except 1e-6, which rounds to 0 at 16 bits either way.

Usage: python3 make_wav_golden.py [out_path]   (default: tests/fixtures/wav_golden.wav)
"""

import struct
import sys
import wave
from pathlib import Path

SAMPLES = [0.0, 0.5, -0.5, 1.0, -1.0, 1.7, -3.0, 0.25, 1e-6, -0.0]
SAMPLE_RATE = 24000


def main() -> None:
    default = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "wav_golden.wav"
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else default
    pcm = b"".join(struct.pack("<h", int(max(-1.0, min(1.0, s)) * 32767)) for s in SAMPLES)
    with wave.open(str(out), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(pcm)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
