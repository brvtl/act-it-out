"""Writes golden fixtures for the TypeScript preprocessing tests from independent reference implementations.

- patchify: numpy transcription of `Lfm3VlVideoProcessor.patchify` (view + permute), on random frames.
- prompt: the checkpoint's own chat_template.jinja rendered in a sandboxed Jinja environment.

Usage: python make_goldens.py <chat_template.jinja> <out_dir>
"""

import json
import sys
from pathlib import Path

import numpy as np
from jinja2 import nodes
from jinja2.ext import Extension, loopcontrols
from jinja2.sandbox import ImmutableSandboxedEnvironment


class GenerationTag(Extension):
    """`{% generation %}…{% endgeneration %}` as transformers' AssistantTracker renders it: the body, unchanged."""

    tags = {"generation"}

    def parse(self, parser):
        lineno = next(parser.stream).lineno
        body = parser.parse_statements(("name:endgeneration",), drop_needle=True)
        return nodes.Scope(body, lineno=lineno)

PATCH, MERGE, TEMPORAL = 16, 2, 2


def patchify(frames_rgba: np.ndarray) -> np.ndarray:
    """frames_rgba: (2, H, W, 4) uint8 -> (H/32, W/32, 2, 2, 1536) float32, as the vision encoder takes it."""
    video = frames_rgba[..., :3].astype(np.float32) / 255.0
    video = (video - 0.5) / 0.5
    video = video.transpose(0, 3, 1, 2)[None]  # (1, T, C, H, W)
    b, t, c, h, w = video.shape
    gh, gw = h // PATCH, w // PATCH
    patches = video.reshape(b, t // TEMPORAL, TEMPORAL, c, gh // MERGE, MERGE, PATCH, gw // MERGE, MERGE, PATCH)
    patches = patches.transpose(0, 1, 4, 7, 5, 8, 3, 2, 6, 9)
    return patches.reshape(gh // MERGE, gw // MERGE, MERGE, MERGE, c * TEMPORAL * PATCH * PATCH)


def write_patchify(out: Path) -> None:
    rng = np.random.default_rng(1234)
    frames = rng.integers(0, 256, size=(2, 64, 96, 4), dtype=np.uint8)
    frames.tofile(out / "patchify_frames_2x64x96_rgba.u8")
    patchify(frames).astype(np.float32).tofile(out / "patchify_expected.f32")


def write_prompt(template_path: Path, out: Path) -> None:
    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=[GenerationTag, loopcontrols])
    env.filters["tojson"] = lambda v, **_: json.dumps(v)
    template = env.from_string(template_path.read_text())
    messages = [
        {"role": "system", "content": "SYSTEM TEXT"},
        {"role": "user", "content": [{"type": "video"}, {"type": "text", "text": "USER TEXT"}]},
    ]
    rendered = template.render(
        messages=messages, add_generation_prompt=True, bos_token="<|startoftext|>", eos_token="<|im_end|>"
    )
    (out / "prompt_template.txt").write_text(rendered)


if __name__ == "__main__":
    template_path, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
    write_patchify(out_dir)
    write_prompt(template_path, out_dir)
