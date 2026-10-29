"""Checks the browser's WebGPU pipeline against ONNX Runtime on the CPU, on the exact same inputs.

The spike page (with ?dump=1) saves the vision encoder inputs it built, the features WebGPU produced, the prompt ids
and the greedy reply. This re-runs the official graphs on the CPU and compares: vision features numerically, and the
decoder's greedy tokens. Catches glue bugs (tensor layout, cache wiring, position ids) that a plausible-looking guess
would hide.

Usage: python check_parity.py <dump.json> <model_dir with onnx/*.onnx> <out.json>
"""

import base64
import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort

HIDDEN = 2048
VIDEO_TOKEN = 125029
EOS = {124900, 125014}


def b64_array(entry: dict) -> np.ndarray:
    return np.frombuffer(base64.b64decode(entry["b64"]), dtype=entry["dtype"]).reshape(entry["dims"])


def session(model_dir: Path, graph: str) -> ort.InferenceSession:
    return ort.InferenceSession(str(model_dir / "onnx" / f"{graph}.onnx"), providers=["CPUExecutionProvider"])


def compare_vision(model_dir: Path, dump: dict) -> tuple[dict, np.ndarray]:
    vision = session(model_dir, "vision_encoder_q8")
    cpu = np.concatenate([vision.run(None, {"pixel_values": b64_array(p)})[0] for p in dump["pixelValues"]])
    gpu = b64_array(dump["videoFeatures"])
    cos = np.sum(cpu * gpu, axis=1) / (np.linalg.norm(cpu, axis=1) * np.linalg.norm(gpu, axis=1))
    return {
        "tokens": int(cpu.shape[0]),
        "maxAbsDiff": float(np.abs(cpu - gpu).max()),
        "meanAbsDiff": float(np.abs(cpu - gpu).mean()),
        "minCosine": float(cos.min()),
    }, cpu


def initial_cache(dec: ort.InferenceSession) -> dict:
    cache = {}
    for i in dec.get_inputs():
        dtype = np.float16 if "float16" in i.type else np.float32
        if i.name.startswith("past_conv."):
            cache[i.name] = np.zeros((1, HIDDEN, 7), dtype)
        elif i.name.startswith("past_key_values."):
            cache[i.name] = np.zeros((1, 4, 0, 128), dtype)
    return cache


def greedy(dec: ort.InferenceSession, ids: list[int], features: np.ndarray, max_new: int) -> tuple[list[int], list[int]]:
    """Greedy text decode on the CPU; returns the reply ids and the first step's top-5."""
    cache, length, out_ids, top5 = initial_cache(dec), 0, [], []
    names = [o.name for o in dec.get_outputs()]
    step_ids, step_feats = ids, features
    for _ in range(max_new):
        n = len(step_ids)
        feeds = {
            "input_ids": np.array([step_ids], np.int64),
            "attention_mask": np.ones((1, length + n), np.int64),
            "position_ids": np.arange(length, length + n, dtype=np.int64)[None],
            "image_features": np.zeros((0, HIDDEN), np.float32),
            "video_features": step_feats,
            "audio_features": np.zeros((0, HIDDEN), np.float32),
            "audio_out_codes": np.zeros((0, 12), np.int64),
            "logits_to_keep": np.array([1], np.int64),
            **cache,
        }
        outs = dict(zip(names, dec.run(None, feeds)))
        logits = outs["logits"][0, -1]
        if not top5:
            top5 = [int(i) for i in np.argsort(-logits)[:5]]
        for k in [k for k in outs if k.startswith("present")]:
            cache[k.replace("present_conv.", "past_conv.").replace("present.", "past_key_values.")] = outs[k]
        length += n
        token = int(np.argmax(logits))
        out_ids.append(token)
        if token in EOS:
            break
        step_ids, step_feats = [token], np.zeros((0, HIDDEN), np.float32)
    return out_ids, top5


def numpy_istft(features: np.ndarray) -> np.ndarray:
    """Reference synthesis with numpy's FFT: polar(exp(log-magnitude), angle) -> irfft -> Hann OLA -> /envelope -> trim."""
    n_fft, hop = 1280, 320
    bins = n_fft // 2 + 1
    spec = np.exp(features[:, :bins]) * np.exp(1j * features[:, bins:])
    frames = np.fft.irfft(spec, n_fft, axis=1)
    window = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n_fft) / n_fft)
    length = (len(frames) - 1) * hop + n_fft
    audio, envelope = np.zeros(length), np.zeros(length)
    for i, frame in enumerate(frames):
        audio[i * hop : i * hop + n_fft] += frame * window
        envelope[i * hop : i * hop + n_fft] += window**2
    trim = (n_fft - hop) // 2
    return (audio / envelope)[trim:-trim]


def compare_speech(model_dir: Path, dump: dict) -> dict:
    if not dump["speechFrames"]:
        return {"skipped": "no speech frames"}
    codes = np.array(dump["speechFrames"], np.int64).T[None]  # (1, 12, frames)
    features = session(model_dir, "codec_decoder").run(None, {"audio_codes": codes})[0][0]
    cpu = numpy_istft(features.astype(np.float64))
    gpu = b64_array(dump["speechAudio"]).astype(np.float64)
    n = min(len(cpu), len(gpu))
    err = cpu[:n] - gpu[:n]
    return {
        "samples": [len(cpu), len(gpu)],
        "maxAbsDiff": float(np.abs(err).max()),
        "snrDb": float(10 * np.log10(np.sum(cpu[:n] ** 2) / max(np.sum(err**2), 1e-20))),
        "cpuRms": float(np.sqrt(np.mean(cpu**2))),
    }


def main() -> None:
    dump_path, model_dir, out_path = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    dump = json.loads(dump_path.read_text())
    vision_report, cpu_features = compare_vision(model_dir, dump)

    report = {"vision": vision_report, "speech": compare_speech(model_dir, dump), "decoder": {}}
    for graph in ("decoder_q4f16", "decoder_q4"):
        try:
            dec = session(model_dir, graph)
            cpu_ids, cpu_top5 = greedy(dec, dump["inputIds"], cpu_features, len(dump["replyIds"]) + 4)
        except Exception as error:  # the fp16 graph may lack CPU kernels; fall back to the q4 bundle
            report["decoder"][graph] = {"error": str(error)[:300]}
            continue
        report["decoder"][graph] = {
            "cpuReplyIds": cpu_ids,
            "gpuReplyIds": dump["replyIds"],
            "replyMatches": cpu_ids == dump["replyIds"],
            "cpuTop5": cpu_top5,
            "gpuTop5": dump["firstTop5"],
        }
        break
    out_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
