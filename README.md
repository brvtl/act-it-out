# Act it out

A charades game where an AI does the guessing. You act something out silently on your webcam, and
[LFM3-Omni-2B](https://huggingface.co/LiquidAI/LFM3-Omni-2B-ONNX) watches the clip and says its guess out loud.
Everything runs in the browser on WebGPU, so no video leaves your machine.

**Status:** feasibility spike. The model runs end to end in the browser; the game itself is not built yet.
The spike page plays one round and measures load time, guess latency and speech speed.

## Requirements

- A browser with WebGPU and `shader-f16` (recent Chrome on macOS, Windows or Linux)
- Tested on an RTX 5090 (Linux) and an Apple M-series Mac
- Node 22 or newer (developed on Node 24)
- About 2.1 GB of model download on first load. The weights are fetched from Hugging Face and kept in the
  browser's Cache API, per origin, so later loads take a couple of seconds.

## Run it

```sh
npm install
npm run dev
```

Open http://localhost:5173/spike/, allow the camera, then:

1. Wait for **Ready when you are** and press **Start round**.
2. Get in frame during the 3-2-1 countdown.
3. Act something out **without talking** for about 4 seconds. The model only sees video.
4. The guess appears on screen and is spoken aloud.

URL parameters: `runs` (repeat each measurement, default 3), `countdown` (seconds, default 3), `frames`
(captured at 2 fps, default 8), `w`/`h` (capture size, default 448×256), `start=auto` (skip the Start button)
and `dump=1` (export tensors for the parity check).

## Benchmark runner

`scripts/run-spike.mjs` drives the spike page unattended in headed Chromium/Chrome, using a `.y4m` video file as
a fake webcam, and writes the result JSON and the spoken answer as a WAV:

```sh
npm run spike -- path/to/clip.y4m out-dir label 'runs=3'
```

It uses `/usr/bin/chromium`, or Google Chrome on macOS; set `CHROMIUM` to override. Measured results and the
CPU parity check are in [spike/results/README.md](spike/results/README.md).

## Development

```sh
npm test               # vitest: unit, golden-fixture and seeded property tests
npm run typecheck
npm run lint
npm run format:check
npm run build
```

Layout:

- `src/model`: preprocessing, prompt rendering, the decode loop and the onnxruntime-web WebGPU runtime.
  [transformers.js](https://github.com/huggingface/transformers.js) has no LFM3-Omni support, so the model runs on raw ORT.
- `src/audio`, `src/video`, `src/game`: WAV encoding, lead-in trim, camera capture and warm-up, round copy and timing.
- `spike/`: the spike page (`index.html`, `ui.ts`, `main.ts`).
- `scripts/`: the benchmark runner, golden-fixture generators and the Python parity check.
- `tests/`: Vitest suites. Pure modules are checked against goldens from independent reference implementations
  (numpy, Jinja, Python `wave`).

The model weights are LiquidAI's and are covered by their license on Hugging Face; they are not part of this repo.
