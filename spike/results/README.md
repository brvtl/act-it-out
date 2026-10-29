# Feasibility spike results (2026-10-28)

RTX 5090 (WebGPU adapter `nvidia/blackwell`, shader-f16), Chromium 156 headed with `--enable-features=Vulkan`,
fake webcam fed a CC BY-SA clip of a man doing jumping jacks and burpees (not committed).
Bundle: `LiquidAI/LFM3-Omni-2B-ONNX@7d6bd3f`, `decoder_q4f16` + `vision_encoder_q8` + `depth_decoder` + `codec_decoder`.

| Metric                              | Value                                                                                                  |
| ----------------------------------- | ------------------------------------------------------------------------------------------------------ |
| Download size                       | 2.14 GB                                                                                                |
| First load (download + sessions)    | 40.8 s + 1.9 s                                                                                         |
| Cached load (Cache API + sessions)  | 0.9 s + 1.8 s                                                                                          |
| Video window                        | 8 frames @ 2 fps, 448×256 → 448 tokens; encode 186 ms warm                                             |
| Time to text guess after the window | ~85–130 ms warm (prefill ~70–105 ms, 549-token prompt); 821 ms on the very first call (shader compile) |
| Decode speed                        | 119 tok/s (128-token generation)                                                                       |
| Native speech (interleaved)         | first audio frame ~90 ms; 1.6 s of audio in ~0.6 s incl. vocoding (RTF 0.39)                           |
| Guess on jumping jacks              | "Jumping" (all 6 runs)                                                                                 |

Parity (`parity.json`, `scripts/parity/check_parity.py`) against ONNX Runtime CPU / numpy on identical inputs:
vision features max |Δ| 8e-6, decoder greedy reply and first-step top-5 identical, codec + iSTFT 117 dB SNR.

The first ~0.85 s of speech is a quiet (≈ −45 dB) low tone before the word; it is present in the CPU reference
too, so it is model output, not a synthesis bug. The game should trim the quiet lead-in.

Note: `load.bytes` in `spike-cold.json` is wrong (0.02 GB): it was counted after freeing the weight buffers.
Fixed before the warm run.

UI round (later): frame 0 used to be black, so capture now waits for a non-black camera frame (reported as
`video.warmup` and `video.firstFrameLuma`) and the runner forces `start=auto&countdown=0`; every other metric is
measured the same way.
