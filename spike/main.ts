import { generate, type DecoderPort, type FrameSampler } from '../src/model/generate';
import { SAMPLE_RATE } from '../src/model/istft';
import { sampleFrame } from '../src/model/maskgit';
import { OmniRuntime } from '../src/model/omni';
import { patchifyPair, tokensPerGroup } from '../src/model/patchify';
import { buildPrompt, expandVideo } from '../src/model/prompt';
import { mulberry32 } from '../src/model/sampling';
import { groupTimestamps } from '../src/model/timestamps';
import { encodeWav } from '../src/audio/wav';
import { trimLeadIn } from '../src/audio/trim';
import { agreement, captureWindowMs, parseCountdown } from '../src/game/phase';
import { captureFrames, waitForPicture } from '../src/video/capture';
import { meanLuma } from '../src/video/luma';
import * as ui from './ui';

const params = new URLSearchParams(location.search);
const RUNS = Number(params.get('runs') ?? 3);
const WIDTH = Number(params.get('w') ?? 448);
const HEIGHT = Number(params.get('h') ?? 256);
const FRAMES = Number(params.get('frames') ?? 8);
const FPS = 2;
const COUNTDOWN = parseCountdown(params.get('countdown'));
const START = params.get('start') === 'auto' ? 'auto' : 'click';

const SYSTEM =
  'You are the guesser in a game of charades. The player acts out a word or phrase without speaking. ' +
  'Watch the video and guess what they are acting out.';
const QUESTION = 'What am I acting out? Answer with just your guess, in a few words.';
const BENCH_QUESTION = 'Describe in detail what the person in the video is doing, step by step.';
const DUMP = params.get('dump') === '1';

const b64 = (a: ArrayBufferView) => {
  let bin = '';
  const bytes = new Uint8Array(a.buffer, a.byteOffset, a.byteLength);
  for (let i = 0; i < bytes.length; i += 0x8000) bin += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
  return btoa(bin);
};
const tensorDump = (a: Float32Array, dims: number[]) => ({ b64: b64(a), dtype: '<f4', dims });

function top5(logits: Float32Array): number[] {
  return Array.from(logits.keys())
    .sort((a, b) => logits[b] - logits[a])
    .slice(0, 5);
}

/** Wraps a conversation to record the prefill's top-5 next tokens for the parity check. */
function recordFirstLogits(conv: DecoderPort) {
  const seen: { top5?: number[] } = {};
  const port: DecoderPort = {
    prefill: async (ids) => {
      const out = await conv.prefill(ids);
      seen.top5 = top5(out.logits);
      return out;
    },
    feedToken: (id) => conv.feedToken(id),
    feedFrame: (codes) => conv.feedFrame(codes),
  };
  return { port, seen };
}

const log = ui.log;
const ms = (t: number) => `${t.toFixed(0)} ms`;
const secs = (t: number) => `${(t / 1000).toFixed(1)} s`;

declare global {
  interface Window {
    __spikeResult?: unknown;
    __spikeError?: string;
    __spikeWav?: string;
    __spikeDump?: unknown;
  }
}

async function adapterInfo() {
  const adapter = await navigator.gpu?.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) throw new Error('WebGPU is not available in this browser');
  const { vendor, architecture, device, description } = adapter.info;
  return { vendor, architecture, device, description, shaderF16: adapter.features.has('shader-f16') };
}

async function startCamera(): Promise<HTMLVideoElement> {
  const video = document.getElementById('cam') as HTMLVideoElement;
  try {
    video.srcObject = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 360 } });
  } catch (e) {
    if (e instanceof DOMException && (e.name === 'NotAllowedError' || e.name === 'SecurityError'))
      throw new Error('Camera blocked: allow camera access and reload.', { cause: e });
    throw e;
  }
  await video.play();
  return video;
}

async function textGuess(model: OmniRuntime, features: Float32Array, prompt: number[], maxNewTokens = 32) {
  const conv = model.conversation(features);
  const recorder = recordFirstLogits(conv);
  const t0 = performance.now();
  let firstTokenAt = 0;
  const out = await generate(
    recorder.port,
    { sample: () => Promise.reject(new Error('text only')) },
    {
      inputIds: prompt,
      speech: false,
      maxNewTokens,
      onText: () => (firstTokenAt ||= performance.now()),
    },
  );
  const end = performance.now();
  conv.dispose();
  const decodeTokens = out.textIds.length - 1;
  return {
    text: model.decode(out.textIds),
    textIds: out.textIds,
    firstTop5: recorder.seen.top5 ?? [],
    tokens: out.textIds.length,
    prefillMs: firstTokenAt - t0,
    totalMs: end - t0,
    decodeTokPerS: decodeTokens > 0 ? (decodeTokens / (end - firstTokenAt)) * 1000 : 0,
  };
}

async function speechGuess(model: OmniRuntime, features: Float32Array, prompt: number[]) {
  const conv = model.conversation(features);
  const rng = mulberry32(42);
  const sampler: FrameSampler = { sample: (hidden, history) => sampleFrame(model, hidden, history, rng) };
  const t0 = performance.now();
  let firstFrameAt = 0;
  const out = await generate(conv, sampler, {
    inputIds: prompt,
    speech: true,
    maxNewTokens: 400,
    onFrame: () => (firstFrameAt ||= performance.now()),
  });
  const genMs = performance.now() - t0;
  conv.dispose();
  const t1 = performance.now();
  const audio = out.frames.length ? await model.speak(out.frames) : new Float32Array(0);
  const vocodeMs = performance.now() - t1;
  const audioSeconds = audio.length / SAMPLE_RATE;
  return {
    result: {
      text: model.decode(out.textIds),
      frames: out.frames.length,
      firstAudioFrameMs: firstFrameAt ? firstFrameAt - t0 : null,
      generateMs: genMs,
      vocodeMs,
      audioSeconds,
      realTimeFactor: audioSeconds ? (genMs + vocodeMs) / 1000 / audioSeconds : null,
    },
    audio,
    frames: out.frames,
  };
}

function webSpeechStatus() {
  const synth = 'speechSynthesis' in window ? window.speechSynthesis : undefined;
  return { available: !!synth, voices: synth?.getVoices().length ?? 0 };
}

async function main() {
  const gpu = await adapterInfo();
  log(`adapter: ${JSON.stringify(gpu)}`);

  // The camera starts before the model loads so the player can line up the shot while it downloads.
  ui.setPhase('loading');
  const video = await startCamera();

  let lastPct = -1;
  const model = await OmniRuntime.load((p) => {
    const pct = Math.floor((p.loaded / p.total) * 10) * 10;
    if (p.loaded === p.total || pct !== lastPct) {
      log(`  ${p.file}: ${pct}%${p.fromCache ? ' (cache)' : ''}`);
      ui.setPhase('loading', `${pct}%`);
    }
    lastPct = pct;
  });
  const t = model.timings;
  log(
    `load: ${(t.bytes / 1e9).toFixed(2)} GB, download ${ms(t.downloadMs)} (cache=${t.fromCache}), sessions ${ms(t.sessionMs)}`,
  );
  ui.stat(t.fromCache ? 'Load (cached)' : 'Model load', secs(t.downloadMs + t.sessionMs));

  ui.setPhase('camera');
  const warmup = await waitForPicture(video);
  log(`warmup: ${warmup.frames} frames, luma ${warmup.luma.toFixed(1)}${warmup.timedOut ? ' (timed out)' : ''}`);

  if (START === 'click') {
    ui.setPhase('ready');
    await ui.waitForStart();
  }
  await ui.countdown(COUNTDOWN);

  const stopTimer = ui.recordingTimer(captureWindowMs(FRAMES, FPS));
  const tCap = performance.now();
  const frames = await captureFrames(video, { count: FRAMES, fps: FPS, width: WIDTH, height: HEIGHT });
  const captureMs = performance.now() - tCap;
  stopTimer();
  ui.setPhase('thinking');
  const firstFrameLuma = meanLuma(frames[0]);
  log(`first frame luma ${firstFrameLuma.toFixed(1)}`);

  const tEnc = performance.now();
  const pairs = [];
  for (let i = 0; i < frames.length; i += 2)
    pairs.push(patchifyPair(frames[i], frames[i + 1] ?? frames[i], WIDTH, HEIGHT));
  const features = await model.encodeVideo(pairs);
  const encodeMs = performance.now() - tEnc;
  const perGroup = tokensPerGroup(WIDTH, HEIGHT);
  const timestamps = groupTimestamps(
    frames.map((_, i) => i),
    FPS,
    2,
  );
  const userTurn = expandVideo(timestamps, perGroup) + QUESTION;
  const textPrompt = model.encode(buildPrompt({ system: SYSTEM, user: userTurn }));
  log(
    `video: ${FRAMES} frames ${WIDTH}x${HEIGHT}, ${features.length / 2048} tokens, encode ${ms(encodeMs)}; prompt ${textPrompt.length} tokens`,
  );
  ui.stat('Video encode', ms(encodeMs));

  const textRuns = [];
  for (let i = 0; i < RUNS; i++) {
    const r = await textGuess(model, features, textPrompt);
    log(
      `text #${i + 1}: "${r.text}" prefill ${ms(r.prefillMs)}, total ${ms(r.totalMs)}, ${r.decodeTokPerS.toFixed(1)} tok/s`,
    );
    textRuns.push(r);
    if (i === 0) {
      ui.showGuess(r.text, `answered in ${ms(r.prefillMs)}`);
      ui.stat('Time to guess', ms(r.prefillMs));
      ui.setPhase('result');
    }
  }
  if (textRuns.length)
    ui.showGuess(
      textRuns[0].text,
      `${agreement(textRuns.map((r) => r.text))} · first answer in ${ms(textRuns[0].prefillMs)}`,
    );

  // A long free-form answer gives a meaningful decode tok/s; the guess itself is only a couple of tokens.
  const benchPrompt = model.encode(
    buildPrompt({ system: SYSTEM, user: expandVideo(timestamps, perGroup) + BENCH_QUESTION }),
  );
  const bench = await textGuess(model, features, benchPrompt, 128);
  log(`bench: ${bench.tokens} tokens, ${bench.decodeTokPerS.toFixed(1)} tok/s decode: "${bench.text.slice(0, 160)}…"`);
  ui.stat('Decode speed', `${bench.decodeTokPerS.toFixed(0)} tok/s`);

  const speechPrompt = model.encode(
    buildPrompt({ system: `${SYSTEM} Respond in interleaved text and audio.`, user: userTurn, speech: true }),
  );
  const speechRuns = [];
  let lastSpeech: Awaited<ReturnType<typeof speechGuess>> | undefined;
  let spoken = false;
  for (let i = 0; i < RUNS; i++) {
    try {
      lastSpeech = await speechGuess(model, features, speechPrompt);
      speechRuns.push(lastSpeech.result);
      log(`speech #${i + 1}: ${JSON.stringify(lastSpeech.result)}`);
      const { audio, result } = lastSpeech;
      // Only the first spoken answer plays; later runs are measured but do not interrupt it.
      if (!spoken && audio.length) {
        spoken = true;
        ui.showSpeech(result.text, encodeWav(trimLeadIn(audio, SAMPLE_RATE), SAMPLE_RATE));
        if (result.firstAudioFrameMs !== null) ui.stat('First audio', ms(result.firstAudioFrameMs));
      }
      if (result.realTimeFactor !== null) ui.stat('Speech RTF', result.realTimeFactor.toFixed(2));
    } catch (e) {
      speechRuns.push({ error: String(e) });
      log(`speech #${i + 1} failed: ${e}`);
    }
  }
  // The saved WAV stays untrimmed so it matches the benchmarked audio; only playback is trimmed.
  if (lastSpeech) window.__spikeWav = b64(encodeWav(lastSpeech.audio, SAMPLE_RATE));

  if (DUMP) {
    window.__spikeDump = {
      pixelValues: pairs.map((p) => tensorDump(p.data, p.dims)),
      videoFeatures: tensorDump(features, [features.length / 2048, 2048]),
      inputIds: textPrompt,
      replyIds: textRuns[0].textIds,
      firstTop5: textRuns[0].firstTop5,
      speechFrames: lastSpeech?.frames.map((f) => Array.from(f)) ?? [],
      speechAudio: lastSpeech ? tensorDump(lastSpeech.audio, [lastSpeech.audio.length]) : null,
    };
  }

  window.__spikeResult = {
    gpu,
    load: t,
    video: {
      frames: FRAMES,
      width: WIDTH,
      height: HEIGHT,
      videoTokens: features.length / 2048,
      captureMs,
      encodeMs,
      firstFrameLuma,
      warmup,
      countdownS: COUNTDOWN,
    },
    promptTokens: textPrompt.length,
    textRuns: textRuns.map(({ text, tokens, prefillMs, totalMs, decodeTokPerS }) => ({
      text,
      tokens,
      prefillMs,
      totalMs,
      decodeTokPerS,
    })),
    bench: { tokens: bench.tokens, decodeTokPerS: bench.decodeTokPerS, text: bench.text },
    speechRuns,
    webSpeech: webSpeechStatus(),
  };
  log('done');
  ui.setPhase('done');
}

main().catch((e) => {
  window.__spikeError = String(e?.stack ?? e);
  log(`ERROR: ${window.__spikeError}`);
  ui.fail(e instanceof Error ? e.message : String(e));
});
