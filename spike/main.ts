import { generate, type FrameSampler } from '../src/model/generate';
import { SAMPLE_RATE } from '../src/model/istft';
import { sampleFrame } from '../src/model/maskgit';
import { OmniRuntime } from '../src/model/omni';
import { patchifyPair, tokensPerGroup } from '../src/model/patchify';
import { buildPrompt, expandVideo } from '../src/model/prompt';
import { mulberry32 } from '../src/model/sampling';
import { groupTimestamps } from '../src/model/timestamps';
import { captureFrames } from '../src/video/capture';

const params = new URLSearchParams(location.search);
const RUNS = Number(params.get('runs') ?? 3);
const WIDTH = Number(params.get('w') ?? 448);
const HEIGHT = Number(params.get('h') ?? 256);
const FRAMES = Number(params.get('frames') ?? 8);
const FPS = 2;

const SYSTEM =
  'You are the guesser in a game of charades. The player acts out a word or phrase without speaking. ' +
  'Watch the video and guess what they are acting out.';
const QUESTION = 'What am I acting out? Answer with just your guess, in a few words.';

const logEl = document.getElementById('log') as HTMLPreElement;
const log = (msg: string) => {
  logEl.textContent += msg + '\n';
  console.log(`[spike] ${msg}`);
};
const ms = (t: number) => `${t.toFixed(0)} ms`;

declare global {
  interface Window {
    __spikeResult?: unknown;
    __spikeError?: string;
    __spikeWav?: string;
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
  video.srcObject = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 360 } });
  await video.play();
  return video;
}

function wavBase64(samples: Float32Array): string {
  const buf = new DataView(new ArrayBuffer(44 + samples.length * 2));
  const str = (o: number, s: string) => [...s].forEach((c, i) => buf.setUint8(o + i, c.charCodeAt(0)));
  str(0, 'RIFF');
  buf.setUint32(4, 36 + samples.length * 2, true);
  str(8, 'WAVEfmt ');
  buf.setUint32(16, 16, true);
  buf.setUint16(20, 1, true);
  buf.setUint16(22, 1, true);
  buf.setUint32(24, SAMPLE_RATE, true);
  buf.setUint32(28, SAMPLE_RATE * 2, true);
  buf.setUint16(32, 2, true);
  buf.setUint16(34, 16, true);
  str(36, 'data');
  buf.setUint32(40, samples.length * 2, true);
  samples.forEach((s, i) => buf.setInt16(44 + i * 2, Math.max(-1, Math.min(1, s)) * 0x7fff, true));
  let bin = '';
  new Uint8Array(buf.buffer).forEach((b) => (bin += String.fromCharCode(b)));
  return btoa(bin);
}

async function textGuess(model: OmniRuntime, features: Float32Array, prompt: number[]) {
  const conv = model.conversation(features);
  const t0 = performance.now();
  let firstTokenAt = 0;
  const out = await generate(
    conv,
    { sample: () => Promise.reject(new Error('text only')) },
    {
      inputIds: prompt,
      speech: false,
      maxNewTokens: 32,
      onText: () => (firstTokenAt ||= performance.now()),
    },
  );
  const end = performance.now();
  conv.dispose();
  const decodeTokens = out.textIds.length - 1;
  return {
    text: model.decode(out.textIds),
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
  };
}

function webSpeechStatus() {
  const synth = 'speechSynthesis' in window ? window.speechSynthesis : undefined;
  return { available: !!synth, voices: synth?.getVoices().length ?? 0 };
}

async function main() {
  const gpu = await adapterInfo();
  log(`adapter: ${JSON.stringify(gpu)}`);

  let lastPct = -1;
  const model = await OmniRuntime.load((p) => {
    const pct = Math.floor((p.loaded / p.total) * 10) * 10;
    if (p.loaded === p.total || pct !== lastPct) log(`  ${p.file}: ${pct}%${p.fromCache ? ' (cache)' : ''}`);
    lastPct = pct;
  });
  const t = model.timings;
  log(
    `load: ${(t.bytes / 1e9).toFixed(2)} GB, download ${ms(t.downloadMs)} (cache=${t.fromCache}), sessions ${ms(t.sessionMs)}`,
  );

  const video = await startCamera();
  const tCap = performance.now();
  const frames = await captureFrames(video, { count: FRAMES, fps: FPS, width: WIDTH, height: HEIGHT });
  const captureMs = performance.now() - tCap;

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

  const textRuns = [];
  for (let i = 0; i < RUNS; i++) {
    const r = await textGuess(model, features, textPrompt);
    log(
      `text #${i + 1}: "${r.text}" prefill ${ms(r.prefillMs)}, total ${ms(r.totalMs)}, ${r.decodeTokPerS.toFixed(1)} tok/s`,
    );
    textRuns.push(r);
  }

  const speechPrompt = model.encode(
    buildPrompt({ system: `${SYSTEM} Respond in interleaved text and audio.`, user: userTurn, speech: true }),
  );
  let speech: unknown;
  try {
    const s = await speechGuess(model, features, speechPrompt);
    speech = s.result;
    window.__spikeWav = wavBase64(s.audio);
    log(`speech: ${JSON.stringify(s.result)}`);
  } catch (e) {
    speech = { error: String(e) };
    log(`speech failed: ${e}`);
  }

  window.__spikeResult = {
    gpu,
    load: t,
    video: { frames: FRAMES, width: WIDTH, height: HEIGHT, videoTokens: features.length / 2048, captureMs, encodeMs },
    promptTokens: textPrompt.length,
    textRuns,
    speech,
    webSpeech: webSpeechStatus(),
  };
  log('done');
}

main().catch((e) => {
  window.__spikeError = String(e?.stack ?? e);
  log(`ERROR: ${window.__spikeError}`);
});
