import * as ort from 'onnxruntime-web/webgpu';
import { Tokenizer } from '@huggingface/tokenizers';
import { HIDDEN_SIZE, NUM_CODEBOOKS, TOKENS } from './constants';
import { fetchCached, type FileProgress } from './fetchCached';
import type { DecoderOutput, DecoderPort } from './generate';
import { istftFromFeatures } from './istft';
import type { DepthPort } from './maskgit';
import type { PatchTensor } from './patchify';

const REPO = 'https://huggingface.co/LiquidAI/LFM3-Omni-2B-ONNX/resolve/7d6bd3fd59a0622e5ce7fefc141145a8bba1435a';
const CACHE_NAME = 'act-it-out-lfm3-omni-2b-onnx-7d6bd3f';
const MAX_PREFILL_CHUNK = 1024;

/** The WebGPU bundle the model card prescribes: decoder_q4f16 with the float32 side graphs of the q4 bundle. */
const GRAPHS = {
  decoder: 'decoder_q4f16',
  vision: 'vision_encoder_q8',
  depth: 'depth_decoder',
  codec: 'codec_decoder',
} as const;
type GraphKey = keyof typeof GRAPHS;

export const MODEL_FILES = [
  ...Object.values(GRAPHS).flatMap((g) => [`onnx/${g}.onnx`, `onnx/${g}.onnx_data`]),
  'tokenizer.json',
  'tokenizer_config.json',
];

export interface LoadProgress {
  file: string;
  loaded: number;
  total: number;
  fromCache: boolean;
}

export interface LoadTimings {
  downloadMs: number;
  sessionMs: number;
  bytes: number;
  fromCache: boolean;
}

type Feeds = Record<string, ort.Tensor>;

const int64 = (values: readonly number[], dims: number[]) =>
  new ort.Tensor('int64', BigInt64Array.from(values, BigInt), dims);
const emptyFeatures = () => new ort.Tensor('float32', new Float32Array(0), [0, HIDDEN_SIZE]);

async function downloadAll(onProgress?: (p: LoadProgress) => void): Promise<Map<string, Uint8Array>> {
  const files = new Map<string, Uint8Array>();
  for (const file of MODEL_FILES) {
    const bytes = await fetchCached(`${REPO}/${file}`, CACHE_NAME, (p: FileProgress) =>
      onProgress?.({ file, loaded: p.loaded, total: p.total, fromCache: p.fromCache }),
    );
    files.set(file, bytes);
  }
  return files;
}

async function createSession(files: Map<string, Uint8Array>, graph: string): Promise<ort.InferenceSession> {
  const model = files.get(`onnx/${graph}.onnx`);
  const data = files.get(`onnx/${graph}.onnx_data`);
  if (!model || !data) throw new Error(`missing files for ${graph}`);
  return ort.InferenceSession.create(model, {
    executionProviders: ['webgpu'],
    graphOptimizationLevel: 'all',
    externalData: [{ path: `${graph}.onnx_data`, data }],
    preferredOutputLocation: graph === GRAPHS.decoder ? 'gpu-buffer' : 'cpu',
  });
}

/** Maps a decoder output cache name to the input it feeds on the next step. */
function pastNameFor(presentName: string): string {
  return presentName.replace(/^present_conv\./, 'past_conv.').replace(/^present\./, 'past_key_values.');
}

function initialCache(session: ort.InferenceSession): Feeds {
  const cache: Feeds = {};
  for (const name of session.inputNames) {
    if (name.startsWith('past_conv.'))
      cache[name] = new ort.Tensor('float16', new Uint16Array(HIDDEN_SIZE * 7), [1, HIDDEN_SIZE, 7]);
    else if (name.startsWith('past_key_values.'))
      cache[name] = new ort.Tensor('float16', new Uint16Array(0), [1, 4, 0, 128]);
  }
  return cache;
}

/** One conversation on the decoder session; its KV and conv caches stay on the GPU between steps. */
class OrtConversation implements DecoderPort {
  private cache: Feeds;
  private length = 0;

  constructor(
    private readonly session: ort.InferenceSession,
    private readonly videoFeatures: Float32Array,
  ) {
    this.cache = initialCache(session);
  }

  async prefill(inputIds: number[]): Promise<DecoderOutput> {
    let out: DecoderOutput | undefined;
    let videoOffset = 0;
    for (let start = 0; start < inputIds.length; start += MAX_PREFILL_CHUNK) {
      const chunk = inputIds.slice(start, start + MAX_PREFILL_CHUNK);
      const videoTokens = chunk.filter((id) => id === TOKENS.video).length;
      const video = this.videoFeatures.subarray(videoOffset * HIDDEN_SIZE, (videoOffset + videoTokens) * HIDDEN_SIZE);
      videoOffset += videoTokens;
      out = await this.step(chunk, new ort.Tensor('float32', video, [videoTokens, HIDDEN_SIZE]));
    }
    if (!out) throw new Error('empty prompt');
    return out;
  }

  feedToken(id: number): Promise<DecoderOutput> {
    return this.step([id]);
  }

  feedFrame(codes: Int32Array): Promise<DecoderOutput> {
    return this.step([TOKENS.audioOut], undefined, int64(Array.from(codes), [1, NUM_CODEBOOKS]));
  }

  dispose(): void {
    for (const t of Object.values(this.cache)) t.dispose();
  }

  private async step(ids: number[], video = emptyFeatures(), audioOut?: ort.Tensor): Promise<DecoderOutput> {
    const total = this.length + ids.length;
    const feeds: Feeds = {
      input_ids: int64(ids, [1, ids.length]),
      attention_mask: int64(Array<number>(total).fill(1), [1, total]),
      position_ids: int64(
        ids.map((_, i) => this.length + i),
        [1, ids.length],
      ),
      image_features: emptyFeatures(),
      video_features: video,
      audio_features: emptyFeatures(),
      audio_out_codes: audioOut ?? int64([], [0, NUM_CODEBOOKS]),
      logits_to_keep: int64([1], [1]),
      ...this.cache,
    };
    const results = await this.session.run(feeds);
    const next: Feeds = {};
    for (const [name, tensor] of Object.entries(results)) {
      if (name.startsWith('present')) next[pastNameFor(name)] = tensor;
    }
    for (const t of Object.values(this.cache)) t.dispose();
    this.cache = next;
    this.length = total;
    const [logits, hidden] = await Promise.all([results.logits.getData(), results.hidden_states.getData()]);
    results.logits.dispose();
    results.hidden_states.dispose();
    return { logits: logits as Float32Array, hidden: hidden as Float32Array };
  }
}

/** The LFM3-Omni-2B model on WebGPU: tokenizer, vision encoder, decoder, speech depth decoder and codec. */
export class OmniRuntime implements DepthPort {
  private constructor(
    readonly tokenizer: Tokenizer,
    private readonly sessions: Record<GraphKey, ort.InferenceSession>,
    readonly timings: LoadTimings,
  ) {}

  static async load(onProgress?: (p: LoadProgress) => void): Promise<OmniRuntime> {
    ort.env.wasm.wasmPaths = { wasm: `${import.meta.env.BASE_URL}ort/ort-wasm-simd-threaded.asyncify.wasm` };
    ort.env.wasm.numThreads = 1;

    let fromCache = true;
    const t0 = performance.now();
    const files = await downloadAll((p) => {
      if (!p.fromCache) fromCache = false;
      onProgress?.(p);
    });
    const t1 = performance.now();
    const bytes = [...files.values()].reduce((n, f) => n + f.byteLength, 0);

    const sessions = {} as Record<GraphKey, ort.InferenceSession>;
    for (const key of Object.keys(GRAPHS) as GraphKey[]) {
      sessions[key] = await createSession(files, GRAPHS[key]);
      files.delete(`onnx/${GRAPHS[key]}.onnx_data`);
    }
    const decodeJson = (name: string) => JSON.parse(new TextDecoder().decode(files.get(name)));
    const tokenizer = new Tokenizer(decodeJson('tokenizer.json'), decodeJson('tokenizer_config.json'));
    return new OmniRuntime(tokenizer, sessions, {
      downloadMs: t1 - t0,
      sessionMs: performance.now() - t1,
      bytes,
      fromCache,
    });
  }

  /** Runs the vision encoder over each temporal patch and concatenates the language-model features. */
  async encodeVideo(pairs: readonly PatchTensor[]): Promise<Float32Array> {
    const parts: Float32Array[] = [];
    for (const pair of pairs) {
      const { image_features } = await this.sessions.vision.run({
        pixel_values: new ort.Tensor('float32', pair.data, pair.dims),
      });
      parts.push((await image_features.getData()) as Float32Array);
      image_features.dispose();
    }
    const out = new Float32Array(parts.reduce((n, p) => n + p.length, 0));
    let offset = 0;
    for (const p of parts) {
      out.set(p, offset);
      offset += p.length;
    }
    return out;
  }

  conversation(videoFeatures: Float32Array): DecoderPort & { dispose(): void } {
    return new OrtConversation(this.sessions.decoder, videoFeatures);
  }

  async logits(hidden: Float32Array, codes: Int32Array, masked: readonly boolean[]): Promise<Float32Array> {
    const { logits } = await this.sessions.depth.run({
      hidden_states: new ort.Tensor('float32', hidden, [1, HIDDEN_SIZE]),
      codes: int64(Array.from(codes), [1, NUM_CODEBOOKS]),
      masked: new ort.Tensor('bool', Uint8Array.from(masked, Number), [1, NUM_CODEBOOKS]),
    });
    return (await logits.getData()) as Float32Array;
  }

  /** Decodes speech frames to 24 kHz audio: codec decoder on the GPU, then the iSTFT on the CPU. */
  async speak(frames: readonly Int32Array[]): Promise<Float32Array> {
    const n = frames.length;
    const codes = new Array<number>(NUM_CODEBOOKS * n);
    frames.forEach((frame, f) => frame.forEach((code, c) => (codes[c * n + f] = code)));
    const { stft_features } = await this.sessions.codec.run({ audio_codes: int64(codes, [1, NUM_CODEBOOKS, n]) });
    const features = (await stft_features.getData()) as Float32Array;
    return istftFromFeatures(features, stft_features.dims[1]);
  }

  decode(ids: readonly number[]): string {
    return this.tokenizer.decode([...ids], { skip_special_tokens: true }).trim();
  }

  encode(text: string): number[] {
    return [...this.tokenizer.encode(text, { add_special_tokens: false }).ids];
  }
}
