import { describe, expect, it } from 'vitest';
import { AUDIO_EOS_CODE, INTERLEAVED_N_AUDIO, INTERLEAVED_N_TEXT, NUM_CODEBOOKS, TOKENS } from '../src/model/constants';
import { generate, type DecoderPort, type FrameSampler } from '../src/model/generate';

const VOCAB = 125100;

/** Logits whose argmax is `id`, with `runnerUp` second so suppression can be observed. */
function logitsFor(id: number, runnerUp = 7): Float32Array {
  const l = new Float32Array(VOCAB);
  l[runnerUp] = 1;
  l[id] = 2;
  return l;
}

type Fed = { kind: 'prefill' | 'token' | 'frame'; value: number[] };

/** A decoder that answers each fed slot with the next scripted text token. */
function scriptedDecoder(script: number[], runnerUp?: number) {
  const fed: Fed[] = [];
  let i = 0;
  const next = () => ({
    logits: logitsFor(script[Math.min(i++, script.length - 1)], runnerUp),
    hidden: new Float32Array(2),
  });
  const port: DecoderPort = {
    async prefill(ids) {
      fed.push({ kind: 'prefill', value: ids });
      return next();
    },
    async feedToken(id) {
      fed.push({ kind: 'token', value: [id] });
      return next();
    },
    async feedFrame(codes) {
      fed.push({ kind: 'frame', value: Array.from(codes) });
      return next();
    },
  };
  return { port, fed };
}

/** Speaks `frames` ordinary frames, then an end-of-speech frame. */
function scriptedSpeech(frames: number) {
  let n = 0;
  const sampler: FrameSampler = {
    async sample() {
      const f = new Int32Array(NUM_CODEBOOKS).fill(n < frames ? n + 1 : 0);
      if (n >= frames) f[3] = AUDIO_EOS_CODE;
      n++;
      return f;
    },
  };
  return sampler;
}

const noSpeech: FrameSampler = {
  async sample() {
    throw new Error('no speech expected');
  },
};

describe('generate (text)', () => {
  it('decodes greedily until <|im_end|> without feeding the final token', async () => {
    const { port, fed } = scriptedDecoder([11, 12, TOKENS.imEnd]);
    const out = await generate(port, noSpeech, { inputIds: [1, 2], speech: false, maxNewTokens: 10 });

    expect(out.textIds).toEqual([11, 12, TOKENS.imEnd]);
    expect(fed.map((f) => f.kind)).toEqual(['prefill', 'token', 'token']);
  });

  it('never opens speech in text-only mode', async () => {
    const { port } = scriptedDecoder([TOKENS.audioStart, TOKENS.imEnd], 42);
    const out = await generate(port, noSpeech, { inputIds: [1], speech: false, maxNewTokens: 2 });
    expect(out.textIds[0]).toBe(42);
  });

  it('stops after maxNewTokens', async () => {
    const { port } = scriptedDecoder([11]);
    const out = await generate(port, noSpeech, { inputIds: [1], speech: false, maxNewTokens: 4 });
    expect(out.textIds).toEqual([11, 11, 11, 11]);
  });
});

describe('generate (interleaved speech)', () => {
  it('alternates 6 text slots with 12 audio frames, then speaks the tail after <|text_end|>', async () => {
    const words = [21, 22, 23, 24, 25, 26];
    const script = [...words, ...Array(INTERLEAVED_N_AUDIO).fill(0), 27, TOKENS.textEnd, 0, 0, 0, 0, TOKENS.imEnd];
    const { port, fed } = scriptedDecoder(script);
    const out = await generate(port, scriptedSpeech(15), {
      inputIds: [1, TOKENS.mixedStart],
      speech: true,
      maxNewTokens: 100,
    });

    const kinds = fed
      .slice(1)
      .map((f) => f.kind[0])
      .join('');
    expect(kinds.slice(0, INTERLEAVED_N_TEXT + INTERLEAVED_N_AUDIO)).toBe('t'.repeat(6) + 'f'.repeat(12));
    expect(out.textIds).toEqual([...words, 27, TOKENS.textEnd, TOKENS.imEnd]);
    expect(out.frames).toHaveLength(15);
  });

  it('feeds an all-EOS frame back and returns to text when speech ends', async () => {
    const { port, fed } = scriptedDecoder([TOKENS.textEnd, 0, 0, TOKENS.imEnd]);
    const out = await generate(port, scriptedSpeech(1), {
      inputIds: [TOKENS.mixedStart],
      speech: true,
      maxNewTokens: 20,
    });

    const frames = fed.filter((f) => f.kind === 'frame').map((f) => f.value);
    expect(frames).toHaveLength(2);
    expect(frames[1]).toEqual(Array(NUM_CODEBOOKS).fill(AUDIO_EOS_CODE));
    expect(out.frames).toHaveLength(1);
    expect(out.textIds.at(-1)).toBe(TOKENS.imEnd);
  });

  it('streams text and frames as they are produced', async () => {
    const { port } = scriptedDecoder([TOKENS.textEnd, TOKENS.imEnd]);
    const seen: string[] = [];
    await generate(port, scriptedSpeech(2), {
      inputIds: [TOKENS.mixedStart],
      speech: true,
      maxNewTokens: 20,
      onText: (id) => seen.push(`t${id}`),
      onFrame: () => seen.push('f'),
    });
    expect(seen).toEqual([`t${TOKENS.textEnd}`, 'f', 'f', `t${TOKENS.imEnd}`]);
  });
});
