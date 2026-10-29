import fc from 'fast-check';
import { describe, expect, it } from 'vitest';
import { trimLeadIn } from '../src/audio/trim';

const FC = { seed: 42, numRuns: 200 };
const SR = 24000;

/** Offset of `out` inside `input` (it is always a view on the same buffer). */
const startOf = (input: Float32Array, out: Float32Array): number =>
  (out.byteOffset - input.byteOffset) / Float32Array.BYTES_PER_ELEMENT;

/**
 * Independent oracle: brute-force RMS of every window in dB, recomputed from scratch per window
 * (no prefix sums). Returns the index of the first window louder than `thresholdDb`, or -1.
 */
function firstLoudWindow(x: Float32Array, sampleRate: number, windowMs = 20, thresholdDb = -35): number {
  const win = Math.min(x.length, Math.max(1, Math.round((windowMs * sampleRate) / 1000)));
  for (let i = 0; i + win <= x.length; i++) {
    let sumSq = 0;
    for (let j = i; j < i + win; j++) {
      const v = Math.max(-1, Math.min(1, x[j]));
      sumSq += v * v;
    }
    const rms = Math.sqrt(sumSq / win);
    if (rms > 0 && 20 * Math.log10(rms) > thresholdDb) return i;
  }
  return -1;
}

/** Sine with the given RMS level in dBFS. */
function tone(seconds: number, db: number, hz = 220, sampleRate = SR): Float32Array {
  const amp = 10 ** (db / 20) * Math.SQRT2;
  const out = new Float32Array(Math.round(seconds * sampleRate));
  for (let i = 0; i < out.length; i++) out[i] = amp * Math.sin((2 * Math.PI * hz * i) / sampleRate);
  return out;
}

const concat = (...parts: Float32Array[]): Float32Array => {
  const out = new Float32Array(parts.reduce((n, p) => n + p.length, 0));
  let o = 0;
  for (const p of parts) {
    out.set(p, o);
    o += p.length;
  }
  return out;
};

/** Signals made of segments at different levels (silence, quiet, borderline, loud). */
const segmented = fc
  .tuple(
    fc.float32Array({ noNaN: true, noDefaultInfinity: true, min: -1, max: 1, maxLength: 3000 }),
    fc.array(fc.constantFrom(0, 1e-3, 0.02, 0.3, 1), { minLength: 1, maxLength: 8 }),
  )
  .map(([raw, levels]) => raw.map((s, i) => s * levels[Math.floor((i * levels.length) / raw.length)]));
const rate = fc.integer({ min: 1000, max: 48000 });

describe('trimLeadIn properties', () => {
  it('returns a suffix of the input that is never longer', () => {
    fc.assert(
      fc.property(segmented, rate, (x, sr) => {
        const out = trimLeadIn(x, sr);
        expect(out.buffer).toBe(x.buffer);
        expect(out.length).toBeLessThanOrEqual(x.length);
        expect(startOf(x, out) + out.length).toBe(x.length);
      }),
      FC,
    );
  });

  it('starts exactly padMs before the first loud window found by a brute-force oracle', () => {
    fc.assert(
      fc.property(segmented, rate, (x, sr) => {
        const loud = firstLoudWindow(x, sr);
        const out = trimLeadIn(x, sr);
        const pad = Math.round((60 * sr) / 1000);
        expect(startOf(x, out)).toBe(loud < 0 ? 0 : Math.max(0, loud - pad));
      }),
      FC,
    );
  });

  it('preserves every sample from the first loud window onwards', () => {
    fc.assert(
      fc.property(segmented, rate, (x, sr) => {
        const loud = firstLoudWindow(x, sr);
        fc.pre(loud >= 0);
        const out = trimLeadIn(x, sr);
        const kept = x.subarray(loud);
        expect(Array.from(out.subarray(out.length - kept.length))).toEqual(Array.from(kept));
      }),
      FC,
    );
  });

  it('returns all-quiet input unchanged (same reference)', () => {
    const q = Math.fround(0.0177); // below the -35 dBFS amplitude (0.01778), so every window RMS is too
    fc.assert(
      fc.property(fc.float32Array({ noNaN: true, min: -q, max: q, maxLength: 3000 }), rate, (x, sr) => {
        expect(trimLeadIn(x, sr)).toBe(x);
      }),
      FC,
    );
  });

  it('is idempotent: trimming a trimmed buffer changes nothing', () => {
    fc.assert(
      fc.property(segmented, rate, (x, sr) => {
        const once = trimLeadIn(x, sr);
        const twice = trimLeadIn(once, sr);
        expect(twice.length).toBe(once.length);
        expect(startOf(x, twice)).toBe(startOf(x, once));
      }),
      FC,
    );
  });
});

describe('trimLeadIn examples', () => {
  it('cuts 0.85 s of -45 dB tone before a loud burst to within window+pad of the onset', () => {
    const x = concat(tone(0.85, -45), tone(0.5, -6, 440));
    const out = trimLeadIn(x, SR);
    const startS = startOf(x, out) / SR;
    expect(startS).toBeGreaterThanOrEqual(0.85 - 0.06 - 0.02);
    expect(startS).toBeLessThanOrEqual(0.85);
    expect(out.length).toBeLessThan(x.length);
  });

  it('keeps a -30 dB lead-in, which is above the -35 dB default threshold', () => {
    const x = concat(tone(0.5, -30), tone(0.5, -6));
    expect(trimLeadIn(x, SR)).toBe(x);
  });

  it('honours custom threshold, window and pad', () => {
    // 1000 zeros, then two samples at 0.01 (-40 dB, quiet), then 0.05 (-26 dB) from index 1002.
    const x = concat(new Float32Array(1000), new Float32Array([0.01, 0.01]), new Float32Array(50).fill(0.05));
    // Under a -20 dB threshold nothing is loud, so nothing is cut.
    expect(trimLeadIn(x, SR, { thresholdDb: -20 })).toBe(x);
    // 1 ms at 1 kHz is a single-sample window, so the onset is exactly index 1002.
    expect(startOf(x, trimLeadIn(x, 1000, { windowMs: 1, padMs: 0 }))).toBe(1002);
    // 5 ms of pad at 1 kHz keeps 5 samples before the onset.
    expect(startOf(x, trimLeadIn(x, 1000, { windowMs: 1, padMs: 5 }))).toBe(997);
  });

  it('returns empty and silent input unchanged', () => {
    const empty = new Float32Array(0);
    expect(trimLeadIn(empty, SR)).toBe(empty);
    const silent = new Float32Array(SR);
    expect(trimLeadIn(silent, SR)).toBe(silent);
  });

  it('keeps the whole buffer when the onset is within padMs of the start', () => {
    const x = concat(new Float32Array(Math.round(0.03 * SR)), tone(0.2, -6));
    expect(trimLeadIn(x, SR)).toBe(x);
  });

  it('treats a buffer shorter than one window as a single window', () => {
    const loud = new Float32Array([0.5, -0.5, 0.5]);
    expect(trimLeadIn(loud, SR)).toBe(loud);
    const short = concat(new Float32Array(3000), new Float32Array([0.9, -0.9]));
    expect(trimLeadIn(short, SR, { windowMs: 1000, padMs: 0 })).toBe(short);
  });

  it('counts NaN as silence and clamps out-of-range samples', () => {
    const x = concat(new Float32Array(SR).fill(Number.NaN), tone(0.2, -6));
    expect(startOf(x, trimLeadIn(x, SR))).toBeGreaterThan(0.9 * SR);
    // A single +Infinity sample clamps to 1; it must not poison detection of later windows.
    const spike = concat(new Float32Array(SR), new Float32Array([Number.POSITIVE_INFINITY]));
    const out = trimLeadIn(spike, SR, { padMs: 0 });
    expect(startOf(spike, out)).toBe(SR - Math.round(0.02 * SR) + 1);
  });

  it('rejects a non-positive sample rate', () => {
    expect(() => trimLeadIn(new Float32Array(10), 0)).toThrow(RangeError);
    expect(() => trimLeadIn(new Float32Array(10), Number.NaN)).toThrow(RangeError);
  });
});
