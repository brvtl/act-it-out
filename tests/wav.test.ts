import { readFileSync } from 'node:fs';
import fc from 'fast-check';
import { describe, expect, it } from 'vitest';
import { encodeWav } from '../src/audio/wav';

// Golden from scripts/goldens/make_wav_golden.py: Python's stdlib `wave` module, an independent encoder.
const fixture = (name: string) => readFileSync(new URL(`./fixtures/${name}`, import.meta.url));
const PROPS = { seed: 42, numRuns: 200 };
const clamp = (s: number) => Math.max(-1, Math.min(1, s));
const ascii = (v: DataView, o: number, n: number) =>
  String.fromCharCode(...Array.from({ length: n }, (_, i) => v.getUint8(o + i)));
const view = (bytes: Uint8Array) => new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
const decode = (bytes: Uint8Array) => {
  const v = view(bytes);
  return Array.from({ length: (bytes.length - 44) / 2 }, (_, i) => v.getInt16(44 + i * 2, true));
};

const samples = fc.float32Array({ noNaN: true });
const rate = fc.integer({ min: 8000, max: 96000 });

describe('encodeWav', () => {
  it('is byte-identical to the Python wave module golden', () => {
    const input = new Float32Array([0, 0.5, -0.5, 1, -1, 1.7, -3, 0.25, 1e-6, -0]);
    const expected = new Uint8Array(fixture('wav_golden.wav'));
    expect(Array.from(encodeWav(input, 24000))).toEqual(Array.from(expected));
  });

  it('encodes an empty signal as a bare 44-byte header', () => {
    const out = encodeWav(new Float32Array(0), 24000);
    expect(out.length).toBe(44);
    expect(view(out).getUint32(40, true)).toBe(0);
  });

  it('sizes the file and its RIFF and data chunks from the sample count', () => {
    fc.assert(
      fc.property(samples, rate, (s, sr) => {
        const out = encodeWav(s, sr);
        const v = view(out);
        expect(out.length).toBe(44 + 2 * s.length);
        expect(v.getUint32(4, true)).toBe(out.length - 8);
        expect(v.getUint32(40, true)).toBe(2 * s.length);
      }),
      PROPS,
    );
  });

  it('writes a mono PCM16 header with the given sample rate', () => {
    fc.assert(
      fc.property(samples, rate, (s, sr) => {
        const v = view(encodeWav(s, sr));
        expect(ascii(v, 0, 4)).toBe('RIFF');
        expect(ascii(v, 8, 4)).toBe('WAVE');
        expect(ascii(v, 12, 4)).toBe('fmt ');
        expect(ascii(v, 36, 4)).toBe('data');
        expect(v.getUint16(20, true)).toBe(1); // PCM
        expect(v.getUint16(22, true)).toBe(1); // mono
        expect(v.getUint32(24, true)).toBe(sr);
        expect(v.getUint32(28, true)).toBe(2 * sr);
        expect(v.getUint16(32, true)).toBe(2);
        expect(v.getUint16(34, true)).toBe(16);
      }),
      PROPS,
    );
  });

  it('quantizes each clamped sample to within one step, keeping its sign', () => {
    fc.assert(
      fc.property(samples, rate, (s, sr) => {
        const decoded = decode(encodeWav(s, sr));
        decoded.forEach((d, i) => {
          const c = clamp(s[i]);
          expect(d).toBeGreaterThanOrEqual(-32767);
          expect(d).toBeLessThanOrEqual(32767);
          expect(d === 0 || Math.sign(d) === Math.sign(c)).toBe(true);
          expect(Math.abs(d / 32767 - c)).toBeLessThan(1 / 32767);
        });
      }),
      PROPS,
    );
  });

  it('is monotone: sorted input decodes to non-decreasing samples', () => {
    fc.assert(
      fc.property(samples, rate, (s, sr) => {
        const decoded = decode(encodeWav(Float32Array.from(s).sort(), sr));
        for (let i = 1; i < decoded.length; i++) expect(decoded[i]).toBeGreaterThanOrEqual(decoded[i - 1]);
      }),
      PROPS,
    );
  });
});
