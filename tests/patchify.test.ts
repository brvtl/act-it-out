import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { patchifyPair, PATCH_DIM } from '../src/model/patchify';

// Golden from scripts/goldens/make_goldens.py: numpy transcription of the reference view + permute.
const W = 96;
const H = 64;
const fixture = (name: string) => readFileSync(new URL(`./fixtures/${name}`, import.meta.url));

describe('patchifyPair', () => {
  const raw = fixture('patchify_frames_2x64x96_rgba.u8');
  const frameBytes = W * H * 4;
  const a = new Uint8ClampedArray(raw.buffer, raw.byteOffset, frameBytes);
  const b = new Uint8ClampedArray(raw.buffer, raw.byteOffset + frameBytes, frameBytes);
  const expectedBytes = fixture('patchify_expected.f32');
  const expected = new Float32Array(expectedBytes.buffer, expectedBytes.byteOffset, expectedBytes.byteLength / 4);

  const out = patchifyPair(a, b, W, H);

  it('has one 2x2 merge block of patches per 32x32 pixels', () => {
    expect(out.dims).toEqual([2, 3, 2, 2, PATCH_DIM]);
  });

  it('matches the reference processor layout and normalization exactly', () => {
    expect(out.data.length).toBe(expected.length);
    let mismatches = 0;
    for (let i = 0; i < expected.length; i++) if (Math.abs(out.data[i] - expected[i]) > 1e-6) mismatches++;
    expect(mismatches).toBe(0);
  });

  it('rejects sizes that are not multiples of 32', () => {
    expect(() => patchifyPair(new Uint8ClampedArray(48 * 32 * 4), new Uint8ClampedArray(48 * 32 * 4), 48, 32)).toThrow(
      /multiple of 32/,
    );
  });
});
