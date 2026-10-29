import fc from 'fast-check';
import { describe, expect, it } from 'vitest';
import { meanLuma } from '../src/video/luma';

const FC = { seed: 42, numRuns: 200 };

const solid = (pixels: number, r: number, g: number, b: number, a = 255) => {
  const out = new Uint8ClampedArray(pixels * 4);
  for (let p = 0; p < pixels; p++) out.set([r, g, b, a], p * 4);
  return out;
};

// Independent oracle: decode into pixel objects, average each pixel's luma (per-pixel, not per-channel sums).
const oracle = (rgba: Uint8ClampedArray): number => {
  const px: { r: number; g: number; b: number; a: number }[] = [];
  for (let i = 0; i + 3 < rgba.length; i += 4) px.push({ r: rgba[i], g: rgba[i + 1], b: rgba[i + 2], a: rgba[i + 3] });
  if (px.length === 0) return 0;
  return px.reduce((acc, { r, g, b }) => acc + (r * 299 + g * 587 + b * 114) / 1000, 0) / px.length;
};

const rgbaArb = fc
  .integer({ min: 0, max: 512 })
  .chain((n) => fc.uint8Array({ minLength: n * 4, maxLength: n * 4 }))
  .map((u) => new Uint8ClampedArray(u));

describe('meanLuma', () => {
  it('is 0 for empty input', () => {
    expect(meanLuma(new Uint8ClampedArray(0))).toBe(0);
  });

  it('is 0 for black and 255 for white', () => {
    expect(meanLuma(solid(16, 0, 0, 0))).toBe(0);
    expect(Math.abs(meanLuma(solid(16, 255, 255, 255)) - 255)).toBeLessThan(1e-9);
  });

  it('weights pure red by 0.299', () => {
    expect(meanLuma(solid(9, 255, 0, 0))).toBeCloseTo(0.299 * 255, 9);
  });

  it('reports an all-black first frame below the warm-up threshold of 8', () => {
    // waitForPicture gates on luma > 8 to skip a cold webcam's black frames. It is DOM-bound, so the gate itself
    // is covered end to end: the runner reports video.firstFrameLuma (127 on the test pattern, 0 before).
    expect(meanLuma(solid(32 * 18, 0, 0, 0))).toBeLessThan(8);
  });

  it('matches a per-pixel oracle', () => {
    fc.assert(
      fc.property(rgbaArb, (rgba) => {
        expect(Math.abs(meanLuma(rgba) - oracle(rgba))).toBeLessThan(1e-9);
      }),
      FC,
    );
  });

  it('stays within [0, 255]', () => {
    fc.assert(
      fc.property(rgbaArb, (rgba) => {
        const l = meanLuma(rgba);
        return l >= 0 && l <= 255 + 1e-9;
      }),
      FC,
    );
  });

  it('ignores alpha', () => {
    fc.assert(
      fc.property(rgbaArb, fc.uint8Array({ minLength: 512, maxLength: 512 }), (rgba, alphas) => {
        const other = rgba.slice();
        for (let i = 3, k = 0; i < other.length; i += 4, k++) other[i] = alphas[k];
        return meanLuma(other) === meanLuma(rgba);
      }),
      FC,
    );
  });
});
