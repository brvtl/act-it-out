/**
 * Mean Rec.601 luma (0.299R + 0.587G + 0.114B) of an RGBA buffer, in 0..255.
 * Alpha is ignored; a trailing partial pixel is ignored; empty input gives 0.
 */
export function meanLuma(rgba: Uint8ClampedArray): number {
  const pixels = Math.floor(rgba.length / 4);
  if (pixels === 0) return 0;
  let r = 0;
  let g = 0;
  let b = 0;
  for (let i = 0; i < pixels * 4; i += 4) {
    r += rgba[i];
    g += rgba[i + 1];
    b += rgba[i + 2];
  }
  return (0.299 * r + 0.587 * g + 0.114 * b) / pixels;
}
