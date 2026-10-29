/** Grabs `count` frames from a playing video at `fps`, scaled to `width`x`height` RGBA. */
export async function captureFrames(
  video: HTMLVideoElement,
  { count, fps, width, height }: { count: number; fps: number; width: number; height: number },
): Promise<Uint8ClampedArray[]> {
  const canvas = new OffscreenCanvas(width, height);
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  if (!ctx) throw new Error('2d canvas unavailable');
  ctx.imageSmoothingQuality = 'high';
  const frames: Uint8ClampedArray[] = [];
  for (let i = 0; i < count; i++) {
    if (i > 0) await new Promise((r) => setTimeout(r, 1000 / fps));
    ctx.drawImage(video, 0, 0, width, height);
    frames.push(ctx.getImageData(0, 0, width, height).data);
  }
  return frames;
}
