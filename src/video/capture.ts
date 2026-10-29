import { meanLuma } from './luma';

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

export interface PictureWait {
  /** Presented frames observed before resolving. */
  frames: number;
  /** Mean luma of the last sampled frame (0..255). */
  luma: number;
  /** Wall time spent waiting. */
  ms: number;
  /** True when the timeout elapsed before a bright enough frame appeared. */
  timedOut: boolean;
}

/**
 * Waits until the camera delivers a real picture: at least `minFrames` presented frames and a
 * downscaled frame whose mean luma exceeds `minLuma` (fresh webcams often emit black frames first).
 * Never throws; on timeout it resolves with `timedOut: true`.
 */
export function waitForPicture(
  video: HTMLVideoElement,
  { minFrames = 3, minLuma = 8, timeoutMs = 3000 }: { minFrames?: number; minLuma?: number; timeoutMs?: number } = {},
): Promise<PictureWait> {
  const t0 = performance.now();
  const canvas = new OffscreenCanvas(32, 18);
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  const useVfc = typeof video.requestVideoFrameCallback === 'function';
  return new Promise((resolve) => {
    let frames = 0;
    let luma = 0;
    let done = false;
    const finish = (timedOut: boolean) => {
      if (done) return;
      done = true;
      clearTimeout(timer);
      resolve({ frames, luma, ms: performance.now() - t0, timedOut });
    };
    const timer = setTimeout(() => finish(true), timeoutMs);
    const schedule = () => {
      if (useVfc) video.requestVideoFrameCallback(onFrame);
      else requestAnimationFrame(onFrame);
    };
    const onFrame = () => {
      if (done) return;
      // rAF fires without a new frame; only count it once the video has data.
      if (useVfc || video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA) frames++;
      if (frames >= minFrames && ctx) {
        try {
          ctx.drawImage(video, 0, 0, 32, 18);
          luma = meanLuma(ctx.getImageData(0, 0, 32, 18).data);
        } catch {
          luma = 0;
        }
        if (luma > minLuma) return finish(false);
      }
      schedule();
    };
    schedule();
  });
}
