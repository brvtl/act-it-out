/**
 * Trims the quiet lead-in off generated speech so playback starts on the first word.
 * Playback only: the saved/benchmarked WAV must keep the untrimmed samples.
 */
export interface TrimOptions {
  /** RMS level a window must exceed to count as speech, in dBFS. Default -35. */
  thresholdDb?: number;
  /** Length of the sliding RMS window, in milliseconds. Default 20. */
  windowMs?: number;
  /** Audio kept before the first loud window, in milliseconds. Default 60. */
  padMs?: number;
}

/**
 * Returns a suffix view of `samples` that starts `padMs` before the first window whose RMS
 * exceeds `thresholdDb`, or `samples` itself when nothing is loud enough (or nothing is cut).
 *
 * The window slides one sample at a time, so detection does not depend on where the input
 * starts: trimming an already-trimmed buffer returns it unchanged. Samples are clamped to
 * [-1, 1] and NaN counts as silence, matching how encodeWav treats out-of-range values.
 */
export function trimLeadIn(samples: Float32Array, sampleRate: number, opts: TrimOptions = {}): Float32Array {
  if (!(sampleRate > 0) || !Number.isFinite(sampleRate)) throw new RangeError(`invalid sampleRate ${sampleRate}`);
  const { thresholdDb = -35, windowMs = 20, padMs = 60 } = opts;
  const n = samples.length;
  if (n === 0) return samples;

  const win = Math.min(n, Math.max(1, Math.round((windowMs * sampleRate) / 1000)));
  const pad = Math.max(0, Math.round((padMs * sampleRate) / 1000));
  const thr = 10 ** (thresholdDb / 20);
  const limit = thr * thr * win; // RMS > thr  <=>  sum of squares > thr^2 * win

  // prefix[i] = sum of squared (clamped) samples [0, i)
  const prefix = new Float64Array(n + 1);
  for (let i = 0; i < n; i++) {
    const s = samples[i];
    const v = Number.isNaN(s) ? 0 : Math.max(-1, Math.min(1, s));
    prefix[i + 1] = prefix[i] + v * v;
  }

  for (let i = 0; i + win <= n; i++) {
    if (prefix[i + win] - prefix[i] > limit) {
      const start = Math.max(0, i - pad);
      return start === 0 ? samples : samples.subarray(start);
    }
  }
  return samples;
}
