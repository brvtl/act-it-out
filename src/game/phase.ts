/** The phases of one round, in the order the spike page walks through them. `error` can follow any phase. */
export type Phase =
  'loading' | 'camera' | 'ready' | 'countdown' | 'recording' | 'thinking' | 'result' | 'done' | 'error';

export interface PhaseCopy {
  /** The status pill text; the UI may append ` · detail`. */
  readonly status: string;
  /** The overlay title; empty where the overlay shows a big number (countdown, recording) or is hidden. */
  readonly title: string;
  /** The overlay hint; empty for `error`, whose hint is the first line of the error message. */
  readonly hint: string;
}

export const PHASE_COPY: Readonly<Record<Phase, PhaseCopy>> = {
  loading: {
    status: 'Loading model',
    title: 'Loading the guesser…',
    hint: 'First visit downloads about 2 GB; after that it loads from cache. Your camera is on so you can line up the shot.',
  },
  camera: {
    status: 'Starting camera…',
    title: 'Camera warming up',
    hint: 'Step back so your head and hands are in frame.',
  },
  ready: {
    status: 'Ready',
    title: 'Ready when you are',
    hint: readyHint(3, 3500),
  },
  countdown: {
    status: 'Get ready',
    title: '',
    hint: 'Get ready to act it out silently. No talking, gestures only.',
  },
  recording: {
    status: 'Recording',
    title: '',
    hint: 'Act it out! No talking: gestures only.',
  },
  thinking: {
    status: 'Thinking…',
    title: 'Hmm…',
    hint: 'Guessing from what it saw.',
  },
  result: { status: 'Measuring speed…', title: '', hint: '' },
  done: { status: 'Done', title: '', hint: '' },
  error: { status: 'Error', title: 'Something went wrong', hint: '' },
};

/** Wall time from the first to the last captured frame at `fps`; a single frame (or none) spans no time. */
export const captureWindowMs = (frames: number, fps: number): number => ((Math.max(frames, 1) - 1) * 1000) / fps;

/** Whole seconds still to go, rounded up so the display reads 3, 2, 1 and only shows 0 once time is up. */
export const secondsLeft = (durationMs: number, elapsedMs: number): number =>
  // The explicit guard keeps "0 iff time is up" even when (d - e) / 1000 underflows to 0.
  elapsedMs >= durationMs ? 0 : Math.max(1, Math.ceil((durationMs - elapsedMs) / 1000));

/**
 * Summarises how consistent repeated runs were: `3/3 runs agree`, or `runs: Jumping ×2, Running ×1`.
 * Answers match trimmed and case-insensitively; each label is the first spelling seen. Groups are listed
 * by count, most common first, ties in first-seen order.
 */
export function agreement(texts: readonly string[]): string {
  if (texts.length === 0) return '';
  const groups = new Map<string, { label: string; count: number }>();
  for (const text of texts) {
    const label = text.trim();
    const key = label.toLowerCase();
    const group = groups.get(key);
    if (group) group.count++;
    else groups.set(key, { label, count: 1 });
  }
  if (groups.size === 1) return `${texts.length}/${texts.length} runs agree`;
  // Array.prototype.sort is stable, so equal counts keep first-seen order.
  const ranked = [...groups.values()].sort((a, b) => b.count - a.count);
  return `runs: ${ranked.map((g) => `${g.label} ×${g.count}`).join(', ')}`;
}

/**
 * Reads the `?countdown=` parameter as whole seconds. Missing, blank, non-numeric, non-finite or negative
 * input yields `fallback`; fractional input is floored.
 */
export function parseCountdown(v: string | null, fallback = 3): number {
  if (v === null || v.trim() === '') return fallback;
  const n = Number(v);
  if (!Number.isFinite(n) || n < 0) return fallback;
  return Math.floor(n);
}

/** The ready-screen instructions for a round with a `countdownS` lead-in and an `actMs` recording window. */
export function readyHint(countdownS: number, actMs: number): string {
  const lead = countdownS > 0 ? `you get ${countdownS} s to get in frame and ` : '';
  return `Think of something to act out. Press Start, then ${lead}about ${Math.max(1, Math.round(actMs / 1000))} s to act. No talking: the AI only watches, it can't hear you.`;
}
