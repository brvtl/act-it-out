import fc from 'fast-check';
import { describe, expect, it } from 'vitest';
import {
  PHASE_COPY,
  agreement,
  captureWindowMs,
  parseCountdown,
  readyHint,
  secondsLeft,
  type Phase,
} from '../src/game/phase';

const params = { seed: 42, numRuns: 200 };
const duration = fc.integer({ min: 1, max: 120_000 });
const elapsed = fc.double({ min: -10_000, max: 200_000, noNaN: true });

describe('captureWindowMs', () => {
  it('spans first to last frame', () => {
    expect(captureWindowMs(8, 2)).toBe(3500);
  });

  it('is zero for a single frame or none', () => {
    expect(captureWindowMs(1, 2)).toBe(0);
    expect(captureWindowMs(0, 2)).toBe(0);
  });
});

describe('secondsLeft', () => {
  it('counts whole seconds up', () => {
    expect(secondsLeft(3500, 0)).toBe(4);
    expect(secondsLeft(3500, 500)).toBe(3);
    expect(secondsLeft(3500, 3499)).toBe(1);
    expect(secondsLeft(3500, 3500)).toBe(0);
    expect(secondsLeft(3500, 9000)).toBe(0);
  });

  it('stays in [0, ceil(d/1000)]', () => {
    fc.assert(
      fc.property(duration, elapsed, (d, e) => {
        const s = secondsLeft(d, e);
        return Number.isInteger(s) && s >= 0 && s <= Math.ceil((d - Math.min(e, 0)) / 1000);
      }),
      params,
    );
    fc.assert(
      fc.property(duration, fc.double({ min: 0, max: 200_000, noNaN: true }), (d, e) => {
        const s = secondsLeft(d, e);
        return s >= 0 && s <= Math.ceil(d / 1000);
      }),
      params,
    );
  });

  it('is non-increasing in elapsed time', () => {
    fc.assert(
      fc.property(duration, elapsed, elapsed, (d, a, b) => {
        const [lo, hi] = a <= b ? [a, b] : [b, a];
        return secondsLeft(d, hi) <= secondsLeft(d, lo);
      }),
      params,
    );
  });

  it('is zero exactly when time is up', () => {
    fc.assert(
      fc.property(duration, elapsed, (d, e) => (secondsLeft(d, e) === 0) === e >= d),
      params,
    );
  });

  it('is zero exactly when time is up, even a hair before the end', () => {
    expect(secondsLeft(3500, 3500 - 1e-9)).toBe(1);
  });

  it('starts at ceil(d/1000)', () => {
    fc.assert(
      fc.property(duration, (d) => secondsLeft(d, 0) === Math.ceil(d / 1000)),
      params,
    );
  });
});

describe('agreement', () => {
  it('reports unanimity, matching trimmed and case-insensitively', () => {
    expect(agreement(['Jumping', 'jumping ', 'Jumping'])).toBe('3/3 runs agree');
  });

  it('tallies disagreeing runs, most common first', () => {
    expect(agreement(['Jumping', 'Running', 'Jumping'])).toBe('runs: Jumping ×2, Running ×1');
    expect(agreement(['Running', 'Jumping', 'jumping'])).toBe('runs: Jumping ×2, Running ×1');
  });

  it('labels each group with its first spelling, trimmed', () => {
    expect(agreement([' running', 'Jumping', 'RUNNING'])).toBe('runs: running ×2, Jumping ×1');
  });

  it('keeps first-seen order for tied counts', () => {
    expect(agreement(['Waving', 'Clapping'])).toBe('runs: Waving ×1, Clapping ×1');
  });

  it('handles a single run and no runs', () => {
    expect(agreement(['Jumping'])).toBe('1/1 runs agree');
    expect(agreement([])).toBe('');
  });
});

describe('parseCountdown', () => {
  it('reads whole seconds and falls back on bad input', () => {
    expect(parseCountdown(null)).toBe(3);
    expect(parseCountdown('0')).toBe(0);
    expect(parseCountdown('5')).toBe(5);
    expect(parseCountdown('-2')).toBe(3);
    expect(parseCountdown('abc')).toBe(3);
    expect(parseCountdown('')).toBe(3);
    expect(parseCountdown('Infinity')).toBe(3);
    expect(parseCountdown('2.7')).toBe(2);
    expect(parseCountdown(null, 0)).toBe(0);
  });

  it('returns a non-negative integer for any string', () => {
    fc.assert(
      fc.property(fc.string(), (s) => {
        const n = parseCountdown(s);
        return Number.isInteger(n) && n >= 0;
      }),
      params,
    );
  });

  it('round-trips any non-negative integer', () => {
    fc.assert(
      fc.property(fc.nat(), (n) => parseCountdown(String(n)) === n),
      params,
    );
  });
});

describe('PHASE_COPY', () => {
  const phases: Phase[] = [
    'loading',
    'camera',
    'ready',
    'countdown',
    'recording',
    'thinking',
    'result',
    'done',
    'error',
  ];

  it('covers every phase with a non-empty status', () => {
    expect(Object.keys(PHASE_COPY).sort()).toEqual([...phases].sort());
    for (const p of phases) expect(PHASE_COPY[p].status.length).toBeGreaterThan(0);
  });

  it('tells the player not to talk while acting', () => {
    expect(PHASE_COPY.countdown.hint).toMatch(/no talking/i);
    expect(PHASE_COPY.recording.hint).toMatch(/no talking/i);
  });

  it('explains on the ready screen that the AI cannot hear', () => {
    expect(PHASE_COPY.ready.hint).toContain("can't hear you");
  });
});

describe('readyHint', () => {
  it('states the default round timing', () => {
    expect(readyHint(3, captureWindowMs(8, 2))).toContain('you get 3 s to get in frame and about 4 s to act');
  });

  it('follows the countdown and frame parameters', () => {
    expect(readyHint(5, captureWindowMs(16, 2))).toContain('you get 5 s to get in frame and about 8 s to act');
    expect(readyHint(0, captureWindowMs(8, 2))).toContain('Press Start, then about 4 s to act');
  });

  it('always says the AI cannot hear', () => {
    expect(readyHint(0, 0)).toContain("can't hear you");
  });
});
