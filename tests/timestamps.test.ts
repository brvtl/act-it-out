import { describe, expect, it } from 'vitest';
import { formatTimestamp, groupTimestamps } from '../src/model/timestamps';

describe('formatTimestamp', () => {
  it('rounds exact binary ties half-to-even like Python', () => {
    expect(formatTimestamp(0.25)).toBe('0.2');
    expect(formatTimestamp(0.75)).toBe('0.8');
    expect(formatTimestamp(1.25)).toBe('1.2');
    expect(formatTimestamp(3.25)).toBe('3.2');
  });

  it('formats non-ties to one decimal', () => {
    expect(formatTimestamp(0)).toBe('0.0');
    expect(formatTimestamp(2.5)).toBe('2.5');
    expect(formatTimestamp(0.35)).toBe('0.3');
    expect(formatTimestamp(1.26)).toBe('1.3');
  });
});

describe('groupTimestamps', () => {
  it('centres each temporal group on its frames', () => {
    expect(groupTimestamps([0, 1, 2, 3, 4, 5, 6, 7], 2, 2)).toEqual([0.25, 1.25, 2.25, 3.25]);
  });

  it('pads an odd frame count by repeating the last frame', () => {
    expect(groupTimestamps([0, 1, 2], 2, 2)).toEqual([0.25, 1]);
  });
});
