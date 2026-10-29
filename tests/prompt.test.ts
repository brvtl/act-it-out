import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { buildPrompt, expandVideo } from '../src/model/prompt';

describe('expandVideo', () => {
  it('wraps each timestamped group and the whole clip in vision spans', () => {
    expect(expandVideo([0.25, 1.25], 2)).toBe(
      '<|vision_start|>' +
        '<0.2 seconds><|vision_start|><|video|><|video|><|vision_end|>' +
        '<1.2 seconds><|vision_start|><|video|><|video|><|vision_end|>' +
        '<|vision_end|>',
    );
  });
});

describe('buildPrompt', () => {
  it('renders the ChatML turn with an empty think block', () => {
    expect(buildPrompt({ system: 'S', user: 'V Q' })).toBe(
      '<|startoftext|><|im_start|>system\nS<|im_end|>\n<|im_start|>user\nV Q<|im_end|>\n<|im_start|>assistant\n<think></think>',
    );
  });

  it('matches the checkpoint chat template after the processor expands the video', () => {
    // Golden from scripts/goldens/make_goldens.py: chat_template.jinja rendered with a video then text.
    const rendered = readFileSync(new URL('./fixtures/prompt_template.txt', import.meta.url), 'utf8');
    const video = expandVideo([0.25, 1.25], 3);
    // The processor strips the template's span pair around the placeholder and its expansion adds exactly one.
    const processed = rendered.replace('<|vision_start|><|video|><|vision_end|>', video);

    expect(buildPrompt({ system: 'SYSTEM TEXT', user: video + 'USER TEXT' })).toBe(processed);
  });

  it('appends the interleaved marker when speech is requested', () => {
    expect(buildPrompt({ system: 'S', user: 'U', speech: true })).toMatch(/<think><\/think><\|mixed_start\|>$/);
  });
});
