import { formatTimestamp } from './timestamps';

const VISION_START = '<|vision_start|>';
const VISION_END = '<|vision_end|>';
const VIDEO = '<|video|>';

/** Expands one video into its timestamped frame groups, matching `Lfm3OmniProcessor._video_expansion`. */
export function expandVideo(timestamps: readonly number[], tokensPerGroup: number): string {
  const groups = timestamps
    .map((t) => `<${formatTimestamp(t)} seconds>` + VISION_START + VIDEO.repeat(tokensPerGroup) + VISION_END)
    .join('');
  return VISION_START + groups + VISION_END;
}

export interface PromptParts {
  system: string;
  user: string;
  /** Open the reply with `<|mixed_start|>` so the model answers in interleaved text and speech. */
  speech?: boolean;
}

/** Renders a single-turn chat the way the checkpoint's chat template does with `add_generation_prompt`. */
export function buildPrompt({ system, user, speech = false }: PromptParts): string {
  return (
    '<|startoftext|>' +
    `<|im_start|>system\n${system}<|im_end|>\n` +
    `<|im_start|>user\n${user}<|im_end|>\n` +
    '<|im_start|>assistant\n<think></think>' +
    (speech ? '<|mixed_start|>' : '')
  );
}
