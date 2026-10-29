/** Token ids and decoding constants of LiquidAI/LFM3-Omni-2B (config.json, generation_config.json, tokenizer.json). */
export const TOKENS = {
  imEnd: 124900,
  textEnd: 125014,
  audioStart: 125012,
  mixedStart: 125015,
  audioOut: 125032,
  video: 125029,
} as const;

export const NUM_CODEBOOKS = 12;
export const CODEBOOK_SIZE = 2017;
export const AUDIO_EOS_CODE = 2016;
export const MASKGIT_SCHEDULE = [4, 4, 4] as const;
export const INTERLEAVED_N_TEXT = 6;
export const INTERLEAVED_N_AUDIO = 12;
export const HIDDEN_SIZE = 2048;
export const VOCAB_SIZE = 128000;

export const SPEECH_SAMPLING = { temperature: 0.4, repetitionPenalty: 1.2, repetitionWindow: 64 } as const;
