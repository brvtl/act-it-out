/** LiquidCodec synthesis parameters (audio_tokenizer/config.json). */
export const N_FFT = 1280;
export const HOP = 320;
export const SAMPLE_RATE = 24000;
const BINS = N_FFT / 2 + 1;
const TRIM = (N_FFT - HOP) / 2;

let tables: { cos: Float32Array; sin: Float32Array; window: Float32Array } | undefined;

function getTables() {
  if (tables) return tables;
  const cos = new Float32Array(N_FFT);
  const sin = new Float32Array(N_FFT);
  const window = new Float32Array(N_FFT);
  for (let n = 0; n < N_FFT; n++) {
    cos[n] = Math.cos((2 * Math.PI * n) / N_FFT);
    sin[n] = Math.sin((2 * Math.PI * n) / N_FFT);
    window[n] = 0.5 - 0.5 * cos[n];
  }
  tables = { cos, sin, window };
  return tables;
}

/** Inverse real DFT of one half spectrum (torch.fft.irfft, norm="backward"), written into `out`. */
function irfft(re: Float32Array, im: Float32Array, out: Float32Array): void {
  const { cos, sin } = getTables();
  const nyquist = N_FFT / 2;
  for (let n = 0; n < N_FFT; n++) {
    let acc = re[0] + (n % 2 === 0 ? re[nyquist] : -re[nyquist]);
    let idx = n;
    for (let k = 1; k < nyquist; k++, idx += n) {
      const j = idx % N_FFT;
      acc += 2 * (re[k] * cos[j] - im[k] * sin[j]);
    }
    out[n] = acc / N_FFT;
  }
}

/**
 * Turns the codec decoder's `stft_features` (`[frames, log-magnitude ‖ angle]`, 1282 per frame) into 24 kHz audio:
 * irfft, Hann synthesis window, overlap-add, envelope normalization and the codec's edge trim.
 */
export function istftFromFeatures(features: Float32Array, frames: number): Float32Array {
  const { window } = getTables();
  const length = (frames - 1) * HOP + N_FFT;
  const audio = new Float32Array(length);
  const envelope = new Float32Array(length);
  const re = new Float32Array(BINS);
  const im = new Float32Array(BINS);
  const frame = new Float32Array(N_FFT);
  for (let f = 0; f < frames; f++) {
    const base = f * BINS * 2;
    for (let k = 0; k < BINS; k++) {
      const mag = Math.exp(features[base + k]);
      const angle = features[base + BINS + k];
      re[k] = mag * Math.cos(angle);
      im[k] = mag * Math.sin(angle);
    }
    irfft(re, im, frame);
    const offset = f * HOP;
    for (let n = 0; n < N_FFT; n++) {
      audio[offset + n] += frame[n] * window[n];
      envelope[offset + n] += window[n] * window[n];
    }
  }
  const out = new Float32Array(length - 2 * TRIM);
  for (let i = 0; i < out.length; i++) out[i] = audio[i + TRIM] / envelope[i + TRIM];
  return out;
}
