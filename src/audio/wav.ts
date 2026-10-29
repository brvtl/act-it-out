/**
 * Encodes mono float samples as a 16-bit PCM RIFF/WAVE file (44-byte header).
 * Samples are clamped to [-1, 1] and scaled by 0x7fff; DataView.setInt16 truncates toward zero.
 */
export function encodeWav(samples: Float32Array, sampleRate: number): Uint8Array {
  const buf = new DataView(new ArrayBuffer(44 + samples.length * 2));
  const str = (o: number, s: string) => [...s].forEach((c, i) => buf.setUint8(o + i, c.charCodeAt(0)));
  str(0, 'RIFF');
  buf.setUint32(4, 36 + samples.length * 2, true);
  str(8, 'WAVEfmt ');
  buf.setUint32(16, 16, true);
  buf.setUint16(20, 1, true);
  buf.setUint16(22, 1, true);
  buf.setUint32(24, sampleRate, true);
  buf.setUint32(28, sampleRate * 2, true);
  buf.setUint16(32, 2, true);
  buf.setUint16(34, 16, true);
  str(36, 'data');
  buf.setUint32(40, samples.length * 2, true);
  samples.forEach((s, i) => buf.setInt16(44 + i * 2, Math.max(-1, Math.min(1, s)) * 0x7fff, true));
  return new Uint8Array(buf.buffer);
}
