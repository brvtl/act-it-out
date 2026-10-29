// Serves onnxruntime-web's WebGPU wasm from our own origin (public/ort) instead of a CDN.
import { copyFileSync, mkdirSync } from 'node:fs';

const file = 'ort-wasm-simd-threaded.asyncify.wasm';
mkdirSync('public/ort', { recursive: true });
copyFileSync(`node_modules/onnxruntime-web/dist/${file}`, `public/ort/${file}`);
