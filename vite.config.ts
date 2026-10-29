import { resolve } from 'node:path';
import { defineConfig } from 'vite';

export default defineConfig({
  optimizeDeps: { exclude: ['onnxruntime-web'] },
  build: {
    target: 'es2022',
    rollupOptions: { input: { spike: resolve(import.meta.dirname, 'spike/index.html') } },
  },
});
