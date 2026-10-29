// Runs the feasibility spike in Chromium on the local GPU, with a video file as the fake webcam.
// Headed on purpose: on this Linux/NVIDIA setup headless Chromium's GPU process segfaults when WebGPU creates its
// context and falls back to SwiftShader, while headed Chromium with Vulkan gets the real adapter.
// Usage: node scripts/run-spike.mjs <fake-cam.y4m> <out-dir> [label] [query]
//   query is the page's URL params (runs, w, h, frames, dump, start, countdown). The runner defaults to
//   start=auto&countdown=0 so a run needs no click; pass e.g. 'start=click&countdown=3' to play a round by hand.
//   CHROMIUM overrides the browser binary (default /usr/bin/chromium, or Google Chrome on macOS).
import { existsSync, mkdirSync, writeFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { chromium } from 'playwright';
import { createServer } from 'vite';

const [camFile, outDir, label = 'run', query = ''] = process.argv.slice(2);
if (!camFile || !outDir) throw new Error('usage: run-spike.mjs <fake-cam.y4m> <out-dir> [label] [query]');
mkdirSync(outDir, { recursive: true });

const q = new URLSearchParams(query);
for (const [k, v] of [
  ['start', 'auto'],
  ['countdown', '0'],
])
  if (!q.has(k)) q.set(k, v);

const MAC_CHROME = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
const executablePath =
  process.env.CHROMIUM ??
  (!existsSync('/usr/bin/chromium') && process.platform === 'darwin' ? MAC_CHROME : '/usr/bin/chromium');

const server = await createServer({ server: { port: 5179, strictPort: true }, logLevel: 'warn' });
await server.listen();

const context = await chromium.launchPersistentContext(resolve('.spike-profile'), {
  executablePath,
  headless: false,
  args: [
    '--enable-unsafe-webgpu',
    '--enable-features=Vulkan',
    '--window-size=900,800',
    '--use-fake-ui-for-media-stream',
    // The spoken answer autoplays; with start=auto there is no click to grant user activation.
    '--autoplay-policy=no-user-gesture-required',
    '--use-fake-device-for-media-stream',
    `--use-file-for-fake-video-capture=${resolve(camFile)}`,
  ],
});

try {
  const page = await context.newPage();
  page.on('console', (m) => m.text().startsWith('[spike]') && console.log(m.text()));
  page.on('pageerror', (e) => console.error('[pageerror]', e.message));
  const started = Date.now();
  await page.goto(`http://localhost:5179/spike/index.html?${q}`);
  await page.waitForFunction(() => window.__spikeResult || window.__spikeError, null, {
    timeout: 45 * 60_000,
    polling: 1000,
  });
  const { result, error, wav, dump } = await page.evaluate(() => ({
    result: window.__spikeResult,
    error: window.__spikeError,
    wav: window.__spikeWav,
    dump: window.__spikeDump,
  }));
  if (error) throw new Error(error);
  const report = { label, wallMs: Date.now() - started, ...result };
  writeFileSync(`${outDir}/spike-${label}.json`, JSON.stringify(report, null, 2));
  if (wav) writeFileSync(`${outDir}/spike-${label}.wav`, Buffer.from(wav, 'base64'));
  if (dump) writeFileSync(`${outDir}/spike-${label}-dump.json`, JSON.stringify(dump));
  console.log(`[runner] wrote ${outDir}/spike-${label}.json`);
} finally {
  await context.close();
  await server.close();
}
