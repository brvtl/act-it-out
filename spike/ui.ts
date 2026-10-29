// The spike page's DOM layer: the only module that knows element ids. main.ts drives it; it holds no game logic.
import { PHASE_COPY, secondsLeft, type Phase } from '../src/game/phase';

const $ = <T extends HTMLElement>(id: string): T => {
  const el = document.getElementById(id);
  if (!el) throw new Error(`#${id} missing from the page`);
  return el as T;
};

let speechUrl: string | undefined;
let speechShown = false;

const setStatus = (text: string) => ($('status').textContent = text);

export function setPhase(phase: Phase, detail?: string): void {
  const copy = PHASE_COPY[phase];
  document.body.dataset.phase = phase;
  setStatus(copy.status + (detail ? ` · ${detail}` : ''));
  $('overlay-title').textContent = copy.title;
  $('overlay-text').textContent = copy.hint;
  if (phase !== 'countdown' && phase !== 'recording') overlayBig('');

  const guess = $('guess');
  if (phase === 'thinking' && guess.classList.contains('pending')) guess.textContent = 'Thinking…';
  if (phase === 'result' && !speechShown) $('speech-text').textContent = 'Saying it out loud…';
  if (phase === 'done' || phase === 'error') $('again').hidden = false;
  if (phase === 'done' && !speechShown) {
    $('speech-text').textContent = '—';
    $('speech-hint').textContent = 'No audio generated.';
  }
}

export function camera(): HTMLVideoElement {
  return $<HTMLVideoElement>('cam');
}

/** Replaces the current phase's overlay hint, for copy that depends on the round's settings. */
export function overlayHint(text: string): void {
  $('overlay-text').textContent = text;
}

export function overlayBig(text: string): void {
  $('overlay-big').textContent = text;
}

export function log(msg: string): void {
  console.log(`[spike] ${msg}`);
  $('log').textContent += msg + '\n';
}

/** Adds or updates one tile in the timings card. */
export function stat(label: string, value: string): void {
  const stats = $('stats');
  let tile = [...stats.children].find((c): c is HTMLElement => c instanceof HTMLElement && c.dataset.key === label);
  if (!tile) {
    tile = document.createElement('div');
    tile.dataset.key = label;
    tile.append(document.createElement('dt'), document.createElement('dd'));
    stats.append(tile);
  }
  const [dt, dd] = tile.children;
  dt.textContent = label;
  dd.textContent = value;
}

/** Enables Start and resolves on its first click; the click also grants the page user activation for audio. */
export function waitForStart(): Promise<void> {
  const button = $<HTMLButtonElement>('start');
  button.disabled = false;
  button.focus();
  return new Promise((resolve) =>
    button.addEventListener(
      'click',
      () => {
        button.disabled = true;
        resolve();
      },
      { once: true },
    ),
  );
}

export async function countdown(seconds: number): Promise<void> {
  if (seconds <= 0) return;
  setPhase('countdown');
  for (let n = seconds; n > 0; n--) {
    overlayBig(String(n));
    await new Promise((r) => setTimeout(r, 1000));
  }
}

/**
 * Shows the recording state for `durationMs`: a CSS-transitioned progress bar plus a seconds-left readout
 * refreshed every 250 ms. The returned function stops the readout.
 */
export function recordingTimer(durationMs: number): () => void {
  setPhase('recording');
  const progress = $('progress');
  progress.style.setProperty('--rec-ms', `${Math.max(0, durationMs)}ms`);
  progress.style.width = '0%';
  void progress.offsetWidth; // commit the 0% start so the transition runs
  progress.style.width = '100%';
  const t0 = performance.now();
  const tick = () => {
    const elapsed = performance.now() - t0;
    const left = secondsLeft(durationMs, elapsed);
    overlayBig(String(left));
    setStatus(`${PHASE_COPY.recording.status} · ${left} s left`);
  };
  tick();
  const id = setInterval(tick, 250);
  return () => clearInterval(id);
}

export function showGuess(text: string, meta: string): void {
  const guess = $('guess');
  guess.textContent = text.trim() || '(no guess)';
  guess.classList.remove('pending');
  guess.style.fontSize = '';
  for (let scale = 0.9; guess.scrollHeight > guess.clientHeight && scale >= 0.4; scale -= 0.1)
    guess.style.fontSize = `calc(var(--guess-size) * ${scale.toFixed(1)})`;
  guess.classList.remove('pop');
  void guess.offsetWidth; // restart the pop animation
  guess.classList.add('pop');
  $('guess-meta').textContent = meta;
}

/** Shows the spoken answer and tries to autoplay it; a blocked autoplay reveals a big play button instead. */
export function showSpeech(text: string, wav: Uint8Array): void {
  speechShown = true;
  const card = $('speech');
  const audio = $<HTMLAudioElement>('speech-audio');
  $('speech-text').textContent = text.trim() || '—';
  if (wav.byteLength <= 44) {
    $('speech-hint').textContent = 'No audio generated.';
    return;
  }
  if (speechUrl) URL.revokeObjectURL(speechUrl);
  // encodeWav allocates a plain ArrayBuffer; the cast only narrows the generic Blob requires.
  speechUrl = URL.createObjectURL(new Blob([wav as Uint8Array<ArrayBuffer>], { type: 'audio/wav' }));
  audio.src = speechUrl;
  audio.hidden = false;
  $('speech-hint').textContent = '';
  audio.play().catch(() => {
    card.dataset.blocked = '';
    $('speech-hint').textContent = 'Your browser blocked autoplay. Press play.';
  });
}

export function fail(msg: string): void {
  setPhase('error');
  $('overlay-text').textContent = msg.split('\n')[0];
  $<HTMLDetailsElement>('log-details').open = true;
}

/** Wires the static controls once at startup. */
function init(): void {
  const audio = $<HTMLAudioElement>('speech-audio');
  const card = $('speech');
  $('speech-play').addEventListener('click', () => void audio.play().catch(() => undefined));
  audio.addEventListener('play', () => {
    delete card.dataset.blocked;
    $('speech-hint').textContent = '';
  });
}
init();
