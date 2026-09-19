import { HumanoidView } from './humanoid-view.js';
import { clearCanvas, drawSkeleton, fitCanvas } from './skeleton.js';
import { LiveSession } from './live.js';

const $ = (id) => document.getElementById(id);
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// ---------------------------------------------------------------- tabs

let live = null;

function showTab(name) {
  const isLive = name === 'live';
  $('tab-upload').setAttribute('aria-selected', String(!isLive));
  $('tab-live').setAttribute('aria-selected', String(isLive));
  $('upload-view').hidden = isLive;
  $('live-view').hidden = !isLive;
  if (isLive) {
    trackToken++; // stop polling any job
    disposePlayer();
    history.replaceState(null, '', '#live');
    live ??= new LiveSession({ onRecording: uploadRecording });
  } else {
    live?.stop();
    if (location.hash === '#live') history.replaceState(null, '', location.pathname);
  }
}

$('tab-upload').addEventListener('click', () => showTab('upload'));
$('tab-live').addEventListener('click', () => showTab('live'));

// ---------------------------------------------------------------- upload + jobs

let selectedFile = null;
let player = null;
let trackToken = 0;

function setFile(file) {
  selectedFile = file;
  $('drop-text').innerHTML = file
    ? `<strong>${escapeHtml(file.name)}</strong> (${(file.size / 1e6).toFixed(1)} MB) · click to change`
    : '<strong>Choose a video</strong> or drop it here';
  $('run-btn').disabled = !file;
}

$('file-input').addEventListener('change', (e) => setFile(e.target.files[0] || null));

const dropZone = $('drop-zone');
for (const type of ['dragenter', 'dragover']) {
  dropZone.addEventListener(type, (e) => { e.preventDefault(); dropZone.classList.add('dragging'); });
}
for (const type of ['dragleave', 'drop']) {
  dropZone.addEventListener(type, () => dropZone.classList.remove('dragging'));
}
dropZone.addEventListener('drop', (e) => {
  e.preventDefault();
  if (e.dataTransfer.files.length) setFile(e.dataTransfer.files[0]);
});

$('upload-form').addEventListener('submit', async (e) => {
  e.preventDefault();
  if (!selectedFile) return;
  $('run-btn').disabled = true;
  try {
    await uploadVideo(selectedFile, { model: $('model-select').value, smooth: $('smooth-select').value });
  } finally {
    $('run-btn').disabled = !selectedFile;
  }
});

/** Upload a video as a new job and start tracking it. Returns the job id, or null on failure. */
async function uploadVideo(file, { model, smooth, duration }) {
  const form = new FormData();
  form.append('file', file);
  form.append('model', model);
  form.append('smooth', smooth);
  if (duration) form.append('duration', duration.toFixed(3));
  hideError();
  showProgress(0, 'Uploading…');
  try {
    const res = await fetch('/api/jobs', { method: 'POST', body: form });
    const body = await res.json();
    if (!res.ok) throw new Error(body.detail || res.statusText);
    refreshHistory();
    track(body.id);
    return body.id;
  } catch (err) {
    hideProgress();
    showError(`Upload failed: ${err.message}`);
    return null;
  }
}

/** Called by the live session when a recording stops: process it like an upload. */
async function uploadRecording(file, { model, duration }) {
  showTab('upload');
  await uploadVideo(file, { model, smooth: '0.2', duration });
}

async function track(id) {
  const token = ++trackToken;
  history.replaceState(null, '', `#job=${id}`);
  hideError();
  disposePlayer();
  markActive(id);
  let lastStatus = null;
  while (token === trackToken) {
    const res = await fetch(`/api/jobs/${id}`);
    if (!res.ok) { hideProgress(); showError('Job not found.'); return; }
    const job = await res.json();
    if (token !== trackToken) return;
    if (job.status === 'done') {
      hideProgress();
      await loadResult(job);
      if (lastStatus !== null) refreshHistory();
      return;
    }
    if (job.status === 'error') {
      hideProgress();
      showError(`Processing failed: ${job.error}`);
      if (lastStatus !== null) refreshHistory();
      return;
    }
    showProgress(job.progress, job.message);
    lastStatus = job.status;
    await sleep(500);
  }
}

async function loadResult(job) {
  const res = await fetch(`/api/jobs/${job.id}/result`);
  if (!res.ok) { showError('Could not load the result.'); return; }
  const result = await res.json();
  $('results').hidden = false;
  $('result-title').textContent = job.filename;
  $('dl-keypoints').href = `/api/jobs/${job.id}/files/keypoints.npz`;
  $('dl-motion').href = `/api/jobs/${job.id}/files/motion.npz`;
  renderStats(result.summary);
  player = new Player(result, job.id);
}

function disposePlayer() {
  if (player) player.dispose();
  player = null;
  $('results').hidden = true;
}

function renderStats(s) {
  const items = [
    ['Frames', `${s.frames}`],
    ['Person detected', `${Math.round((100 * s.detected_frames) / s.frames)}%`],
    ['Frame rate', `${s.fps} fps`],
    ['Mean IK error', `${s.mean_error_cm} cm`],
    ['Max IK error', `${s.max_error_cm} cm`],
    ['Processing time', `${s.total_seconds} s`],
  ];
  $('stats').innerHTML = items
    .map(([label, value]) => `<div class="stat"><div class="label">${label}</div><div class="value">${value}</div></div>`)
    .join('');
}

async function refreshHistory() {
  const res = await fetch('/api/jobs');
  if (!res.ok) return;
  const jobs = await res.json();
  $('history-card').hidden = jobs.length === 0;
  $('history').innerHTML = '';
  for (const job of jobs) {
    const li = document.createElement('li');
    li.dataset.id = job.id;
    const when = new Date(job.created * 1000).toLocaleString();
    let status = '';
    if (job.status === 'error') status = '<span class="status-error">failed</span> · ';
    else if (job.status !== 'done') status = '<span class="status-running">processing</span> · ';
    const err = job.summary ? ` · ${job.summary.mean_error_cm} cm` : '';
    li.innerHTML = `<span>${escapeHtml(job.filename)}</span><span class="meta">${status}${job.model}${err} · ${when}</span>`;
    li.addEventListener('click', () => track(job.id));
    $('history').appendChild(li);
  }
  const current = new URLSearchParams(location.hash.slice(1)).get('job');
  if (current) markActive(current);
}

function markActive(id) {
  for (const li of $('history').children) li.classList.toggle('active', li.dataset.id === id);
}

function showProgress(fraction, text) {
  $('progress').hidden = false;
  $('progress-fill').style.width = `${Math.round(fraction * 100)}%`;
  $('progress-text').textContent = text;
}
function hideProgress() { $('progress').hidden = true; }
function showError(text) { $('error-text').textContent = text; $('error-text').hidden = false; }
function hideError() { $('error-text').hidden = true; }

function escapeHtml(s) {
  return s.replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c]);
}

// ---------------------------------------------------------------- playback

class Player {
  constructor(result, jobId) {
    this.r = result;
    this.fps = result.fps;
    this.n = result.n_frames;
    this.t = 0;
    this.frame = -1;
    this.playing = false;
    this.speed = Number($('speed-select').value);
    this.videoOk = false;
    this.dirty = true;
    this.abort = new AbortController();
    const signal = this.abort.signal;

    // Video: the master clock when the browser can play it.
    this.video = $('video');
    this.video.loop = true;
    this.video.playbackRate = this.speed;
    this.video.addEventListener('loadeddata', () => {
      this.videoOk = true;
      $('video-note').hidden = true;
      this.video.currentTime = this.t;
      if (this.playing) this.video.play().catch(() => {});
      this.dirty = true;
    }, { signal });
    this.video.addEventListener('error', () => {
      this.videoOk = false;
      $('video-note').hidden = false;
    }, { signal });
    $('video-note').hidden = true;
    this.video.src = `/api/jobs/${jobId}/video`;

    this.overlay = $('overlay');
    this.view = new HumanoidView($('three-stage'), result.geoms, result.root[0]);

    // Controls
    const scrubber = $('scrubber');
    scrubber.max = this.n - 1;
    scrubber.value = 0;
    scrubber.addEventListener('input', () => this.seek(Number(scrubber.value)), { signal });
    $('play-btn').addEventListener('click', () => (this.playing ? this.pause() : this.play()), { signal });
    $('speed-select').addEventListener('change', (e) => {
      this.speed = Number(e.target.value);
      this.video.playbackRate = this.speed;
    }, { signal });
    for (const id of ['show-keypoints', 'show-targets']) {
      $(id).addEventListener('change', () => { this.dirty = true; }, { signal });
    }
    document.addEventListener('keydown', (e) => {
      if (e.target.closest('input, select, textarea, button')) return;
      if (e.code === 'Space') { e.preventDefault(); this.playing ? this.pause() : this.play(); }
      else if (e.code === 'ArrowRight') { this.pause(); this.seek(Math.min(this.frame + 1, this.n - 1)); }
      else if (e.code === 'ArrowLeft') { this.pause(); this.seek(Math.max(this.frame - 1, 0)); }
    }, { signal });

    this.resizeObserver = new ResizeObserver(() => { fitCanvas(this.overlay); this.dirty = true; });
    this.resizeObserver.observe($('video-stage'));
    fitCanvas(this.overlay);
    this.pause();

    this.last = performance.now();
    this.raf = requestAnimationFrame((now) => this.tick(now));
  }

  play() {
    this.playing = true;
    $('play-btn').textContent = '⏸';
    $('play-btn').setAttribute('aria-label', 'Pause');
    if (this.videoOk) this.video.play().catch(() => {});
  }

  pause() {
    this.playing = false;
    $('play-btn').textContent = '▶';
    $('play-btn').setAttribute('aria-label', 'Play');
    this.video.pause();
  }

  seek(frame) {
    this.t = (frame + 0.5) / this.fps;
    if (this.videoOk) this.video.currentTime = this.t;
    this.dirty = true;
  }

  tick(now) {
    if (this.playing) {
      if (this.videoOk) {
        this.t = this.video.currentTime;
      } else {
        this.t += ((now - this.last) / 1000) * this.speed;
        if (this.t >= this.n / this.fps) this.t = 0;
      }
    }
    this.last = now;
    const frame = Math.min(this.n - 1, Math.max(0, Math.floor(this.t * this.fps + 1e-3)));
    if (frame !== this.frame || this.dirty) {
      this.frame = frame;
      this.dirty = false;
      this.drawFrame(frame);
    }
    this.raf = requestAnimationFrame((t) => this.tick(t));
  }

  drawFrame(f) {
    this.view.setPose(this.r.poses[f], this.r.root[f]);
    this.view.setTargets($('show-targets').checked ? this.r.targets[f] : null);
    $('scrubber').value = f;
    $('frame-label').textContent =
      `${f + 1}/${this.n} · ${(f / this.fps).toFixed(2)} s · ${this.r.error_cm[f].toFixed(1)} cm`;

    if (!$('show-keypoints').checked) { clearCanvas(this.overlay); return; }
    let size = this.r.video_size;
    if (this.videoOk && this.video.videoWidth) size = [this.video.videoWidth, this.video.videoHeight];
    drawSkeleton(this.overlay, this.r.landmarks_2d[f], this.r.visibility[f], size);
  }

  dispose() {
    cancelAnimationFrame(this.raf);
    this.abort.abort();
    this.resizeObserver.disconnect();
    this.video.pause();
    this.video.removeAttribute('src');
    this.video.load();
    clearCanvas(this.overlay);
    this.view.dispose();
  }
}

// ---------------------------------------------------------------- startup

refreshHistory();
if (location.hash === '#live') {
  showTab('live');
} else {
  const initial = new URLSearchParams(location.hash.slice(1)).get('job');
  if (initial) track(initial);
}
