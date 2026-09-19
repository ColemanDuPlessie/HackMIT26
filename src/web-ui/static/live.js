import { FilesetResolver, PoseLandmarker } from '@mediapipe/tasks-vision';
import { HumanoidView } from './humanoid-view.js';
import { clearCanvas, drawSkeleton, fitCanvas } from './skeleton.js';

const $ = (id) => document.getElementById(id);

// Keep in sync with the "@mediapipe/tasks-vision" entry in index.html's import map.
const WASM_URL = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.0.1/wasm';
const MODEL_URL = (variant) =>
  `https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_${variant}/float16/latest/pose_landmarker_${variant}.task`;
const REPLY_TIMEOUT_MS = 1000; // give up on a lost WebSocket reply after this long
const RECORDER_TYPES = ['video/webm;codecs=vp9', 'video/webm;codecs=vp8', 'video/webm', 'video/mp4'];

// Landmarkers are expensive to create, so keep one per model variant for the page's lifetime.
// VIDEO mode requires strictly increasing timestamps per landmarker, tracked in lastTimestamp.
let filesetPromise = null;
const landmarkers = {};
let lastTimestamp = 0;
let geomsPromise = null;

function getLandmarker(variant) {
  filesetPromise ??= FilesetResolver.forVisionTasks(WASM_URL);
  landmarkers[variant] ??= filesetPromise.then((fileset) => createLandmarker(fileset, variant));
  landmarkers[variant].catch(() => { delete landmarkers[variant]; });
  return landmarkers[variant];
}

async function createLandmarker(fileset, variant) {
  const options = (delegate) => ({
    baseOptions: { modelAssetPath: MODEL_URL(variant), delegate },
    runningMode: 'VIDEO',
    numPoses: 1,
  });
  try {
    return await PoseLandmarker.createFromOptions(fileset, options('GPU'));
  } catch (err) {
    console.warn('MediaPipe GPU delegate failed; falling back to CPU.', err);
    return PoseLandmarker.createFromOptions(fileset, options('CPU'));
  }
}

function getGeoms() {
  geomsPromise ??= fetch('/api/model').then((r) => r.json()).then((m) => m.geoms);
  geomsPromise.catch(() => { geomsPromise = null; });
  return geomsPromise;
}

/** Running average for timing stats: a plain mean over the first 10 samples, then an EMA. */
class Avg {
  constructor() { this.value = null; this.n = 0; }
  add(x) {
    this.n++;
    this.value = this.value == null ? x : this.value + Math.max(0.1, 1 / this.n) * (x - this.value);
  }
}
const newStats = () => ({ fps: new Avg(), detect: new Avg(), rtt: new Avg(), ik: new Avg(), error: new Avg() });

/**
 * Webcam -> MediaPipe (in the browser) -> /api/live WebSocket (IK on the server) -> 3D view.
 * Frames are sent stop-and-wait: a new one only after the previous reply, so latency stays bounded
 * and the server skips frames when it can't keep up.
 */
export class LiveSession {
  constructor({ onRecording }) {
    this.onRecording = onRecording;
    this.running = false;
    this.session = 0;
    this.video = $('live-video');
    this.overlay = $('live-overlay');

    $('live-start').addEventListener('click', () => (this.running ? this.stop() : this.start()));
    $('live-model').addEventListener('change', () => { if (this.running) this.restart(); });
    $('live-smoothing').addEventListener('change', () => this.sendConfig());
    $('live-mirror').addEventListener('change', () => this.applyMirror());
    $('live-record').addEventListener('click', () => (this.recorder ? this.finishRecording() : this.startRecording()));
    this.applyMirror();
    this.setStatus('Camera off. Press Start camera; your browser will ask for permission.');
  }

  async start() {
    const session = ++this.session;
    const stale = () => session !== this.session;
    this.setButtons({ starting: true });
    this.stats = newStats();
    this.warmedUp = false;
    this.tracking = false;

    try {
      this.setStatus('Starting camera…');
      this.stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: 'user' },
        audio: false,
      });
      if (stale()) return this.releaseCamera();
      this.video.srcObject = this.stream;
      await this.video.play();

      this.setStatus('Loading pose model (first time downloads it)…');
      const [landmarker, geoms] = await Promise.all([getLandmarker($('live-model').value), getGeoms()]);
      if (stale()) return this.releaseCamera();
      this.landmarker = landmarker;
      this.view = new HumanoidView($('live-three'), geoms);
      this.view.setOpacity(0.3);
      this.applyMirror();
    } catch (err) {
      if (stale()) return;
      this.releaseCamera();
      this.setButtons({ running: false });
      this.setStatus(cameraErrorMessage(err), true);
      return;
    }

    this.running = true;
    this.setButtons({ running: true });
    this.setStatus('Looking for a person…');
    this.connect(session);
    this.scheduleFrame(session);
  }

  stop() {
    this.session++;
    this.running = false;
    if (this.recorder) this.discardRecording();
    this.ws?.close();
    this.ws = null;
    this.inFlight = false;
    this.releaseCamera();
    this.view?.dispose();
    this.view = null;
    clearCanvas(this.overlay);
    this.setButtons({ running: false });
    this.setStatus('Camera off.');
    $('live-stats').textContent = '';
  }

  async restart() {
    this.stop();
    await this.start();
  }

  releaseCamera() {
    this.stream?.getTracks().forEach((t) => t.stop());
    this.stream = null;
    this.video.srcObject = null;
  }

  // ------------------------------------------------------------ server connection

  connect(session) {
    const ws = new WebSocket(`${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/api/live`);
    this.ws = ws;
    this.inFlight = false;
    ws.onopen = () => this.sendConfig();
    ws.onmessage = (ev) => { if (this.ws === ws) this.onReply(JSON.parse(ev.data)); };
    ws.onclose = () => {
      if (session !== this.session || !this.running) return;
      this.setStatus('Lost connection to the server; retrying…', true);
      setTimeout(() => { if (session === this.session && this.running) this.connect(session); }, 1000);
    };
  }

  sendConfig() {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({ type: 'config', smoothing: $('live-smoothing').value }));
    }
  }

  onReply(msg) {
    this.inFlight = false;
    this.stats.rtt.add(performance.now() - this.sentAt);
    if (msg.type === 'error') {
      console.warn('Live IK error:', msg.message);
      return;
    }
    if (!this.view) return;
    if (msg.lost) {
      this.view.setOpacity(0.3);
      this.tracking = false;
      return;
    }
    this.tracking = true;
    this.view.setOpacity(1);
    this.view.setPose(msg.poses, msg.root);
    this.view.setTargets($('live-targets').checked ? msg.targets : null);
    this.stats.ik.add(msg.ik_ms);
    this.stats.error.add(msg.error_cm);
  }

  // ------------------------------------------------------------ per-frame loop

  scheduleFrame(session) {
    const next = () => { if (session === this.session && this.running) this.onFrame(session); };
    if (this.video.requestVideoFrameCallback) this.video.requestVideoFrameCallback(next);
    else requestAnimationFrame(next);
  }

  onFrame(session) {
    const now = performance.now();
    if (this.video.readyState >= 2 && this.video.currentTime !== this.lastVideoTime) {
      this.lastVideoTime = this.video.currentTime;
      // Gaps over 1 s (e.g. the first MediaPipe call warming up the GPU) aren't the camera's rate.
      if (this.lastFrameAt && now - this.lastFrameAt < 1000) this.stats.fps.add(1000 / (now - this.lastFrameAt));
      this.lastFrameAt = now;
      this.processFrame(now);
    }
    this.updateStatus(now);
    this.scheduleFrame(session);
  }

  processFrame(now) {
    const timestamp = Math.max(Math.round(now), lastTimestamp + 1);
    lastTimestamp = timestamp;
    const t0 = performance.now();
    const result = this.landmarker.detectForVideo(this.video, timestamp);
    if (this.warmedUp) this.stats.detect.add(performance.now() - t0);
    this.warmedUp = true; // the first call includes one-time GPU setup

    const image = result.landmarks[0];
    const world = result.worldLandmarks[0];
    const visibility = world ? world.map((p) => p.visibility ?? 1) : null;
    fitCanvas(this.overlay);
    drawSkeleton(this.overlay, image?.map((p) => [p.x, p.y]), image?.map((p) => p.visibility ?? 1),
      [this.video.videoWidth, this.video.videoHeight]);

    if (this.inFlight && now - this.sentAt > REPLY_TIMEOUT_MS) this.inFlight = false;
    if (this.ws?.readyState === WebSocket.OPEN && !this.inFlight) {
      this.ws.send(JSON.stringify({
        type: 'frame',
        t: timestamp / 1000,
        world: world ? world.map((p) => [p.x, p.y, p.z]) : null,
        visibility,
      }));
      this.inFlight = true;
      this.sentAt = performance.now();
    }
  }

  // ------------------------------------------------------------ recording

  startRecording() {
    const mimeType = RECORDER_TYPES.find((t) => MediaRecorder.isTypeSupported(t));
    this.chunks = [];
    this.recorder = new MediaRecorder(this.stream, mimeType ? { mimeType } : undefined);
    this.recorder.ondataavailable = (e) => { if (e.data.size) this.chunks.push(e.data); };
    this.recorder.start(1000);
    this.recordStart = performance.now();
    this.setButtons({ running: true });
  }

  finishRecording() {
    const recorder = this.recorder;
    const duration = (performance.now() - this.recordStart) / 1000;
    const model = $('live-model').value;
    recorder.onstop = () => {
      const type = recorder.mimeType || 'video/webm';
      const ext = type.includes('mp4') ? 'mp4' : 'webm';
      const stamp = new Date().toISOString().slice(0, 19).replace(/[:T]/g, '-');
      const file = new File(this.chunks, `webcam-${stamp}.${ext}`, { type });
      this.chunks = [];
      this.onRecording(file, { model, duration });
    };
    this.recorder = null;
    recorder.stop();
    this.stop();
  }

  discardRecording() {
    this.recorder.onstop = null;
    this.recorder.stop();
    this.recorder = null;
    this.chunks = [];
  }

  // ------------------------------------------------------------ UI

  applyMirror() {
    const mirrored = $('live-mirror').checked;
    $('live-stage').classList.toggle('mirrored', mirrored);
    this.view?.setMirrored(mirrored);
  }

  setButtons({ running = false, starting = false }) {
    const start = $('live-start');
    start.disabled = starting;
    start.textContent = starting ? 'Starting…' : running ? 'Stop camera' : 'Start camera';
    const record = $('live-record');
    record.disabled = !running;
    record.textContent = this.recorder ? '■ Stop & process' : '● Record';
    record.classList.toggle('recording', Boolean(this.recorder));
  }

  setStatus(text, isError = false) {
    const el = $('live-status');
    el.textContent = text;
    el.classList.toggle('error', isError);
  }

  updateStatus(now) {
    if (now - (this.lastStatusAt ?? 0) < 250) return;
    this.lastStatusAt = now;
    const { fps, detect, rtt, ik, error } = this.stats;
    const parts = [];
    if (fps.value != null) parts.push(`camera ${fps.value.toFixed(0)} fps`);
    if (detect.value != null) parts.push(`MediaPipe ${detect.value.toFixed(0)} ms`);
    if (rtt.value != null) parts.push(`server round trip ${rtt.value.toFixed(0)} ms (IK ${(ik.value ?? 0).toFixed(1)} ms)`);
    if (error.value != null && this.tracking) parts.push(`IK error ${error.value.toFixed(1)} cm`);
    $('live-stats').textContent = parts.join(' · ');
    if (this.ws?.readyState !== WebSocket.OPEN) return;
    if (this.recorder) {
      const secs = Math.floor((now - this.recordStart) / 1000);
      this.setStatus(`Recording ${Math.floor(secs / 60)}:${String(secs % 60).padStart(2, '0')}. Press Stop & process to run the full pipeline on it.`);
    } else {
      this.setStatus(this.tracking ? 'Tracking.' : 'Looking for a person…');
    }
  }
}

function cameraErrorMessage(err) {
  if (err?.name === 'NotAllowedError') return 'Camera permission was denied. Allow camera access for this page and try again.';
  if (err?.name === 'NotFoundError') return 'No camera was found.';
  if (err?.name === 'NotReadableError') return 'The camera is in use by another application.';
  if (!navigator.mediaDevices) return 'Camera access needs a secure page (http://localhost, http://127.0.0.1 or https).';
  return `Could not start: ${err?.message || err}`;
}
