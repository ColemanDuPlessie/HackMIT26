import { cameraErrorMessage, getLandmarker, nextTimestamp } from './live.js';
import { clearCanvas, drawSkeleton, fitCanvas } from './skeleton.js';

const $ = (id) => document.getElementById(id);
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const clamp = (x, lo, hi) => Math.min(hi, Math.max(lo, x));

// ---------------------------------------------------------------- scoring

// Bones compared between the user and the dancer: [from, to, weight]. Arms carry most dance moves;
// the torso barely changes between poses, so it counts for little.
const BONES = [
  [11, 13, 2], [13, 15, 2], [12, 14, 2], [14, 16, 2], // arms
  [23, 25, 1], [25, 27, 1], [24, 26, 1], [26, 28, 1], // legs
  [11, 12, 0.25], [23, 24, 0.25], [11, 23, 0.25], [12, 24, 0.25], // torso
];
const MIN_VISIBILITY = 0.5;
const MIN_WEIGHT = 3; // visible bone weight needed for a score (e.g. both arms, or legs + torso)
const Z_WEIGHT = 0.3; // MediaPipe's depth is noisier than x/y, so it counts for less
const FULL_ANGLE = 12; // degrees of bone misalignment that still score 100%...
const ZERO_ANGLE = 50; // ...and that score 0%
// Many bones line up even between unrelated poses (an arbitrary pose from a sample dance averages
// ~0.6), so the weighted bone score is stretched so that SCORE_FLOOR reads 0% and SCORE_CEIL 100%.
const SCORE_FLOOR = 0.4;
const SCORE_CEIL = 0.95;
const LAG_BEFORE = 0.4; // seconds the user may trail the dancer (reaction time)
const LAG_AFTER = 0.1; // seconds the user may lead
const SMOOTH_TAU = 0.25; // seconds; time constant of the meter's smoothing
const COUNTDOWN_S = 3;
const GHOST_COLOR = 'rgba(255, 255, 255, 0.75)';

// Landmark index of the same point on the other side of the body, for mirroring.
const SWAP = Array.from({ length: 33 }, (_, i) => i);
for (const [a, b] of [[1, 4], [2, 5], [3, 6], [7, 8], [9, 10]]) { SWAP[a] = b; SWAP[b] = a; }
for (let i = 11; i < 33; i += 2) { SWAP[i] = i + 1; SWAP[i + 1] = i; }

/** Reference frames as {world, image, vis}; mirrored swaps left/right so the user can copy a mirror image. */
function buildFrames(ref, mirrored) {
  return ref.world.map((world, f) => {
    const image = ref.image[f];
    const vis = ref.visibility[f];
    if (!mirrored) return { world, image, vis };
    return {
      world: world.map((_, i) => { const p = world[SWAP[i]]; return p[0] === null ? p : [-p[0], p[1], p[2]]; }),
      image: image.map((_, i) => { const p = image[SWAP[i]]; return p[0] === null ? p : [1 - p[0], p[1]]; }),
      vis: vis.map((_, i) => vis[SWAP[i]]),
    };
  });
}

function direction(world, a, b) {
  const dx = world[b][0] - world[a][0];
  const dy = world[b][1] - world[a][1];
  const dz = (world[b][2] - world[a][2]) * Z_WEIGHT;
  const n = Math.hypot(dx, dy, dz);
  return n > 1e-6 ? [dx / n, dy / n, dz / n] : null;
}

/** How well two poses ({world, vis}) match, 0..1, from bone directions; null if too little is visible. */
function poseScore(user, ref) {
  let total = 0;
  let weight = 0;
  for (const [a, b, w] of BONES) {
    if (Math.min(user.vis[a], user.vis[b], ref.vis[a], ref.vis[b]) < MIN_VISIBILITY) continue;
    if (ref.world[a][0] === null || ref.world[b][0] === null) continue;
    const u = direction(user.world, a, b);
    const r = direction(ref.world, a, b);
    if (!u || !r) continue;
    const angle = (Math.acos(clamp(u[0] * r[0] + u[1] * r[1] + u[2] * r[2], -1, 1)) * 180) / Math.PI;
    total += w * clamp((ZERO_ANGLE - angle) / (ZERO_ANGLE - FULL_ANGLE), 0, 1);
    weight += w;
  }
  if (weight < MIN_WEIGHT) return null;
  return clamp((total / weight - SCORE_FLOOR) / (SCORE_CEIL - SCORE_FLOOR), 0, 1);
}

/** Best score against the dancer around time t, allowing for reaction lag. Returns {score, frame}. */
function matchAt(user, frames, fps, t) {
  const center = clamp(Math.floor(t * fps), 0, frames.length - 1);
  const lo = Math.max(0, Math.floor((t - LAG_BEFORE) * fps));
  const hi = Math.min(frames.length - 1, Math.ceil((t + LAG_AFTER) * fps));
  let best = { score: null, frame: center };
  for (let f = lo; f <= hi; f++) {
    const s = poseScore(user, frames[f]);
    if (s !== null && (best.score === null || s > best.score)) best = { score: s, frame: f };
  }
  return best;
}

/**
 * The dancer's 2D keypoints moved and scaled onto the user's body (matching hip center and torso
 * length), in the webcam's normalized image coords. Null when either torso isn't visible.
 */
function fitGhost(refFrame, refSize, userImage, userVis, camSize) {
  const torso = [11, 12, 23, 24];
  if (torso.some((i) => refFrame.vis[i] < MIN_VISIBILITY || userVis[i] < MIN_VISIBILITY)) return null;
  const toPx = (p, [w, h]) => [p[0] * w, p[1] * h];
  const anchors = (pts, size) => {
    const [ls, rs, lh, rh] = torso.map((i) => toPx(pts[i], size));
    const hip = [(lh[0] + rh[0]) / 2, (lh[1] + rh[1]) / 2];
    const shoulder = [(ls[0] + rs[0]) / 2, (ls[1] + rs[1]) / 2];
    return { hip, length: Math.hypot(shoulder[0] - hip[0], shoulder[1] - hip[1]) };
  };
  const r = anchors(refFrame.image, refSize);
  const u = anchors(userImage, camSize);
  if (r.length < 1) return null;
  const s = u.length / r.length;
  return refFrame.image.map((p) => {
    if (p[0] === null) return p;
    const [x, y] = toPx(p, refSize);
    return [(u.hip[0] + (x - r.hip[0]) * s) / camSize[0], (u.hip[1] + (y - r.hip[1]) * s) / camSize[1]];
  });
}

// ---------------------------------------------------------------- state

const refVideo = $('ref-video');
const camVideo = $('cam-video');
let ref = null; // {jobId, fps, size, frames: {plain, mirrored}}
let loadToken = 0;

const cam = { on: false, session: 0, stream: null, landmarker: null, lastVideoTime: -1 };
let smoothed = 0;
let lastScoreAt = null;
let hint = 'match';
let run = { sum: 0, count: 0 }; // raw scores while the video plays, for the final score
let finalScore = null;
let countdownToken = 0;
let counting = false;

// ---------------------------------------------------------------- reference video

async function refreshJobs(selectId) {
  const res = await fetch('/api/jobs');
  const jobs = res.ok ? (await res.json()).filter((j) => j.status === 'done') : [];
  const select = $('job-select');
  select.innerHTML = '';
  const placeholder = new Option(jobs.length ? 'Choose a video…' : 'No processed videos yet: upload one', '');
  select.add(placeholder);
  for (const job of jobs) {
    const when = new Date(job.created * 1000).toLocaleString();
    select.add(new Option(`${job.filename} · ${when}`, job.id));
  }
  if (selectId && jobs.some((j) => j.id === selectId)) select.value = selectId;
  return jobs;
}

$('job-select').addEventListener('change', (e) => { if (e.target.value) loadReference(e.target.value); });

$('demo-file').addEventListener('change', async (e) => {
  const file = e.target.files[0];
  e.target.value = '';
  if (file) await uploadAndProcess(file);
});

async function uploadAndProcess(file) {
  const token = ++loadToken;
  const form = new FormData();
  form.append('file', file);
  form.append('model', $('demo-model').value);
  form.append('smooth', '0.2');
  showProgress(0, `Uploading ${file.name}…`);
  setStatus('');
  try {
    const res = await fetch('/api/jobs', { method: 'POST', body: form });
    const body = await res.json();
    if (!res.ok) throw new Error(body.detail || res.statusText);
    while (token === loadToken) {
      const job = await (await fetch(`/api/jobs/${body.id}`)).json();
      if (job.status === 'done') break;
      if (job.status === 'error') throw new Error(job.error);
      showProgress(job.progress, job.message);
      await sleep(500);
    }
    if (token !== loadToken) return;
    hideProgress();
    await refreshJobs(body.id);
    await loadReference(body.id);
  } catch (err) {
    if (token !== loadToken) return;
    hideProgress();
    setStatus(`Processing failed: ${err.message}`, true);
  }
}

async function loadReference(jobId) {
  const token = ++loadToken;
  hideProgress();
  stopPlayback();
  setStatus('Loading the dancer’s keypoints…');
  const [res, jobRes] = await Promise.all([fetch(`/api/jobs/${jobId}/reference`), fetch(`/api/jobs/${jobId}`)]);
  if (token !== loadToken) return;
  if (!res.ok) { setStatus('Could not load that video’s keypoints.', true); return; }
  const data = await res.json();
  const job = jobRes.ok ? await jobRes.json() : { filename: 'Dancer' };
  if (token !== loadToken) return;

  ref = {
    jobId,
    fps: data.fps,
    size: data.video_size,
    frames: { plain: buildFrames(data, false), mirrored: buildFrames(data, true) },
  };
  $('job-select').value = jobId;
  $('ref-caption').textContent = `Dancer · ${job.filename}`;
  history.replaceState(null, '', `?job=${jobId}`);
  refVideo.src = `/api/jobs/${jobId}/video`;
  refVideo.playbackRate = Number($('demo-speed').value);
  resetRun();
  $('play-btn').disabled = false;
  $('restart-btn').disabled = false;
  setStatus(cam.on ? 'Ready. Press Play and copy the dancer.' : 'Ready. Start the camera, then press Play.');
}

refVideo.addEventListener('error', () => {
  if (ref) setStatus('Your browser can’t play this video format; try converting it to .mp4.', true);
});
refVideo.addEventListener('ended', () => {
  updatePlayButton();
  if (run.count > 0) {
    finalScore = run.sum / run.count;
    setStatus(`Finished! Your average match was ${Math.round(finalScore * 100)}%. Press Play to go again.`);
  }
});
refVideo.addEventListener('play', updatePlayButton);
refVideo.addEventListener('pause', updatePlayButton);

// ---------------------------------------------------------------- playback

$('play-btn').addEventListener('click', togglePlay);
$('restart-btn').addEventListener('click', () => startFromTop());
$('demo-speed').addEventListener('change', (e) => { refVideo.playbackRate = Number(e.target.value); });
document.addEventListener('keydown', (e) => {
  if (e.code !== 'Space' || e.target.closest('input, select, textarea, button') || !ref) return;
  e.preventDefault();
  togglePlay();
});

function togglePlay() {
  if (!ref) return;
  if (counting) { stopPlayback(); return; }
  if (!refVideo.paused) { refVideo.pause(); return; }
  if (refVideo.ended || refVideo.currentTime === 0) startFromTop();
  else refVideo.play().catch(() => {});
}

async function startFromTop() {
  const token = ++countdownToken;
  refVideo.pause();
  refVideo.currentTime = 0;
  resetRun();
  counting = true;
  updatePlayButton();
  if (!cam.on) setStatus('Tip: start the camera to get scored.');
  const el = $('countdown');
  el.hidden = false;
  for (let n = COUNTDOWN_S; n > 0; n--) {
    el.textContent = n;
    await sleep(1000);
    if (token !== countdownToken) return;
  }
  el.hidden = true;
  counting = false;
  resetRun();
  refVideo.play().catch(() => {});
  updatePlayButton();
}

function stopPlayback() {
  countdownToken++;
  counting = false;
  $('countdown').hidden = true;
  refVideo.pause();
  updatePlayButton();
}

function resetRun() {
  run = { sum: 0, count: 0 };
  finalScore = null;
}

function updatePlayButton() {
  const playing = counting || !refVideo.paused;
  $('play-btn').textContent = playing ? '⏸ Pause' : '▶ Play';
}

// ---------------------------------------------------------------- camera

$('cam-btn').addEventListener('click', () => (cam.on ? stopCamera() : startCamera()));
$('demo-model').addEventListener('change', async (e) => {
  if (!cam.on) return;
  const session = cam.session;
  const landmarker = await getLandmarker(e.target.value).catch(() => null);
  if (landmarker && session === cam.session) cam.landmarker = landmarker;
});
$('show-ghost').addEventListener('change', () => clearCanvas($('ghost-overlay')));

async function startCamera() {
  const session = ++cam.session;
  const btn = $('cam-btn');
  btn.disabled = true;
  btn.textContent = 'Starting…';
  try {
    setStatus('Starting camera…');
    cam.stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: 'user' },
      audio: false,
    });
    if (session !== cam.session) return releaseCamera();
    camVideo.srcObject = cam.stream;
    await camVideo.play();
    setStatus('Loading pose model (first time downloads it)…');
    cam.landmarker = await getLandmarker($('demo-model').value);
    if (session !== cam.session) return releaseCamera();
  } catch (err) {
    if (session !== cam.session) return;
    releaseCamera();
    btn.disabled = false;
    btn.textContent = 'Start camera';
    setStatus(cameraErrorMessage(err), true);
    return;
  }
  cam.on = true;
  btn.disabled = false;
  btn.textContent = 'Stop camera';
  setStatus(ref ? 'Camera on. Press Play and copy the dancer.' : 'Camera on. Pick a video above.');
  scheduleFrame(session);
}

function stopCamera() {
  cam.session++;
  cam.on = false;
  releaseCamera();
  clearCanvas($('cam-overlay'));
  clearCanvas($('ghost-overlay'));
  $('cam-btn').textContent = 'Start camera';
  setStatus('Camera off.');
}

function releaseCamera() {
  cam.stream?.getTracks().forEach((t) => t.stop());
  cam.stream = null;
  camVideo.srcObject = null;
}

function scheduleFrame(session) {
  const next = () => { if (session === cam.session && cam.on) onCameraFrame(session); };
  if (camVideo.requestVideoFrameCallback) camVideo.requestVideoFrameCallback(next);
  else requestAnimationFrame(next);
}

function onCameraFrame(session) {
  if (camVideo.readyState >= 2 && camVideo.currentTime !== cam.lastVideoTime) {
    cam.lastVideoTime = camVideo.currentTime;
    processCameraFrame(performance.now());
  }
  scheduleFrame(session);
}

function processCameraFrame(now) {
  const result = cam.landmarker.detectForVideo(camVideo, nextTimestamp(now));
  const image = result.landmarks[0]?.map((p) => [p.x, p.y]);
  const world = result.worldLandmarks[0];
  const vis = world?.map((p) => p.visibility ?? 1);
  const camSize = [camVideo.videoWidth, camVideo.videoHeight];

  const overlay = $('cam-overlay');
  fitCanvas(overlay);
  drawSkeleton(overlay, image, vis, camSize);

  let raw = null;
  let ghost = null;
  if (!world) {
    hint = 'step into view';
  } else if (!ref) {
    hint = 'pick a video';
  } else {
    const frames = $('mirror-moves').checked ? ref.frames.mirrored : ref.frames.plain;
    const user = { world: world.map((p) => [p.x, p.y, p.z]), vis };
    const { score, frame } = matchAt(user, frames, ref.fps, refVideo.currentTime);
    raw = score;
    hint = score === null ? 'show more of your body' : 'match';
    if (score === null && Math.max(...frames[frame].vis) < MIN_VISIBILITY) hint = 'dancer not in view';
    const points = $('show-ghost').checked && fitGhost(frames[frame], ref.size, image, vis, camSize);
    if (points) ghost = { points, vis: frames[frame].vis };
  }

  const ghostCanvas = $('ghost-overlay');
  fitCanvas(ghostCanvas);
  if (ghost) drawSkeleton(ghostCanvas, ghost.points, ghost.vis, camSize, { color: GHOST_COLOR, lineWidth: 6 });
  else clearCanvas(ghostCanvas);

  // Smooth toward the raw score (0 when there's nothing to score) so the meter doesn't flicker.
  const dt = lastScoreAt === null ? 0 : (now - lastScoreAt) / 1000;
  lastScoreAt = now;
  smoothed += (1 - Math.exp(-dt / SMOOTH_TAU)) * ((raw ?? 0) - smoothed);

  if (ref && !counting && !refVideo.paused) {
    run.sum += raw ?? 0;
    run.count++;
  }
}

// ---------------------------------------------------------------- display loop

function render() {
  // Dancer keypoints over the reference video (unmirrored, as the video is shown).
  const refOverlay = $('ref-overlay');
  fitCanvas(refOverlay);
  if (ref) {
    const frames = ref.frames.plain;
    const f = clamp(Math.floor(refVideo.currentTime * ref.fps), 0, frames.length - 1);
    let size = ref.size;
    if (refVideo.videoWidth) size = [refVideo.videoWidth, refVideo.videoHeight];
    drawSkeleton(refOverlay, frames[f].image, frames[f].vis, size);
  } else {
    clearCanvas(refOverlay);
  }

  let value = null;
  let label = hint;
  if (finalScore !== null) { value = finalScore; label = 'final score'; }
  else if (cam.on) value = smoothed;
  else label = 'camera off';
  $('meter-fill').style.setProperty('--score', (value ?? 0).toFixed(3));
  $('score-value').textContent = value === null ? '–' : `${Math.round(value * 100)}%`;
  $('score-label').textContent = label;
  requestAnimationFrame(render);
}

// ---------------------------------------------------------------- UI helpers

function setStatus(text, isError = false) {
  const el = $('demo-status');
  el.textContent = text;
  el.classList.toggle('error', isError);
}

function showProgress(fraction, text) {
  $('demo-progress').hidden = false;
  $('demo-progress-fill').style.width = `${Math.round(fraction * 100)}%`;
  $('demo-progress-text').textContent = text;
}

function hideProgress() { $('demo-progress').hidden = true; }

// ---------------------------------------------------------------- startup

(async () => {
  const requested = new URLSearchParams(location.search).get('job');
  const jobs = await refreshJobs(requested);
  const initial = requested && jobs.some((j) => j.id === requested) ? requested : jobs[0]?.id;
  if (initial) await loadReference(initial);
  else setStatus('Upload a video of someone dancing to get started.');
})();
requestAnimationFrame(render);
