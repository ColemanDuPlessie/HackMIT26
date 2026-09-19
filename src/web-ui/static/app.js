import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const $ = (id) => document.getElementById(id);

// MediaPipe pose landmark connections, and which landmarks are on the subject's left.
const CONNECTIONS = [
  [0, 1], [1, 2], [2, 3], [3, 7], [0, 4], [4, 5], [5, 6], [6, 8], [9, 10],
  [11, 12], [11, 13], [13, 15], [15, 17], [15, 19], [15, 21], [17, 19],
  [12, 14], [14, 16], [16, 18], [16, 20], [16, 22], [18, 20],
  [11, 23], [12, 24], [23, 24], [23, 25], [24, 26], [25, 27], [26, 28],
  [27, 29], [28, 30], [29, 31], [30, 32], [27, 31], [28, 32],
];
const LEFT = new Set([1, 2, 3, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31]);
const COLOR_LEFT = '#ffa94d';
const COLOR_RIGHT = '#4dabf7';
const COLOR_CENTER = '#e6e9ee';
const MIN_VISIBILITY = 0.5;
// Camera offset from the pelvis: in front of the subject (+x) and slightly to their left (+y),
// so the 3D view roughly matches the video's viewpoint.
const CAMERA_OFFSET = new THREE.Vector3(3.0, 1.0, 0.6);

// Face landmarks (0-10) are drawn neutral; body landmarks by side.
const colorOf = (i) => (i <= 10 ? COLOR_CENTER : LEFT.has(i) ? COLOR_LEFT : COLOR_RIGHT);

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

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
  const form = new FormData();
  form.append('file', selectedFile);
  form.append('model', $('model-select').value);
  form.append('smooth', $('smooth-select').value);

  $('run-btn').disabled = true;
  hideError();
  showProgress(0, 'Uploading…');
  try {
    const res = await fetch('/api/jobs', { method: 'POST', body: form });
    const body = await res.json();
    if (!res.ok) throw new Error(body.detail || res.statusText);
    refreshHistory();
    track(body.id);
  } catch (err) {
    hideProgress();
    showError(`Upload failed: ${err.message}`);
  } finally {
    $('run-btn').disabled = !selectedFile;
  }
});

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
    this.setupThree();

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

    this.resizeObserver = new ResizeObserver(() => this.resize());
    this.resizeObserver.observe($('three-stage'));
    this.resizeObserver.observe($('video-stage'));
    this.resize();
    this.pause();

    this.last = performance.now();
    this.raf = requestAnimationFrame((now) => this.tick(now));
  }

  setupThree() {
    const stage = $('three-stage');
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.shadowMap.enabled = true;
    stage.appendChild(renderer.domElement);

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0b0e12);

    const camera = new THREE.PerspectiveCamera(45, 1, 0.05, 100);
    camera.up.set(0, 0, 1); // MuJoCo is z-up
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    const root = new THREE.Vector3(...this.r.root[0]);
    controls.target.copy(root);
    camera.position.copy(root).add(CAMERA_OFFSET);

    const hemi = new THREE.HemisphereLight(0xffffff, 0x303640, 1.4);
    hemi.position.set(0, 0, 1);
    scene.add(hemi);
    const sun = new THREE.DirectionalLight(0xffffff, 1.6);
    sun.position.set(2, 1.5, 5);
    sun.castShadow = true;
    sun.shadow.mapSize.set(1024, 1024);
    Object.assign(sun.shadow.camera, { left: -3, right: 3, top: 3, bottom: -3 });
    scene.add(sun);

    const ground = new THREE.Mesh(
      new THREE.PlaneGeometry(20, 20),
      new THREE.MeshStandardMaterial({ color: 0x1a2028, roughness: 0.95 }),
    );
    ground.receiveShadow = true;
    scene.add(ground);
    const grid = new THREE.GridHelper(20, 40, 0x3a4450, 0x262d36);
    grid.rotation.x = Math.PI / 2; // GridHelper lies in XZ; MuJoCo floor is XY
    grid.position.z = 0.001;
    scene.add(grid);

    this.meshes = this.r.geoms.map((g) => {
      let geometry;
      if (g.type === 'capsule') {
        geometry = new THREE.CapsuleGeometry(g.size[0], 2 * g.size[1], 6, 16).rotateX(Math.PI / 2); // Y -> Z axis
      } else if (g.type === 'box') {
        geometry = new THREE.BoxGeometry(2 * g.size[0], 2 * g.size[1], 2 * g.size[2]);
      } else {
        geometry = new THREE.SphereGeometry(g.size[0], 24, 16);
      }
      const material = new THREE.MeshStandardMaterial({
        color: new THREE.Color(g.rgba[0], g.rgba[1], g.rgba[2]),
        roughness: 0.55,
      });
      const mesh = new THREE.Mesh(geometry, material);
      mesh.castShadow = true;
      scene.add(mesh);
      return mesh;
    });

    this.targetMesh = new THREE.InstancedMesh(
      new THREE.SphereGeometry(0.022, 12, 8),
      new THREE.MeshBasicMaterial({ color: 0x3ecf8e }),
      33,
    );
    scene.add(this.targetMesh);
    this.dummy = new THREE.Object3D();

    Object.assign(this, { renderer, scene, camera, controls, ground, grid });
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
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
    this.raf = requestAnimationFrame((t) => this.tick(t));
  }

  drawFrame(f) {
    const pose = this.r.poses[f];
    this.meshes.forEach((mesh, i) => {
      const o = i * 7;
      mesh.position.set(pose[o], pose[o + 1], pose[o + 2]);
      mesh.quaternion.set(pose[o + 4], pose[o + 5], pose[o + 6], pose[o + 3]); // MuJoCo wxyz -> three xyzw
    });

    const showTargets = $('show-targets').checked;
    const targets = this.r.targets[f];
    for (let j = 0; j < 33; j++) {
      const p = targets[j];
      const visible = showTargets && p[0] !== null;
      this.dummy.position.set(visible ? p[0] : 0, visible ? p[1] : 0, visible ? p[2] : 0);
      this.dummy.scale.setScalar(visible ? 1 : 0);
      this.dummy.updateMatrix();
      this.targetMesh.setMatrixAt(j, this.dummy.matrix);
    }
    this.targetMesh.instanceMatrix.needsUpdate = true;

    // Follow the pelvis, keeping the user's chosen camera offset.
    const root = new THREE.Vector3(...this.r.root[f]);
    const delta = root.sub(this.controls.target);
    this.controls.target.add(delta);
    this.camera.position.add(delta);

    $('scrubber').value = f;
    $('frame-label').textContent =
      `${f + 1}/${this.n} · ${(f / this.fps).toFixed(2)} s · ${this.r.error_cm[f].toFixed(1)} cm`;
    this.drawOverlay(f);
  }

  drawOverlay(f) {
    const canvas = this.overlay;
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (!$('show-keypoints').checked) return;

    const dpr = window.devicePixelRatio || 1;
    const cw = canvas.width / dpr;
    const ch = canvas.height / dpr;
    let [vw, vh] = this.r.video_size;
    if (this.videoOk && this.video.videoWidth) [vw, vh] = [this.video.videoWidth, this.video.videoHeight];
    // Match the video's object-fit: contain placement.
    const s = Math.min(cw / vw, ch / vh);
    const ox = (cw - vw * s) / 2;
    const oy = (ch - vh * s) / 2;
    const lm = this.r.landmarks_2d[f];
    const vis = this.r.visibility[f];
    const ok = (i) => lm[i][0] !== null && vis[i] >= MIN_VISIBILITY;
    const px = (i) => [ox + lm[i][0] * vw * s, oy + lm[i][1] * vh * s];

    ctx.save();
    ctx.scale(dpr, dpr);
    ctx.lineWidth = 3;
    ctx.lineCap = 'round';
    for (const [a, b] of CONNECTIONS) {
      if (!ok(a) || !ok(b)) continue;
      ctx.strokeStyle = colorOf(a) === colorOf(b) ? colorOf(a) : COLOR_CENTER;
      const [x1, y1] = px(a);
      const [x2, y2] = px(b);
      ctx.beginPath();
      ctx.moveTo(x1, y1);
      ctx.lineTo(x2, y2);
      ctx.stroke();
    }
    for (let i = 0; i < 33; i++) {
      if (!ok(i)) continue;
      const [x, y] = px(i);
      ctx.fillStyle = colorOf(i);
      ctx.beginPath();
      ctx.arc(x, y, 3.5, 0, 2 * Math.PI);
      ctx.fill();
    }
    ctx.restore();
  }

  resize() {
    const stage = $('three-stage');
    const { width, height } = stage.getBoundingClientRect();
    if (width > 0 && height > 0) {
      this.renderer.setSize(width, height);
      this.camera.aspect = width / height;
      this.camera.updateProjectionMatrix();
    }
    const vs = $('video-stage').getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    this.overlay.width = Math.round(vs.width * dpr);
    this.overlay.height = Math.round(vs.height * dpr);
    this.dirty = true;
  }

  dispose() {
    cancelAnimationFrame(this.raf);
    this.abort.abort();
    this.resizeObserver.disconnect();
    this.video.pause();
    this.video.removeAttribute('src');
    this.video.load();
    this.overlay.getContext('2d').clearRect(0, 0, this.overlay.width, this.overlay.height);
    this.scene.traverse((obj) => {
      if (obj.geometry) obj.geometry.dispose();
      if (obj.material) obj.material.dispose();
    });
    this.controls.dispose();
    this.renderer.dispose();
    this.renderer.domElement.remove();
  }
}

// ---------------------------------------------------------------- startup

refreshHistory();
const initial = new URLSearchParams(location.hash.slice(1)).get('job');
if (initial) track(initial);
