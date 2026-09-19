// Draws MediaPipe 2D pose landmarks over a video shown with object-fit: contain.

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

// Face landmarks (0-10) are drawn neutral; body landmarks by side.
const colorOf = (i) => (i <= 10 ? COLOR_CENTER : LEFT.has(i) ? COLOR_LEFT : COLOR_RIGHT);

/** Match the canvas backing store to its displayed size. */
export function fitCanvas(canvas) {
  const rect = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  const w = Math.round(rect.width * dpr);
  const h = Math.round(rect.height * dpr);
  if (canvas.width !== w || canvas.height !== h) {
    canvas.width = w;
    canvas.height = h;
  }
}

export function clearCanvas(canvas) {
  canvas.getContext('2d').clearRect(0, 0, canvas.width, canvas.height);
}

/**
 * landmarks: 33 [x, y] in normalized image coords (null entries allowed); visibility: 33 numbers;
 * videoSize: [width, height] of the source frames. `color` draws everything in one color instead
 * of by side, and `lineWidth` scales the strokes and dots.
 */
export function drawSkeleton(canvas, landmarks, visibility, videoSize, { color = null, lineWidth = 3 } = {}) {
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (!landmarks) return;

  const dpr = window.devicePixelRatio || 1;
  const cw = canvas.width / dpr;
  const ch = canvas.height / dpr;
  const [vw, vh] = videoSize;
  const s = Math.min(cw / vw, ch / vh);
  const ox = (cw - vw * s) / 2;
  const oy = (ch - vh * s) / 2;
  const ok = (i) => landmarks[i][0] !== null && visibility[i] >= MIN_VISIBILITY;
  const px = (i) => [ox + landmarks[i][0] * vw * s, oy + landmarks[i][1] * vh * s];

  ctx.save();
  ctx.scale(dpr, dpr);
  ctx.lineWidth = lineWidth;
  ctx.lineCap = 'round';
  for (const [a, b] of CONNECTIONS) {
    if (!ok(a) || !ok(b)) continue;
    ctx.strokeStyle = color ?? (colorOf(a) === colorOf(b) ? colorOf(a) : COLOR_CENTER);
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
    ctx.fillStyle = color ?? colorOf(i);
    ctx.beginPath();
    ctx.arc(x, y, lineWidth + 0.5, 0, 2 * Math.PI);
    ctx.fill();
  }
  ctx.restore();
}
