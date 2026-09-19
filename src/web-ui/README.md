# web-ui

Local web page for the [pose-estimation](../pose-estimation) pipeline: upload a video of one person,
watch progress, then play back the result. The page shows the video with MediaPipe's 2D keypoints
drawn over it, next to an interactive 3D view of the fitted MuJoCo humanoid, on one shared timeline.

## Run

```bash
cd src/web-ui
uv sync
uv run server.py          # open http://127.0.0.1:8000  (--port to change)
```

`pose-estimation` is installed as an editable dependency, so changes there are picked up on restart.
The page loads three.js and MediaPipe's JS build from the jsDelivr CDN, so the browser needs internet
access. The first run of each pose model also downloads it (~5–30 MB).

## Using it

- Choose or drop a video (.mp4, .mov, .m4v, .webm, .avi, .mkv), pick a pose model and smoothing, and click **Run pipeline**.
  Jobs run one at a time; a 10 s clip takes several seconds.
- **Playback:** play/pause, scrub, speed; **Space** toggles play, **←/→** step frames (when focus isn't on a control).
  Drag to orbit the 3D view, scroll to zoom; the camera follows the pelvis.
- **Toggles:** keypoint overlay on the video, and the green IK target spheres in 3D.
- **Downloads:** `keypoints.npz` and `motion.npz`, the same files the command-line scripts produce
  (open `motion.npz` with `../pose-estimation/visualize.py`).
- **Previous runs** are listed below and survive server restarts. Each job lives in `jobs/<id>/` (gitignored);
  delete a folder to remove it.

## Live webcam

The **Live webcam** tab runs the pipeline on your camera in real time:

- MediaPipe runs **in the browser** (WebGL, falling back to CPU). Only the 33 landmarks are sent to
  the server over a WebSocket (`/api/live`); the server solves IK and returns the humanoid's pose.
  Frames are sent one at a time, each after the previous reply, so latency stays bounded and frames
  are skipped when IK can't keep up.
- Live IK can only use past frames: it uses a One-Euro filter (the **Smoothing** setting trades lag for
  steadiness) and holds the last pose through brief detection dropouts, resetting after 0.5 s.
- **Mirror** flips both the webcam and the humanoid like a selfie camera. The status line shows camera
  fps, MediaPipe time, server round trip and IK error.
- **Record** saves the webcam video; **Stop & process** uploads it as a normal job, so it gets the smoother
  batch pipeline, playback and downloads. The frame rate comes from the recording's length, since
  browser WebM files don't store a reliable one.

Browsers only allow camera access on secure pages: `http://localhost` / `http://127.0.0.1` work, but
opening the page from another device on your network needs HTTPS.

If the browser can't decode the uploaded format (e.g. some .avi/.mkv files), the pipeline still runs and
the page shows the keypoints on a blank background, with the 3D view on its own clock.

## API

| Method | Path | |
|---|---|---|
| POST | `/api/jobs` | multipart: `file`, `model` (lite/full/heavy), `smooth` (seconds), optional `duration` (seconds; sets fps = frames / duration) → `{id}` |
| GET | `/api/jobs` | all jobs, newest first |
| GET | `/api/jobs/{id}` | status, progress, message, summary |
| GET | `/api/jobs/{id}/result` | playback data: per-frame geom poses, IK targets, 2D landmarks, errors |
| GET | `/api/jobs/{id}/video` | the uploaded video (supports range requests) |
| GET | `/api/jobs/{id}/files/{keypoints.npz,motion.npz}` | downloads |
| GET | `/api/model` | the humanoid's geoms, for the live view |
| WS | `/api/live` | streaming IK; message format in `server.py` (`live()` docstring) |
