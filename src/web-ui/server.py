"""Local web UI for the pose-estimation pipeline.

Usage:
    uv run server.py            # then open http://127.0.0.1:8000

Uploads are processed one at a time in a background worker. Each job gets a folder
under jobs/<id>/ holding the input video, keypoints.npz, motion.npz, result.json
(the data the browser viewer plays back) and meta.json (status), so finished jobs
survive server restarts.
"""

import argparse
import json
import shutil
import sys
import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field, fields
from functools import lru_cache
from pathlib import Path

import cv2
import mujoco
import numpy as np
import uvicorn
from fastapi import Body, FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

import extract_keypoints
import retarget

ROOT = Path(__file__).parent
# The reward code is a plain script folder rather than a package, so import it by path.
sys.path.insert(0, str(ROOT.parent / "reward-calculation"))
import trajectory_similarity  # noqa: E402
JOBS_DIR = ROOT / "jobs"
STATIC_DIR = ROOT / "static"
VIDEO_EXTS = {".mp4", ".mov", ".m4v", ".webm", ".avi", ".mkv"}
DOWNLOADS = {"keypoints.npz", "motion.npz"}
# Share of the progress bar given to landmark extraction; IK takes the rest.
EXTRACT_SHARE = 0.85
# /livedemo's dance_similarity scoring: seconds of recent motion compared per request, the
# shortest window worth scoring, and the longest gap in webcam frames that is interpolated over.
DANCE_WINDOW_S = 2.0
DANCE_MIN_WINDOW_S = 0.5
DANCE_MAX_GAP_S = 0.3
# Live smoothing presets -> One-Euro min_cutoff (Hz); lower is smoother but laggier.
LIVE_SMOOTHING = {"low": 3.0, "medium": 1.5, "high": 0.7}
GEOM_TYPES = {mujoco.mjtGeom.mjGEOM_CAPSULE: "capsule", mujoco.mjtGeom.mjGEOM_BOX: "box",
              mujoco.mjtGeom.mjGEOM_SPHERE: "sphere"}


@dataclass
class Job:
    id: str
    filename: str
    video: str
    model: str
    smooth: float
    # Recording length in seconds, for browser recordings whose container frame rate is unreliable;
    # the frame rate is then computed as decoded frames / duration.
    duration: float | None = None
    created: float = field(default_factory=time.time)
    status: str = "queued"  # queued | extracting | retargeting | done | error
    progress: float = 0.0
    message: str = "Waiting for worker"
    error: str | None = None
    summary: dict | None = None

    @property
    def dir(self) -> Path:
        return JOBS_DIR / self.id

    def save(self):
        (self.dir / "meta.json").write_text(json.dumps(asdict(self)))


jobs: dict[str, Job] = {}
jobs_lock = threading.Lock()
worker = ThreadPoolExecutor(max_workers=1)


def load_saved_jobs():
    known = {f.name for f in fields(Job)}
    for meta in JOBS_DIR.glob("*/meta.json"):
        # Ignore keys from older/newer versions of Job so old jobs still load.
        job = Job(**{k: v for k, v in json.loads(meta.read_text()).items() if k in known})
        if job.status not in ("done", "error"):  # interrupted by a restart
            job.status, job.error, job.message = "error", "Server restarted before the job finished", "Failed"
        jobs[job.id] = job


def update(job: Job, **changes):
    with jobs_lock:
        for k, v in changes.items():
            setattr(job, k, v)


def run_job(job: Job):
    try:
        t0 = time.time()
        update(job, status="extracting", message="Detecting pose (MediaPipe)")

        def on_extract(done: int, total: int):
            if total:
                update(job, progress=EXTRACT_SHARE * done / total, message=f"Detecting pose: frame {done}/{total}")
            else:  # frame count unknown (e.g. browser recordings)
                update(job, message=f"Detecting pose: frame {done}")

        kp = extract_keypoints.extract(str(job.dir / job.video), extract_keypoints.ensure_model(job.model),
                                       progress=on_extract)
        if len(kp["visibility"]) == 0:
            raise ValueError("Could not read any frames from the video.")
        if not (kp["visibility"].sum(axis=1) > 0).any():
            raise ValueError("No person was detected in the video.")
        if job.duration:
            kp["fps"] = np.float32(len(kp["visibility"]) / job.duration)
        np.savez(job.dir / "keypoints.npz", **kp)
        t_extract = time.time() - t0

        update(job, status="retargeting", message="Solving IK")

        def on_ik(done: int, total: int):
            update(job, progress=EXTRACT_SHARE + (1 - EXTRACT_SHARE) * done / total,
                   message=f"Solving IK: frame {done}/{total}")

        motion = retarget.retarget(kp, smooth=job.smooth, progress=on_ik)
        np.savez(job.dir / "motion.npz", **motion)

        result = build_result(kp, motion, job.dir / job.video)
        detected = int((kp["visibility"].sum(axis=1) > 0).sum())
        result["summary"] = summary = {
            "frames": len(motion["qpos"]),
            "detected_frames": detected,
            "fps": round(float(motion["fps"]), 2),
            "mean_error_cm": round(float(motion["error"].mean() * 100), 2),
            "max_error_cm": round(float(motion["error"].max() * 100), 2),
            "extract_seconds": round(t_extract, 1),
            "total_seconds": round(time.time() - t0, 1),
        }
        (job.dir / "result.json").write_text(json.dumps(result, separators=(",", ":")))
        update(job, status="done", progress=1.0, message="Done", summary=summary)
    except Exception as e:
        traceback.print_exc()
        update(job, status="error", message="Failed", error=f"{type(e).__name__}: {e}")
    job.save()


def build_result(kp: dict, motion: dict, video_path: Path) -> dict:
    """Everything the browser needs to play the result back without MuJoCo."""
    model = mujoco.MjModel.from_xml_path(str(motion["model_path"]))
    data = mujoco.MjData(model)
    geom_ids = body_geoms(model)
    poses = [geom_poses(model, data, geom_ids, q) for q in motion["qpos"]]

    return {
        "fps": float(motion["fps"]),
        "n_frames": len(motion["qpos"]),
        "video_size": video_size(video_path),
        "geoms": geom_info(model, geom_ids),
        "poses": poses,  # per frame: [x, y, z, qw, qx, qy, qz] for each geom, flattened
        "root": _r(motion["qpos"][:, :3]),
        "targets": _r(motion["targets"]),  # NaN (untracked landmarks) -> null
        "error_cm": _r(motion["error"] * 100, 2),
        "landmarks_2d": _r(kp["image"][:, :, :2]),
        "visibility": _r(kp["visibility"], 2),
    }


def video_size(path: Path) -> list[int]:
    cap = cv2.VideoCapture(str(path))
    size = [int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))]
    cap.release()
    return size


def body_geoms(model: mujoco.MjModel) -> list[int]:
    """Ids of the humanoid's geoms (everything except world-attached ones like the floor)."""
    return [g for g in range(model.ngeom) if model.geom_bodyid[g] != 0]


def geom_info(model: mujoco.MjModel, geom_ids: list[int]) -> list[dict]:
    return [{"type": GEOM_TYPES.get(int(model.geom_type[g]), "sphere"),
             "size": _r(model.geom_size[g]), "rgba": _r(model.geom_rgba[g], 3)} for g in geom_ids]


def geom_poses(model: mujoco.MjModel, data: mujoco.MjData, geom_ids: list[int], qpos: np.ndarray) -> list:
    """Flattened [x, y, z, qw, qx, qy, qz] per geom for one qpos."""
    data.qpos[:] = qpos
    mujoco.mj_kinematics(model, data)
    quat = np.zeros(4)
    out = []
    for g in geom_ids:
        mujoco.mju_mat2Quat(quat, data.geom_xmat[g])
        out.extend(_r(np.concatenate([data.geom_xpos[g], quat])))
    return out


def _r(a, digits: int = 4):
    """Round to a JSON-friendly nested list, mapping NaN to None."""
    a = np.round(np.asarray(a, dtype=np.float64), digits)
    return np.where(np.isnan(a), None, a).tolist()


app = FastAPI(title="Pose estimation UI")


@app.post("/api/jobs")
def create_job(file: UploadFile = File(...), model: str = Form("full"), smooth: float = Form(0.2),
               duration: float | None = Form(None)):
    ext = Path(file.filename or "").suffix.lower()
    if ext not in VIDEO_EXTS:
        raise HTTPException(400, f"Unsupported file type '{ext}'. Use one of: {', '.join(sorted(VIDEO_EXTS))}")
    if model not in ("lite", "full", "heavy"):
        raise HTTPException(400, "model must be lite, full or heavy")
    if not 0 <= smooth <= 2:
        raise HTTPException(400, "smooth must be between 0 and 2 seconds")
    if duration is not None and not 0.1 <= duration <= 3600:
        raise HTTPException(400, "duration must be between 0.1 and 3600 seconds")

    job = Job(id=uuid.uuid4().hex[:12], filename=Path(file.filename).name, video=f"input{ext}",
              model=model, smooth=smooth, duration=duration)
    job.dir.mkdir(parents=True)
    with open(job.dir / job.video, "wb") as out:
        shutil.copyfileobj(file.file, out)
    job.save()
    with jobs_lock:
        jobs[job.id] = job
    worker.submit(run_job, job)
    return {"id": job.id}


@app.get("/api/jobs")
def list_jobs():
    with jobs_lock:
        return sorted((asdict(j) for j in jobs.values()), key=lambda j: j["created"], reverse=True)


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    with jobs_lock:
        return asdict(_job(job_id))


@app.get("/api/jobs/{job_id}/result")
def get_result(job_id: str):
    path = _job(job_id).dir / "result.json"
    if not path.exists():
        raise HTTPException(404, "Result not ready")
    return FileResponse(path, media_type="application/json")


@app.get("/api/jobs/{job_id}/reference")
def get_reference(job_id: str):
    """MediaPipe landmarks of a finished job, for scoring a webcam against it on /livedemo."""
    job = _job(job_id)
    path = job.dir / "keypoints.npz"
    if job.status != "done" or not path.exists():
        raise HTTPException(404, "Keypoints not ready")
    kp = np.load(path)
    return {
        "fps": float(kp["fps"]),
        "n_frames": len(kp["visibility"]),
        "video_size": video_size(job.dir / job.video),
        "world": _r(kp["world"], 3),  # NaN (no detection) -> null
        "image": _r(kp["image"][:, :, :2], 4),
        "visibility": _r(kp["visibility"], 2),
    }


@lru_cache(maxsize=8)
def reference_world(job_id: str) -> tuple[np.ndarray, float]:
    kp = np.load(_job(job_id).dir / "keypoints.npz")
    return kp["world"].astype(np.float64), float(kp["fps"])


@app.post("/api/jobs/{job_id}/dance-similarity")
def score_dance(job_id: str, body: dict = Body(...)):
    """Score recent webcam motion with reward-calculation's dance_similarity.

    Body: {"t": video time (s), "times": [video time of each webcam frame],
           "world": [[[x, y, z] * 33] per frame]}  (MediaPipe world landmarks, already mirrored if wanted)
    The webcam frames are resampled onto the reference's frame times over the last DANCE_WINDOW_S
    seconds before t, and compared with the same reference frames. Returns dance_similarity's
    result, or {"final_score": null, "reason"} when there isn't enough to compare.
    """
    job = _job(job_id)
    if job.status != "done" or not (job.dir / "keypoints.npz").exists():
        raise HTTPException(404, "Keypoints not ready")
    try:
        t = float(body["t"])
        times = np.asarray(body["times"], dtype=np.float64)
        world = np.asarray(body["world"], dtype=np.float64)
    except (KeyError, TypeError, ValueError) as e:
        raise HTTPException(400, f"Bad request body: {e}")
    if world.ndim != 3 or world.shape[1:] != (33, 3) or times.shape != (len(world),):
        raise HTTPException(400, "world must be (frames, 33, 3) with one time per frame")
    if not (np.isfinite(world).all() and np.isfinite(times).all()):
        raise HTTPException(400, "times and world must be finite")

    ref, fps = reference_world(job_id)
    order = np.argsort(times, kind="stable")
    times, world = times[order], world[order]
    if len(times) < 2:
        return {"final_score": None, "reason": "not enough webcam frames"}

    # Reference frames in the window that the webcam samples cover without long gaps.
    start = max(t - DANCE_WINDOW_S, times[0] - 1 / fps)
    end = min(t, times[-1] + 1 / fps)
    frames = np.arange(max(0, int(np.ceil(start * fps))), min(len(ref) - 1, int(end * fps)) + 1)
    if len(frames) < DANCE_MIN_WINDOW_S * fps:
        return {"final_score": None, "reason": "warming up"}
    frame_times = frames / fps
    in_window = (times >= frame_times[0] - DANCE_MAX_GAP_S) & (times <= frame_times[-1] + DANCE_MAX_GAP_S)
    if np.diff(times[in_window]).max(initial=0) > DANCE_MAX_GAP_S:
        return {"final_score": None, "reason": "lost track of you"}

    flat = world.reshape(len(world), -1)
    generated = np.stack([np.interp(frame_times, times, flat[:, c]) for c in range(flat.shape[1])], axis=1)
    generated = generated.reshape(len(frames), 33, 3)

    reference = ref[frames]
    valid = np.isfinite(reference).all(axis=(1, 2))
    if valid.mean() < 0.5:
        return {"final_score": None, "reason": "dancer not in view"}
    if not valid.all():  # fill frames where the dancer wasn't detected
        idx = np.flatnonzero(valid)
        ref_flat = reference[valid].reshape(len(idx), -1)
        reference = np.stack([np.interp(np.arange(len(frames)), idx, ref_flat[:, c])
                              for c in range(ref_flat.shape[1])], axis=1).reshape(len(frames), 33, 3)

    result = trajectory_similarity.dance_similarity(reference, generated, fps)
    if not np.isfinite(result["final_score"]):
        return {"final_score": None, "reason": "could not score"}
    out = {k: (int(v) if k == "timing_shift_frames" else round(float(v), 4)) for k, v in result.items()}
    out["window_s"] = round(len(frames) / fps, 2)
    return out


@app.get("/api/jobs/{job_id}/video")
def get_video(job_id: str):
    job = _job(job_id)
    return FileResponse(job.dir / job.video)


@app.get("/api/jobs/{job_id}/files/{name}")
def get_file(job_id: str, name: str):
    path = _job(job_id).dir / name
    if name not in DOWNLOADS or not path.exists():
        raise HTTPException(404, "File not found")
    stem = Path(_job(job_id).filename).stem
    return FileResponse(path, filename=f"{stem}_{name}")


def _job(job_id: str) -> Job:
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    return job


@app.get("/api/model")
def get_model():
    """Static geometry of the default humanoid, for the live view."""
    model = mujoco.MjModel.from_xml_path(str(retarget.DEFAULT_MODEL))
    return {"geoms": geom_info(model, body_geoms(model))}


@app.websocket("/api/live")
async def live(ws: WebSocket):
    """Streaming IK. The client sends MediaPipe world landmarks and gets back humanoid geom poses.

    Client -> server:
        {"type": "frame", "t": seconds, "world": [[x, y, z] * 33] | null, "visibility": [33 floats]}
        {"type": "config", "smoothing": "low" | "medium" | "high"}
    Server -> client (one reply per frame message):
        {"type": "pose", "t", "poses", "root", "targets", "error_cm", "ik_ms"}  or  {"type": "pose", "t", "lost": true}
        {"type": "error", "message"}
    The client should wait for each reply before sending the next frame, so latency stays bounded.
    """
    await ws.accept()
    streamer = retarget.StreamingRetargeter(min_cutoff=LIVE_SMOOTHING["medium"])
    model = streamer.model
    data = mujoco.MjData(model)
    geom_ids = body_geoms(model)

    def step(world, visibility, t):
        t0 = time.perf_counter()
        out = streamer.step(world, visibility, t)
        if out is None:
            return {"type": "pose", "t": t, "lost": True}
        return {"type": "pose", "t": t, "poses": geom_poses(model, data, geom_ids, out["qpos"]),
                "root": _r(out["qpos"][:3]), "targets": _r(out["targets"]),
                "error_cm": round(out["error"] * 100, 2), "ik_ms": round((time.perf_counter() - t0) * 1000, 1)}

    try:
        while True:
            msg = await ws.receive_json()
            if msg.get("type") == "config":
                streamer.set_smoothing(LIVE_SMOOTHING.get(msg.get("smoothing"), LIVE_SMOOTHING["medium"]))
                continue
            if msg.get("type") != "frame":
                continue
            try:
                t = float(msg["t"])
                world = msg.get("world")
                visibility = msg.get("visibility")
                if world is not None:
                    world = np.asarray(world, dtype=np.float64)
                    if world.shape != (33, 3) or not np.isfinite(world).all():
                        raise ValueError("world must be 33 finite [x, y, z] landmarks")
                    visibility = np.asarray(visibility, dtype=np.float64) if visibility is not None else None
                    if visibility is not None and visibility.shape != (33,):
                        raise ValueError("visibility must have 33 values")
                reply = await run_in_threadpool(step, world, visibility, t)
            except KeyError as e:
                reply = {"type": "error", "message": f"missing field {e}"}
            except (TypeError, ValueError) as e:
                reply = {"type": "error", "message": str(e)}
            await ws.send_json(reply)
    except WebSocketDisconnect:
        pass


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/livedemo")
def livedemo():
    return FileResponse(STATIC_DIR / "livedemo.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    # Browsers only allow webcam access on https pages (or localhost), so serving to other devices
    # on the network (e.g. /livedemo at http://<ip>:8000) needs a certificate.
    parser.add_argument("--ssl-certfile", help="serve over https with this certificate (PEM)")
    parser.add_argument("--ssl-keyfile", help="private key for --ssl-certfile (PEM)")
    args = parser.parse_args()
    JOBS_DIR.mkdir(exist_ok=True)
    load_saved_jobs()
    scheme = "https" if args.ssl_certfile else "http"
    print(f"Open {scheme}://{args.host}:{args.port}  (live demo: {scheme}://{args.host}:{args.port}/livedemo)")
    uvicorn.run(app, host=args.host, port=args.port, ssl_certfile=args.ssl_certfile, ssl_keyfile=args.ssl_keyfile)


if __name__ == "__main__":
    main()
