"""Play back a saved pose file (motion.npz from retarget.py) on its MuJoCo model.

Usage:
    uv run mjpython visualize.py motion.npz              # interactive viewer (macOS needs mjpython)
    uv run visualize.py motion.npz --render out.mp4      # headless render to video
    uv run visualize.py motion.npz --render out.png --start 120   # single frame

Viewer controls:
    Space        pause / resume
    Left/Right   step one frame (pauses)
    Up/Down      double / halve playback speed
    Home         jump to first frame
"""

import argparse
import threading
import time
from pathlib import Path

import mujoco
import numpy as np

DEFAULT_MODEL = Path(__file__).parent / "humanoid.xml"
TARGET_RGBA = np.array([0.1, 0.9, 0.3, 0.8], dtype=np.float32)

KEY_SPACE, KEY_RIGHT, KEY_LEFT, KEY_DOWN, KEY_UP, KEY_HOME = 32, 262, 263, 264, 265, 268


def load(motion_path: str, model_path: str | None = None):
    """Return (model, qpos, targets or None, fps, error or None) for a saved motion file."""
    data = np.load(motion_path)
    if model_path is None:
        saved = Path(str(data["model_path"])) if "model_path" in data else None
        model_path = saved if saved is not None and saved.exists() else DEFAULT_MODEL
    model = mujoco.MjModel.from_xml_path(str(model_path))
    qpos = data["qpos"]
    if qpos.shape[1] != model.nq:
        raise ValueError(f"qpos has {qpos.shape[1]} columns but {model_path} has nq={model.nq}")
    targets = data["targets"] if "targets" in data else None
    error = data["error"] if "error" in data else None
    return model, qpos, targets, float(data["fps"]), error


def add_target_markers(scene: mujoco.MjvScene, points: np.ndarray):
    """Draw a small sphere at each finite point (e.g. IK targets) into a scene."""
    for p in points:
        if scene.ngeom >= scene.maxgeom:
            return
        if not np.all(np.isfinite(p)):
            continue
        mujoco.mjv_initGeom(scene.geoms[scene.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE,
                            np.array([0.02, 0, 0]), p, np.eye(3).ravel(), TARGET_RGBA)
        scene.ngeom += 1


def render(model: mujoco.MjModel, qpos: np.ndarray, targets: np.ndarray | None, fps: float, path: str,
           start: int = 0, width: int = 1280, height: int = 720):
    """Render to an .mp4 (all frames from `start`) or a single-frame .png/.jpg."""
    import cv2

    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height, width)
    cam = _default_camera()
    single = Path(path).suffix.lower() in (".png", ".jpg", ".jpeg")
    frames = [start] if single else range(start, len(qpos))
    writer = None if single else cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    for f in frames:
        data.qpos[:] = qpos[f]
        mujoco.mj_forward(model, data)
        if _has_free_root(model):
            cam.lookat[:] = data.qpos[:3]
        renderer.update_scene(data, cam)
        if targets is not None:
            add_target_markers(renderer.scene, targets[f])
        img = cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR)
        if single:
            cv2.imwrite(path, img)
        else:
            writer.write(img)
    if writer is not None:
        writer.release()
    renderer.close()
    print(f"Rendered {path}")


def view(model: mujoco.MjModel, qpos: np.ndarray, targets: np.ndarray | None, fps: float,
         error: np.ndarray | None = None, start: int = 0, speed: float = 1.0, follow: bool = True):
    """Interactive playback in the MuJoCo passive viewer."""
    import mujoco.viewer

    n = len(qpos)
    state = {"t": start / fps, "paused": False, "speed": speed}
    lock = threading.Lock()

    def on_key(key: int):
        with lock:
            frame = int(state["t"] * fps) % n
            if key == KEY_SPACE:
                state["paused"] = not state["paused"]
            elif key in (KEY_RIGHT, KEY_LEFT):
                state["paused"] = True
                state["t"] = ((frame + (1 if key == KEY_RIGHT else -1)) % n + 0.5) / fps
            elif key == KEY_UP:
                state["speed"] = min(state["speed"] * 2, 8.0)
            elif key == KEY_DOWN:
                state["speed"] = max(state["speed"] / 2, 0.125)
            elif key == KEY_HOME:
                state["t"] = 0.0

    data = mujoco.MjData(model)
    with mujoco.viewer.launch_passive(model, data, key_callback=on_key) as viewer:
        cam = _default_camera()
        viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = cam.distance, cam.azimuth, cam.elevation
        last = time.time()
        while viewer.is_running():
            now = time.time()
            with lock:
                if not state["paused"]:
                    state["t"] += (now - last) * state["speed"]
                frame = int(state["t"] * fps) % n
                paused, spd = state["paused"], state["speed"]
            last = now

            with viewer.lock():
                data.qpos[:] = qpos[frame]
                data.qvel[:] = 0
                mujoco.mj_forward(model, data)
                if follow and _has_free_root(model):
                    viewer.cam.lookat[:] = data.qpos[:3]
                viewer.user_scn.ngeom = 0
                if targets is not None:
                    add_target_markers(viewer.user_scn, targets[frame])
                status = f"{frame + 1}/{n}\n{frame / fps:.2f}s\n{spd:g}x{'  (paused)' if paused else ''}"
                if error is not None:
                    status += f"\n{error[frame] * 100:.1f} cm"
                viewer.set_texts((mujoco.mjtFontScale.mjFONTSCALE_150, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                                  "Frame\nTime\nSpeed" + ("\nIK err" if error is not None else ""), status))
            viewer.sync()
            time.sleep(max(0.0, 1.0 / 60 - (time.time() - now)))


def _default_camera() -> mujoco.MjvCamera:
    cam = mujoco.MjvCamera()
    cam.distance, cam.azimuth, cam.elevation = 3.5, 150.0, -15.0
    cam.lookat[:] = [0, 0, 0.9]
    return cam


def _has_free_root(model: mujoco.MjModel) -> bool:
    return model.njnt > 0 and model.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("motion", help=".npz from retarget.py (needs qpos and fps)")
    parser.add_argument("--model", help="MJCF to use (default: model_path saved in the file, else humanoid.xml)")
    parser.add_argument("--render", metavar="PATH", help="render headlessly to .mp4, or one frame to .png")
    parser.add_argument("--start", type=int, default=0, help="first frame to show")
    parser.add_argument("--speed", type=float, default=1.0, help="initial playback speed")
    parser.add_argument("--no-targets", action="store_true", help="hide IK target markers")
    parser.add_argument("--no-follow", action="store_true", help="don't move the camera with the root")
    args = parser.parse_args()

    model, qpos, targets, fps, error = load(args.motion, args.model)
    if not 0 <= args.start < len(qpos):
        parser.error(f"--start must be in [0, {len(qpos) - 1}]")
    if args.no_targets:
        targets = None
    if args.render:
        render(model, qpos, targets, fps, args.render, start=args.start)
    else:
        view(model, qpos, targets, fps, error, start=args.start, speed=args.speed, follow=not args.no_follow)


if __name__ == "__main__":
    main()
