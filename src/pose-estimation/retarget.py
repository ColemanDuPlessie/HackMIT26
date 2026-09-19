"""Retarget MediaPipe landmarks (from extract_keypoints.py) onto a MuJoCo humanoid with mink IK.

Usage:
    uv run retarget.py keypoints.npz -o motion.npz [--render motion.mp4]
    uv run mjpython retarget.py keypoints.npz --view     # macOS needs mjpython for the viewer

Pipeline:
    1. Fill missing frames and convert MediaPipe camera axes to MuJoCo (z-up, facing +x, left +y).
    2. Temporal smoothing (Savitzky-Golay).
    3. Rebuild the skeleton with the model's bone lengths, keeping only the bone
       directions from MediaPipe, so IK targets are always reachable.
    4. Place the lowest foot point on the floor each frame (no horizontal root motion;
       MediaPipe world landmarks are hip-centered).
    5. Per-frame IK: one position task per mp_<i> site, plus a light posture prior.

Output .npz contents:
    qpos       (T, nq) MuJoCo joint positions for the model
    targets    (T, 33, 3) IK target positions in world frame
    error      (T,) mean site-to-target distance after IK, meters
    fps        scalar
    joint_names, model_path
"""

import argparse
import time
from pathlib import Path
from typing import Callable, Mapping

import mink
import mujoco
import numpy as np
from scipy.signal import savgol_filter

import visualize

DEFAULT_MODEL = Path(__file__).parent / "humanoid.xml"

# Kinematic tree over MediaPipe landmarks, used to rebuild the skeleton with model
# bone lengths. "hip_c"/"sho_c" are virtual midpoints of the hips/shoulders.
PARENTS = {
    "sho_c": "hip_c", 23: "hip_c", 24: "hip_c",
    11: "sho_c", 12: "sho_c", 0: "sho_c", 7: "sho_c", 8: "sho_c",
    13: 11, 15: 13, 17: 15, 19: 15, 21: 15,
    14: 12, 16: 14, 18: 16, 20: 16, 22: 16,
    25: 23, 27: 25, 29: 27, 31: 27,
    26: 24, 28: 26, 30: 28, 32: 28,
}
TRACKED = [k for k in PARENTS if isinstance(k, int)]
FOOT_POINTS = [27, 28, 29, 30, 31, 32]
# Relative IK weights; face and hand points are noisier in MediaPipe.
BASE_COST = {i: 1.0 for i in TRACKED} | {i: 0.3 for i in (0, 7, 8, 17, 18, 19, 20, 21, 22)}


def mp_to_mujoco(p: np.ndarray) -> np.ndarray:
    """MediaPipe (x right, y down, z away from camera) -> MuJoCo (x toward camera, y = subject's left, z up)."""
    return np.stack([-p[..., 2], p[..., 0], -p[..., 1]], axis=-1)


def fill_missing(x: np.ndarray) -> np.ndarray:
    """Linearly interpolate NaN frames along time, per coordinate."""
    x = x.copy()
    t = np.arange(x.shape[0])
    flat = x.reshape(x.shape[0], -1)
    for c in range(flat.shape[1]):
        ok = ~np.isnan(flat[:, c])
        if not ok.any():
            raise ValueError("Keypoints contain a landmark that is never detected.")
        flat[:, c] = np.interp(t, t[ok], flat[ok, c])
    return flat.reshape(x.shape)


def with_virtual(p: np.ndarray) -> dict:
    """Map node name -> (..., 3) positions, including virtual hip/shoulder centers."""
    nodes = {i: p[..., i, :] for i in TRACKED}
    nodes["hip_c"] = 0.5 * (p[..., 23, :] + p[..., 24, :])
    nodes["sho_c"] = 0.5 * (p[..., 11, :] + p[..., 12, :])
    return nodes


def rest_pose_sites(model: mujoco.MjModel) -> np.ndarray:
    """(33, 3) world positions of the mp_<i> sites at qpos0 (NaN for untracked landmarks)."""
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    pos = np.full((33, 3), np.nan)
    for i in TRACKED:
        pos[i] = data.site(f"mp_{i}").xpos
    return pos


def rescale_skeleton(kp: np.ndarray, rest: np.ndarray) -> np.ndarray:
    """Rebuild (T, 33, 3) keypoints with the model's bone lengths, rooted at the origin."""
    src, ref = with_virtual(kp), with_virtual(rest)
    out = {"hip_c": np.zeros_like(src["hip_c"])}
    pending = dict(PARENTS)
    while pending:  # resolve nodes whose parent is already placed
        for node, parent in list(pending.items()):
            if parent in out:
                d = src[node] - src[parent]
                d /= np.linalg.norm(d, axis=-1, keepdims=True) + 1e-9
                out[node] = out[parent] + d * np.linalg.norm(ref[node] - ref[parent])
                del pending[node]
    targets = np.full_like(kp, np.nan)
    for i in TRACKED:
        targets[:, i] = out[i]
    return targets


def ground(targets: np.ndarray, rest: np.ndarray, fps: float) -> np.ndarray:
    """Shift each frame vertically so its lowest foot point matches the model's rest-pose sole height."""
    floor_z = np.nanmin(rest[FOOT_POINTS, 2])
    shift = floor_z - targets[:, FOOT_POINTS, 2].min(axis=1)
    win = _savgol_window(len(shift), fps, 0.15)
    if win:
        shift = savgol_filter(shift, win, 2)
    return targets + np.array([0, 0, 1]) * shift[:, None, None]


def _savgol_window(n: int, fps: float, seconds: float) -> int:
    win = int(round(seconds * fps)) | 1  # odd
    win = min(win, n if n % 2 else n - 1)
    return win if win >= 5 else 0


def root_quat_from_targets(t: np.ndarray) -> np.ndarray:
    """Pelvis orientation (wxyz) facing the direction implied by hips and shoulders."""
    left = t[23] - t[24]
    up = 0.5 * (t[11] + t[12]) - 0.5 * (t[23] + t[24])
    up /= np.linalg.norm(up)
    left -= left.dot(up) * up
    left /= np.linalg.norm(left)
    fwd = np.cross(left, up)
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, np.column_stack([fwd, left, up]).ravel())
    return quat


def solve(model: mujoco.MjModel, targets: np.ndarray, visibility: np.ndarray,
          iters: int = 20, first_iters: int = 200, posture_cost: float = 0.02,
          progress: Callable[[int, int], None] | None = None) -> tuple[np.ndarray, np.ndarray]:
    config = mink.Configuration(model)
    tasks = {i: mink.FrameTask(f"mp_{i}", "site", position_cost=BASE_COST[i], orientation_cost=0.0)
             for i in TRACKED}
    # Posture prior resolves limb twist ambiguity; zero cost on the free joint so it
    # doesn't pull the root toward the origin.
    posture_costs = np.full(model.nv, posture_cost)
    posture_costs[:6] = 0.0
    posture = mink.PostureTask(model, cost=posture_costs)
    posture.set_target_from_configuration(config)
    limits = [mink.ConfigurationLimit(model)]
    all_tasks = [*tasks.values(), posture]

    q = model.qpos0.copy()
    q[0:3] = 0.5 * (targets[0, 23] + targets[0, 24])
    q[3:7] = root_quat_from_targets(targets[0])
    config.update(q)

    site_ids = [model.site(f"mp_{i}").id for i in TRACKED]
    qpos = np.zeros((len(targets), model.nq))
    err = np.zeros(len(targets))
    for f in range(len(targets)):
        for i, task in tasks.items():
            task.set_target(mink.SE3.from_translation(targets[f, i]))
            task.set_position_cost(BASE_COST[i] * max(float(visibility[f, i]), 0.1))
        for _ in range(first_iters if f == 0 else iters):
            vel = mink.solve_ik(config, all_tasks, 0.01, "daqp", damping=1e-3, limits=limits)
            config.integrate_inplace(vel, 0.01)
            if np.linalg.norm(vel) * 0.01 < 1e-5:
                break
        qpos[f] = config.q
        err[f] = np.linalg.norm(config.data.site_xpos[site_ids] - targets[f, TRACKED], axis=1).mean()
        if progress:
            progress(f + 1, len(targets))
    return qpos, err


def retarget(keypoints: Mapping, model_path: str | Path = DEFAULT_MODEL, smooth: float = 0.2, iters: int = 20,
             progress: Callable[[int, int], None] | None = None) -> dict:
    """Full pipeline from extract_keypoints output to a motion dict (the contents of motion.npz)."""
    fps = float(keypoints["fps"])
    world = mp_to_mujoco(fill_missing(np.asarray(keypoints["world"], dtype=np.float64)))
    visibility = keypoints["visibility"]
    win = _savgol_window(len(world), fps, smooth) if smooth > 0 else 0
    if win:
        world = savgol_filter(world, win, 2, axis=0)

    model = mujoco.MjModel.from_xml_path(str(model_path))
    rest = rest_pose_sites(model)
    targets = ground(rescale_skeleton(world, rest), rest, fps)
    qpos, err = solve(model, targets, visibility, iters=iters, progress=progress)
    return dict(qpos=qpos, targets=targets, error=err, fps=np.float32(fps),
                joint_names=[model.joint(j).name for j in range(model.njnt)],
                model_path=str(Path(model_path).resolve()))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("keypoints", help=".npz from extract_keypoints.py")
    parser.add_argument("-o", "--output", default="motion.npz")
    parser.add_argument("--model", default=str(DEFAULT_MODEL), help="MJCF with mp_<i> sites")
    parser.add_argument("--smooth", type=float, default=0.2, help="smoothing window in seconds (0 disables)")
    parser.add_argument("--iters", type=int, default=20, help="IK iterations per frame")
    parser.add_argument("--render", metavar="MP4", help="render the result to a video file (see visualize.py)")
    parser.add_argument("--view", action="store_true", help="play back in the MuJoCo viewer (use mjpython on macOS)")
    args = parser.parse_args()

    t0 = time.time()
    motion = retarget(np.load(args.keypoints), args.model, smooth=args.smooth, iters=args.iters)
    qpos, targets, err, fps = motion["qpos"], motion["targets"], motion["error"], float(motion["fps"])
    print(f"Retargeted {len(qpos)} frames in {time.time() - t0:.1f}s; "
          f"mean site error {err.mean() * 100:.1f} cm (max {err.max() * 100:.1f} cm)")
    np.savez(args.output, **motion)
    print(f"Saved {args.output}")

    if args.render or args.view:
        model = mujoco.MjModel.from_xml_path(args.model)
    if args.render:
        visualize.render(model, qpos, targets, fps, args.render)
    if args.view:
        visualize.view(model, qpos, targets, fps, err)


if __name__ == "__main__":
    main()
