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


class IKSolver:
    """Per-frame IK onto the mp_<i> sites, warm-started from the previous frame's solution."""

    def __init__(self, model: mujoco.MjModel, iters: int = 20, first_iters: int = 200, posture_cost: float = 0.02):
        self.model, self.iters, self.first_iters = model, iters, first_iters
        self.config = mink.Configuration(model)
        self.tasks = {i: mink.FrameTask(f"mp_{i}", "site", position_cost=BASE_COST[i], orientation_cost=0.0)
                      for i in TRACKED}
        # Posture prior resolves limb twist ambiguity; zero cost on the free joint so it
        # doesn't pull the root toward the origin.
        posture_costs = np.full(model.nv, posture_cost)
        posture_costs[:6] = 0.0
        posture = mink.PostureTask(model, cost=posture_costs)
        posture.set_target_from_configuration(self.config)
        self.limits = [mink.ConfigurationLimit(model)]
        self.all_tasks = [*self.tasks.values(), posture]
        self.site_ids = [model.site(f"mp_{i}").id for i in TRACKED]
        self.initialized = False

    def reset(self):
        """Forget the previous pose; the next step() re-initializes the root from its targets."""
        self.initialized = False

    def step(self, targets: np.ndarray, visibility: np.ndarray) -> tuple[np.ndarray, float]:
        """Solve for one frame of (33, 3) targets. Returns (qpos copy, mean site error in meters)."""
        if not self.initialized:
            q = self.model.qpos0.copy()
            q[0:3] = 0.5 * (targets[23] + targets[24])
            q[3:7] = root_quat_from_targets(targets)
            self.config.update(q)
        for i, task in self.tasks.items():
            task.set_target(mink.SE3.from_translation(targets[i]))
            task.set_position_cost(BASE_COST[i] * max(float(visibility[i]), 0.1))
        for _ in range(self.iters if self.initialized else self.first_iters):
            vel = mink.solve_ik(self.config, self.all_tasks, 0.01, "daqp", damping=1e-3, limits=self.limits)
            self.config.integrate_inplace(vel, 0.01)
            if np.linalg.norm(vel) * 0.01 < 1e-5:
                break
        self.initialized = True
        err = np.linalg.norm(self.config.data.site_xpos[self.site_ids] - targets[TRACKED], axis=1).mean()
        return self.config.q.copy(), float(err)


def solve(model: mujoco.MjModel, targets: np.ndarray, visibility: np.ndarray,
          iters: int = 20, first_iters: int = 200, posture_cost: float = 0.02,
          progress: Callable[[int, int], None] | None = None) -> tuple[np.ndarray, np.ndarray]:
    solver = IKSolver(model, iters, first_iters, posture_cost)
    qpos = np.zeros((len(targets), model.nq))
    err = np.zeros(len(targets))
    for f in range(len(targets)):
        qpos[f], err[f] = solver.step(targets[f], visibility[f])
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


class OneEuroFilter:
    """One-Euro filter (Casiez et al. 2012) over arrays: low lag when moving fast, smooth when still.

    min_cutoff (Hz) sets smoothing at rest (lower = smoother); beta sets how quickly the cutoff
    rises with speed (units of Hz per m/s here, since inputs are in meters).
    """

    def __init__(self, min_cutoff: float = 1.5, beta: float = 1.0, d_cutoff: float = 1.0):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.reset()

    def reset(self):
        self.x = self.dx = self.t = None

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x: np.ndarray, t: float) -> np.ndarray:
        if self.x is None:
            self.x, self.dx, self.t = x.copy(), np.zeros_like(x), t
            return x.copy()
        dt = t - self.t if t > self.t else 1.0 / 30
        self.t = t
        dx = (x - self.x) / dt
        self.dx = self.dx + self._alpha(self.d_cutoff, dt) * (dx - self.dx)
        cutoff = self.min_cutoff + self.beta * np.abs(self.dx)
        self.x = self.x + self._alpha(cutoff, dt) * (x - self.x)
        return self.x.copy()


class StreamingRetargeter:
    """Causal, frame-at-a-time version of retarget() for live input.

    Differences from the batch pipeline, which can look ahead: One-Euro filtering instead of
    Savitzky-Golay, an exponential moving average for foot grounding, and no gap filling. After
    `hold` seconds without a detection the solver resets, so a new person starts cleanly.
    """

    def __init__(self, model_path: str | Path = DEFAULT_MODEL, min_cutoff: float = 1.5, beta: float = 1.0,
                 ground_tau: float = 0.1, hold: float = 0.5, iters: int = 20):
        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.rest = rest_pose_sites(self.model)
        self.floor_z = np.nanmin(self.rest[FOOT_POINTS, 2])
        self.solver = IKSolver(self.model, iters=iters)
        self.filter = OneEuroFilter(min_cutoff, beta)
        self.ground_tau, self.hold = ground_tau, hold
        self.shift = None
        self.last_seen = None

    def set_smoothing(self, min_cutoff: float):
        self.filter.min_cutoff = min_cutoff

    def reset(self):
        self.solver.reset()
        self.filter.reset()
        self.shift = self.last_seen = None

    def step(self, world: np.ndarray | None, visibility: np.ndarray | None, t: float) -> dict | None:
        """world: (33, 3) MediaPipe world landmarks (camera axes) or None if no person; t in seconds.

        Returns {qpos, targets, error} or None when there is no pose to show.
        """
        if world is None:
            if self.last_seen is not None and t - self.last_seen > self.hold:
                self.reset()
            return None
        dt = t - self.last_seen if self.last_seen is not None else None
        self.last_seen = t

        pts = self.filter(mp_to_mujoco(np.asarray(world, dtype=np.float64)), t)
        targets = rescale_skeleton(pts[None], self.rest)[0]
        shift = self.floor_z - targets[FOOT_POINTS, 2].min()
        if self.shift is None or dt is None or dt <= 0:
            self.shift = shift
        else:
            self.shift += (1 - np.exp(-dt / self.ground_tau)) * (shift - self.shift)
        targets[:, 2] += self.shift

        vis = np.ones(33) if visibility is None else np.asarray(visibility, dtype=np.float64)
        qpos, err = self.solver.step(targets, vis)
        return dict(qpos=qpos, targets=targets, error=err)


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
