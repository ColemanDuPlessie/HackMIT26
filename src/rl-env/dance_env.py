"""Gymnasium environment: make the MuJoCo humanoid physically perform a retargeted dance.

The supervised model in ../supervised-training produces *kinematic* motion — joint angles with
no physics behind them, so it can float, skate or strike poses no body could hold. This
environment asks a policy to reproduce such a trajectory with torques, under gravity and
contact, which is the DeepMimic formulation and the one place in this project where RL has a
job supervision can't do: there is no ground-truth torque to regress onto.

    observation  the humanoid's own state (root height and orientation, joint angles and
                 velocities) plus the next few reference frames, so the policy can see where
                 it is meant to be going
    action       a target-angle offset per actuated DOF, added to the reference pose and
                 converted to torque by a PD controller (34 values in [-1, 1])
    reward       DeepMimic's four terms: pose, velocity, end-effector and root
    episode      one reference clip, starting at a random frame, ending when the clip runs out
                 or the humanoid falls

Two details do most of the work and are easy to leave out by accident:

* **Reference state initialisation** — starting at a random frame of the clip, in that frame's
  pose. Without it the policy only ever sees the first second and never learns the rest.
* **Early termination** — ending the episode when the humanoid falls, so it stops collecting
  reward for lying on the floor doing a convincing impression of a dancer. "Fallen" is measured
  against the reference, not against a fixed height: these clips crouch to 0.38 m and leave the
  floor entirely, and an absolute threshold calls both of those a fall.

Usage:
    uv run dance_env.py                     # random-policy smoke test + Gymnasium's env checker
    uv run dance_env.py --motion path/to/motion.npz --render frames.mp4
"""

import argparse
from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from humanoid_actuated import joint_layout, load_actuated, rotvec_between

ROOT = Path(__file__).resolve().parent
# Feet and hands: the joints a viewer notices first, so they get their own reward term.
END_EFFECTORS = ("l_foot", "r_foot", "l_hand", "r_hand")
# How far ahead the policy is shown the reference, in frames (at the clip's own frame rate).
LOOKAHEAD = (1, 4, 10)


class DanceTrackingEnv(gym.Env):
    """Track a reference `qpos` trajectory with torques."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(self, motion: dict | str | Path, control_hz: float = 30.0,
                 fall_margin: float = 0.25, max_reference_speed: float = 20.0,
                 action_scale: float = 0.4, kp: float = 60.0, kd: float = 5.0,
                 render_mode: str | None = None):
        super().__init__()
        if not isinstance(motion, dict):
            with np.load(motion) as data:
                motion = {k: data[k] for k in data.files}
        self.reference = np.asarray(motion["qpos"], dtype=np.float64)
        self.reference_fps = float(motion["fps"])
        self.frames = len(self.reference)

        self.model = load_actuated()
        self.data = mujoco.MjData(self.model)
        self.layout = joint_layout(self.model)
        self.kp, self.kd = kp, kd
        self.action_scale = action_scale
        self.fall_margin = fall_margin
        self.max_reference_speed = max_reference_speed

        # Physics runs at the model's timestep; the policy acts at control_hz.
        self.substeps = max(1, int(round(1.0 / (control_hz * self.model.opt.timestep))))
        self.frames_per_step = self.reference_fps / control_hz

        self.ee_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
                       for name in END_EFFECTORS]
        self.ee_ids = [i for i in self.ee_ids if i >= 0]
        self.reference_velocity = self._reference_velocity()
        self.standing_height = float(np.median(self.reference[:, 2]))
        self.torso_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "torso")

        self.action_space = spaces.Box(-1.0, 1.0, (self.model.nu,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, (len(self._observe(0.0)),), dtype=np.float32)
        self.render_mode = render_mode
        self._renderer = None

    # ------------------------------------------------------------------ reference

    def _reference_velocity(self) -> np.ndarray:
        """Finite-difference qvel for each reference frame, for the velocity reward."""
        velocity = np.zeros((self.frames, self.model.nv))
        dt = 1.0 / self.reference_fps
        for f in range(self.frames - 1):
            mujoco.mj_differentiatePos(self.model, velocity[f], dt,
                                       self.reference[f], self.reference[f + 1])
        velocity[-1] = velocity[-2] if self.frames > 1 else 0.0
        # Retargeting jitter produces occasional spikes of >100 rad/s. Starting an episode with
        # one of those throws the humanoid across the room before the policy acts at all.
        return np.clip(velocity, -self.max_reference_speed, self.max_reference_speed)

    def _reference_at(self, frame: float) -> np.ndarray:
        """Reference pose at a fractional frame (nearest frame; poses are dense at 60 fps)."""
        return self.reference[int(np.clip(round(frame), 0, self.frames - 1))]

    # ------------------------------------------------------------------ gym API

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        # Reference state initialisation: start anywhere in the clip, in that frame's pose.
        start = options.get("frame") if options else None
        self.frame = float(self.np_random.integers(0, max(1, self.frames - 2)) if start is None else start)
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = self._reference_at(self.frame)
        self.data.qvel[:] = self.reference_velocity[int(self.frame)]
        mujoco.mj_forward(self.model, self.data)
        return self._observe(self.frame).astype(np.float32), {}

    def step(self, action: np.ndarray):
        target = self._reference_at(self.frame + self.frames_per_step)
        action = np.clip(action, -1.0, 1.0)
        for _ in range(self.substeps):
            self.data.ctrl[:] = self._pd_torques(target, action)
            mujoco.mj_step(self.model, self.data)

        self.frame += self.frames_per_step
        frame_index = int(np.clip(round(self.frame), 0, self.frames - 1))
        if not np.isfinite(self.data.qpos).all():
            # MuJoCo resets itself when the solver diverges, which otherwise shows up as an
            # episode where nothing the policy did mattered. Treat it as a fall.
            return (self._observe(self.frame).astype(np.float32), 0.0, True, False,
                    {"frame": frame_index, "fallen": True, "diverged": True,
                     "reward_pose": 0.0, "reward_velocity": 0.0,
                     "reward_end_effector": 0.0, "reward_root": 0.0})
        reward, parts = self._reward(frame_index)
        fallen = self._fallen(frame_index)
        truncated = self.frame >= self.frames - 1
        return (self._observe(self.frame).astype(np.float32), reward, bool(fallen), bool(truncated),
                {"frame": frame_index, "fallen": bool(fallen), "diverged": False, **parts})

    def _fallen(self, frame: int) -> bool:
        """Below the reference by more than `fall_margin`, or tipped past horizontal."""
        sunk = self.data.qpos[2] < self.reference[frame][2] - self.fall_margin
        upright = self.data.xmat[self.torso_id].reshape(3, 3)[2, 2] if self.torso_id >= 0 else 1.0
        return bool(sunk or upright < 0.0)

    # ------------------------------------------------------------------ control

    def _pd_torques(self, target_qpos: np.ndarray, action: np.ndarray) -> np.ndarray:
        """Torques driving each joint toward the reference pose, offset by the policy's action.

        Acting as an offset from the reference (rather than an absolute target) means a
        zero action already tracks the clip, so the policy starts from something sane and
        learns the corrections that keep the humanoid upright.
        """
        torques = np.zeros(self.model.nu)
        for joint in self.layout:
            offset = action[joint["ctrl"]: joint["ctrl"] + joint["width"]] * self.action_scale
            velocity = self.data.qvel[joint["qvel"]: joint["qvel"] + joint["width"]]
            if joint["ball"]:
                current = self.data.qpos[joint["qpos"]: joint["qpos"] + 4]
                wanted = target_qpos[joint["qpos"]: joint["qpos"] + 4]
                error = rotvec_between(current, wanted) + offset
            else:
                error = (target_qpos[joint["qpos"]] - self.data.qpos[joint["qpos"]]) + offset
            torques[joint["ctrl"]: joint["ctrl"] + joint["width"]] = self.kp * error - self.kd * velocity
        return np.clip(torques, self.model.actuator_ctrlrange[:, 0], self.model.actuator_ctrlrange[:, 1])

    # ------------------------------------------------------------------ reward

    def _reward(self, frame: int) -> tuple[float, dict]:
        reference = self.reference[frame]
        pose_error = sum(
            float(np.sum(rotvec_between(self.data.qpos[j["qpos"]: j["qpos"] + 4],
                                        reference[j["qpos"]: j["qpos"] + 4]) ** 2)) if j["ball"]
            else float((reference[j["qpos"]] - self.data.qpos[j["qpos"]]) ** 2)
            for j in self.layout)
        velocity_error = float(np.mean((self.data.qvel[6:] - self.reference_velocity[frame][6:]) ** 2))
        root_error = float(np.sum((self.data.qpos[:3] - reference[:3]) ** 2))

        ee_error = 0.0
        if self.ee_ids:
            here = np.array([self.data.xpos[i] for i in self.ee_ids])
            saved = self.data.qpos.copy()
            self.data.qpos[:] = reference
            mujoco.mj_kinematics(self.model, self.data)
            there = np.array([self.data.xpos[i] for i in self.ee_ids])
            self.data.qpos[:] = saved
            mujoco.mj_kinematics(self.model, self.data)
            ee_error = float(np.mean(np.sum((here - there) ** 2, axis=1)))

        parts = {"pose": np.exp(-2.0 * pose_error), "velocity": np.exp(-0.1 * velocity_error),
                 "end_effector": np.exp(-40.0 * ee_error), "root": np.exp(-20.0 * root_error)}
        reward = (0.65 * parts["pose"] + 0.10 * parts["velocity"]
                  + 0.15 * parts["end_effector"] + 0.10 * parts["root"])
        return float(reward), {f"reward_{k}": float(v) for k, v in parts.items()}

    # ------------------------------------------------------------------ observation

    def _observe(self, frame: float) -> np.ndarray:
        """Own state, plus where the reference will be a few frames from now.

        Root x/y are dropped: a dance is the same dance wherever on the floor it happens, and
        the differences to the reference carry the position information that does matter.
        """
        own = np.concatenate([self.data.qpos[2:], self.data.qvel])
        ahead = [self._reference_at(frame + k * self.frames_per_step)[2:] - self.data.qpos[2:]
                 for k in LOOKAHEAD]
        phase = [frame / max(1, self.frames - 1)]
        return np.concatenate([own, *ahead, phase])

    # ------------------------------------------------------------------ rendering

    def render(self):
        if self.render_mode != "rgb_array":
            return None
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, 480, 640)
        self._renderer.update_scene(self.data)
        return self._renderer.render()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None


def make_env(motion: str | Path, **kwargs) -> DanceTrackingEnv:
    return DanceTrackingEnv(motion, **kwargs)


def _demo_motion() -> Path:
    """A retargeted clip to test against: the demos folder has two."""
    found = sorted((ROOT.parents[1] / "demos").glob("*/ground_truth_motion.npz"))
    if not found:
        raise SystemExit("No motion.npz found; pass --motion, or make one with "
                         "../pose-estimation/retarget.py")
    return found[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--motion", type=Path, help="retargeted motion .npz (default: a demos clip)")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--zero-action", action="store_true",
                        help="apply no offset, i.e. pure PD tracking of the reference")
    args = parser.parse_args()

    motion = args.motion or _demo_motion()
    env = DanceTrackingEnv(motion)
    print(f"{Path(motion).parent.name}: {env.frames} frames at {env.reference_fps:.1f} fps")
    print(f"obs {env.observation_space.shape}  action {env.action_space.shape}  "
          f"{env.substeps} physics steps per control step")

    from gymnasium.utils.env_checker import check_env
    check_env(env, skip_render_check=True)
    print("gymnasium env checker passed")

    for episode in range(args.episodes):
        obs, _ = env.reset(seed=episode)
        total, steps, parts = 0.0, 0, {}
        while True:
            action = np.zeros(env.action_space.shape) if args.zero_action else env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total += reward
            steps += 1
            parts = info
            if terminated or truncated:
                break
        print(f"episode {episode}: {steps:4d} steps, return {total:7.2f}, "
              f"mean {total / steps:.3f}, {'fell' if parts['fallen'] else 'finished the clip'}"
              f"  (pose {parts['reward_pose']:.2f} ee {parts['reward_end_effector']:.2f})")
    env.close()


if __name__ == "__main__":
    main()
