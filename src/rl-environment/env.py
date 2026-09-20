"""`DanceEnv`: the music -> dance model's rollout, framed as a Markov decision process.

The environment is built out of the two pipelines that already exist in this repo, and adds
nothing to either:

    video -> pose   `prepare_data.py` cached (audio, motion) pairs. The motion is the
                    *reference* dance: it seeds each episode and scores it (`reward.py`).
    audio -> pose   the rollout loop from `MotionTransformer.generate`, cut open so that the
                    per-frame pose delta becomes an action instead of a forward pass.

The MDP:

    state       the music around frame t (with lookahead) and the recent pose history
    action      a_t in R^99, the pose delta in *normalised* units --- exactly what
                `MotionTransformer.delta` emits, so a supervised checkpoint is already a
                policy for this environment and a fresh run starts from it rather than from
                noise (see `policy.py`)
    transition  pose[t] = pose[t-1] + a_t; the music advances one frame
    reward      `reward.DanceReward` --- musicality, imitation, and the plausibility terms
    episode     one clip, seeded with the dancer's real first `seed_frames`, ending at the
                end of the music or early if the body comes apart

Two differences from `generate()` are deliberate:

  * **The state is a fixed window, not the whole prefix.** `generate()` re-attends over every
    frame so far, which is O(T^2) and non-Markov. Here the policy sees the last
    `pose_history` frames, so every step costs the same and value bootstrapping is sound.
  * **The music is visible slightly ahead of the pose.** `audio_future` frames of lookahead
    are in the observation: a dancer hears the bar coming and moves *into* the beat, and a
    strictly causal policy can only ever react late. The supervised model is causal, so the
    lookahead arrives as a zero-initialised extra input (`policy.py`) that starts out doing
    nothing.

Usage:

    from env import DanceEnv, VecDanceEnv
    env = DanceEnv.from_checkpoint(Path("../supervised-training/checkpoints/model.pt"))
    obs, info = env.reset()
    obs, reward, terminated, truncated, info = env.step(action)
    env.save_episode(Path("episode.npz"))   # -> retarget.py / the web UI, like any clip
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "supervised-training"))

import features  # noqa: E402
from dataset import Stats, load_clips  # noqa: E402

from reward import DanceReward, RewardConfig  # noqa: E402

try:  # gymnasium is an optional dependency: the env is usable (and testable) without it
    from gymnasium import spaces
    HAVE_GYM = True
except ImportError:  # pragma: no cover - exercised only where gymnasium is absent
    HAVE_GYM = False

    class _Box:
        def __init__(self, low, high, shape, dtype=np.float32):
            self.low, self.high, self.shape, self.dtype = low, high, tuple(shape), dtype

        def __repr__(self):
            return f"Box({self.low}, {self.high}, {self.shape})"

    class _Dict(dict):
        pass

    class spaces:  # type: ignore[no-redef]
        Box, Dict = _Box, _Dict


@dataclass
class EnvConfig:
    pose_history: int = 30      # frames of pose in the observation (0.5 s at 60 fps)
    audio_future: int = 30      # frames of lookahead; the dancer hears the bar coming
    frames_per_step: int = 1    # frames emitted per action (see `step`)
    seed_frames: int = 30       # real frames that start the rollout
    max_seconds: float = 0.0    # 0 = dance the whole clip
    action_limit: float = 5.0   # clip on the delta, in normalised units
    beat_horizon_s: float = 2.0  # cap on the "seconds to the next beat" observation


class DanceEnv:
    """One clip per episode. Gymnasium's five-tuple API, without requiring gymnasium."""

    def __init__(self, clips: list[dict], stats: Stats, cfg: EnvConfig | None = None,
                 reward_cfg: RewardConfig | None = None, seed: int = 0):
        if not clips:
            raise ValueError("DanceEnv needs at least one prepared clip.")
        self.clips, self.stats = clips, stats
        self.cfg = cfg or EnvConfig()
        self.reward_cfg = reward_cfg or RewardConfig()
        self.rng = np.random.default_rng(seed)

        if self.cfg.seed_frames < 1:
            raise ValueError("seed_frames must be at least 1: the rollout needs a pose to start from.")

        c = self.cfg
        self.observation_space = spaces.Dict({
            "audio": spaces.Box(-np.inf, np.inf, (c.pose_history + c.audio_future, features.AUDIO_DIM)),
            "pose": spaces.Box(-np.inf, np.inf, (c.pose_history, features.MOTION_DIM)),
            "beat": spaces.Box(0.0, c.beat_horizon_s, (2,)),
        })
        self.action_space = spaces.Box(-c.action_limit, c.action_limit,
                                       (c.frames_per_step, features.MOTION_DIM))
        self.clip = None

    # ------------------------------------------------------------------ construction
    @classmethod
    def from_checkpoint(cls, checkpoint: Path, cache: Path | None = None,
                        split: str = "val", **kwargs) -> "DanceEnv":
        """Build an env over a checkpoint's held-out clips, with its normalisation statistics.

        Reusing the checkpoint's `val_clips` matters: fine-tuning on clips the supervised
        model was trained on would let RL recover memorised choreography rather than learn
        anything, and the `split="val"` default keeps the two stages honest about the same
        held-out songs.
        """
        import torch

        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
        cache = cache or ROOT.parent / "supervised-training" / "cache"
        held_out = set(ckpt["val_clips"])
        names = sorted(p.stem for p in Path(cache).glob("*.npz"))
        if split == "val":
            names = [n for n in names if n in held_out]
        elif split == "train":
            names = [n for n in names if n not in held_out]
        elif split != "all":
            raise ValueError(f"split must be train, val or all; got {split!r}")
        if not names:
            raise FileNotFoundError(f"No {split} clips in {cache}.")
        return cls(load_clips(Path(cache), names), Stats.from_dict(ckpt["stats"]), **kwargs)

    # ------------------------------------------------------------------ episode
    def reset(self, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        name = (options or {}).get("clip")
        if name is not None:
            self.clip = next(c for c in self.clips if c["name"] == name)
        else:
            self.clip = self.clips[self.rng.integers(len(self.clips))]

        clip, cfg = self.clip, self.cfg
        self.fps = float(clip["fps"])
        self.audio_raw = clip["audio"]
        self.audio = self.stats.normalise_audio(clip["audio"]).astype(np.float32)

        length = len(clip["motion"])
        if cfg.max_seconds:
            length = min(length, cfg.seed_frames + int(round(cfg.max_seconds * self.fps)))
        if length <= cfg.seed_frames + 1:
            raise ValueError(f"{clip['name']} is too short for seed_frames={cfg.seed_frames}.")
        self.length = length

        # The rollout buffer, in both representations: normalised (what actions act on) and
        # metres (what the reward reads). Seeded with what the dancer actually did.
        self.motion = np.zeros((length, features.MOTION_DIM), dtype=np.float32)
        self.world = np.zeros((length, features.NUM_LANDMARKS, 3), dtype=np.float64)
        real = clip["motion"][:length]
        self.motion[: cfg.seed_frames] = self.stats.normalise_motion(real[: cfg.seed_frames])
        self.world[: cfg.seed_frames] = features.unflatten_motion(real[: cfg.seed_frames])

        self.reward = DanceReward(features.unflatten_motion(real), clip["audio"][:length],
                                  self.fps, self.reward_cfg)
        self.beat_seconds = self._beat_distances(self.reward.beats, length)

        self.t = cfg.seed_frames - 1  # index of the last filled frame
        self.total_reward = 0.0
        return self._obs(), {"clip": clip["name"], "frames": length, "fps": self.fps}

    def step(self, action: np.ndarray):
        """Apply `frames_per_step` pose deltas.

        Chunking several frames into one action shortens the horizon (a 16 s clip at 60 fps is
        960 single-frame steps) at the cost of the policy not being able to react inside the
        chunk. One frame per step is the honest default; raise it if credit assignment over a
        full clip turns out to be the binding constraint.
        """
        if self.clip is None:
            raise RuntimeError("Call reset() before step().")
        cfg = self.cfg
        deltas = np.asarray(action, dtype=np.float32).reshape(cfg.frames_per_step,
                                                              features.MOTION_DIM)
        deltas = np.clip(deltas, -cfg.action_limit, cfg.action_limit)

        reward, terminated, parts = 0.0, False, {}
        for delta in deltas:
            t = self.t + 1
            self.motion[t] = self.motion[t - 1] + delta
            self.world[t] = features.unflatten_motion(
                self.stats.denormalise_motion(self.motion[t][None]))[0]
            self.t = t

            step_reward, parts, terminated = self.reward.step(self.world, t)
            reward += step_reward
            if terminated or t >= self.length - 1:
                break

        self.total_reward += reward
        truncated = not terminated and self.t >= self.length - 1
        info = {"terms": parts, "frame": self.t, "clip": self.clip["name"]}
        if terminated or truncated:
            info["episode"] = {"r": self.total_reward, "l": self.t - cfg.seed_frames + 1,
                               "frac": (self.t + 1) / self.length}
        return self._obs(), reward, terminated, truncated, info

    # ------------------------------------------------------------------ observation
    def _obs(self) -> dict:
        c, t = self.cfg, self.t
        return {
            # Audio aligned with the pose history, plus lookahead past the current frame.
            "audio": _window(self.audio, t - c.pose_history + 1, t + c.audio_future + 1),
            "pose": _window(self.motion, t - c.pose_history + 1, t + 1),
            "beat": self.beat_seconds[min(t, self.length - 1)],
        }

    def _beat_distances(self, beats: np.ndarray, length: int) -> np.ndarray:
        """(length, 2): seconds since the previous beat and until the next one, both capped.

        Phase within the bar is the single most useful thing to tell a dancer, and deriving it
        from the one-hot channel inside the observation window would cost the policy capacity
        for no reason.
        """
        horizon = self.cfg.beat_horizon_s
        out = np.full((length, 2), horizon, dtype=np.float32)
        if len(beats) == 0:
            return out
        frames = np.arange(length)
        after = np.searchsorted(beats, frames, side="right")
        prev_ok, next_ok = after > 0, after < len(beats)
        out[prev_ok, 0] = (frames[prev_ok] - beats[after[prev_ok] - 1]) / self.fps
        out[next_ok, 1] = (beats[after[next_ok]] - frames[next_ok]) / self.fps
        return np.clip(out, 0.0, horizon)

    # ------------------------------------------------------------------ output
    def save_episode(self, path: Path) -> Path:
        """Write the episode as an `extract_keypoints`-shaped .npz.

        The same file `generate.py` writes, so an episode goes straight into
        `pose-estimation/retarget.py` and the web UI: watching a rollout is the only way to
        catch a policy that is farming the reward while dancing badly.
        """
        np.savez(path, **features.to_keypoints(self.world[: self.t + 1].astype(np.float32), self.fps))
        return path


def _window(array: np.ndarray, start: int, stop: int) -> np.ndarray:
    """`array[start:stop]` with edge padding, so observations near either end keep their shape."""
    index = np.clip(np.arange(start, stop), 0, len(array) - 1)
    return array[index].astype(np.float32)


class VecDanceEnv:
    """`n` independent `DanceEnv`s stepped in lockstep, with automatic reset.

    A plain Python loop: the cost of a step here is the policy's forward pass over the whole
    batch, not the environment, so there is nothing to gain from subprocesses.

    On a finished episode the returned observation is already the *next* episode's, and the
    terminal one is handed back in `info["final_obs"]`. PPO needs that distinction: a
    truncated episode (the music ran out) must bootstrap its value, a terminated one (the
    body came apart) must not.
    """

    def __init__(self, clips: list[dict], stats: Stats, n: int = 8, cfg: EnvConfig | None = None,
                 reward_cfg: RewardConfig | None = None, seed: int = 0):
        self.envs = [DanceEnv(clips, stats, cfg, reward_cfg, seed=seed + i) for i in range(n)]
        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space
        self.n = n
        self.last_obs: dict | None = None

    def reset(self):
        self.last_obs = _stack([env.reset()[0] for env in self.envs])
        return self.last_obs

    def step(self, actions: np.ndarray):
        obs, rewards, terminated, truncated, infos = [], [], [], [], []
        for env, action in zip(self.envs, actions):
            o, r, term, trunc, info = env.step(action)
            if term or trunc:
                info["final_obs"] = o
                o, _ = env.reset()
            obs.append(o)
            rewards.append(r)
            terminated.append(term)
            truncated.append(trunc)
            infos.append(info)
        return (_stack(obs), np.array(rewards, dtype=np.float32),
                np.array(terminated), np.array(truncated), infos)


def _stack(observations: list[dict]) -> dict:
    return {key: np.stack([o[key] for o in observations]) for key in observations[0]}
