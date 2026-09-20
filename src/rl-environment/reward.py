"""Per-frame reward for the dance environment, over MediaPipe world landmarks.

Everything here consumes the same `(T, 33, 3)` hip-centred metres that `prepare_data.py`
caches, `reward-calculation` scores and `retarget.py` animates, so a reward term can be
sanity-checked against a real dance with no conversion.

Axis convention is MediaPipe's: x right, y *down*, z away from the camera. Height is
therefore `-y` (see `retarget.mp_to_mujoco`, which maps mujoco-z to `-y`), and the ground
plane is x-z. Getting this backwards silently inverts the foot-skate term, so it is spelled
out in `height()` rather than inlined.

Every penalty is calibrated against the *reference dancer's own* value for that quantity in
the clip being danced to, at a high percentile, so that **replaying the real dance scores
about zero on every penalty term**. That invariant is the whole point of the design and is
worth re-checking after any change here:

  * An absolute threshold would mean something different for a breakdance clip than for a
    waacking one, and one set of weights could not cover both.
  * MediaPipe's own noise makes a real dancer's bone lengths wobble by ~5% frame to frame. A
    penalty calibrated on anything tighter charges the reference tens of points per clip for
    measurement error, which silently teaches the policy to move *less* than a human.

So the thresholds are the reference's 90th percentile: only frames genuinely worse than the
real dancer's worst tenth cost anything.

The terms, and what failure each exists to punish:

    musicality   stopping off the beat            (the thing we actually want)
    imitate      not resembling the reference     (windowed `dance_similarity`)
    bone         limbs stretching                 (autoregressive drift's first symptom)
    energy       decaying to a mean pose, or flailing
    jerk         per-frame jitter
    foot         feet sliding while planted

"""

from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "reward-calculation"))
import trajectory_similarity  # noqa: E402

# Segments whose length a human cannot change. Face and hand points are left out: MediaPipe
# is noisy enough there that the "violation" would mostly be measurement error.
BONES = [
    (11, 12), (23, 24), (11, 23), (12, 24),           # shoulders, hips, torso sides
    (11, 13), (13, 15), (12, 14), (14, 16),           # arms
    (23, 25), (25, 27), (24, 26), (26, 28),           # legs
    (27, 29), (27, 31), (28, 30), (28, 32),           # feet
    (0, 11), (0, 12),                                 # neck
]
_BONE_A = np.array([a for a, _ in BONES])
_BONE_B = np.array([b for _, b in BONES])

FOOT_POINTS = [27, 28, 29, 30, 31, 32]  # ankles, heels, toes

EPS = 1e-8


@dataclass
class RewardConfig:
    """Weights and thresholds. Defaults are a starting point, not tuned values.

    `w_imitate` is the dial between the two things this environment can be asked for: at 0 the
    policy is free to invent any dance that fits the music, at 2 it is being asked to reproduce
    this particular choreography and the RL is mostly a drift-correction on the supervised model.
    """

    w_musicality: float = 1.0
    w_imitate: float = 0.3
    w_bone: float = 1.0
    w_energy: float = 0.5
    w_jerk: float = 0.2
    w_foot: float = 0.3
    alive: float = 0.1  # paid every surviving frame, so early termination actually costs something

    beat_tolerance_s: float = 0.10   # same tolerance evaluate.py's beat_alignment uses
    off_beat_scale: float = 0.5      # a hit off the beat costs this much of what an on-beat hit earns
    energy_band: tuple[float, float] = (0.5, 2.5)  # allowed multiples of the reference's speed
    tolerance_percentile: float = 90.0  # of the reference's own frames, below which a penalty is 0
    jerk_allowance: float = 1.5      # further multiples of that threshold that cost nothing
    contact_height: float = 0.08     # metres above the floor still treated as a planted foot
    imitate_every: int = 30          # frames between (expensive) dance_similarity calls
    imitate_window_s: float = 2.0
    bone_terminate: float = 0.30     # bone error *above the reference's own* that ends the episode
    terminate_penalty: float = 5.0


def windowed_similarity(reference: np.ndarray, generated: np.ndarray, fps: float,
                        max_shift_s: float = 0.15) -> float:
    """`dance_similarity` over a short window, with a short timing search.

    `trajectory_similarity.dance_similarity` hard-codes a +/-1 s alignment search, which is
    right when scoring a whole clip against a whole clip but wrong as a reward: a policy a
    beat behind the music would be shifted back into agreement and paid in full. Same four
    terms and the same equal weighting, over a timing window narrow enough that lateness
    still costs something.
    """
    ts = trajectory_similarity
    ref, gen, timing, _, _ = ts.align_timing(reference, generated, fps, max_shift_seconds=max_shift_s)
    position, _ = ts.position_similarity(ref, gen)
    angle, _ = ts.angle_similarity(ref, gen)
    trajectory, _ = ts.trajectory_similarity(ref, gen)
    return float((position * angle * trajectory * timing) ** 0.25)


def height(world: np.ndarray) -> np.ndarray:
    """Metres above the hips. MediaPipe's y axis points down, so height is -y."""
    return -world[..., 1]


def bone_lengths(world: np.ndarray) -> np.ndarray:
    """(..., 33, 3) -> (..., n_bones) segment lengths in metres."""
    return np.linalg.norm(world[..., _BONE_A, :] - world[..., _BONE_B, :], axis=-1)


def joint_speed(world: np.ndarray, fps: float) -> np.ndarray:
    """(T, 33, 3) -> (T,) mean joint speed in m/s, with speed[0] = 0."""
    step = np.linalg.norm(np.diff(world, axis=0), axis=2).mean(axis=1) * fps
    return np.concatenate([[0.0], step])


def foot_skate(world: np.ndarray, fps: float, floor: float, contact_height: float) -> np.ndarray:
    """(T,) horizontal distance travelled per second by foot points that are on the floor.

    A planted foot should not move. Points are weighted by how close to the floor they are, so
    the term fades out rather than switching on and off between frames.
    """
    feet = world[:, FOOT_POINTS, :]
    horizontal = np.linalg.norm(np.diff(feet[:, :, [0, 2]], axis=0), axis=2) * fps
    contact = np.clip(1.0 - (height(feet[1:]) - floor) / max(contact_height, EPS), 0.0, 1.0)
    return np.concatenate([[0.0], (horizontal * contact).sum(axis=1)])


class DanceReward:
    """Reward for one episode: one reference clip, one soundtrack.

    Constructed from the reference dance so every term can be expressed relative to it, then
    stepped frame by frame as the policy writes into a shared `(T, 33, 3)` buffer.
    """

    def __init__(self, reference: np.ndarray, audio: np.ndarray, fps: float,
                 cfg: RewardConfig | None = None):
        self.cfg = cfg or RewardConfig()
        self.reference = np.asarray(reference, dtype=np.float64)
        self.fps = float(fps)

        # --- what the real dancer did, used to calibrate the penalties --------------------
        # Each threshold is a high percentile of the reference's own per-frame value, so
        # replaying the reference costs ~nothing and only genuinely worse frames are charged.
        tol = self.cfg.tolerance_percentile
        self.rest_bones = np.median(bone_lengths(self.reference), axis=0)
        ref_bone = self._bone_error(self.reference)
        self.bone_tolerance = float(np.percentile(ref_bone, tol))

        ref_speed = joint_speed(self.reference, self.fps)
        self.ref_speed = max(float(ref_speed.mean()), EPS)
        self.hit_scale = max(float(np.abs(np.diff(ref_speed)).mean()), EPS)
        self.ref_jerk = max(float(np.percentile(self._jerk(self.reference), tol)), EPS)
        # 5th percentile rather than the minimum: one bad MediaPipe frame should not define the floor.
        self.floor = float(np.percentile(height(self.reference[:, FOOT_POINTS, :]), 5))
        ref_skate = foot_skate(self.reference, self.fps, self.floor, self.cfg.contact_height)
        self.ref_skate = max(float(np.percentile(ref_skate, tol)), EPS)

        # --- where the music's beats are --------------------------------------------------
        beats = np.flatnonzero(audio[:, -1] > 0.5)  # the beat one-hot from features.py
        tolerance = max(1, int(round(self.cfg.beat_tolerance_s * self.fps)))
        self.on_beat = np.zeros(len(audio), dtype=bool)
        for b in beats:
            self.on_beat[max(0, b - tolerance): b + tolerance + 1] = True
        self.beats = beats

        self.window = max(4, int(round(self.cfg.imitate_window_s * self.fps)))
        self._imitate = 0.0  # held between the frames that actually recompute it

    def _bone_error(self, world: np.ndarray) -> np.ndarray:
        """Mean relative deviation of each frame's bones from this dancer's own skeleton."""
        return (np.abs(bone_lengths(world) - self.rest_bones).mean(axis=-1)
                / max(self.rest_bones.mean(), EPS))

    @staticmethod
    def _jerk(world: np.ndarray) -> np.ndarray:
        """Mean per-joint third difference; the quantity a human eye reads as jitter."""
        if len(world) < 4:
            return np.zeros(max(len(world), 1))
        return np.linalg.norm(np.diff(world, n=3, axis=0), axis=2).mean(axis=1)

    def step(self, world: np.ndarray, t: int) -> tuple[float, dict, bool]:
        """Reward for frame `t` of `world`, which must be filled up to and including `t`.

        Returns (reward, per-term components for logging, terminated).
        """
        cfg = self.cfg
        frame = world[t]
        if not np.isfinite(frame).all():
            return -cfg.terminate_penalty, {"nonfinite": 1.0}, True

        parts: dict[str, float] = {}

        # Bone lengths: stretch beyond what MediaPipe's own noise produces on this dancer.
        bone_err = float(self._bone_error(frame))
        parts["bone"] = -cfg.w_bone * max(0.0, bone_err - self.bone_tolerance)
        if bone_err > self.bone_tolerance + cfg.bone_terminate:
            parts["terminate"] = -cfg.terminate_penalty
            return sum(parts.values()), parts, True

        parts["alive"] = cfg.alive

        # Musicality: a "hit" is a sharp deceleration, which is what reads as dancing *to* a
        # beat. Reward it when it lands on one and charge for it when it doesn't, so the
        # policy cannot farm the term by stopping constantly.
        if t >= 2:
            speed = joint_speed(world[t - 2: t + 1], self.fps)
            hit = max(0.0, float(speed[1] - speed[2])) / self.hit_scale
            sign = 1.0 if self.on_beat[min(t, len(self.on_beat) - 1)] else -cfg.off_beat_scale
            parts["musicality"] = cfg.w_musicality * sign * min(hit, 3.0)

        # Energy: punish both collapse toward a mean pose (the autoregressive failure mode)
        # and flailing, but nothing inside the band, so the policy picks its own dynamics.
        lo, hi = cfg.energy_band
        start = max(0, t - self.window + 1)
        if t - start >= 2:
            level = float(joint_speed(world[start: t + 1], self.fps)[1:].mean()) / self.ref_speed
            parts["energy"] = -cfg.w_energy * (max(0.0, lo - level) + max(0.0, level - hi))

        if t >= 3:
            excess = float(self._jerk(world[t - 3: t + 1])[0]) / self.ref_jerk - cfg.jerk_allowance
            parts["jerk"] = -cfg.w_jerk * max(0.0, excess)

        if t >= 1:
            skate = float(foot_skate(world[t - 1: t + 1], self.fps, self.floor,
                                     cfg.contact_height)[1]) / self.ref_skate
            parts["foot"] = -cfg.w_foot * max(0.0, skate - 1.0)

        # Imitation is the expensive term (an O(shift x window) alignment plus per-frame joint
        # angles), so it is recomputed on a stride and held constant in between. It is a
        # *level*, not an increment: paying it every frame is deliberate, since the policy
        # should be rewarded for staying near the reference rather than for arriving once.
        if cfg.w_imitate and t >= self.window and t % cfg.imitate_every == 0:
            sl = slice(t - self.window + 1, t + 1)
            self._imitate = windowed_similarity(self.reference[sl], world[sl], self.fps)
        if cfg.w_imitate:
            parts["imitate"] = cfg.w_imitate * self._imitate

        return float(sum(parts.values())), parts, False
