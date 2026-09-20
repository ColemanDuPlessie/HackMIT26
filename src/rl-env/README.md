# rl-env

A Gymnasium environment where a policy has to make the MuJoCo humanoid **physically perform** a
retargeted dance — torques, gravity and contact, rather than poses replayed frame by frame.

This is the one part of the project where RL earns its place. Everything upstream has a
supervised target: MediaPipe learns landmarks, IK solves poses, the music model regresses onto
real dances. Torques have no ground truth, and balance and contact can't be regressed onto
anyway, so the DeepMimic formulation is the right tool.

**This is a scaffold.** The env runs, passes Gymnasium's checker and trains under PPO, but no
policy has been trained to convergence — expect millions of steps for a single clip.

```bash
cd src/rl-env
uv sync
uv run dance_env.py                     # env checker + random-policy rollouts
uv run dance_env.py --zero-action       # pure PD tracking, the no-policy baseline
uv run train_ppo.py --check             # 2k steps, proves the PPO loop runs
uv run train_ppo.py --steps 2000000 --envs 8
```

Reference motion is any `motion.npz` from `../pose-estimation/retarget.py` — the two clips in
`../../demos/` are picked up automatically.

## The environment

| | |
|---|---|
| **Observation** (237) | root height, joint angles and velocities, and the reference pose 1, 4 and 10 frames ahead as *differences* from the current pose, plus clip phase |
| **Action** (34) | a target-angle offset per actuated DOF, in [-1, 1]. Zero means "track the reference exactly", so the policy learns corrections rather than a controller from scratch |
| **Reward** | DeepMimic's four terms — pose 0.65, end-effector 0.15, velocity 0.10, root 0.10 — each `exp(-k · error²)`, so 1.0 is perfect tracking |
| **Episode** | one clip, starting at a random frame, ending when the clip runs out or the humanoid falls |

Two details do most of the work: **reference state initialisation** (start anywhere in the clip,
so the policy sees the whole thing rather than only the first second) and **early termination**
(end the episode on a fall, so lying on the floor stops paying). Falling is judged *against the
reference*, not an absolute height — these clips crouch to 0.38 m and leave the floor, and a
fixed threshold calls both a fall.

## What the source model needed

`../pose-estimation/humanoid.xml` exists to be posed by IK, so three things had to change, all
in `humanoid_actuated.py` and none of them edits to the original file:

1. **No actuators at all** (`nu = 0`). One motor is injected per rotational DOF: three per ball
   joint, one per hinge, 34 in total, with per-body-part torque limits.
2. **Collisions disabled** (`contype="0"` on every body geom), which is right for IK and means
   the humanoid falls through the floor under physics.
3. **A 10 ms timestep and 0.01 armature**, fine for replaying poses and immediately unstable
   under torque control — MuJoCo reports "huge value in QACC" and silently resets, which looks
   exactly like actions having no effect. Now 2 ms, Newton solver, armature 0.05.

Reference velocities are also clipped to 20 rad/s: retargeting jitter produces spikes above
100 rad/s, and starting an episode on one throws the humanoid across the room.

## Where it stands

With no policy at all (`--zero-action`, pure PD tracking) the humanoid manages **0.6–0.9 s**
before falling, at ~0.2 mean reward. Random actions do slightly worse. That gap is the problem
PPO has to solve; a trained policy should hold the clip to the end at 0.6+.

Worth knowing before spending GPU time:

- **Throughput** is ~17 physics steps per control step at 30 Hz control. Use `--envs 8` or more;
  MJX or Brax is the answer if that isn't enough.
- **The reference is imperfect.** It comes from MediaPipe plus IK, so it contains jitter and
  poses no real body could hold. A policy that tracks it perfectly is not the goal.
- **Start with one clip.** Getting a single dance to hold up is the milestone; a policy that
  generalises across clips is a much larger undertaking.
