# rl-environment

Music in, dance out — as a Markov decision process. `DanceEnv` reuses the two pipelines that
already exist here, unchanged:

```
video -> pose                          audio -> model -> pose
(prepare_data.py cached clips)         (MotionTransformer.generate, cut open)
        |                                       |
        |  the reference dance:                 |  the rollout loop: each per-frame
        |  seeds each episode, scores it        |  pose delta becomes an *action*
        v                                       v
                        DanceEnv  ->  reward.py  ->  PPO (train_rl.py)
```

**Status: the environment is built and verified; no policy has been trained with it.** The
numbers below are measured, but they measure the *environment*, not a result.

## Why RL at all

`supervised-training/README.md` argues, correctly, that policy gradients are the wrong tool
for learning choreography from labelled clips: every frame of every clip is already a target,
and trading that for one scalar per rollout costs orders of magnitude in samples for nothing.
It then names the two jobs RL does earn — a MuJoCo policy that has to physically track a
trajectory, and fine-tuning the model on a non-differentiable reward, the way Bailando does.

This is the second job, and nothing else. **The policy is initialised from `model.pt` and
kept near it by a KL penalty.** A run is measured in minutes. The three things it targets are
exactly the three the supervised loss cannot express:

| failure | why L1 can't fix it | term here |
|---|---|---|
| dances off the beat | beat alignment is a discrete matching, not a gradient | `musicality` |
| decays to a mean pose after a few seconds | teacher forcing never shows the model its own drift | `energy`, and early termination |
| feet slide, limbs stretch | no term in the loss mentions the floor or bone length | `foot`, `bone` |

The first and third are `supervised-training`'s known weak points 2 and 4.

## The MDP

| | |
|---|---|
| **state** | the music around frame *t* with lookahead, the last `pose_history` poses, and the phase of the bar (seconds since the last beat, seconds to the next) |
| **action** | `a_t ∈ R^99`, the pose delta in normalised units |
| **transition** | `pose[t] = pose[t-1] + a_t`; the music advances one frame |
| **reward** | `reward.DanceReward`, below |
| **episode** | one clip, seeded with the dancer's real first 30 frames, ending with the music or early if the body comes apart |

Three choices are worth defending:

**The action is the supervised model's own output.** `MotionTransformer` predicts a *delta*
from the previous pose (`model.py`'s `self.delta`), and that delta is the action. So a trained
checkpoint is already a policy for this environment, with no adapter: fine-tuning starts from
a model that dances rather than from noise, and a zero-variance policy reproduces `generate()`
exactly. The smoke test asserts this — a fresh `ActorCritic`'s mean action is bit-for-bit the
supervised prediction.

**The state is a fixed window, not the whole prefix.** `generate()` re-attends over every
frame so far: O(T²), and not Markov, so a value function over it would be estimating something
that depends on history it cannot see. Here every step costs the same and bootstrapping is
sound. The cost is that the policy forgets what it did more than half a second ago, which is
real — see weak points.

**The music is visible ~0.5 s ahead of the pose.** A dancer hears the bar coming and moves
*into* the beat; a strictly causal policy can only ever react late, which caps `musicality`
below what a human scores. The supervised backbone is causal, so the lookahead enters through
a separate zero-initialised path (`policy.py`) that contributes nothing at step 0 and can be
learned into.

## Reward

Six terms, in `reward.py`, all computed on the same `(T, 33, 3)` hip-centred metres the rest
of the repo uses, so any of them can be checked against a real dance with no conversion.

| term | rewards / punishes |
|---|---|
| `musicality` | a *hit* — a sharp deceleration — landing within 100 ms of a beat, and charges for one that doesn't, so the policy can't farm it by stopping constantly |
| `imitate` | windowed `dance_similarity` against the reference (see below) |
| `bone` | limb segments changing length |
| `energy` | windowed speed leaving a band around the reference's; catches both collapse and flailing |
| `jerk` | third-difference jitter |
| `foot` | foot points moving horizontally while they're on the floor |

### The calibration invariant

**Replaying the real dance must score about zero on every penalty.** Each threshold is the
90th percentile of the reference dancer's *own* per-frame value for that quantity, so only
frames genuinely worse than their worst tenth cost anything.

This is not a detail. MediaPipe's own noise makes a real dancer's bone lengths wobble by ~5%
frame to frame; the first version of this file used absolute thresholds and charged the
reference dance 78 points per clip for measurement error, which would have taught a policy to
move *less* than a human in order to score well. `train_rl.py --smoke-test` asserts the
invariant.

`imitate` deliberately does not use `trajectory_similarity.dance_similarity` directly: that
function searches ±1 s for the best timing alignment, which is right for scoring a whole clip
but wrong as a reward, since a policy a beat behind the music would be shifted back into
agreement and paid in full. `windowed_similarity` is the same four terms with the same
weights over a ±0.15 s search.

`--w-imitate` is the dial between the two things this environment can be asked for: at 0 the
policy may invent any dance that fits the music, at 2 it is being asked to reproduce this
particular choreography and the RL is mostly drift correction.

## What it measures today

8 s of a held-out clip, `checkpoints/model.pt` (11.4M params, the current run):

| policy | return | what happened |
|---|---|---|
| the dancer's real motion, replayed | **+180.8** | penalties ≈ 0 (`bone` −0.4, `foot` −4.8, `jerk` −0.2), `musicality` +21.2 |
| frozen pose | −8.6 | `energy` −101: standing still is heavily punished |
| the supervised checkpoint | ≈ −27 | **terminated around frame 225 of 480** — bone error blew past the threshold |

That last row is the interesting one: the checkpoint's autoregressive rollout comes apart in
under four seconds, and its `musicality` is *negative* — it is decelerating off the beat more
often than on it. That is `supervised-training`'s weak point 2 turned into a number, and it is
the thing this environment exists to fix.

## Running it

```bash
cd src/rl-environment
uv sync
uv run train_rl.py --smoke-test          # no dataset needed: shapes, reward terms, 3 PPO updates
uv run train_rl.py --minutes 20          # fine-tune ../supervised-training/checkpoints/model.pt
uv run train_rl.py --w-imitate 0         # free dance instead of imitation
```

Outputs land in `runs/`: `policy.pt`, and `rollout.npz` in `extract_keypoints`' format, so an
episode goes straight into the existing viewer —

```bash
uv run ../pose-estimation/retarget.py runs/rollout.npz -o motion.npz --render out.mp4
```

**Watch the rollouts.** The reward is a proxy and a policy will find ways to score well that
do not look like dancing; `--kl-coef` is the brake, and your eyes are the detector.

By default the env runs on the checkpoint's *held-out* clips (`--split val`). Fine-tuning on
clips the supervised model was trained on would let RL recover memorised choreography and
report it as learning.

Throughput on an M-series CPU is ~230 env steps/s single-threaded, dominated by the policy
forward pass, so 16 envs in lockstep is roughly free. MPS helps once the batch is large.

## A second environment, not built: physical tracking

The other job RL earns. `retarget.py` already turns any dance — real or generated — into
`qpos` for `humanoid.xml`, which is the reference trajectory a DeepMimic-style tracking policy
needs. The env would be: state = humanoid `qpos`/`qvel` plus the next few reference frames,
action = joint torques, reward = the usual pose/velocity/end-effector/centre-of-mass tracking
terms, early termination on falling. That buys physically plausible contact, balance, and the
end of foot skate for free, which no kinematic term here can guarantee.

Two concrete things block it, both in `humanoid.xml`:

1. **There are no actuators.** The file has no `<actuator>` section at all — it is built for
   IK, where joints are set directly. Torque control needs a `<motor>` per joint, with gears,
   and ball joints need three.
2. **There are no contacts.** Every body geom is `contype="0" conaffinity="0"` (the `<default>`
   block); only the floor collides. Nothing would push back on a foot.

Both are additive — a separate `humanoid_actuated.xml` leaves the retargeting path untouched.
It is a bigger change than it looks, though: torque-controlled humanoid tracking is a
days-to-weeks project, not an afternoon, and it needs the kinematic model to be producing
decent trajectories first. Hence the ordering.

## Known weak points

Roughly in the order they'll bite:

1. **The reward is a proxy, and PPO is very good at proxies.** `musicality` rewards
   decelerating on the beat, which a metronomic twitch satisfies. `--kl-coef` (0.1) is what
   stops the policy walking away from the supervised prior; in the 4-minute CPU run above the
   KL still climbed steadily to ~2 nats, so it likely wants raising before any real run.
   Watch `runs/rollout.npz` rather than the return.
2. **Half a second of memory.** `pose_history=30` means the policy cannot represent an
   eight-count, which is the unit choreography is actually built from. Raising it is cheap in
   memory and quadratic in attention; the real fix is a recurrent state or a phase input
   derived from the bar, not a longer window.
3. **Credit assignment over 960 steps.** A 16 s clip at 60 fps is a long episode for a dense
   reward this noisy. `EnvConfig.frames_per_step` chunks several frames into one action and
   shortens the horizon, at the cost of the policy not reacting inside the chunk; `policy.py`
   only implements the single-frame case, so a chunked head is still to write.
4. **No global translation.** Inherited from MediaPipe world landmarks, which are hip-centred:
   the dancer never travels, so no reward term can mention where they are on the floor. Fixing
   it means a root-motion channel in `features.py`, which changes every stage.
5. **`imitate` costs real time.** Joint angles are computed per frame in Python
   (`trajectory_similarity.get_joint_angles`), so it is recomputed on a stride of 30 frames
   and held constant in between. Vectorising it would make a denser imitation signal affordable.
