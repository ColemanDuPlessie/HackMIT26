# supervised-training

Music in, dance out. A sequence model trained on the clips from [`dataset/download_aist.py`](../../dataset/download_aist.py):
the soundtrack is the input, the dancer's MediaPipe world landmarks (from
[`pose-estimation`](../pose-estimation)) are the target, and generations are scored with
[`reward-calculation`](../reward-calculation).

**This is a sketch:** every file runs and the shapes line up (`train.py --smoke-test` checks that
end to end), but nothing here has been trained on real data yet. Treat the hyperparameters as
starting points, not as tuned values.

## Why supervised and not RL

Every clip is a labelled example: this music, that dance. Policy gradients would throw away the
frame-by-frame supervision and learn from a single scalar per rollout, which costs orders of
magnitude more samples for no gain. RL earns its place in two other jobs: a MuJoCo policy that
has to *physically* track a trajectory (balance and contact have no supervised target), and
fine-tuning this model on a non-differentiable reward such as beat alignment, the way Bailando
does. Both are easier once this model exists.

## Pipeline

```
dataset/aist/clips/*.mp4
        |  prepare_data.py   (MediaPipe landmarks + librosa features, aligned per frame)
        v
cache/<clip>.npz             audio (T, 35), motion (T, 99), fps
        |  train.py
        v
checkpoints/model.pt         weights + normalisation stats + held-out clip names
        |  generate.py       any song -> keypoints .npz (-> retarget.py -> MuJoCo humanoid)
        |  evaluate.py       dance_similarity + beat alignment on held-out clips
```

```bash
cd src/supervised-training
uv sync
uv run train.py --smoke-test        # no dataset needed: checks shapes and that loss falls
uv run prepare_data.py              # the slow step: MediaPipe over every clip
uv run train.py --epochs 100
uv run evaluate.py --baselines
uv run generate.py some_song.mp3 --retarget danced_motion.npz
```

## Representation

- **Motion:** MediaPipe world landmarks, 33 joints x 3 axes = 99 numbers per frame, hip-centred
  metres. Chosen because `dance_similarity` already consumes exactly this, and `retarget.py`
  turns it into the MuJoCo humanoid, so a generation can be scored *and* watched with no glue code.
  The catch: world landmarks carry no global translation, so the dancer never travels. Fine for
  AIST's fixed-camera routines; a travelling dance needs a root-motion channel.
- **Audio:** the 35-dim AIST++ baseline feature (envelope, 20 MFCC, 12 chroma, onset one-hot,
  beat one-hot) at the motion frame rate, so `audio[t]` belongs to `motion[t]`.
- **Model:** a causal transformer predicting `pose[t]` from `audio[0..t]` and `pose[0..t-1]`,
  as a *delta* from the previous pose. ~11M parameters at the default size.
- **Loss:** L1 on position, velocity and acceleration. Position alone produces motion that is
  right on average and visibly jittery.

## Known weak points

Roughly in the order they'll bite:

1. **Data volume.** AIST++ baselines train on tens of hours. A handful of clips will overfit
   within minutes; hold out whole *songs* (the `mXXX` field of the clip name), not just clips,
   or the model can memorise a soundtrack it will be tested on.
2. **Autoregressive drift.** Feeding the model its own output compounds error, and dances tend
   to decay toward a mean pose after a few seconds. `--noise` mitigates it; a diffusion or
   VQ-VAE formulation (EDGE, Bailando) removes it properly, and is the upgrade I'd make first
   if generations collapse.
3. **One-to-many.** Many good dances fit one song, so a low `dance_similarity` against the
   original doesn't mean the dance is bad. Keep it as a training-progress signal, and judge
   quality with beat alignment plus your own eyes.
4. **Foot skate.** Nothing constrains feet to the floor. A foot-contact loss, or the MuJoCo
   tracking policy, is the fix.
