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
./run_session.sh                    # one ~1 h session: download -> prepare -> train
```

Or the stages by hand:

```bash
python3 ../../dataset/download_aist.py --situations sBM --cameras c01 --limit 600
uv run prepare_data.py --workers 4 --model lite     # the slow step
uv run train.py --minutes 55 --max-uses 5
uv run evaluate.py --baselines
uv run generate.py some_song.mp3 --retarget danced_motion.npz
```

## Running a session

`train.py` is built around a budget and a ledger rather than epochs:

- **`--minutes`** caps wall-clock time. The run stops cleanly at the cap, after the current
  shard, and Ctrl-C does the same.
- **`--max-uses`** (5) caps how often a clip may *ever* be trained on. The checkpoint counts
  each clip's passes, and every run takes the least-used clips first, so the dataset is
  consumed evenly and nothing is over-trained.
- **Resuming is the default.** Re-running the same command continues from
  `checkpoints/checkpoint.pt`: same step count, same optimiser state, same LR schedule,
  same held-out validation clips, and it automatically picks up clips prepared since.
  `--fresh` starts over. The learning rate is derived from the global step
  (warmup then cosine toward `--total-steps`), so a run that stops early doesn't distort it.
- **Outputs:** `checkpoint.pt` (resumable, includes optimiser + ledger), `model.pt` (the final
  weights, what `generate.py` and `evaluate.py` read) and `best.pt` (best validation loss).

Training stops early once every prepared clip has hit the cap. That's the normal outcome
here, not an error: prepare more clips, or raise `--max-uses`.

### Where an hour actually goes

Measured on an M-series laptop (MPS), default model, 16 s clips at 60 fps:

| Stage | Rate | 600 clips |
|---|---|---|
| Download (`c01` solo) | ~16 MB/clip | ~5 GB |
| MediaPipe prep | ~5 s/clip (2 workers), ~3 s (4) | 30-50 min |
| Training, 5 passes/clip | ~0.25 s/clip (80 windows/s) | **~2.5 min** |

Preprocessing dominates; the GPU is not the constraint. All of AIST's solo front-camera
footage (~1,700 clips) is about 7 minutes of training at the default size. So to spend an
hour on *compute* rather than data, either enlarge the model (`--d-model 768 --layers 12` is
~8x the cost, ~85M params) or overlap windows (`--stride 120` doubles the steps per clip).
Both trade against overfitting on a dataset this small, which is exactly what `--max-uses`
and the permanently held-out validation clips exist to expose.

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
   or the model can memorise a soundtrack it will be tested on. `train.py` currently holds out
   the first `--val-clips` clips by name, which is weaker than that.
2. **Autoregressive drift.** Feeding the model its own output compounds error, and dances tend
   to decay toward a mean pose after a few seconds. `--noise` mitigates it; a diffusion or
   VQ-VAE formulation (EDGE, Bailando) removes it properly, and is the upgrade I'd make first
   if generations collapse.
3. **One-to-many.** Many good dances fit one song, so a low `dance_similarity` against the
   original doesn't mean the dance is bad. Keep it as a training-progress signal, and judge
   quality with beat alignment plus your own eyes.
4. **Foot skate.** Nothing constrains feet to the floor. A foot-contact loss, or the MuJoCo
   tracking policy, is the fix.
