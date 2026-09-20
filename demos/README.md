# demos

Two paired examples from the v2 model (`src/supervised-training/checkpoints-v2/model.pt`,
25 sweeps over 1,024 clips): what the dancer actually did, and what the model invents from
that clip's **audio alone**.

| | Split | Clip | dance_similarity |
|---|---|---|---|
| `val_gJS_sBM_c01_d03_mJS2_ch01_00` | held out | jazz, dancer d03, song mJS2 | **0.402** |
| `train_gJS_sBM_c01_d01_mJS0_ch01_00` | trained on | jazz, dancer d01, song mJS0 | **0.483** |

The held-out clip is the *median* of the ten scored validation clips, not the best one; its
song and its dancer appear nowhere in training. The training clip is the same genre for a
like-for-like comparison. The gap between them (0.402 vs 0.483) is roughly the model's
memorisation advantage on material it has seen.

For reference, on these held-out clips a frozen pose scores 0.371 and an unrelated dance 0.284.

## What's in each folder

| File | |
|---|---|
| `ground_truth.mp4` | the source clip, with its music |
| `ground_truth_keypoints.npz` | MediaPipe world landmarks, as `extract_keypoints` writes them |
| `ground_truth_motion.npz` | those landmarks fitted to the MuJoCo humanoid |
| `generated_keypoints.npz` | the model's dance for the same audio |
| `generated_motion.npz` | the same, fitted to the humanoid |
| `scores.json` | dance_similarity (with its parts) and beat alignment |

Watch either side:

```bash
cd src/pose-estimation
uv run visualize.py ../../demos/val_gJS_sBM_c01_d03_mJS2_ch01_00/generated_motion.npz
```

**One caveat about "from audio alone":** the rollout is seeded with the first 30 frames (0.5 s)
of the real dance, because the model predicts each pose from the previous one and needs somewhere
to start. Everything after that half second is the model's own, driven by the music.

Regenerate or add more:

```bash
cd src/supervised-training
uv run export_demo.py <clip_name> --label val    # or --label train
```
