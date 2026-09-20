# demos

Two paired examples from the v2 model (`src/supervised-training/checkpoints-v2/model.pt`,
25 sweeps over 1,024 clips): what the dancer actually did, and what the model invents from
that clip's **audio alone**.

Both clips are the **middle chunk** (`_01` of three) of a longer `sFM` recording, so the dancing
is underway rather than an intro.

| | Split | Clip | Model | Frozen pose | Wrong dance |
|---|---|---|---|---|---|
| `val_gJB_sFM_c01_d07_mJB0_ch01_01` | held out (0 passes) | break, dancer d07, song mJB0 | **0.160** | 0.111 | 0.064 |
| `train_gJS_sFM_c01_d01_mJS0_ch01_01` | trained (25 passes) | jazz, dancer d01, song mJS0 | **0.150** | 0.153 | 0.065 |

## Read these numbers carefully

They look far worse than the 0.40 the model averages on its held-out set, and the reason is
instructive: **most of that set is barely moving.** These middles average 3.66 and 2.05 cm of
joint movement per frame; the `sBM` opening chunks the score is usually computed on average
**0.36 cm/frame** — ten times stiller, often a dancer standing and waiting for the music. On a
near-static clip a frozen pose scores 0.45 by itself, and beating it is easy.

On genuinely active dancing the picture is sober: the model beats a frozen pose by half on the
held-out clip (0.160 vs 0.111) and merely *ties* it on a clip it trained on 25 times
(0.150 vs 0.153). It is well clear of an unrelated dance in both cases (0.064, 0.065), so it is
not producing noise — it is producing plausible but wrong choreography.

Two consequences worth acting on:

1. The headline 0.404 is inflated by low-motion clips. A movement-weighted metric, or simply
   reporting motion alongside score, would stop that.
2. These are `sFM` ("advanced") recordings while training was ~97% `sBM` ("basic"), so they are
   also out of distribution. The two effects are tangled here and worth separating.

## What's in each folder

| File | |
|---|---|
| `ground_truth.mp4` | the source clip, with its music |
| `ground_truth_keypoints.npz` | MediaPipe world landmarks, as `extract_keypoints` writes them |
| `ground_truth_motion.npz` | those landmarks fitted to the MuJoCo humanoid |
| `generated_keypoints.npz` | the model's dance for the same audio |
| `generated_motion.npz` | the same, fitted to the humanoid |
| `ground_truth_humanoid.mp4` | the pose estimate from the video, rendered on the humanoid, with the song |
| `generated_humanoid.mp4` | the model's dance, rendered the same way, with the same song |
| `scores.json` | dance_similarity (with its parts) and beat alignment |

The two rendered mp4s are the quickest way to judge a result: same music, same camera, one driven
by the real dancer's pose estimate and one by the model.

```bash
cd src/pose-estimation
uv run visualize.py ../../demos/val_gJB_sFM_c01_d07_mJB0_ch01_01/generated_motion.npz
```

**One caveat about "from audio alone":** the rollout is seeded with the first 30 frames (0.5 s) of
the real dance, because the model predicts each pose from the previous one and needs somewhere to
start. Everything after that half second is the model's own, driven by the music.

Regenerate, or add more:

```bash
cd src/supervised-training
uv run export_demo.py <clip_name> --label val        # or --label train
cd ../pose-estimation                                 # then render each side
uv run visualize.py <demo>/generated_motion.npz --render out.mp4 --no-targets
ffmpeg -i out.mp4 -i <demo>/ground_truth.mp4 -map 0:v:0 -map 1:a:0 \
  -c:v libx264 -pix_fmt yuv420p -c:a aac -shortest <demo>/generated_humanoid.mp4
```

(the re-encode is because `visualize.py` writes mp4v, which many players and browsers refuse).
