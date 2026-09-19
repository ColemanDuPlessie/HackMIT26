# pose-estimation

Single-person video → MuJoCo `qpos`, using MediaPipe Pose Landmarker for 3D keypoints and
[mink](https://github.com/kevinzakka/mink) for inverse kinematics.

## Setup

Requires [uv](https://docs.astral.sh/uv/). Python 3.10–3.12 (tested on 3.12; MediaPipe wheels lag new Python releases);
uv fetches a compatible interpreter automatically.

```bash
cd src/pose-estimation
uv sync
```

MediaPipe is pinned to `<0.11`: version 1.0.1 crashes on macOS in its Metal helper even with the CPU delegate.

## Usage

```bash
# 1. Video -> MediaPipe landmarks (downloads the model to models/ on first run)
uv run extract_keypoints.py input.mp4 -o keypoints.npz          # --model lite|full|heavy

# 2. Landmarks -> MuJoCo qpos for humanoid.xml
uv run retarget.py keypoints.npz -o motion.npz --render motion.mp4

# 3. Play back a saved pose file (macOS needs mjpython for the interactive viewer)
uv run mjpython visualize.py motion.npz
uv run visualize.py motion.npz --render motion.mp4          # headless video
uv run visualize.py motion.npz --render frame.png --start 120  # single frame
```

Viewer controls: **Space** pause/resume, **←/→** step a frame, **↑/↓** double/halve speed, **Home** restart.
The overlay shows frame, time, speed and IK error. Other flags: `--speed`, `--no-targets`, `--no-follow`,
`--model` (defaults to the model path saved in the file, falling back to `humanoid.xml`).

`motion.npz` contains `qpos (T, nq)`, `targets (T, 33, 3)`, per-frame IK `error`, `fps`, and `joint_names`.
Green spheres in the render/viewer are the IK targets.

## Using from Python

The package is installable (`web-ui` uses it as an editable dependency). Main entry points:

```python
import extract_keypoints, retarget
kp = extract_keypoints.extract("input.mp4", extract_keypoints.ensure_model("full"), progress=print)
motion = retarget.retarget(kp, smooth=0.2, progress=print)   # same contents as motion.npz
```

## Files

- `extract_keypoints.py`: runs MediaPipe in VIDEO mode, saves hip-centered metric world landmarks,
  normalized image landmarks, and per-landmark visibility.
- `humanoid.xml`: simple humanoid (free root; ball joints at spine, neck, shoulders, wrists, hips, ankles;
  hinge elbows/knees). Sites `mp_<i>` mark where MediaPipe landmark `i` sits on the body.
- `retarget.py`: axis conversion → smoothing → rescale skeleton to model bone lengths → foot grounding → per-frame IK.
  `--render`/`--view` are shortcuts to `visualize.py`.
- `visualize.py`: plays back a saved `motion.npz` in the MuJoCo viewer, or renders it to video/image.

## Using a different model

Any MJCF works if it has sites named `mp_<i>` for landmarks 0, 7, 8, and 11–32. Bone lengths are read
from the model's rest pose, so targets are rescaled to fit it. Pass it with `--model path/to/model.xml`.
If the model has no free joint at the root, `solve()` in `retarget.py` needs adjusting (it assumes
`qpos[0:7]` is the root).

## Limitations

- **No horizontal root motion.** MediaPipe world landmarks are hip-centered, so the character stays in place;
  only vertical motion is recovered (via foot grounding). The `image` landmarks in `keypoints.npz` could be used
  to estimate translation.
- **Monocular depth is noisy**, so expect errors for limbs pointing toward/away from the camera.
- **Kinematic only**: no physics, so foot sliding and impossible balance are not corrected.
- Limb twist is weakly constrained (resolved by a posture prior and the hand/foot points).
