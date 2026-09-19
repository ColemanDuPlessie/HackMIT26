"""Extract per-frame 3D pose landmarks from a single-person video with MediaPipe.

Usage:
    uv run extract_keypoints.py input.mp4 -o keypoints.npz [--model heavy]

Output .npz contents:
    world       (T, 33, 3) metric landmarks, hip-centered, camera axes (x right, y down, z away)
    image       (T, 33, 3) normalized image coords (x, y in [0, 1]; z relative depth)
    visibility  (T, 33)    per-landmark visibility in [0, 1]
    fps         scalar
Frames with no detection are NaN in world/image and 0 in visibility.
"""

import argparse
import urllib.request
from pathlib import Path
from typing import Callable

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions, RunningMode

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_{0}/float16/latest/pose_landmarker_{0}.task"
)
MODEL_DIR = Path(__file__).parent / "models"
NUM_LANDMARKS = 33


def ensure_model(variant: str) -> Path:
    path = MODEL_DIR / f"pose_landmarker_{variant}.task"
    if not path.exists():
        MODEL_DIR.mkdir(exist_ok=True)
        print(f"Downloading {path.name}...")
        urllib.request.urlretrieve(MODEL_URL.format(variant), path)
    return path


def extract(video_path: str, model_path: Path,
            progress: Callable[[int, int], None] | None = None) -> dict:
    """Run MediaPipe on every frame. `progress(done, total)` is called per frame if given."""
    opts = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path), delegate=BaseOptions.Delegate.CPU),
        running_mode=RunningMode.VIDEO,
        num_poses=1,
    )
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    world, image, vis = [], [], []
    with PoseLandmarker.create_from_options(opts) as landmarker:
        i = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            res = landmarker.detect_for_video(mp_img, int(i * 1000 / fps))
            if res.pose_world_landmarks:
                w, n = res.pose_world_landmarks[0], res.pose_landmarks[0]
                world.append([[l.x, l.y, l.z] for l in w])
                image.append([[l.x, l.y, l.z] for l in n])
                vis.append([l.visibility for l in w])
            else:
                world.append(np.full((NUM_LANDMARKS, 3), np.nan))
                image.append(np.full((NUM_LANDMARKS, 3), np.nan))
                vis.append(np.zeros(NUM_LANDMARKS))
            i += 1
            if progress:
                progress(i, max(total, i))
    cap.release()

    vis = np.asarray(vis, dtype=np.float32)
    detected = int((vis.sum(axis=1) > 0).sum())
    print(f"Processed {i} frames at {fps:.1f} fps; person detected in {detected}.")
    return dict(
        world=np.asarray(world, dtype=np.float32),
        image=np.asarray(image, dtype=np.float32),
        visibility=vis,
        fps=np.float32(fps),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("video")
    parser.add_argument("-o", "--output", default="keypoints.npz")
    parser.add_argument("--model", choices=["lite", "full", "heavy"], default="heavy")
    args = parser.parse_args()

    data = extract(args.video, ensure_model(args.model))
    np.savez(args.output, **data)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
