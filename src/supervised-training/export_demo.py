"""Export a ground-truth clip next to the dance the model invents for the same music.

For one prepared clip this writes, side by side, what the dancer actually did and what the
model produces from that clip's audio alone: MediaPipe world landmarks for each, both fitted
to the MuJoCo humanoid, the source video, and the dance_similarity scores between them.

Usage:
    uv run export_demo.py gJS_sBM_c01_d03_mJS2_ch01_00 --label val --out ../../demos
    uv run export_demo.py <clip> --checkpoint checkpoints-v2/model.pt --seed-frames 30

Everything lands in <out>/<label>_<clip>/:
    ground_truth.mp4              the source video (with its music)
    ground_truth_keypoints.npz    landmarks as extract_keypoints writes them
    ground_truth_motion.npz       those landmarks retargeted to the humanoid
    generated_keypoints.npz       the model's dance for the same audio
    generated_motion.npz          retargeted the same way
    scores.json                   dance_similarity and beat alignment

View either motion.npz with ../pose-estimation/visualize.py.
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

import features
from dataset import load_clips
from evaluate import beat_alignment, score
from generate import load_checkpoint

ROOT = Path(__file__).resolve().parent


def retarget_motion(world: np.ndarray, fps: float) -> dict:
    sys.path.insert(0, str(ROOT.parent / "pose-estimation"))
    import retarget
    return retarget.retarget(features.to_keypoints(world, fps))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("clip", help="prepared clip name, without .npz")
    parser.add_argument("--label", default="demo", help="prefix for the output folder, e.g. val or train")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints-v2" / "model.pt")
    parser.add_argument("--cache", type=Path, default=ROOT / "cache")
    parser.add_argument("--clips", type=Path, default=ROOT.parents[1] / "dataset" / "aist" / "clips")
    parser.add_argument("--out", type=Path, default=ROOT.parents[1] / "demos")
    parser.add_argument("--seed-frames", type=int, default=30,
                        help="ground-truth frames used to start the rollout (0.5 s at 60 fps)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    model, stats, _ = load_checkpoint(args.checkpoint, args.device)
    clip = load_clips(args.cache, [args.clip])[0]
    fps = clip["fps"]

    audio = torch.from_numpy(stats.normalise_audio(clip["audio"]).astype(np.float32))[None].to(args.device)
    seed = torch.from_numpy(
        stats.normalise_motion(clip["motion"][: args.seed_frames]).astype(np.float32))[None].to(args.device)
    print(f"Generating {len(clip['motion']) / fps:.1f} s for {args.clip}…")
    generated = features.unflatten_motion(
        stats.denormalise_motion(model.generate(audio, seed=seed))[0].cpu().numpy())
    truth = features.unflatten_motion(clip["motion"])

    out = args.out / f"{args.label}_{args.clip}"
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "ground_truth_keypoints.npz", **features.to_keypoints(truth, fps))
    np.savez(out / "generated_keypoints.npz", **features.to_keypoints(generated, fps))

    source = args.clips / f"{args.clip}.mp4"
    if source.exists():
        shutil.copyfile(source, out / "ground_truth.mp4")

    for name, world in (("ground_truth", truth), ("generated", generated)):
        print(f"Retargeting {name} to the humanoid…")
        np.savez(out / f"{name}_motion.npz", **retarget_motion(world, fps))

    result = score(truth, generated, fps)
    summary = {
        "clip": args.clip,
        "split": args.label,
        "checkpoint": str(args.checkpoint),
        "seconds": round(len(truth) / fps, 2),
        "fps": round(fps, 2),
        "seed_frames": args.seed_frames,
        "dance_similarity": {k: round(float(v), 4) for k, v in result.items()},
        "beat_alignment": {"generated": round(beat_alignment(generated, clip["audio"], fps), 3),
                           "ground_truth": round(beat_alignment(truth, clip["audio"], fps), 3)},
    }
    (out / "scores.json").write_text(json.dumps(summary, indent=2))
    print(f"\n{out}: dance_similarity {result['final_score']:.3f} "
          f"(position {result['position_score']:.2f} angle {result['angle_score']:.2f} "
          f"trajectory {result['trajectory_score']:.2f} timing {result['timing_score']:.2f})")


if __name__ == "__main__":
    main()
