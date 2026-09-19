"""Generate a dance for a piece of music with a trained checkpoint.

Usage:
    uv run generate.py song.mp3 --out danced.npz
    uv run generate.py clip.mp4 --seed-from cache/clip.npz --retarget danced_motion.npz

Writes an extract_keypoints-shaped .npz (world/image/visibility/fps), so the result can be
fed to ../pose-estimation/retarget.py and viewed in the web UI like any other clip.
With --retarget it also runs the IK itself and writes motion.npz for visualize.py.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

import features
from dataset import Stats
from model import ModelConfig, MotionTransformer
from prepare_data import load_audio

ROOT = Path(__file__).resolve().parent


def load_checkpoint(path: Path, device: str):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = MotionTransformer(ModelConfig(**ckpt["config"])).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, Stats.from_dict(ckpt["stats"]), float(ckpt["fps"])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("audio", type=Path, help="music to dance to (any ffmpeg-readable file)")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints" / "model.pt")
    parser.add_argument("--out", type=Path, default=ROOT / "generated.npz")
    parser.add_argument("--seconds", type=float, help="only dance the first N seconds")
    parser.add_argument("--seed-from", type=Path,
                        help="prepared .npz whose first frames start the rollout; "
                             "without it the dance starts from the dataset's mean pose")
    parser.add_argument("--seed-frames", type=int, default=30)
    parser.add_argument("--retarget", type=Path, help="also solve IK and write this motion.npz")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if not args.checkpoint.exists():
        sys.exit(f"No checkpoint at {args.checkpoint}; run train.py first.")
    model, stats, fps = load_checkpoint(args.checkpoint, args.device)

    wav = load_audio(args.audio, features.SAMPLE_RATE)
    if args.seconds:
        wav = wav[: int(args.seconds * features.SAMPLE_RATE)]
    audio = features.audio_features(wav, features.SAMPLE_RATE, fps)
    audio_t = torch.from_numpy(stats.normalise_audio(audio).astype(np.float32))[None].to(args.device)

    seed = None
    if args.seed_from:
        with np.load(args.seed_from) as z:
            seed_motion = stats.normalise_motion(z["motion"][: args.seed_frames])
        seed = torch.from_numpy(seed_motion.astype(np.float32))[None].to(args.device)

    print(f"Generating {len(audio) / fps:.1f} s at {fps:.1f} fps on {args.device}…")
    motion = model.generate(audio_t, seed=seed)
    world = features.unflatten_motion(stats.denormalise_motion(motion)[0].cpu().numpy())

    np.savez(args.out, **features.to_keypoints(world, fps))
    print(f"Wrote {args.out} ({len(world)} frames)")

    if args.retarget:
        sys.path.insert(0, str(ROOT.parent / "pose-estimation"))
        import retarget
        with np.load(args.out) as kp:
            motion_npz = retarget.retarget(dict(kp))
        np.savez(args.retarget, **motion_npz)
        print(f"Wrote {args.retarget}; view it with "
              f"`uv run ../pose-estimation/visualize.py {args.retarget}`")


if __name__ == "__main__":
    main()
