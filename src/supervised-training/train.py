"""Train the music -> motion model on prepared clips.

Usage:
    uv run train.py                                  # cache/ -> checkpoints/model.pt
    uv run train.py --epochs 200 --window 240 --batch 16 --device cuda
    uv run train.py --smoke-test                     # 20 steps on random data, no dataset needed

Loss = position + velocity + acceleration, all on normalised coordinates:
position alone gives motion that matches on average but jitters, because a small per-frame
error looks fine in position and awful in velocity. The derivative terms are what make the
output watchable, and are standard in the motion-generation literature.

Input poses get Gaussian noise during training (--noise). The model is fed its own outputs
at generation time, so training only on ground-truth inputs leaves it helpless once it drifts
off the data manifold; noise is the cheap version of scheduled sampling.
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import DanceWindows, Stats, load_clips, split_clips
from model import ModelConfig, MotionTransformer

ROOT = Path(__file__).resolve().parent


def shift_inputs(motion: torch.Tensor) -> torch.Tensor:
    """Poses fed to the model: pose[t-1] at position t, with the first frame repeated."""
    return torch.cat([motion[:, :1], motion[:, :-1]], dim=1)


def motion_loss(pred: torch.Tensor, target: torch.Tensor, w_vel: float, w_acc: float) -> dict:
    pos = torch.nn.functional.l1_loss(pred, target)
    vel = torch.nn.functional.l1_loss(pred.diff(dim=1), target.diff(dim=1))
    acc = torch.nn.functional.l1_loss(pred.diff(n=2, dim=1), target.diff(n=2, dim=1))
    return {"loss": pos + w_vel * vel + w_acc * acc, "pos": pos, "vel": vel, "acc": acc}


def run_epoch(model, loader, cfg, optimiser=None, scheduler=None):
    train = optimiser is not None
    model.train(train)
    totals, n = {}, 0
    for audio, motion in loader:
        audio, motion = audio.to(cfg.device), motion.to(cfg.device)
        prev = shift_inputs(motion)
        if train and cfg.noise > 0:
            prev = prev + cfg.noise * torch.randn_like(prev)
        with torch.set_grad_enabled(train):
            pred = model(prev, audio)
            parts = motion_loss(pred, motion, cfg.w_vel, cfg.w_acc)
        if train:
            optimiser.zero_grad(set_to_none=True)
            parts["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip)
            optimiser.step()
            scheduler.step()
        for k, v in parts.items():
            totals[k] = totals.get(k, 0.0) + v.item() * len(audio)
        n += len(audio)
    return {k: v / n for k, v in totals.items()}


def smoke_test(args):
    """Check the plumbing without a dataset: shapes, a few optimisation steps, a rollout."""
    torch.manual_seed(0)
    cfg = ModelConfig(d_model=64, n_layers=2, n_heads=2, d_ff=128, max_len=128)
    model = MotionTransformer(cfg).to(args.device)
    audio = torch.randn(4, 64, cfg.audio_dim, device=args.device)
    motion = torch.randn(4, 64, cfg.motion_dim, device=args.device).cumsum(dim=1) * 0.01
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    first = last = None
    for step in range(20):
        parts = motion_loss(model(shift_inputs(motion), audio), motion, args.w_vel, args.w_acc)
        opt.zero_grad(set_to_none=True)
        parts["loss"].backward()
        opt.step()
        first = first if first is not None else parts["loss"].item()
        last = parts["loss"].item()
    rollout = model.generate(audio[:1], seed=motion[:1, :5])
    params = sum(p.numel() for p in model.parameters())
    print(f"loss {first:.4f} -> {last:.4f} over 20 steps ({params/1e6:.2f}M params)")
    print(f"rollout {tuple(rollout.shape)} finite={bool(torch.isfinite(rollout).all())}")
    assert rollout.shape == (1, 64, cfg.motion_dim) and torch.isfinite(rollout).all()
    assert last < first, "loss did not go down"
    print("smoke test passed")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cache", type=Path, default=ROOT / "cache")
    parser.add_argument("--out", type=Path, default=ROOT / "checkpoints")
    parser.add_argument("--window", type=int, default=240, help="frames per window (240 = 4 s at 60 fps)")
    parser.add_argument("--stride", type=int, default=30)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--noise", type=float, default=0.02, help="input pose noise, in normalised units")
    parser.add_argument("--w-vel", type=float, default=1.0, dest="w_vel")
    parser.add_argument("--w-acc", type=float, default=0.5, dest="w_acc")
    parser.add_argument("--clip", type=float, default=1.0)
    parser.add_argument("--holdout", type=float, default=0.15)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else
                                     "mps" if torch.backends.mps.is_available() else "cpu")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    if args.smoke_test:
        smoke_test(args)
        return

    clips = load_clips(args.cache)
    train_clips, val_clips = split_clips(clips, args.holdout)
    stats = Stats.fit(train_clips)
    train_set = DanceWindows(train_clips, stats, args.window, args.stride)
    val_set = DanceWindows(val_clips, stats, args.window, args.window)
    print(f"{len(clips)} clips -> {len(train_set)} train / {len(val_set)} val windows on {args.device}")

    train_loader = DataLoader(train_set, batch_size=args.batch, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=args.batch)

    cfg = ModelConfig(max_len=max(args.window, 512))
    model = MotionTransformer(cfg).to(args.device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimiser, max_lr=args.lr, total_steps=args.epochs * max(1, len(train_loader)))

    args.out.mkdir(parents=True, exist_ok=True)
    best = float("inf")
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr = run_epoch(model, train_loader, args, optimiser, scheduler)
        va = run_epoch(model, val_loader, args)
        print(f"epoch {epoch:3d}  train {tr['loss']:.4f} (pos {tr['pos']:.4f} vel {tr['vel']:.4f})"
              f"  val {va['loss']:.4f}  {time.time() - t0:.1f}s")
        if va["loss"] < best:
            best = va["loss"]
            torch.save({"model": model.state_dict(), "config": cfg.to_dict(),
                        "stats": stats.to_dict(), "fps": float(np.mean([c["fps"] for c in clips])),
                        "val_loss": best, "val_clips": [c["name"] for c in val_clips]},
                       args.out / "model.pt")
            print(f"  saved {args.out / 'model.pt'} (val {best:.4f})")


if __name__ == "__main__":
    main()
