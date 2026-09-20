"""Train the music -> motion model with a capped number of passes per clip, resumably.

No clip is trained on more than --max-uses times, ever (5 by default; one sweep = one use). The checkpoint carries
a ledger counting how often each clip has been used, so a later run continues where the last
one stopped: least-used clips first, including clips prepared after the last run finished.
The cap is what keeps a small dataset from being memorised by a comparatively large model.

Usage:
    uv run train.py --minutes 55                 # fresh run; stops at 55 min or when every clip hits the cap
    uv run train.py --minutes 55                 # again later: resumes from the ledger, least-used clips first
    uv run train.py --fresh                      # ignore an existing checkpoint and start over
    uv run train.py --smoke-test                 # 20 steps on random data, no dataset needed

Clips are processed in shards (--shard-clips) so memory stays bounded; the checkpoint is
written after each shard, and Ctrl-C saves before exiting. A shard interrupted mid-way is not
counted, so its clips keep their old count and come round again.

Outputs in --out:
    checkpoint.pt   model + optimiser + ledger; what --resume reads
    model.pt        final weights only, for generate.py / evaluate.py
    best.pt         weights at the lowest validation loss so far
"""

import argparse
import json
import random
import re
import signal
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import DanceWindows, Stats, load_clips
from model import ModelConfig, MotionTransformer

ROOT = Path(__file__).resolve().parent
# Config that must not change between runs: the ledger and optimiser state assume them.
FROZEN = ("window", "batch", "lr", "lr_high_mult", "lr_high_sweeps", "lr_decay_sweeps")


def model_config(args) -> ModelConfig:
    """Architecture for a fresh run. Cost scales ~ d_model^2 * layers, so these are the dials
    for making a run fill a time budget when the dataset is too small to fill it by itself."""
    return ModelConfig(d_model=args.d_model, n_layers=args.layers, n_heads=args.heads,
                       d_ff=4 * args.d_model, max_len=max(args.window, 512))


def shift_inputs(motion: torch.Tensor) -> torch.Tensor:
    """Poses fed to the model: pose[t-1] at position t, with the first frame repeated."""
    return torch.cat([motion[:, :1], motion[:, :-1]], dim=1)


def motion_loss(pred: torch.Tensor, target: torch.Tensor, w_vel: float, w_acc: float) -> dict:
    pos = torch.nn.functional.l1_loss(pred, target)
    vel = torch.nn.functional.l1_loss(pred.diff(dim=1), target.diff(dim=1))
    acc = torch.nn.functional.l1_loss(pred.diff(n=2, dim=1), target.diff(n=2, dim=1))
    return {"loss": pos + w_vel * vel + w_acc * acc, "pos": pos, "vel": vel, "acc": acc}


def lr_at(sweep_pos: float, base: float, high_mult: float, high_sweeps: float,
          decay_sweeps: float) -> float:
    """Learning rate at a fractional sweep position (1.5 = halfway through the second sweep).

    `high_mult` x base while the model is still seeing each clip for the first few times,
    then a linear decay to base over the next `decay_sweeps`, then base. Tying the shape to
    sweeps rather than steps keeps it meaningful when the dataset (and so the steps per
    sweep) changes between runs.
    """
    if sweep_pos < high_sweeps:
        return base * high_mult
    if sweep_pos < high_sweeps + decay_sweeps:
        t = (sweep_pos - high_sweeps) / decay_sweeps
        return base * (high_mult + (1.0 - high_mult) * t)
    return base


def choose_val_clips(names: list[str], songs_per_genre: int, seed: int) -> list[str]:
    """Hold out whole songs, spread across every genre.

    Clips of one song share a soundtrack, so splitting by clip leaks the music into training;
    splitting by genre alone leaves the metric blind to most of the distribution. AIST names
    look like gBR_sBM_c01_d04_mBR0_ch01_00 -> genre gBR, song mBR0. Names that don't parse
    fall back to taking the first few clips.
    """
    songs: dict[tuple[str, str], list[str]] = {}
    for name in names:
        parts = name.split("_")
        if len(parts) >= 5 and re.fullmatch(r"g[A-Z]{2}", parts[0]) and re.fullmatch(r"m\w+", parts[4]):
            songs.setdefault((parts[0], parts[4]), []).append(name)
    if not songs:
        return names[: max(1, songs_per_genre)]
    rng = random.Random(seed)
    val: list[str] = []
    for genre in sorted({g for g, _ in songs}):
        choices = sorted(song for g, song in songs if g == genre)
        for song in rng.sample(choices, min(songs_per_genre, len(choices))):
            val += songs[(genre, song)]
    return sorted(val)


def fit_stats(cache: Path) -> Stats:
    """Per-channel mean/std over every prepared clip, in one streaming pass (bounded memory).

    Computed once, on the first run, and frozen in the checkpoint: later runs see different
    clips, and normalisation that drifts between runs would invalidate the optimiser state.
    """
    sums = {}
    for path in sorted(cache.glob("*.npz")):
        with np.load(path) as z:
            for key in ("audio", "motion"):
                x = z[key].astype(np.float64)
                if key not in sums:
                    sums[key] = [np.zeros(x.shape[1]), np.zeros(x.shape[1]), 0]
                sums[key][0] += x.sum(0)
                sums[key][1] += (x ** 2).sum(0)
                sums[key][2] += len(x)
    out = {}
    for key, (total, sq, n) in sums.items():
        mean = total / n
        out[key] = (mean, np.sqrt(np.maximum(sq / n - mean ** 2, 0)) + 1e-6)
    return Stats(out["audio"][0], out["audio"][1], out["motion"][0], out["motion"][1])


def evaluate(model, loader, args, device) -> dict:
    model.eval()
    totals, n = {}, 0
    with torch.no_grad():
        for audio, motion in loader:
            audio, motion = audio.to(device), motion.to(device)
            parts = motion_loss(model(shift_inputs(motion), audio), motion, args.w_vel, args.w_acc)
            for k, v in parts.items():
                totals[k] = totals.get(k, 0.0) + v.item() * len(audio)
            n += len(audio)
    return {k: v / max(1, n) for k, v in totals.items()}


def save(path: Path, payload: dict):
    """Write via a temporary file so an interrupted save can't corrupt the checkpoint."""
    tmp = path.with_suffix(".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def slim(state: dict, model, stats, fps, val_clips) -> dict:
    """The checkpoint format generate.py and evaluate.py expect: weights, config, stats."""
    return {"model": model.state_dict(), "config": state["config"]["model"], "stats": stats.to_dict(),
            "fps": fps, "val_clips": val_clips, "step": state["step"],
            "clip_passes": sum(state["uses"].values())}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cache", type=Path, default=ROOT / "cache")
    parser.add_argument("--out", type=Path, default=ROOT / "checkpoints")
    parser.add_argument("--minutes", type=float, default=55.0, help="wall-clock budget for this run")
    parser.add_argument("--shard-clips", type=int, default=16, help="clips held in memory at once")
    parser.add_argument("--window", type=int, default=240, help="frames per window (240 = 4 s at 60 fps)")
    parser.add_argument("--stride", type=int, default=0,
                        help="window start spacing; 0 = no overlap, so each frame is trained on once")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4, help="the rate the schedule decays to")
    parser.add_argument("--lr-high-mult", type=float, default=3.0, dest="lr_high_mult",
                        help="multiple of --lr used for the first sweeps")
    parser.add_argument("--lr-high-sweeps", type=float, default=5.0, dest="lr_high_sweeps",
                        help="sweeps held at the high rate")
    parser.add_argument("--lr-decay-sweeps", type=float, default=15.0, dest="lr_decay_sweeps",
                        help="sweeps over which it decays linearly back to --lr")
    parser.add_argument("--warmup", type=int, default=0,
                        help="steps ramping into the high rate at the very start (0 = none)")
    parser.add_argument("--noise", type=float, default=0.02, help="input pose noise, in normalised units")
    parser.add_argument("--w-vel", type=float, default=1.0, dest="w_vel")
    parser.add_argument("--w-acc", type=float, default=0.5, dest="w_acc")
    parser.add_argument("--clip-grad", type=float, default=1.0)
    parser.add_argument("--snapshot-every", type=int, default=0,
                        help="keep a numbered copy of the weights every N sweeps, for plotting progress")
    parser.add_argument("--val-songs", type=int, default=1, dest="val_songs",
                        help="songs held out per genre, with all their clips (first run only)")
    parser.add_argument("--val-seed", type=int, default=0, help="which songs --val-songs picks")
    parser.add_argument("--max-uses", type=int, default=5,
                        help="how often one clip may ever be trained on, across all runs")
    parser.add_argument("--d-model", type=int, default=ModelConfig.d_model, dest="d_model")
    parser.add_argument("--layers", type=int, default=ModelConfig.n_layers)
    parser.add_argument("--heads", type=int, default=ModelConfig.n_heads)
    parser.add_argument("--fresh", action="store_true", help="start over, ignoring any checkpoint")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else
                                     "mps" if torch.backends.mps.is_available() else "cpu")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    args.stride = args.stride or args.window

    if args.smoke_test:
        smoke_test(args)
        return

    args.out.mkdir(parents=True, exist_ok=True)
    ckpt_path = args.out / "checkpoint.pt"
    device = args.device

    clips_available = sorted(p.stem for p in args.cache.glob("*.npz"))
    if not clips_available:
        raise SystemExit(f"No prepared clips in {args.cache}; run prepare_data.py first.")

    # ---------------------------------------------------------------- resume or start
    if ckpt_path.exists() and not args.fresh:
        state = torch.load(ckpt_path, map_location=device, weights_only=False)
        stored = state["config"]["model"]
        if (stored["d_model"], stored["n_layers"], stored["n_heads"]) != (args.d_model, args.layers, args.heads):
            print(f"Note: keeping the checkpoint's architecture (d_model={stored['d_model']}, "
                  f"layers={stored['n_layers']}, heads={stored['n_heads']}); --fresh to change it.")
        for key in FROZEN:
            if state["config"][key] != getattr(args, key):
                raise SystemExit(f"--{key.replace('_', '-')} is {getattr(args, key)} but the checkpoint "
                                 f"used {state['config'][key]}; pass the same value or --fresh.")
        stats = Stats.from_dict(state["stats"])
        used = state["uses"]
        print(f"Resuming at step {state['step']}: {len(used)} clips used, "
              f"{sum(used.values())} passes so far (cap {args.max_uses} each)")
    else:
        stats = fit_stats(args.cache)
        # Held out by whole song, recorded in the checkpoint, never trained on in any run.
        val_names = choose_val_clips(clips_available, args.val_songs, args.val_seed)
        state = {"config": {k: getattr(args, k) for k in FROZEN} | {"model": model_config(args).to_dict(),
                                                                   "stride": args.stride},
                 "stats": stats.to_dict(), "uses": {}, "val": val_names,
                 "step": 0, "best_val": float("inf"), "history": [], "curve": []}
        held_songs = sorted({"_".join(n.split("_")[:1] + n.split("_")[4:5]) for n in val_names})
        print(f"Fresh run. Held out {len(val_names)} clips from {len(held_songs)} songs: "
              f"{', '.join(held_songs)}")

    model = MotionTransformer(ModelConfig(**state["config"]["model"])).to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    if "model_state" in state:
        model.load_state_dict(state["model_state"])
        optimiser.load_state_dict(state["optimiser"])

    # ---------------------------------------------------------------- data
    val_clips = load_clips(args.cache, state["val"])
    val_loader = DataLoader(DanceWindows(val_clips, stats, args.window, args.window), batch_size=args.batch)
    fps = float(np.mean([c["fps"] for c in val_clips]))

    uses = state["uses"]
    held_out = set(state["val"])

    def next_queue() -> list[str]:
        """Clips still under the cap, least-used first, so passes spread evenly."""
        under = [n for n in clips_available if n not in held_out and uses.get(n, 0) < args.max_uses]
        return sorted(under, key=lambda n: (uses.get(n, 0), n))

    queue = next_queue()
    if not queue:
        raise SystemExit(f"Every prepared clip has been used {args.max_uses} times. Prepare more "
                         f"clips (prepare_data.py), raise --max-uses, or pass --fresh.")
    budget = sum(args.max_uses - uses.get(n, 0) for n in queue)
    print(f"{len(clips_available)} prepared, {len(queue)} below the cap ({budget} passes left) "
          f"-> this run's budget: {args.minutes:.0f} min on {device}")

    frames = int(np.median([len(c["motion"]) for c in val_clips]))
    windows_per_clip = max(1, (frames - args.window) // args.stride + 1)
    planned = max(1, budget * windows_per_clip // args.batch)
    print(f"~{planned} steps planned ({windows_per_clip} windows/clip). LR "
          f"{args.lr * args.lr_high_mult:.2e} for {args.lr_high_sweeps:g} sweeps, then linear to "
          f"{args.lr:.2e} over {args.lr_decay_sweeps:g} more"
          + (f", after {args.warmup} warmup steps" if args.warmup else ""))

    # ---------------------------------------------------------------- train
    stop = {"now": False}
    signal.signal(signal.SIGINT, lambda *_: stop.update(now=True))
    deadline = time.time() + args.minutes * 60
    run = {"started": time.time(), "steps": 0, "clips": 0, "loss": None}

    sweep = 0
    # A clip's use count is how many sweeps it has had, so the least-used clip says where the
    # schedule is; a resumed run then continues along the curve rather than restarting it.
    sweeps_done = min((uses.get(n, 0) for n in queue), default=0)
    while queue and not stop["now"] and time.time() < deadline:
      sweep += 1
      shards = max(1, -(-len(queue) // args.shard_clips))
      for shard_index, start in enumerate(range(0, len(queue), args.shard_clips)):
        if stop["now"] or time.time() >= deadline:
            break
        names = queue[start : start + args.shard_clips]
        shard = load_clips(args.cache, names)
        loader = DataLoader(DanceWindows(shard, stats, args.window, args.stride),
                            batch_size=args.batch, shuffle=True, drop_last=False)

        model.train()
        losses = []
        for audio, motion in loader:
            if stop["now"] or time.time() >= deadline:
                break
            audio, motion = audio.to(device), motion.to(device)
            prev = shift_inputs(motion)
            if args.noise > 0:
                prev = prev + args.noise * torch.randn_like(prev)
            parts = motion_loss(model(prev, audio), motion, args.w_vel, args.w_acc)
            # Sweeps already completed, plus how far this sweep has got.
            sweep_pos = sweeps_done + shard_index / shards
            lr = lr_at(sweep_pos, args.lr, args.lr_high_mult, args.lr_high_sweeps, args.lr_decay_sweeps)
            if args.warmup and state["step"] < args.warmup:
                lr *= (state["step"] + 1) / args.warmup
            for group in optimiser.param_groups:
                group["lr"] = lr
            optimiser.zero_grad(set_to_none=True)
            parts["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            optimiser.step()
            state["step"] += 1
            run["steps"] += 1
            losses.append(parts["loss"].item())
        else:
            # Only a shard trained to the end is counted, so an interrupted shard isn't charged a pass.
            for name in names:
                uses[name] = uses.get(name, 0) + 1
            run["clips"] += len(names)

        val = evaluate(model, val_loader, args, device)
        run["loss"] = float(np.mean(losses)) if losses else run["loss"]
        left = max(0.0, deadline - time.time()) / 60
        print(f"sweep {sweeps_done + 1}  lr {optimiser.param_groups[0]['lr']:.2e}  "
              f"step {state['step']:6d}  passes {sum(uses.values()):5d}  "
              f"train {run['loss']:.4f}  val {val['loss']:.4f}  {left:.0f} min left")

        # One point per sweep, so progress can be plotted without re-running anything.
        state.setdefault("curve", []).append(
            {"sweep": sweeps_done + 1, "lr": optimiser.param_groups[0]["lr"], "step": state["step"], "minutes": round((time.time() - run["started"]) / 60, 2),
             "train_loss": run["loss"], "val_loss": val["loss"], "passes": sum(uses.values())})
        state["model_state"] = model.state_dict()
        state["optimiser"] = optimiser.state_dict()
        save(ckpt_path, state)
        save(args.out / "model.pt", slim(state, model, stats, fps, state["val"]))
        if val["loss"] < state["best_val"]:
            state["best_val"] = val["loss"]
            save(args.out / "best.pt", slim(state, model, stats, fps, state["val"]))
      if args.snapshot_every and sweep % args.snapshot_every == 0:
          snap_dir = args.out / "snapshots"
          snap_dir.mkdir(exist_ok=True)
          save(snap_dir / f"step{state['step']:06d}.pt", slim(state, model, stats, fps, state["val"]))
      sweeps_done += 1
      queue = next_queue()  # clips that have hit the cap drop out; the rest come round again

    # ---------------------------------------------------------------- finish
    minutes = (time.time() - run["started"]) / 60
    state["config"]["schedule"] = {"lr": args.lr, "high_mult": args.lr_high_mult,
                                   "high_sweeps": args.lr_high_sweeps,
                                   "decay_sweeps": args.lr_decay_sweeps, "warmup": args.warmup}
    state["history"].append({"minutes": round(minutes, 1), "steps": run["steps"],
                             "passes": run["clips"], "final_train_loss": run["loss"],
                             "best_val": state["best_val"], "ended": time.strftime("%Y-%m-%d %H:%M")})
    save(ckpt_path, state)
    save(args.out / "model.pt", slim(state, model, stats, fps, state["val"]))
    remaining = sum(args.max_uses - u for u in
                    (uses.get(n, 0) for n in clips_available if n not in held_out))
    print(f"\n{'Interrupted' if stop['now'] else 'Done'} after {minutes:.1f} min: "
          f"{run['steps']} steps over {run['clips']} clip passes "
          f"({sum(uses.values())} total; {remaining} passes still allowed under the cap).")
    print(f"Final model: {args.out / 'model.pt'}  best val: {state['best_val']:.4f} "
          f"({args.out / 'best.pt'})")
    print(f"Resume with the same command; history: {json.dumps(state['history'][-1])}")


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
    print("lr by sweep:", {s: round(lr_at(s, 3e-4, 3.0, 5, 15), 7) for s in (0, 4.9, 5, 12.5, 20, 25)})
    assert rollout.shape == (1, 64, cfg.motion_dim) and torch.isfinite(rollout).all()
    assert last < first, "loss did not go down"
    print("smoke test passed")


if __name__ == "__main__":
    main()
