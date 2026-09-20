"""Score a run's snapshots and plot how it progressed.

Each snapshot kept by `train.py --snapshot-every N` is used to dance the held-out clips,
scored with reward-calculation's dance_similarity, and drawn against the two baselines that
say whether the model is doing anything at all: holding one pose, and an unrelated dance.
Validation loss (recorded every sweep) is drawn underneath on its own panel.

Usage:
    uv run plot_progress.py                                  # checkpoints/ -> progress.png
    uv run plot_progress.py --out checkpoints-cap50 --png cap50.png
"""

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import features  # noqa: E402
from dataset import load_clips  # noqa: E402
from evaluate import score  # noqa: E402
from generate import load_checkpoint  # noqa: E402

ROOT = Path(__file__).resolve().parent
# One hue for the measured series; everything else is recessive ink, so identity is never
# carried by colour alone (each panel has a single series, named by its title).
INK, MUTED, GRID = "#1f2933", "#7b8794", "#e4e7eb"
SERIES = "#2f6fdb"
BASELINE_FROZEN, BASELINE_OTHER = "#9aa5b1", "#616e7c"


def score_snapshot(path: Path, clips: list[dict], device: str, seed_frames: int) -> float:
    model, stats, _ = load_checkpoint(path, device)
    scores = []
    for clip in clips:
        audio = torch.from_numpy(stats.normalise_audio(clip["audio"]).astype(np.float32))[None].to(device)
        seed = torch.from_numpy(
            stats.normalise_motion(clip["motion"][:seed_frames]).astype(np.float32))[None].to(device)
        generated = features.unflatten_motion(
            stats.denormalise_motion(model.generate(audio, seed=seed))[0].cpu().numpy())
        scores.append(score(features.unflatten_motion(clip["motion"]), generated, clip["fps"])["final_score"])
    return float(np.mean(scores))


def baselines(clips: list[dict]) -> tuple[float, float]:
    """Frozen pose, and dancing a different clip's choreography: the bars to clear."""
    frozen, other = [], []
    for i, clip in enumerate(clips):
        reference = features.unflatten_motion(clip["motion"])
        frozen.append(score(reference, np.repeat(reference[:1], len(reference), axis=0),
                            clip["fps"])["final_score"])
        wrong = features.unflatten_motion(clips[(i + 1) % len(clips)]["motion"])
        n = min(len(reference), len(wrong))
        other.append(score(reference[:n], wrong[:n], clip["fps"])["final_score"])
    return float(np.mean(frozen)), float(np.mean(other))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=ROOT / "checkpoints", help="a run's output directory")
    parser.add_argument("--cache", type=Path, default=ROOT / "cache")
    parser.add_argument("--png", type=Path, default=ROOT / "progress.png")
    parser.add_argument("--seed-frames", type=int, default=30)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    state = torch.load(args.out / "checkpoint.pt", map_location="cpu", weights_only=False)
    curve = state.get("curve", [])
    clips = load_clips(args.cache, state["val"])
    snapshots = sorted((args.out / "snapshots").glob("sweep*.pt"))
    if not snapshots:
        raise SystemExit(f"No snapshots in {args.out / 'snapshots'}; train with --snapshot-every N.")

    by_sweep = {point["sweep"]: point for point in curve}
    xs, ys = [], []
    for path in snapshots:
        sweep = int(path.stem.replace("sweep", ""))
        point = by_sweep.get(sweep)
        value = score_snapshot(path, clips, args.device, args.seed_frames)
        xs.append(point["minutes"] if point else len(xs))
        ys.append(value)
        print(f"{path.name}: sweep {sweep}, step {point['step'] if point else '?'}, "
              f"dance_similarity {value:.3f}")
    frozen, other = baselines(clips)
    print(f"baselines: frozen pose {frozen:.3f}, a different dance {other:.3f}")

    fig, (top, bottom) = plt.subplots(2, 1, figsize=(8, 6.5), sharex=True,
                                      gridspec_kw={"height_ratios": [3, 2], "hspace": 0.18})
    fig.patch.set_facecolor("white")

    top.axhline(other, color=BASELINE_OTHER, lw=1.5, ls=(0, (5, 3)), zorder=1)
    top.axhline(frozen, color=BASELINE_FROZEN, lw=1.5, ls=(0, (5, 3)), zorder=1)
    top.annotate(f"a different dance  {other:.2f}", (xs[-1], other), xytext=(-4, 5),
                 textcoords="offset points", ha="right", color=BASELINE_OTHER, fontsize=9)
    top.annotate(f"frozen pose  {frozen:.2f}", (xs[-1], frozen), xytext=(-4, -13),
                 textcoords="offset points", ha="right", color=BASELINE_FROZEN, fontsize=9)
    top.plot(xs, ys, color=SERIES, lw=2, marker="o", ms=5, zorder=3)
    top.annotate(f"{ys[-1]:.2f}", (xs[-1], ys[-1]), xytext=(6, 0), textcoords="offset points",
                 va="center", color=INK, fontsize=10, fontweight="medium")
    top.set_ylabel("dance_similarity", color=MUTED, fontsize=10)
    top.set_title("Held-out score over the run", color=INK, fontsize=13, loc="left", pad=10)
    top.set_ylim(0, max(0.45, max(ys + [frozen, other]) * 1.25))

    if curve:
        bottom.plot([p["minutes"] for p in curve], [p["val_loss"] for p in curve],
                    color=SERIES, lw=2, zorder=3)
    bottom.set_ylabel("validation loss", color=MUTED, fontsize=10)
    bottom.set_xlabel("minutes of training", color=MUTED, fontsize=10)
    bottom.set_title("Validation loss per sweep", color=INK, fontsize=11, loc="left", pad=8)

    for ax in (top, bottom):
        ax.set_facecolor("white")
        ax.grid(axis="y", color=GRID, lw=1)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        ax.tick_params(colors=MUTED, labelsize=9, length=0)

    passes = sum(state["uses"].values())
    fig.text(0.125, 0.955, f"{len(state['uses'])} clips, {passes} passes, {state['step']} steps, "
                           f"{len(clips)} held-out clips", color=MUTED, fontsize=9.5)
    fig.savefig(args.png, dpi=160, bbox_inches="tight", facecolor="white")
    print(f"\nWrote {args.png}")


if __name__ == "__main__":
    main()
