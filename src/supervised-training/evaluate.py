"""Score generated dances against the real ones with reward-calculation's dance_similarity.

Usage:
    uv run evaluate.py                       # every held-out clip in the checkpoint
    uv run evaluate.py --clips cache/foo.npz --baselines

For each clip the model dances to that clip's music, and the result is compared with what
the dancer actually did. Two caveats worth keeping in mind while reading the numbers:

  * Choreography is one-to-many. A good dance that isn't *this* dance scores badly, so
    dance_similarity measures imitation, not quality. Beat alignment is reported alongside
    because it rewards dancing on the beat without demanding one specific routine.
  * --baselines prints reference points: the real dance against itself (the ceiling, 1.0),
    against a frozen pose, and against a different clip's dance. A model that can't beat
    "mean pose held still" has learnt nothing, and that is a surprisingly common outcome.
  * Clips differ enormously in how much dancing they contain - an opening chunk where the
    dancer waits for the music moves a tenth as much as a mid-song chunk - and a frozen pose
    scores well on the still ones. So alongside the plain mean this reports a mean weighted by
    each clip's motion, and the mean margin over each clip's own frozen-pose baseline. Those
    two are the numbers to quote; the plain mean flatters a model that barely moves.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

import features
from dataset import load_clips, sample_across_genres
from generate import load_checkpoint

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "reward-calculation"))
import trajectory_similarity  # noqa: E402


def beat_alignment(world: np.ndarray, audio: np.ndarray, fps: float, tolerance_s: float = 0.1) -> float:
    """Share of the music's beats that have a motion 'hit' within `tolerance_s`.

    The AIST++ papers report this (as Beat Alignment Score) because it captures musicality
    independently of which choreography was danced.
    """
    beats = np.flatnonzero(audio[:, -1] > 0.5)  # the beat one-hot channel from features.py
    hits = features.kinematic_beats(world, fps)
    if len(beats) == 0 or len(hits) == 0:
        return float("nan")
    gaps = np.abs(beats[:, None] - hits[None, :]).min(axis=1) / fps
    return float((gaps <= tolerance_s).mean())


def score(reference: np.ndarray, generated: np.ndarray, fps: float) -> dict:
    n = min(len(reference), len(generated))  # dance_similarity needs matching shapes
    return trajectory_similarity.dance_similarity(reference[:n], generated[:n], fps)


def frozen_baseline(reference: np.ndarray, fps: float) -> float:
    """What this clip scores against its own first pose, held for the whole clip."""
    return float(score(reference, np.repeat(reference[:1], len(reference), axis=0), fps)["final_score"])


def weighted_means(rows: list[dict]) -> dict:
    """Plain mean, motion-weighted mean, and mean margin over each clip's frozen pose."""
    motion = np.array([r["motion"] for r in rows])
    scores = np.array([r["score"] for r in rows])
    margins = scores - np.array([r["frozen"] for r in rows])
    weights = motion / motion.sum() if motion.sum() > 0 else np.full(len(rows), 1 / len(rows))
    return {"mean": float(scores.mean()), "weighted": float((scores * weights).sum()),
            "margin": float(margins.mean()), "motion_cm": float(motion.mean() * 100)}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints" / "model.pt")
    parser.add_argument("--cache", type=Path, default=ROOT / "cache")
    parser.add_argument("--clips", type=Path, nargs="*", help="specific prepared .npz files")
    parser.add_argument("--seed-frames", type=int, default=30)
    parser.add_argument("--baselines", action="store_true")
    parser.add_argument("--limit", type=int, default=10,
                        help="held-out clips to score, sampled across genres (0 = all)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if not args.checkpoint.exists():
        sys.exit(f"No checkpoint at {args.checkpoint}; run train.py first.")
    model, stats, _ = load_checkpoint(args.checkpoint, args.device)
    held_out = set(torch.load(args.checkpoint, map_location="cpu", weights_only=False)["val_clips"])

    clips = load_clips(args.cache)
    if args.clips:
        wanted = {p.stem for p in args.clips}
        clips = [c for c in clips if c["name"] in wanted]
    else:
        clips = [c for c in clips if c["name"] in held_out]
    if not clips:
        sys.exit("No clips to evaluate.")
    chosen = set(sample_across_genres([c["name"] for c in clips], args.limit))
    clips = [c for c in clips if c["name"] in chosen]
    print(f"Scoring {len(clips)} of {len(held_out)} held-out clips "
          f"({len({c['name'].split('_')[0] for c in clips})} genres)\n")

    rows = []
    for clip in clips:
        fps = clip["fps"]
        audio = torch.from_numpy(stats.normalise_audio(clip["audio"]).astype(np.float32))[None].to(args.device)
        seed = torch.from_numpy(
            stats.normalise_motion(clip["motion"][: args.seed_frames]).astype(np.float32))[None].to(args.device)

        motion = model.generate(audio, seed=seed)
        generated = features.unflatten_motion(stats.denormalise_motion(motion)[0].cpu().numpy())
        reference = features.unflatten_motion(clip["motion"])

        result = score(reference, generated, fps)
        rows.append({"name": clip["name"], "score": float(result["final_score"]),
                     "motion": features.motion_amount(reference),
                     "frozen": frozen_baseline(reference, fps),
                     "beat": beat_alignment(generated, clip["audio"], fps),
                     "beat_real": beat_alignment(reference, clip["audio"], fps)})
        print(f"{clip['name']:40s} score {result['final_score']:.3f}  "
              f"(position {result['position_score']:.2f} angle {result['angle_score']:.2f} "
              f"trajectory {result['trajectory_score']:.2f} timing {result['timing_score']:.2f})  "
              f"motion {rows[-1]['motion'] * 100:.2f} cm/frame  frozen {rows[-1]['frozen']:.2f}  "
              f"beat {rows[-1]['beat']:.2f} vs real {rows[-1]['beat_real']:.2f}")

    by_genre: dict[str, list[float]] = {}
    for row in rows:
        by_genre.setdefault(row["name"].split("_")[0], []).append(row["score"])
    if len(by_genre) > 1:
        print("\nby genre: " + "  ".join(f"{g} {np.mean(v):.3f}" for g, v in sorted(by_genre.items())))

    agg = weighted_means(rows)
    print(f"\nmean dance_similarity      {agg['mean']:.3f} over {len(rows)} clips")
    print(f"motion-weighted mean       {agg['weighted']:.3f}  "
          f"(clips average {agg['motion_cm']:.2f} cm/frame of movement)")
    print(f"mean margin over frozen    {agg['margin']:+.3f}  "
          f"(per clip, against its own frozen pose)")
    print(f"mean beat alignment        {np.nanmean([r['beat'] for r in rows]):.3f} "
          f"(real dances {np.nanmean([r['beat_real'] for r in rows]):.3f})")

    if args.baselines:
        clip = clips[0]
        fps, reference = clip["fps"], features.unflatten_motion(clip["motion"])
        frozen = np.repeat(reference[:1], len(reference), axis=0)
        other = features.unflatten_motion(
            next(c for c in load_clips(args.cache) if c["name"] != clip["name"])["motion"])
        print(f"\nbaselines on {clip['name']}:")
        print(f"  itself          {score(reference, reference, fps)['final_score']:.3f}")
        print(f"  frozen pose     {score(reference, frozen, fps)['final_score']:.3f}")
        print(f"  a different dance {score(reference, other, fps)['final_score']:.3f}")


if __name__ == "__main__":
    main()
