"""Turn dance clips into training pairs: per-clip audio features + MediaPipe world landmarks.

For every clip (mp4 with its music, as produced by dataset/download_aist.py) this runs the
same MediaPipe pass as the rest of the pipeline, extracts the soundtrack with ffmpeg, and
caches one .npz holding aligned (audio, motion) arrays.

Usage:
    uv run prepare_data.py                              # ../../dataset/aist/clips -> cache/
    uv run prepare_data.py --clips DIR --out DIR --model heavy --workers 4

Output per clip, cache/<name>.npz:
    audio       (T, 35) float32   music features at the clip's frame rate
    motion      (T, 99) float32   world landmarks, hip-centred, gaps interpolated
    detected    (T,)    bool      frames where MediaPipe actually found the dancer
    fps         scalar
Clips whose .npz already exists are skipped, so the run is resumable.
"""

import argparse
import subprocess
import sys
import tempfile
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pose-estimation"))
import extract_keypoints  # noqa: E402

import features  # noqa: E402

ROOT = Path(__file__).resolve().parent
DEFAULT_CLIPS = ROOT.parents[1] / "dataset" / "aist" / "clips"
# Clips where MediaPipe loses the dancer this often are dropped: the landmarks would be
# mostly interpolation, which teaches the model nothing.
MIN_DETECTED = 0.9


def load_audio(video: Path, sr: int) -> np.ndarray:
    """Mono waveform of a video's soundtrack, via ffmpeg (already required by the downloader)."""
    with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", str(video),
             "-vn", "-ac", "1", "-ar", str(sr), "-f", "wav", tmp.name],
            check=True,
        )
        import soundfile as sf
        wav, file_sr = sf.read(tmp.name, dtype="float32")
    assert file_sr == sr, (file_sr, sr)
    return wav


def prepare_clip(video: Path, out_dir: Path, model: str) -> tuple[str, str]:
    """Returns (clip name, status) so the parent can report without importing heavy deps."""
    out = out_dir / f"{video.stem}.npz"
    if out.exists():
        return video.stem, "cached"
    try:
        kp = extract_keypoints.extract(str(video), extract_keypoints.ensure_model(model))
        world, detected = features.fill_gaps(kp["world"])
        if detected.mean() < MIN_DETECTED:
            return video.stem, f"skipped (dancer found in {detected.mean():.0%} of frames)"

        fps = float(kp["fps"])
        wav = load_audio(video, features.SAMPLE_RATE)
        audio = features.audio_features(wav, features.SAMPLE_RATE, fps, n_frames=len(world))

        np.savez(out, audio=audio, motion=features.flatten_motion(world),
                 detected=detected, fps=np.float32(fps))
        return video.stem, f"ok ({len(world)} frames @ {fps:.1f} fps)"
    except Exception as e:
        traceback.print_exc()
        return video.stem, f"failed: {type(e).__name__}: {e}"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--clips", type=Path, default=DEFAULT_CLIPS)
    parser.add_argument("--out", type=Path, default=ROOT / "cache")
    parser.add_argument("--model", default="full", choices=["lite", "full", "heavy"])
    parser.add_argument("--workers", type=int, default=2, help="clips in parallel (each loads its own model)")
    args = parser.parse_args()

    videos = sorted(p for p in args.clips.glob("*.mp4"))
    if not videos:
        sys.exit(f"No .mp4 clips in {args.clips}. Run dataset/download_aist.py first.")
    args.out.mkdir(parents=True, exist_ok=True)
    print(f"Preparing {len(videos)} clips -> {args.out}")

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(prepare_clip, v, args.out, args.model) for v in videos]
        for i, future in enumerate(as_completed(futures), 1):
            name, status = future.result()
            print(f"[{i}/{len(videos)}] {name}: {status}")


if __name__ == "__main__":
    main()
