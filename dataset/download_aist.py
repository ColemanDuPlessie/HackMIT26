"""Download the solo-dancer AIST Dance DB videos and cut them into clips of at most 20 s.

Uses the refined (trimmed) 10 Mbps videos that AIST++ is built on. Only the two solo,
fixed-camera situations are included by default:
    sBM  basic dances     ~10.8k videos, 7-12 s each   (~28 h, ~128 GB across all 9 cameras)
    sFM  advanced dances  ~1.9k videos, 28-48 s each   (~19 h, ~89 GB across all 9 cameras)
The other situations (battle, cypher, group, showcase, moving camera) have several people
or a moving camera, so they are excluded.

Each video is split into the fewest equal-length chunks that are all <= --max-len, e.g. a
48 s video becomes 3 x 16 s. Chunks are re-encoded (H.264 + AAC) so cuts are frame exact;
videos already short enough are copied as-is. The embedded music track is kept.

Output (next to this script):
    aist/raw/<name>.mp4             downloaded videos (deleted after cutting unless --keep-raw)
    aist/clips/<name>_<NN>.mp4      clips
    aist/manifest.csv               one row per clip with genre/camera/dancer/music/offsets

The run is resumable: videos whose clips already exist are skipped.

Usage:
    python download_aist.py                       # everything (~47 h of video)
    python download_aist.py --cameras c01         # front camera only (~5 h, ~24 GB download)
    python download_aist.py --situations sFM --genres gBR gPO --workers 8

Requires ffmpeg/ffprobe on PATH. Python standard library only.
AIST Dance DB license: research and non-commercial use only.
"""

import argparse
import csv
import io
import math
import shutil
import subprocess
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

LIST_URL = "https://aistdancedb.ongaaccel.jp/data/video_refined/10M/refined_10M_{}_all.csv"
SOLO_SITUATIONS = ["sBM", "sFM"]
ROOT = Path(__file__).resolve().parent / "aist"
MANIFEST_FIELDS = ["clip", "source", "genre", "situation", "camera", "dancer", "music",
                   "choreography", "start", "duration"]


def fetch_list(situation: str) -> list[dict]:
    with urllib.request.urlopen(LIST_URL.format(situation)) as r:
        return list(csv.DictReader(io.StringIO(r.read().decode())))


def probe_duration(path: Path) -> float:
    out = subprocess.check_output(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                                   "-of", "csv=p=0", str(path)])
    return float(out)


def download(url: str, dest: Path, expected_size: int) -> None:
    if dest.exists() and dest.stat().st_size == expected_size:
        return
    part = dest.with_suffix(".part")
    with urllib.request.urlopen(url) as r, open(part, "wb") as f:
        shutil.copyfileobj(r, f, length=1 << 20)
    if part.stat().st_size != expected_size:
        part.unlink()
        raise IOError(f"size mismatch for {dest.name}")
    part.rename(dest)


def cut(src: Path, clip_dir: Path, n: int) -> list[tuple[Path, float, float]]:
    """Split src into n equal chunks. Returns (clip, start, duration) per chunk."""
    length = probe_duration(src) / n
    clips = []
    for i in range(n):
        out = clip_dir / f"{src.stem}_{i:02d}.mp4"
        tmp = out.with_name(out.stem + ".tmp.mp4")
        if n == 1:
            shutil.copyfile(src, tmp)
        else:
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{i * length:.3f}", "-i", str(src),
                            "-t", f"{length:.3f}", "-c:v", "libx264", "-preset", "fast", "-crf", "18",
                            "-c:a", "aac", "-b:a", "192k", str(tmp)], check=True)
        tmp.rename(out)
        clips.append((out, i * length, length))
    return clips


def process(row: dict, args) -> list[dict]:
    name = row["BASE_NAME"]
    raw = ROOT / "raw" / row["FILE_NAME"]
    clip_dir = ROOT / "clips"
    sec = float(row["SEC"])
    n = max(1, math.ceil(sec / args.max_len))
    done = [clip_dir / f"{name}_{i:02d}.mp4" for i in range(n)]
    if all(p.exists() for p in done):  # finished on a previous run
        chunks = [(p, i * sec / n, sec / n) for i, p in enumerate(done)]
    else:
        download(row["URL"], raw, int(row["FILE_SIZE"]))
        chunks = cut(raw, clip_dir, n)
        if not args.keep_raw:
            raw.unlink()
    meta = {"source": name, "genre": row["GENRE"], "situation": row["SITUATION"],
            "camera": row["CAMERA"], "dancer": row["DANCER"], "music": row["MUSIC"],
            "choreography": row["CHOREOGRAPHY"]}
    return [{"clip": p.name, **meta, "start": f"{s:.3f}", "duration": f"{d:.3f}"}
            for p, s, d in chunks]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--situations", nargs="+", default=SOLO_SITUATIONS, help="default: sBM sFM")
    ap.add_argument("--cameras", nargs="+", help="e.g. c01 (front). Default: all c01-c09")
    ap.add_argument("--genres", nargs="+", help="e.g. gBR gPO. Default: all 10")
    ap.add_argument("--max-len", type=float, default=20.0, help="max clip length in seconds")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--keep-raw", action="store_true", help="keep downloaded videos after cutting")
    ap.add_argument("--dry-run", action="store_true", help="print what would be downloaded and exit")
    args = ap.parse_args()

    if not args.dry_run and not (shutil.which("ffmpeg") and shutil.which("ffprobe")):
        sys.exit("ffmpeg and ffprobe must be on PATH")

    rows = [r for s in args.situations for r in fetch_list(s)]
    rows = [r for r in rows
            if (not args.cameras or r["CAMERA"] in args.cameras)
            and (not args.genres or r["GENRE"] in args.genres)]
    hours = sum(float(r["SEC"]) for r in rows) / 3600
    gb = sum(int(r["FILE_SIZE"]) for r in rows) / 1e9
    print(f"{len(rows)} videos, {hours:.1f} h, {gb:.1f} GB to download")
    if args.dry_run:
        return

    (ROOT / "raw").mkdir(parents=True, exist_ok=True)
    (ROOT / "clips").mkdir(parents=True, exist_ok=True)
    manifest, failed = [], []
    with ThreadPoolExecutor(args.workers) as ex:
        futures = {ex.submit(process, r, args): r["BASE_NAME"] for r in rows}
        for i, fut in enumerate(as_completed(futures), 1):
            name = futures[fut]
            try:
                manifest += fut.result()
            except Exception as e:
                failed.append(name)
                print(f"FAILED {name}: {e}", file=sys.stderr)
            print(f"[{i}/{len(rows)}] {name}", flush=True)

    with open(ROOT / "manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        w.writeheader()
        w.writerows(sorted(manifest, key=lambda m: m["clip"]))
    print(f"{len(manifest)} clips in {ROOT / 'clips'}; {len(failed)} videos failed"
          + (" (rerun to retry)" if failed else ""))


if __name__ == "__main__":
    main()
