#!/usr/bin/env bash
# The next training run: from scratch on every prepared clip, 25 sweeps, song-level
# validation across all genres, and the sharply decaying learning rate.
#
#   ./run_next.sh              # train, then score and plot
#   MINUTES=90 ./run_next.sh   # more headroom if the machine is busy
#
# Fresh model, so it goes to its own directory and leaves previous runs alone.
set -uo pipefail
cd "$(dirname "$0")"

OUT=${OUT:-checkpoints-v2}
MINUTES=${MINUTES:-75}     # ceiling only; 25 sweeps over ~1000 clips is ~35 min on an M-series GPU
SWEEPS=${SWEEPS:-25}       # one sweep = one pass over every clip, so this is also --max-uses
SNAPSHOT=${SNAPSHOT:-5}
WARMUP=${WARMUP:-200}      # steps ramping up to 9e-4; a cold start at 3x the base rate can diverge
SCORE_CLIPS=${SCORE_CLIPS:-10}

echo "=== $(date +%T): $(ls cache/*.npz | wc -l | tr -d ' ') clips prepared"
uv run train.py --out "$OUT" --fresh \
  --minutes "$MINUTES" --max-uses "$SWEEPS" --snapshot-every "$SNAPSHOT" \
  --val-songs 1 \
  --lr 3e-4 --lr-high-mult 3 --lr-high-sweeps 5 --lr-decay-sweeps 15 --warmup "$WARMUP"

echo "=== scoring $(date +%T)"
uv run evaluate.py --checkpoint "$OUT/model.pt" --limit "$SCORE_CLIPS" --baselines

echo "=== plotting $(date +%T)"
uv run plot_progress.py --out "$OUT" --png progress-v2.png --limit "$SCORE_CLIPS"
echo "=== done $(date +%T)"
