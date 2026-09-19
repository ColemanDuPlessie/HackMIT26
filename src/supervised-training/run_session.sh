#!/usr/bin/env bash
# One ~1 hour session: fetch clips, prepare them, then train under a wall-clock cap.
#
# Every stage is resumable and safe to re-run: downloads skip clips already on disk, prep
# skips clips already in cache/, and training continues from checkpoints/checkpoint.pt with
# its ledger of how often each clip has been used (never more than MAX_USES).
#
#   ./run_session.sh                      # defaults below
#   CLIPS=1200 MINUTES=55 ./run_session.sh
#   SKIP_DOWNLOAD=1 ./run_session.sh      # only prepare and train what's already downloaded
#
# Where the hour actually goes: MediaPipe preprocessing is ~3-5 s per clip, while training
# the default model on one clip (5 passes, non-overlapping windows) is ~0.25 s. Training
# stops early when every prepared clip has hit the cap -- that is expected, not a failure.
# To spend the budget on compute instead of data, raise D_MODEL/LAYERS (cost scales roughly
# with D_MODEL^2 x LAYERS) or set STRIDE=120 for overlapping windows.
set -euo pipefail
cd "$(dirname "$0")"

CLIPS=${CLIPS:-600}            # target number of downloaded clips
WORKERS=${WORKERS:-4}          # parallel MediaPipe workers; ~1 core each
POSE_MODEL=${POSE_MODEL:-lite} # lite is ~2x faster than full and fine for training data
MINUTES=${MINUTES:-55}         # wall-clock cap for the training stage
MAX_USES=${MAX_USES:-5}        # how often one clip may ever be trained on
PREP_LIMIT=${PREP_LIMIT:-$CLIPS}
D_MODEL=${D_MODEL:-384}
LAYERS=${LAYERS:-6}
STRIDE=${STRIDE:-0}            # 0 = windows don't overlap
CLIP_DIR=${CLIP_DIR:-../../dataset/aist/clips}

echo "=== 1/3 download (target $CLIPS clips)"
have=$(ls "$CLIP_DIR"/*.mp4 2>/dev/null | wc -l | tr -d ' ')
if [ "${SKIP_DOWNLOAD:-0}" = "1" ] || [ "$have" -ge "$CLIPS" ]; then
  echo "$have clips on disk; skipping download"
else
  # sBM clips are short (7-12 s) so videos ~= clips; --limit samples across genres.
  python3 ../../dataset/download_aist.py --situations sBM --cameras c01 \
    --limit "$((CLIPS - have))" --workers 4
fi

echo "=== 2/3 prepare (up to $PREP_LIMIT new clips, $WORKERS workers, $POSE_MODEL model)"
uv run prepare_data.py --clips "$CLIP_DIR" --model "$POSE_MODEL" \
  --workers "$WORKERS" --limit "$PREP_LIMIT"

echo "=== 3/3 train (cap ${MINUTES} min, max $MAX_USES uses per clip)"
uv run train.py --minutes "$MINUTES" --max-uses "$MAX_USES" \
  --stride "$STRIDE" --d-model "$D_MODEL" --layers "$LAYERS"

echo
echo "Checkpoints in checkpoints/ (model.pt is the final model, best.pt the best validation loss)."
echo "Score it:    uv run evaluate.py --baselines"
echo "Dance to it: uv run generate.py <song> --retarget danced_motion.npz"
