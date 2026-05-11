#!/bin/bash
# Run training, then mirror the new run folder to GCS.
# Usage: bash pose_bert/scripts/train_and_upload.sh [args forwarded to train.py]
# Override bucket with: BUCKET=my-bucket bash pose_bert/scripts/train_and_upload.sh ...
set -euo pipefail

BUCKET="${BUCKET:-mabe-challenge-ido}"

# 1. Train — propagates failure so we won't upload a broken run
python3 -m pose_bert.model.train "$@"

# 2. Resolve the run folder just created via the 'latest' symlink
RUN_DIR=$(readlink -f checkpoints/pose_bert/latest)
RUN_NAME=$(basename "$RUN_DIR")

# 3. Mirror to bucket (same folder structure as local)
echo "Uploading $RUN_DIR -> gs://$BUCKET/runs/$RUN_NAME"
gsutil -m rsync -r "$RUN_DIR" "gs://$BUCKET/runs/$RUN_NAME"
echo "Upload done."
