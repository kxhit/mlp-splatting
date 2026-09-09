#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "Usage: $0 DATA_DIR CHECKPOINT [RESULT_JSON]" >&2
  exit 2
fi

data_dir=$1
checkpoint=$2
result_json=${3:-results/room0_lseg.json}

python evaluate.py \
  --config configs/replica/room0_lseg.json \
  --data "$data_dir" \
  --checkpoint "$checkpoint" \
  --disable-lpips \
  --output "$result_json"

python - "$result_json" <<'PY'
import json
import sys

result = json.load(open(sys.argv[1], encoding="utf-8"))
psnr = result["mean_psnr_db"]
frames = result["num_frames_evaluated"]
if frames != 10:
    raise SystemExit(f"expected 10 views, got {frames}")
if psnr is None or not 36.18 <= psnr <= 36.27:
    raise SystemExit(f"expected PSNR in [36.18, 36.27] dB, got {psnr}")
print(f"PASS: {frames} views, mean PSNR {psnr:.4f} dB")
PY
