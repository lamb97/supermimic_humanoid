#!/usr/bin/env bash
# 第 2 步: 下载 LAFAN1 的 G1 重定向 CSV (HF lvhaidong/LAFAN1_Retargeting_Dataset 的 g1/, 40 条, 不需要登录)
# -> data/raw/lafan1/g1/*.csv -> 50 Hz npz: data/npz/lafan1/ (40 条, 共 2.451 h). 纯 CPU, 几分钟.
set -euo pipefail
source "$(dirname "$0")/env.sh"
RAW="$SM_ROOT/data/raw/lafan1"
"$PY" - "$RAW" <<'EOF'
import sys
from huggingface_hub import snapshot_download
snapshot_download("lvhaidong/LAFAN1_Retargeting_Dataset", repo_type="dataset", allow_patterns=["g1/*.csv"], local_dir=sys.argv[1])
EOF
echo "CSV: $(ls "$RAW"/g1/*.csv | wc -l) 条"
"$PY" scripts/data/csv_to_npz_batch.py --input-dir "$RAW/g1" --output-dir "$SM_ROOT/data/npz/lafan1" --workers "${WORKERS:-8}"
echo "npz: $(ls "$SM_ROOT"/data/npz/lafan1/*.npz | wc -l) 条"
