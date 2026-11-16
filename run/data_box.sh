#!/usr/bin/env bash
# 第 2 步: 箱子交互数据. 纯 CPU.
#   OmniRetarget robot-object (HF omniretarget/OmniRetarget_Dataset 的 robot-object.zip, MIT, 不需要登录):
#   G1 29 自由度 + OMOMO 大箱子, 30 fps, 1826 条 (312 条原始 + 增广), 2.98 h.
# 产出:
#   data/npz/omniretarget_box/             50 Hz npz + 箱子轨迹; manifest.csv (全部), manifest_original.csv (312 条原始, 评测用),
#                                          manifest_push.csv / manifest_kick.csv (只训推 / 只训踢的子集; keep 列只对该类为 1)
#                                          每条按轨迹分类: 搬 carry / 推 push / 踢 kick / other (规则见 omniretarget_to_npz.py)
#   data/npz/omniretarget_box_static_all/  每条前面接 "待机静止 1 s + 1.5 s 过渡 + 1 s 渐入" (真机从待机姿势起步用; STATIC=0 跳过)
set -euo pipefail
source "$(dirname "$0")/env.sh"
RAW="$SM_ROOT/data/raw/omniretarget"
BOX="$SM_ROOT/data/npz/omniretarget_box"
W="${WORKERS:-8}"
"$PY" - "$RAW" <<'EOF'
import sys
from huggingface_hub import hf_hub_download
hf_hub_download("omniretarget/OmniRetarget_Dataset", "robot-object.zip", repo_type="dataset", local_dir=sys.argv[1])
EOF
"$PY" scripts/data/omniretarget_to_npz.py --zip-path "$RAW/robot-object.zip" --output-dir "$BOX" --workers "$W"
"$PY" - "$BOX" <<'EOF'
import csv, sys
d = sys.argv[1]
rows = list(csv.DictReader(open(f"{d}/manifest.csv", newline="")))
for c in ("push", "kick"):
    with open(f"{d}/manifest_{c}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow({**r, "keep": int(r["category"] == c)})
    print(c, sum(r["category"] == c for r in rows), "条")
EOF
if [ "${STATIC:-1}" = 1 ]; then
  OMP_NUM_THREADS=1 "$PY" scripts/data/static_start_prefix.py --src-dir "$BOX" --out-dir "$SM_ROOT/data/npz/omniretarget_box_static_all" --workers "$W"
fi
