#!/usr/bin/env bash
# 导出真机 motion_tracking_controller (legged_control2) 能直接加载的 ONNX: 每条片段一个 .onnx,
# 参考动作烘进图里按 time_step 查表; 同时在训练仿真里闭环对拍 (ONNX vs torch), 再用部署同版本 onnxruntime 1.20 校验.
# 需要 GPU (对拍跑训练仿真) 和 run/deploy_venvs.sh 建好的 venvs/ort120.
# 用法: run/export_onnx.sh <ckpt> <npz 目录> <输出目录> <片段名...>
# 环境变量: VARIANT 文件名前缀 (默认 policy); START_STEP 参考从第几帧开始 (静止起步片段用 0;
#           不设 = 1–6 s 内离默认站姿最近的帧); PARITY_STEPS 对拍步数 (默认 300).
set -euo pipefail
source "$(dirname "$0")/env.sh"
CKPT="${1:?ckpt}"; NPZ="${2:?npz 目录}"; OUT="${3:?输出目录}"; shift 3
[ $# -gt 0 ] || { echo "至少给一条片段名"; exit 1; }
EXTRA=()
[ -n "${START_STEP:-}" ] && EXTRA+=(--start-step "$START_STEP")
[ -n "${PARITY_STEPS:-}" ] && EXTRA+=(--parity-steps "$PARITY_STEPS")
mkdir -p "$OUT"
"$PY" tools/export_deploy.py --ckpt "$CKPT" --variant "${VARIANT:-policy}" --clips "$@" --npz-dir "$NPZ" --out-dir "$OUT" "${EXTRA[@]}"
"$DEPLOY_VENVS/ort120/bin/python" tools/ort_check.py "$OUT"
