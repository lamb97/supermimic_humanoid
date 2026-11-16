#!/usr/bin/env bash
# 部署侧的两个小 venv (不碰训练 venv):
#   venvs/ort120       onnxruntime 1.20.1 (= 真机 legged_control2 链接的版本) + MuJoCo 3.2.7 (= 部署端 Docker sim2sim 的版本)
#                      -> tools/ort_check.py, tools/sim2sim.py --model deploy
#   venvs/ort120_mj316 同上但 MuJoCo 3.16 (= 训练版本), 能加载 export 存下的训练模型 .mjb -> tools/sim2sim.py --model train
set -euo pipefail
source "$(dirname "$0")/env.sh"
COMMON=("onnxruntime==1.20.1" "numpy<2.3" pillow imageio imageio-ffmpeg)
for spec in "ort120 mujoco==3.2.7" "ort120_mj316 mujoco==3.16.0"; do
  set -- $spec
  [ -x "$DEPLOY_VENVS/$1/bin/python" ] || uv venv -q --python 3.12 "$DEPLOY_VENVS/$1"
  uv pip install -q --python "$DEPLOY_VENVS/$1/bin/python" "${COMMON[@]}" "$2"
  "$DEPLOY_VENVS/$1/bin/python" -c "import onnxruntime, mujoco; print('$1', 'onnxruntime', onnxruntime.__version__, 'mujoco', mujoco.__version__)"
done
