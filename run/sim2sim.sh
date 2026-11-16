#!/usr/bin/env bash
# CPU MuJoCo sim2sim: 按部署语义 (legged_control2 观测拼接 / 2 ms 步长 / mujoco_ros2_control PD 力矩截断) 跑导出的 ONNX.
# 用法: run/sim2sim.sh <deploy|mesh|train> <npz 目录> <输出目录> <onnx 文件...>
#   deploy  部署端 unitree_description 的 g1.xml (先跑 run/fetch_unitree_description.sh)
#   mesh    同上, 但手/腕碰撞换成训练用的网格手胶囊 (unitree 的手胶囊比真橡胶手胖, 搬箱子会抓不住)
#   train   export 时存下的训练模型 .mjb (MuJoCo 3.16): 与 deploy 对照, 区分 "机器人模型差异" 和 "部署管线差异"
# 环境变量: DEPLOY_XML (默认在 $SM_ROOT/deploy_src 下找 g1.xml); RENDER=0 不出视频; 其余参数见 tools/sim2sim.py --help
# 无显示器出视频要 EGL (MUJOCO_GL=egl, 默认); 没 GPU 的机器可装 Mesa 走软件 EGL.
set -euo pipefail
source "$(dirname "$0")/env.sh"
MODE="${1:?deploy|mesh|train}"; NPZ="${2:?npz 目录}"; OUT="${3:?输出目录}"; shift 3
XML="${DEPLOY_XML:-$(find "$SM_ROOT/deploy_src" -path '*mjcf/g1.xml' 2>/dev/null | head -1)}"
[ -f "$XML" ] || { echo "找不到部署 g1.xml: 先跑 run/fetch_unitree_description.sh 或设 DEPLOY_XML"; exit 1; }
VENV=ort120; MODEL=deploy; HANDS=unitree; ARGS=()
[ "$MODE" = mesh ] && HANDS=mesh
[ "$MODE" = train ] && VENV=ort120_mj316 && MODEL=train
[ "${RENDER:-1}" = 0 ] && ARGS+=(--no-render)
export MUJOCO_GL="${MUJOCO_GL:-egl}"
mkdir -p "$OUT"
"$DEPLOY_VENVS/$VENV/bin/python" tools/sim2sim.py --onnx "$@" --npz-dir "$NPZ" --deploy-xml "$XML" --model "$MODEL" \
  --hands "$HANDS" --out-dir "$OUT" --tag "sim2sim_$MODE" "${ARGS[@]}" ${SIM2SIM_ARGS:-}
