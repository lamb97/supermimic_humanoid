#!/usr/bin/env bash
# 第 3 步: 在 LAFAN1 40 条上训练, 结束后自动跑整段 benchmark (论文附录 F 协议).
# 用法: run/train.sh <ori|anchor|paper> [小时数, 默认 4] [其余参数原样传给训练脚本]
#   ori     锚点只加相对朝向 6 维 (IMU 可观测, 真机不需要位置估计) —— 真机部署用的就是这一版
#   anchor  锚点相对位置 3 + 朝向 6 = 9 维 (上游开关, 真机需要位置估计)
#   paper   论文原始观测 (无锚点; 上游实验说明会有 ~1.7 deg/s 的 yaw 漂移)
# 环境变量: RUN=运行名 (默认 lafan1_<variant>_<时间>); RESUME=<已有运行名> 从其 ckpt_latest.pt 续训 (开关须一致)
# 训练按墙钟时间停 (--schedule.max-hours); 换更慢的卡要相应加时间, 对照下面的迭代数.
set -euo pipefail
source "$(dirname "$0")/env.sh"
VARIANT="${1:?用法: run/train.sh <ori|anchor|paper> [小时数]}"
HOURS="${2:-4}"
shift $(( $# < 2 ? $# : 2 ))
case "$VARIANT" in
  ori) FLAGS=(--env.anchor-obs --env.anchor-ori-only) ;;
  anchor) FLAGS=(--env.anchor-obs) ;;
  paper) FLAGS=() ;;
  *) echo "variant 只能是 ori / anchor / paper"; exit 1 ;;
esac
NPZ="$SM_ROOT/data/npz/lafan1"
RUN="${RUN:-lafan1_${VARIANT}_$(date +%m%d_%H%M)}"
RESARG=()
if [ -n "${RESUME:-}" ]; then RUN="${RESUME}_resume_$(date +%m%d_%H%M)"; RESARG=(--resume "$SM_ROOT/logs/$RESUME/ckpt_latest.pt"); fi
echo "== 训练 $RUN (${FLAGS[*]:-论文观测}), ${HOURS} h"
"$PY" -m supermimic.scripts.train --run-name "$RUN" --log-root "$SM_ROOT/logs" --data.npz-dir "$NPZ" \
  "${FLAGS[@]}" "${RESARG[@]}" --schedule.max-hours "$HOURS" "$@"
echo "== benchmark (40 条整段, 只测 ckpt_latest)"
"$PY" scripts/eval_benchmark.py --ckpt "$SM_ROOT/logs/$RUN/ckpt_latest.pt" --npz-dir "$NPZ"
