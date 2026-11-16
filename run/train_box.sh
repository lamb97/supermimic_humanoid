#!/usr/bin/env bash
# 第 3 步: 箱子交互训练 (搬 / 推 / 踢一个策略), 结束后在 312 条原始片段上评测 (箱子终止从第 0 步开, 无噪声 / DR).
# 世界模型和 critic 总是含箱子; 锚点观测固定用 "只加朝向 6 维" (真机不需要位置估计); 手是 mjlab G1 自带的橡胶手胶囊.
# 用法: run/train_box.sh <boxobs|nobox> [小时数, 默认 8] [其余参数原样传给训练脚本]
#   boxobs  策略看箱子 (骨盆系箱子位姿 + 每个目标帧的参考箱子位姿)  —— 仿真里最好
#   nobox   策略不看箱子, 只跟踪人形动作 (真机拿不到箱子位姿时用这个)
# 环境变量:
#   BOX_TERM_START  箱子偏离终止从第几个 env step 开启 (默认 0 = 一开始就追箱子; 代码默认 10000, 实测 0 明显更好)
#   DATA_NAME       data/npz 下的数据目录: omniretarget_box (默认, 原片段) | omniretarget_box_static_all (静止起步版)
#   DATA_SUBSET     all (默认) | push | kick
#   RUN             运行名 (默认 box_<boxobs|nobox>_<数据>_<子集>_bts<N>_<时间>)
#   RESUME          已有运行名: 从其 ckpt_latest.pt 续训 (开关须与原 run 一致; 回放池不存档, 续训会先补满再更新)
set -euo pipefail
source "$(dirname "$0")/env.sh"
OBS="${1:?用法: run/train_box.sh <boxobs|nobox> [小时数]}"
HOURS="${2:-8}"
shift $(( $# < 2 ? $# : 2 ))
case "$OBS" in
  boxobs) OBSARG=(--env.policy-box-obs) ;;
  nobox) OBSARG=(--env.no-policy-box-obs) ;;
  *) echo "第一个参数只能是 boxobs / nobox"; exit 1 ;;
esac
BTS="${BOX_TERM_START:-0}"
DATA_NAME="${DATA_NAME:-omniretarget_box}"
DATA="$SM_ROOT/data/npz/$DATA_NAME"
SUBSET="${DATA_SUBSET:-all}"
SUBARG=()
[ "$SUBSET" != all ] && SUBARG=(--data.manifest "$DATA/manifest_${SUBSET}.csv")
RUN="${RUN:-box_${OBS}_${DATA_NAME#omniretarget_}_${SUBSET}_bts${BTS}_$(date +%m%d_%H%M)}"
RESARG=()
if [ -n "${RESUME:-}" ]; then RUN="${RESUME}_resume_$(date +%m%d_%H%M)"; RESARG=(--resume "$SM_ROOT/logs/$RESUME/ckpt_latest.pt"); fi
echo "== 训练 $RUN, ${HOURS} h"
"$PY" -m supermimic.scripts.train --run-name "$RUN" --log-root "$SM_ROOT/logs" --data.npz-dir "$DATA" "${SUBARG[@]}" "${RESARG[@]}" \
  --env.box "${OBSARG[@]}" --env.box-term-start-step "$BTS" --env.anchor-obs --env.anchor-ori-only \
  --schedule.max-hours "$HOURS" "$@"
echo "== benchmark: 312 条原始片段, 按 搬/推/踢 分类"
"$PY" scripts/eval_benchmark.py --ckpt "$SM_ROOT/logs/$RUN/ckpt_latest.pt" --npz-dir "$DATA" --manifest "$DATA/manifest_original.csv"
