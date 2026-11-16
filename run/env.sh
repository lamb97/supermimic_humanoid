# 公共环境变量. run/*.sh 开头都会 source 它; 手动跑命令前也可以 `source run/env.sh`.
#
# SM_ROOT  数据 / 训练日志 / 导出结果的根目录. 默认 = 仓库根目录 (data/ logs/ deploy/ venvs/ 都在 .gitignore 里).
#          想放到大盘: export SM_ROOT=/big/disk/supermimic —— 仓库下会建 data -> $SM_ROOT/data 软链 (单元测试按 data/ 找数据).
# UV_PROJECT_ENVIRONMENT  训练 venv, 默认 <仓库>/.venv (uv sync 建).
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export SM_ROOT="${SM_ROOT:-$REPO_ROOT}"
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$REPO_ROOT/.venv}"
export PY="$UV_PROJECT_ENVIRONMENT/bin/python"
# 部署侧的小 venv (onnxruntime 1.20 + 部署用 MuJoCo 版本), 由 run/deploy_venvs.sh 建
export DEPLOY_VENVS="${DEPLOY_VENVS:-$SM_ROOT/venvs}"
# 系统 PYTHONPATH 里若有 ROS 的 python 包会让 pytest / 训练崩; 别的 venv / conda 会让 uv 选错环境
unset PYTHONPATH VIRTUAL_ENV CONDA_PREFIX CONDA_DEFAULT_ENV
mkdir -p "$SM_ROOT/data" "$SM_ROOT/logs" "$SM_ROOT/deploy"
if [ "$SM_ROOT" != "$REPO_ROOT" ] && [ ! -e "$REPO_ROOT/data" ]; then ln -s "$SM_ROOT/data" "$REPO_ROOT/data"; fi
cd "$REPO_ROOT"
