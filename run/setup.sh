#!/usr/bin/env bash
# 第 1 步: 装训练环境. Python 3.12 + torch 2.14.1 (CUDA 13.0) + mujoco 3.16 + mujoco-warp + mjlab, 版本全部锁在 uv.lock.
# 需要: uv (curl -LsSf https://astral.sh/uv/install.sh | sh), NVIDIA 驱动 >= 580 (CUDA 13).
set -euo pipefail
source "$(dirname "$0")/env.sh"
command -v uv >/dev/null || { echo "缺 uv: curl -LsSf https://astral.sh/uv/install.sh | sh"; exit 1; }
uv sync --frozen --group dev
"$PY" - <<'EOF'
import importlib.metadata as m
import torch
for p in ["torch", "mujoco", "mujoco-warp", "warp-lang", "mjlab"]:
    print(f"{p:12s}", m.version(p))
print("CUDA 可用:", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
EOF
