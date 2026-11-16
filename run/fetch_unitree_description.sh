#!/usr/bin/env bash
# 下载部署端用的 G1 模型: legged_control2 apt 源 (qiayuanl/legged_buildfarm, noble-jazzy-amd64) 里的
# ros-jazzy-unitree-description, 只解包不安装. sim2sim 的 --deploy-xml 用其中的 mjcf/g1.xml.
# 需要 curl, ar (binutils), 支持 --zstd 的 tar. 我们验证时用的版本是 1.3.0-2026.10.06.16.27.
set -euo pipefail
source "$(dirname "$0")/env.sh"
BASE=https://github.com/qiayuanl/legged_buildfarm/raw/noble-jazzy-amd64
D="$SM_ROOT/deploy_src"
mkdir -p "$D/unitree_description"
curl -fsSL "$BASE/Packages" -o "$D/Packages"
F=$(awk '/^Package: ros-jazzy-unitree-description$/{p=1} p && /^Filename:/{print $2; exit}' "$D/Packages")
V=$(awk '/^Package: ros-jazzy-unitree-description$/{p=1} p && /^Version:/{print $2; exit}' "$D/Packages")
echo "ros-jazzy-unitree-description $V"
curl -fsSL "$BASE/${F#./}" -o "$D/ud.deb"
cd "$D/unitree_description"
ar x ../ud.deb
tar -xf data.tar.*
XML=$(find "$D/unitree_description" -path '*mjcf/g1.xml' | head -1)
echo "DEPLOY_XML=$XML"
