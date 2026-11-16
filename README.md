# SuperMimic G1 —— 纯动作跟踪 (`motion` 分支)

用 SuperMimic 方法(可微世界模型 + 短视界 BPTT,arXiv 2610.28480)训练 Unitree G1 的全身动作跟踪策略，并导出给真机 `motion_tracking_controller`(legged_control2)直接加载的 ONNX。

| 分支 | 内容 |
|---|---|
| **`motion`(本分支)** | 纯动作跟踪:LAFAN1 40 条，导出 ONNX,sim2sim,真机翻转 demo 窗口 |
| [`object`](https://github.com/lamb97/supermimic_humanoid/tree/object) | 物体交互：一个策略同时学搬、推、踢箱子(OmniRetarget 数据) |

代码底座是社区复现 [CuboiLeo/SuperMimic-Reproduced](https://github.com/CuboiLeo/SuperMimic-Reproduced)(mjlab / MuJoCo Warp,MIT),本分支保留它的全部提交历史。它原来的 README 移到了 [docs/UPSTREAM_README.md](docs/UPSTREAM_README.md),里面有代码结构、论文对应关系和实验记录([docs/](docs/))。在它之上我们做了三件事：

1. **锚点只加朝向**(`--env.anchor-ori-only`):目标里只加锚点(躯干)相对朝向 6 维，不加位置。朝向用 IMU 就能观测，真机不需要位置估计。
2. **续训修复**:回放池不进存档，用 `--resume` 续训时先把回放池补满再更新，否则第一步会因为空池采样而崩。
3. **部署工具**(`tools/`,`run/`):导出 ONNX、用部署同版本的 onnxruntime 校验、按部署语义跑 CPU MuJoCo sim2sim、截真机翻转窗口。

## 结果(我们的运行，作为复现对照)

LAFAN1 40 条，单张 H200 训 4 小时(约 5.8 万轮，速度约 4.0 it/s),用论文附录 F 协议整段评测，只测 `ckpt_latest`:

| 观测 | 命令 | 归一化回报 | 失败 |
|---|---|---|---|
| 锚点位置 + 朝向 9 维 | `run/train.sh anchor 4` | 0.894 | 2/40 |
| **锚点只加朝向 6 维(真机用这版)** | `run/train.sh ori 4` | 0.837 | 1/40 |

两版只有 `root_pos` 一项不同(81% 对 18%),原因是没有位置反馈时整体位置会漂移;局部动作和朝向跟踪完全一样(身体位置 97.5%、身体朝向 86%)。论文的数字是 0.82。

训练按墙钟时间停(`--schedule.max-hours`)。换成更慢的卡，要按迭代数折算时间：参考速度 H200 约 4.0 到 4.5 it/s,A100 约 2.9 it/s;另外开头 torch.compile 预热约 6 分钟。

## 复现步骤

需要 Linux、NVIDIA GPU(驱动 ≥ 580,因为用的是 CUDA 13 版 torch)、[uv](https://docs.astral.sh/uv/),以及能访问 Hugging Face 和 GitHub 的网络。数据都是公开的，不需要登录。

```bash
git clone -b motion https://github.com/lamb97/supermimic_humanoid.git && cd supermimic_humanoid
# 可选: 数据 / 日志 / 导出结果放到别的盘 (默认就放仓库目录下, 已在 .gitignore 里)
# export SM_ROOT=/big/disk/supermimic

bash run/setup.sh          # 1. 装环境: uv sync (Python 3.12, torch 2.14.1+cu130, mujoco 3.16, mjlab 固定版本, 全部锁在 uv.lock)
bash run/data_lafan.sh     # 2. 下载 LAFAN1 G1 CSV (HF lvhaidong/LAFAN1_Retargeting_Dataset) -> data/npz/lafan1 (40 条, 2.451 h)
.venv/bin/python -m pytest tests -q   # 3. 单元测试 (FK / 解析头 / 世界模型 / 想象 等对拍)
bash run/train.sh ori 4    # 4. 训练 4 h, 结束后自动评测; 日志和存档在 logs/<运行名>/
```

`run/train.sh` 的用法：

```bash
run/train.sh <ori|anchor|paper> [小时数] [其余参数原样传给训练脚本]
#   ori    锚点只加朝向 6 维   anchor  锚点位置+朝向 9 维   paper  论文原始观测 (无锚点)
RUN=my_run run/train.sh ori 4                  # 指定运行名
RESUME=my_run run/train.sh ori 4               # 从 logs/my_run/ckpt_latest.pt 接着训 4 h (开关要和原运行一致)
run/train.sh ori 4 --schedule.checkpoint-interval 2000   # 其余训练参数见: .venv/bin/python -m supermimic.scripts.train --help
```

单独评测某个存档，或者看回放:

```bash
.venv/bin/python scripts/eval_benchmark.py --ckpt logs/<运行名>/ckpt_latest.pt --npz-dir data/npz/lafan1
.venv/bin/python scripts/play.py --help       # 交互式回放 (无显示器时 --viewer viser, 浏览器打开 http://localhost:8080)
MUJOCO_GL=egl .venv/bin/python tools/render_clips.py --ckpt logs/<运行名>/ckpt_latest.pt --tag ori --npz-dir data/npz/lafan1 --clips walk1_subject1 --out-dir videos
```

多卡训练、BONES-SEED 全量数据等上游功能见 [docs/UPSTREAM_README.md](docs/UPSTREAM_README.md)。

## 部署：导出 ONNX 和 sim2sim

```bash
bash run/deploy_venvs.sh                 # 部署侧两个小 venv: onnxruntime 1.20.1 + MuJoCo 3.2.7 / 3.16 (不碰训练环境)
bash run/fetch_unitree_description.sh    # 下载部署端的 G1 模型 (legged_control2 apt 源里的 ros-jazzy-unitree-description, 只解包)

# 导出: 每条片段一个 ONNX, 参考动作烘在图里; 同时在训练仿真里闭环对拍, 再用 onnxruntime 1.20 校验
VARIANT=b_ori_only bash run/export_onnx.sh logs/<运行名>/ckpt_latest.pt data/npz/lafan1 deploy/onnx walk1_subject1 dance1_subject1

# sim2sim (CPU MuJoCo, 按部署语义: legged_control2 观测拼接 / 2 ms 步长 / PD 力矩截断)
bash run/sim2sim.sh deploy data/npz/lafan1 deploy/sim2sim deploy/onnx/b_ori_only_walk1_subject1.onnx   # 部署端 g1.xml
bash run/sim2sim.sh train  data/npz/lafan1 deploy/sim2sim deploy/onnx/b_ori_only_walk1_subject1.onnx   # 训练模型对照
```

每条片段导出这几个文件:`<名>.onnx`(给 `motion_tracking_controller`)、`<名>.deploy.json`(观测项名、增益、起始帧等)、`<名>.initial_pose.json`(起始关节角)、`<名>.parity.npz`(对拍记录),以及目录里共用一份的 `train_model.mjb`(训练模型，给 `sim2sim.sh train` 用)。

导出要点(和真机对齐时会用到):
- 部署侧 onnxruntime 是 1.20(只支持 IR ≤ 10、opset ≤ 21),所以导出用旧版导出器，参数是 opset 17、IR 8。
- 观测项名对应 legged_control2 自带的 `joint_pos`、`joint_vel`、`base_ang_vel`、`projected_gravity`、`actions`(历史 10 帧，旧的在前，reset 时用首帧填满)和 `motion_anchor_ori_b`;9 维版还多一项 `motion_anchor_pos_b`。部署端不需要改 C++。
- 未来 10 帧目标在 ONNX 图里按 `time_step` 查表合成;观测缩放随图导出。
- 起始帧默认取 1 到 6 s 内离默认站姿最近的一帧;设 `START_STEP=0` 就从第 0 帧开始(静止起步的片段用这个)。

### 真机翻转 demo 窗口(可选)

LAFAN1 里没有空中空翻，只有 `dance1_subject1` 的两次手撑侧手翻(约 106.5 s、115.3 s)和 `dance2_subject4` 的贴地后滚翻(58.3 到 61.9 s)。真机是从待机姿势静止接管的，所以要把片段截成 5 s 左右，首尾接上"待机静止 → 过渡 → 渐入"和对称的收尾，再导出:

```bash
.venv/bin/python tools/lafan_window.py --src data/npz/lafan1/dance1_subject1.npz --name dance1_cartwheel \
    --start-range 103.3 105.6 --end-range 108.4 110.6 --out-dir data/npz/lafan1_flip_windows
VARIANT=b_ori_only START_STEP=0 PARITY_STEPS=2000 bash run/export_onnx.sh logs/<运行名>/ckpt_latest.pt \
    data/npz/lafan1_flip_windows deploy/flip_onnx dance1_cartwheel
bash run/sim2sim.sh deploy data/npz/lafan1_flip_windows deploy/flip_sim deploy/flip_onnx/b_ori_only_dance1_cartwheel.onnx
```

我们的结果:3 个窗口(两次侧手翻、一次后滚翻)在部署模型、部署模型加网格手、训练模型三套 sim2sim 里都完整播完、没有摔。侧手翻会向左后方移动约 3.4 m(备选那次横移约 5 m),后滚翻基本在原地。

## 验证状态

上面的结果用的是同一份代码，在 SLURM 集群上训练、评测、导出、sim2sim 得到的。`run/` 里的脚本是从当时的作业脚本改写成的通用 shell,命令和参数都没变，但**改写后还没有在全新机器上从头完整跑过一遍**。哪一步报错，就把那个脚本里的命令拿出来单独跑，方便定位。

## 目录

```
run/      复现用的 shell 脚本 (env / setup / 数据 / 训练 / 部署 venv / 部署模型下载 / 导出 / sim2sim)
tools/    export_deploy.py, ort_check.py, sim2sim.py, render_clips.py, lafan_window.py (+ static_start_prefix.py 的副本)
src/      supermimic 包 (上游代码 + anchor_ori_only + 续训修复)
scripts/  上游的数据转换 / 评测 / 回放脚本
docs/     上游 README 与实验记录
```

数据许可：LAFAN1 的 G1 重定向数据来自 HF `lvhaidong/LAFAN1_Retargeting_Dataset`,原始 LAFAN1 是 Ubisoft 的 CC BY-NC-ND 4.0;本仓库不包含任何数据或存档。
