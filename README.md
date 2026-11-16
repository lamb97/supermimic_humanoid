# SuperMimic G1 —— 物体交互：搬 / 推 / 踢箱子(`object` 分支)

在 SuperMimic(可微世界模型 + 短视界 BPTT,arXiv 2610.28480)上加入箱子：仿真、世界模型和 critic 都包含箱子，**一个策略同时学搬、推、踢**。数据是 OmniRetarget 的 robot-object 子集。

| 分支 | 内容 |
|---|---|
| [`motion`](https://github.com/lamb97/supermimic_humanoid/tree/motion) | 纯动作跟踪:LAFAN1,导出 ONNX,sim2sim,真机翻转 demo |
| **`object`(本分支)** | 物体交互：箱子搬、推、踢，一个策略 |

本分支 = `motion` 分支的代码(社区复现 [CuboiLeo/SuperMimic-Reproduced](https://github.com/CuboiLeo/SuperMimic-Reproduced) 加上"锚点只加朝向"和续训修复)+ 箱子改动。上游原 README 在 [docs/UPSTREAM_README.md](docs/UPSTREAM_README.md)。所有箱子开关默认关闭，全关时和纯动作版行为一致。

## 改了什么

- **箱子**(`src/supermimic/envs/box.py`):OMOMO 大箱子。网格坐标系和箱子本身不对齐，所以用最小体积有向包围盒拟合出真实箱子(32.5 × 33.8 × 36.3 cm,在网格系里转了 170.9°)作碰撞体。名义质量 0.75 kg,训练时质量随机化 0.5 到 1.0 kg,摩擦也随机化。箱子用 condim 3、priority 1,这样手和箱子之间有摩擦(G1 非脚部碰撞体默认 condim 1、没有摩擦，手会抓不住箱子)。
- **开关**:
  - `--env.box`:仿真、世界模型、critic 加入箱子。
  - `--env.policy-box-obs`:策略也看箱子，包括骨盆系下的箱子位姿和每个目标帧的参考箱子位姿。
  - `--env.box-term-start-step N`:箱子偏离参考超过 1 m 或 45° 时终止，从第 N 个 env step 开始生效。
- **世界模型**:状态 71 → 84 维(加箱子 13 维);输入特征加箱子相对量 15 维、手脚到箱子的向量 12 维;输出 82 维;终止头加箱子项。ξ 加箱子质量和摩擦。
- **奖励**:箱子位置 exp(σ 0.3,权重 1)和朝向 exp(σ 0.4,权重 1),照 holosoma(OmniRetarget 官方代码)的物体项，没有接触奖励。
- **评测**:`scripts/eval_benchmark.py --manifest` 按搬、推、踢分类统计箱子误差和成功率。
- **数据脚本**:`scripts/data/omniretarget_to_npz.py`(zip → 50 Hz npz + 按轨迹分类)、`scripts/data/static_start_prefix.py`(每条片段前面接上真机待机姿势起步的过渡段)。

## 结果(我们的运行，作为复现对照)

评测设置:312 条原始片段(不含增广),无噪声、无域随机化，箱子终止从第 0 步开。**箱子成功** = 没有失败，且最后一帧箱子位置误差 < 0.2 m。所有运行都是单张 H200 训 20 小时，共 10 到 14 万轮(最好那版 14.4 万轮;速度约 1.8 到 2.1 it/s,比纯动作版慢，因为世界模型特征要算 FK)。

| 策略看箱子 | 箱子终止开启 | 数据 | 箱子成功 | 搬 / 推 / 踢 | 失败 |
|---|---|---|---|---|---|
| 否 | 第 1 万步 | 原片段 | 48.1% | 57.4 / 38.7 / 36.4 | 55 |
| 是 | 第 1 万步 | 原片段 | 72.4% | 84.5 / 65.8 / 45.5 | 18 |
| 否 | 第 0 步 | 原片段 | 68.6% | 81.3 / 63.1 / 36.4 | 39 |
| **是** | **第 0 步** | 原片段 | **83.0%** | **89.7 / 82.0 / 61.4** | 9 |
| 否 | 第 0 步 | 静止起步版(在静止起步片段上评) | 70.8% | 81.9 / 68.5 / 36.4 | 31 |

结论：策略看箱子更好;箱子终止从第 0 步开更好;踢最难。只用推的数据训，推的成功率 69.4%(全量不看箱子 63.1%);只用踢的数据训，踢 61.4%(36.4%);单项策略在其他类上基本为 0。

最好的那版是先训 8 h、再用 `RESUME` 续训 12 h 得到的(集群单个任务时限的原因)。学习率按迭代数调度，所以在一台机器上直接训 20 h 是等价的，只是不用中途补满回放池。

## 复现步骤

需要 Linux、NVIDIA GPU(驱动 ≥ 580,CUDA 13 版 torch)、[uv](https://docs.astral.sh/uv/),以及能访问 Hugging Face 和 GitHub 的网络。数据公开，不需要登录。

```bash
git clone -b object https://github.com/lamb97/supermimic_humanoid.git && cd supermimic_humanoid
# 可选: export SM_ROOT=/big/disk/supermimic   (数据 / 日志 / 导出结果放别的盘; 默认在仓库目录下)

bash run/setup.sh        # 1. 装环境 (uv sync, 版本锁在 uv.lock)
bash run/data_box.sh     # 2. 下载 OmniRetarget robot-object.zip 并转换, 几分钟 (见下)
bash run/data_lafan.sh   #    (可选) LAFAN1, 只有上游的单元测试需要
.venv/bin/python -m pytest tests -q          # 3. 单元测试 (含 tests/test_box.py)
bash run/train_box.sh boxobs 20              # 4. 最好的配置: 策略看箱子, 箱子终止从第 0 步, 20 h, 结束自动评测
```

`run/data_box.sh` 的产出:
- `data/npz/omniretarget_box/`:1826 条 50 Hz npz(312 条原始 + 增广,共 2.98 h)。清单有 `manifest.csv`(全部)、`manifest_original.csv`(312 条原始，评测用)、`manifest_push.csv` 和 `manifest_kick.csv`(只训推或只训踢)。分类按轨迹判定，原始片段里搬 155、推 111、踢 44、其他 2。注意文件名都叫 largebox,不代表动作类型。
- `data/npz/omniretarget_box_static_all/`:同样 1826 条，每条前面接上"待机姿势静止 1 s + 1.5 s 过渡 + 1 s 渐入"。真机是从待机姿势接管的，所以部署模型用这份数据训。设 `STATIC=0` 可以跳过这一步。

`run/train_box.sh` 的用法:

```bash
run/train_box.sh <boxobs|nobox> [小时数, 默认 8] [其余参数原样传给训练脚本]
BOX_TERM_START=10000 run/train_box.sh boxobs 20                      # 箱子终止从第 1 万步开 (默认 0)
DATA_SUBSET=push run/train_box.sh nobox 20                           # 只用推 (或 kick)
DATA_NAME=omniretarget_box_static_all run/train_box.sh nobox 20      # 静止起步版 (真机部署用的那版)
RESUME=<运行名> run/train_box.sh boxobs 12                            # 续训 (开关要一致)
```

单独评测(按类别输出，结果写到存档旁边的 `benchmark_<存档名>.json`):

```bash
.venv/bin/python scripts/eval_benchmark.py --ckpt logs/<运行名>/ckpt_latest.pt \
    --npz-dir data/npz/omniretarget_box --manifest data/npz/omniretarget_box/manifest_original.csv
MUJOCO_GL=egl .venv/bin/python tools/render_clips.py --ckpt logs/<运行名>/ckpt_latest.pt --tag boxobs \
    --best-per-category 3 --npz-dir data/npz/omniretarget_box --out-dir videos   # 每类挑 3 条箱子误差最小的成功片段出视频
```

## 部署：导出 ONNX 和 sim2sim

真机拿不到箱子位姿，所以部署用**不看箱子**的策略(静止起步版):ONNX 的输入和纯动作版完全一样，`motion_tracking_controller` 不需要改。导出脚本会拒绝看箱子的存档。箱子要按参考轨迹第一帧的位置摆好。

```bash
bash run/deploy_venvs.sh                # onnxruntime 1.20.1 + MuJoCo 3.2.7 / 3.16 小 venv
bash run/fetch_unitree_description.sh   # 部署端 G1 模型 (ros-jazzy-unitree-description, 只解包)

C=sub10_largebox_060_original_static    # 例: 一条搬箱子 (静止起步版片段名 = <原名>_static)
VARIANT=nobox_static START_STEP=0 bash run/export_onnx.sh logs/<静止起步 nobox 运行>/ckpt_latest.pt \
    data/npz/omniretarget_box_static_all deploy/onnx $C
SIM2SIM_ARGS="--box-mass 0.6" bash run/sim2sim.sh deploy data/npz/omniretarget_box_static_all deploy/sim2sim deploy/onnx/nobox_static_$C.onnx
SIM2SIM_ARGS="--box-mass 0.6" bash run/sim2sim.sh mesh   data/npz/omniretarget_box_static_all deploy/sim2sim deploy/onnx/nobox_static_$C.onnx
SIM2SIM_ARGS="--box-mass 0.6" bash run/sim2sim.sh train  data/npz/omniretarget_box_static_all deploy/sim2sim deploy/onnx/nobox_static_$C.onnx
```

sim2sim 三种模式:
- `deploy`:部署端的 `unitree_description` g1.xml。
- `mesh`:同一个模型，但手和手腕的碰撞体换成训练用的网格手胶囊。
- `train`:训练模型对照。

箱子默认 0.6 kg(和我们的真箱子一样),成功判据和训练评测一致。`--no-box-term` 表示只在摔倒时结束。

**已知问题：部署模型的手太胖。** unitree 的 g1.xml 手部碰撞胶囊半径 5 cm,够不到指尖;训练用的橡胶手胶囊是 3.5 cm,真橡胶手宽 6.7 cm、长到 17 cm。结果是用部署模型搬箱子时容易抓歪或抓不住。我们挑了 219 条在训练仿真里成功的搬和推片段跑 sim2sim:`deploy` 101 条成功,`mesh` 212 条,`train` 211 条，所有运行都没有摔倒。所以搬箱失败的原因是部署模型的手部几何，不是导出或时序问题。真机建议先做推，再做搬。

## 已知问题

- **回报采样器偏向片段末尾。** `bin_return` 是折扣 return-to-go,片段结尾不做 bootstrap,所以每条片段最后一个 bin 的回报天然最低，按"难度"采样时 90% 以上的起点落在片段最后 1 s。OmniRetarget 片段平均只有 5.9 s,影响明显：大约一半起点被浪费，片段开头练得少。可能的修法是按剩余步数的可达上限归一化回报，但这会改上游的共享采样器，目前**没有修**。
- **静止起步的过渡段可能扫到箱子。** 过渡段的手臂路径没有避开箱子，约 11% 的片段在过渡时手腕穿进箱子超过 1 cm。

## 验证状态

上面的结果用的是同一份代码，在 SLURM 集群上训练、评测、导出、sim2sim 得到的。`run/` 里的脚本是从当时的作业脚本改写成的通用 shell,命令和参数都没变，但**改写后还没有在全新机器上从头完整跑过一遍**。哪一步报错，就把那个脚本里的命令拿出来单独跑，方便定位。

## 目录

```
run/      复现脚本 (env / setup / 数据 / 训练 / 部署 venv / 部署模型下载 / 导出 / sim2sim)
tools/    export_deploy.py, ort_check.py, sim2sim.py (含箱子), render_clips.py (含箱子和参考箱子 ghost)
src/      supermimic 包 (上游 + anchor_ori_only + 续训修复 + 箱子)
scripts/  数据转换 (含 omniretarget_to_npz.py / static_start_prefix.py)、评测、回放
tests/    上游测试 + test_box.py
```

数据许可:OmniRetarget 数据集([HF omniretarget/OmniRetarget_Dataset](https://huggingface.co/datasets/omniretarget/OmniRetarget_Dataset))数据卡标注 MIT,机器人-物体轨迹源自 OMOMO。本仓库不包含任何数据或存档。
