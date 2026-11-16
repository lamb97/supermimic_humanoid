# SuperMimic G1 — motion tracking (`motion` branch)

Whole-body motion tracking for the Unitree G1 with the SuperMimic method (differentiable world model + short-horizon BPTT, arXiv 2610.28480), plus export to ONNX files that the real-robot `motion_tracking_controller` (legged_control2) loads directly.

| Branch | Contents |
|---|---|
| **`motion` (this branch)** | Motion tracking: LAFAN1 (40 clips), ONNX export, sim2sim, real-robot flip demo windows |
| [`object`](https://github.com/lamb97/supermimic_humanoid/tree/object) | Object interaction: one policy that carries, pushes and kicks a box (OmniRetarget data) |

The code base is the community reproduction [CuboiLeo/SuperMimic-Reproduced](https://github.com/CuboiLeo/SuperMimic-Reproduced) (mjlab / MuJoCo Warp, MIT), with its full commit history. Its original README is in [docs/UPSTREAM_README.md](docs/UPSTREAM_README.md); it covers the code layout, the mapping to the paper, and the experiment log ([docs/](docs/)). On top of it this branch adds three things:

1. **Orientation-only anchor** (`--env.anchor-ori-only`): the goal gets only the anchor (torso) relative orientation (6-D), not its position. The orientation is observable from the IMU, so the real robot needs no position estimate.
2. **Resume fix**: the replay buffer is not saved in checkpoints. With `--resume`, training now refills the buffer before updating again; before the fix the first update sampled an empty buffer and crashed.
3. **Deployment tools** (`tools/`, `run/`): ONNX export, a check with the same onnxruntime version as the robot, a CPU MuJoCo sim2sim that follows the deployment stack, and real-robot flip windows.

## Results (our runs, for comparison)

LAFAN1, 40 clips, one H200 for 4 h (about 58k iterations at about 4.0 it/s). Evaluated with the paper's Appendix F protocol (full clips), `ckpt_latest` only:

| Observation | Command | Normalized return | Failures |
|---|---|---|---|
| Anchor position + orientation (9-D) | `run/train.sh anchor 4` | 0.894 | 2/40 |
| **Anchor orientation only (6-D, used on the real robot)** | `run/train.sh ori 4` | 0.837 | 1/40 |

The two runs differ only in `root_pos` (81% vs 18%): without position feedback the global position drifts. Local motion and orientation tracking are the same (body position 97.5%, body orientation 86%). The paper reports 0.82.

Training stops on wall-clock time (`--schedule.max-hours`). On a slower GPU, scale the time by iterations: about 4.0–4.5 it/s on an H200 and about 2.9 it/s on an A100, plus about 6 min of torch.compile warm-up at the start.

## Reproduce

Requirements: Linux, an NVIDIA GPU with driver ≥ 580 (the torch build is CUDA 13), [uv](https://docs.astral.sh/uv/), and network access to Hugging Face and GitHub. All data is public; no login is needed.

```bash
git clone -b motion https://github.com/lamb97/supermimic_humanoid.git && cd supermimic_humanoid
# Optional: keep data / logs / exports on another disk (default: inside the repo, already in .gitignore)
# export SM_ROOT=/big/disk/supermimic

bash run/setup.sh          # 1. environment: uv sync (Python 3.12, torch 2.14.1+cu130, mujoco 3.16, pinned mjlab; all locked in uv.lock)
bash run/data_lafan.sh     # 2. LAFAN1 G1 CSVs (HF lvhaidong/LAFAN1_Retargeting_Dataset) -> data/npz/lafan1 (40 clips, 2.451 h)
.venv/bin/python -m pytest tests -q   # 3. unit tests (FK / analytic heads / world model / imagination parity checks)
bash run/train.sh ori 4    # 4. train for 4 h, then evaluate; logs and checkpoints go to logs/<run name>/
```

`run/train.sh` usage:

```bash
run/train.sh <ori|anchor|paper> [hours] [any extra arguments are passed to the trainer]
#   ori    anchor orientation only (6-D)   anchor  anchor position + orientation (9-D)   paper  paper observations (no anchor)
RUN=my_run run/train.sh ori 4                  # set the run name
RESUME=my_run run/train.sh ori 4               # continue from logs/my_run/ckpt_latest.pt for 4 more hours (use the same switches)
run/train.sh ori 4 --schedule.checkpoint-interval 2000   # all trainer options: .venv/bin/python -m supermimic.scripts.train --help
```

Evaluate a checkpoint or watch it play:

```bash
.venv/bin/python scripts/eval_benchmark.py --ckpt logs/<run name>/ckpt_latest.pt --npz-dir data/npz/lafan1
.venv/bin/python scripts/play.py --help       # interactive replay (headless: --viewer viser, then open http://localhost:8080)
MUJOCO_GL=egl .venv/bin/python tools/render_clips.py --ckpt logs/<run name>/ckpt_latest.pt --tag ori --npz-dir data/npz/lafan1 --clips walk1_subject1 --out-dir videos
```

Multi-GPU training, the full BONES-SEED dataset and other upstream features are described in [docs/UPSTREAM_README.md](docs/UPSTREAM_README.md).

## Deployment: ONNX export and sim2sim

```bash
bash run/deploy_venvs.sh                 # two small deploy-side venvs: onnxruntime 1.20.1 + MuJoCo 3.2.7 / 3.16 (training env untouched)
bash run/fetch_unitree_description.sh    # deploy-side G1 model (ros-jazzy-unitree-description from the legged_control2 apt source, unpacked only)

# Export: one ONNX per clip with the reference baked in; closed-loop parity check in the training sim, then a check with onnxruntime 1.20
VARIANT=b_ori_only bash run/export_onnx.sh logs/<run name>/ckpt_latest.pt data/npz/lafan1 deploy/onnx walk1_subject1 dance1_subject1

# sim2sim (CPU MuJoCo, deployment semantics: legged_control2 observation layout / 2 ms step / PD torque clamping)
bash run/sim2sim.sh deploy data/npz/lafan1 deploy/sim2sim deploy/onnx/b_ori_only_walk1_subject1.onnx   # deploy-side g1.xml
bash run/sim2sim.sh train  data/npz/lafan1 deploy/sim2sim deploy/onnx/b_ori_only_walk1_subject1.onnx   # training model, for comparison
```

Each clip exports `<name>.onnx` (for `motion_tracking_controller`), `<name>.deploy.json` (observation term names, gains, start frame, ...), `<name>.initial_pose.json` (start joint angles) and `<name>.parity.npz` (parity record). The directory also gets one shared `train_model.mjb` (the training model, used by `sim2sim.sh train`).

Export details (relevant when matching the robot side):
- The robot runs onnxruntime 1.20 (IR ≤ 10, opset ≤ 21), so the export uses the legacy exporter with opset 17 and IR 8.
- Observation terms map to legged_control2's built-in `joint_pos`, `joint_vel`, `base_ang_vel`, `projected_gravity`, `actions` (history 10, oldest first, filled with the first value on reset) and `motion_anchor_ori_b`. The 9-D variant also uses `motion_anchor_pos_b`. No C++ changes are needed on the robot.
- The 10 future goal frames are looked up inside the ONNX graph by `time_step`. Observation scaling is exported with the graph.
- By default the start frame is the frame closest to the default standing pose within 1–6 s. `START_STEP=0` starts at frame 0 (use it for static-start clips).

### Real-robot flip windows (optional)

LAFAN1 has no aerial flips. It has two hand-supported cartwheels in `dance1_subject1` (around 106.5 s and 115.3 s) and a ground back roll in `dance2_subject4` (58.3–61.9 s). The real robot takes over from the standby pose at rest, so each window is cut to about 5 s and wrapped with "standby hold → transition → ease-in" at the start and the mirror at the end, then exported:

```bash
.venv/bin/python tools/lafan_window.py --src data/npz/lafan1/dance1_subject1.npz --name dance1_cartwheel \
    --start-range 103.3 105.6 --end-range 108.4 110.6 --out-dir data/npz/lafan1_flip_windows
VARIANT=b_ori_only START_STEP=0 PARITY_STEPS=2000 bash run/export_onnx.sh logs/<run name>/ckpt_latest.pt \
    data/npz/lafan1_flip_windows deploy/flip_onnx dance1_cartwheel
bash run/sim2sim.sh deploy data/npz/lafan1_flip_windows deploy/flip_sim deploy/flip_onnx/b_ori_only_dance1_cartwheel.onnx
```

Our result: all 3 windows (two cartwheels, one back roll) played to the end without a fall in all three sim2sim setups (deploy model, deploy model with mesh hands, training model). The cartwheel travels about 3.4 m back-left (the alternative one about 5 m sideways); the back roll stays roughly in place.

## Verification status

The results above come from this code, trained, evaluated, exported and run in sim2sim on a SLURM cluster. The scripts in `run/` are those cluster job scripts rewritten as plain shell, with the same commands and arguments, but **the rewritten scripts have not yet been run end to end on a fresh machine**. If a step fails, run the commands inside that script one at a time to find the problem.

## Layout

```
run/      reproduction scripts (env / setup / data / training / deploy venvs / deploy model download / export / sim2sim)
tools/    export_deploy.py, ort_check.py, sim2sim.py, render_clips.py, lafan_window.py (+ a copy of static_start_prefix.py)
src/      the supermimic package (upstream + anchor_ori_only + resume fix)
scripts/  upstream data conversion / evaluation / replay scripts
docs/     upstream README and experiment log
```

Data license: the LAFAN1 G1 retargeting comes from HF `lvhaidong/LAFAN1_Retargeting_Dataset`; the original LAFAN1 is Ubisoft's, under CC BY-NC-ND 4.0. This repository contains no data and no checkpoints.
