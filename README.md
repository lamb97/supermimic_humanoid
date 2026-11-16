# SuperMimic G1 — object interaction: carry / push / kick a box (`object` branch)

SuperMimic (differentiable world model + short-horizon BPTT, arXiv 2610.28480) extended with a box: the simulation, the world model and the critic all include the box, and **one policy learns to carry, push and kick it**. The data is the robot-object subset of OmniRetarget.

| Branch | Contents |
|---|---|
| [`motion`](https://github.com/lamb97/supermimic_humanoid/tree/motion) | Motion tracking: LAFAN1, ONNX export, sim2sim, real-robot flip demos |
| **`object` (this branch)** | Object interaction: one policy that carries, pushes and kicks a box |

This branch = the `motion` branch code (the community reproduction [CuboiLeo/SuperMimic-Reproduced](https://github.com/CuboiLeo/SuperMimic-Reproduced) plus the orientation-only anchor and the resume fix) + the box changes. The original upstream README is in [docs/UPSTREAM_README.md](docs/UPSTREAM_README.md). All box switches are off by default; with all of them off, training behaves like the motion-only version.

## What changed

- **Box** (`src/supermimic/envs/box.py`): the OMOMO large box. Its mesh frame is not aligned with the box itself, so the collision geom is a minimum-volume oriented bounding box fit (32.5 × 33.8 × 36.3 cm, rotated 170.9° inside the mesh frame). Nominal mass 0.75 kg; training randomizes the mass in 0.5–1.0 kg and also the friction. The box uses condim 3 and priority 1 so that hand–box contacts have friction (G1's non-foot collision geoms default to condim 1, which is frictionless, and the hands cannot grip).
- **Switches**:
  - `--env.box`: add the box to the simulation, the world model and the critic.
  - `--env.policy-box-obs`: the policy also sees the box: its pose in the pelvis frame and the reference box pose for every goal frame.
  - `--env.box-term-start-step N`: terminate when the box is more than 1 m or 45° off its reference, starting at env step N.
- **World model**: state 71 → 84 (+13 for the box). Input features gain 15 box-relative terms and 12 hand/foot-to-box vectors. Output is 82-D. The termination head gets box terms. ξ gains the box mass and friction.
- **Rewards**: box position exp (σ 0.3, weight 1) and orientation exp (σ 0.4, weight 1), following the object terms in holosoma (the official OmniRetarget code). There is no contact reward.
- **Evaluation**: `scripts/eval_benchmark.py --manifest` reports box error and success rate per category (carry / push / kick).
- **Data scripts**: `scripts/data/omniretarget_to_npz.py` (zip → 50 Hz npz + classification by trajectory) and `scripts/data/static_start_prefix.py` (prepends a transition from the real robot's standby pose to every clip).

## Results (our runs, for comparison)

Evaluation setup: the 312 original clips (no augmentations), no noise, no domain randomization, box termination on from step 0. **Box success** = no failure, and the final box position error < 0.2 m. Every run trained for 20 h on one H200, for 100k–144k iterations (144k for the best run). Speed is about 1.8–2.1 it/s, slower than motion-only because the world-model features need FK.

| Policy sees box | Box termination from | Data | Box success | Carry / push / kick | Failures |
|---|---|---|---|---|---|
| no | step 10k | original clips | 48.1% | 57.4 / 38.7 / 36.4 | 55 |
| yes | step 10k | original clips | 72.4% | 84.5 / 65.8 / 45.5 | 18 |
| no | step 0 | original clips | 68.6% | 81.3 / 63.1 / 36.4 | 39 |
| **yes** | **step 0** | original clips | **83.0%** | **89.7 / 82.0 / 61.4** | 9 |
| no | step 0 | static-start set (evaluated on static-start clips) | 70.8% | 81.9 / 68.5 / 36.4 | 31 |

Takeaways: seeing the box helps; box termination from step 0 helps; kicking is the hardest. Training on push data only gives 69.4% on push (vs 63.1% for the full-data run without box observation). Training on kick data only gives 61.4% on kick (vs 36.4%). These single-category policies score close to 0 on the other categories.

The best run trained for 8 h and then resumed with `RESUME` for 12 h, because of the cluster's per-job time limit. The learning-rate schedule follows iterations, so a single 20 h run on one machine is equivalent, minus the mid-run replay-buffer refill.

## Reproduce

Requirements: Linux, an NVIDIA GPU with driver ≥ 580 (the torch build is CUDA 13), [uv](https://docs.astral.sh/uv/), and network access to Hugging Face and GitHub. The data is public; no login is needed.

```bash
git clone -b object https://github.com/lamb97/supermimic_humanoid.git && cd supermimic_humanoid
# Optional: export SM_ROOT=/big/disk/supermimic   (data / logs / exports on another disk; default: inside the repo)

bash run/setup.sh        # 1. environment (uv sync, versions locked in uv.lock)
bash run/data_box.sh     # 2. download OmniRetarget robot-object.zip and convert it, a few minutes (see below)
bash run/data_lafan.sh   #    (optional) LAFAN1, only needed by the upstream unit tests
.venv/bin/python -m pytest tests -q          # 3. unit tests (including tests/test_box.py)
bash run/train_box.sh boxobs 20              # 4. best setup: policy sees the box, box termination from step 0, 20 h, then evaluation
```

`run/data_box.sh` produces:
- `data/npz/omniretarget_box/`: 1826 clips at 50 Hz (312 originals + augmentations, 2.98 h). Manifests: `manifest.csv` (all), `manifest_original.csv` (the 312 originals, for evaluation), `manifest_push.csv` and `manifest_kick.csv` (train on push or kick only). Categories come from the trajectories. Among the originals: 155 carry, 111 push, 44 kick, 2 other. Note that every file name says largebox; the name does not tell you the action.
- `data/npz/omniretarget_box_static_all/`: the same 1826 clips, each prefixed with "1 s standing still in the standby pose + 1.5 s transition + 1 s ease-in". The real robot takes over from the standby pose, so the deployed model trains on this set. `STATIC=0` skips this step.

`run/train_box.sh` usage:

```bash
run/train_box.sh <boxobs|nobox> [hours, default 8] [any extra arguments are passed to the trainer]
BOX_TERM_START=10000 run/train_box.sh boxobs 20                      # box termination from step 10k (default 0)
DATA_SUBSET=push run/train_box.sh nobox 20                           # push data only (or kick)
DATA_NAME=omniretarget_box_static_all run/train_box.sh nobox 20      # static-start set (the version we deploy)
RESUME=<run name> run/train_box.sh boxobs 12                         # resume (use the same switches)
```

Evaluate on its own (per-category output, written next to the checkpoint as `benchmark_<checkpoint name>.json`):

```bash
.venv/bin/python scripts/eval_benchmark.py --ckpt logs/<run name>/ckpt_latest.pt \
    --npz-dir data/npz/omniretarget_box --manifest data/npz/omniretarget_box/manifest_original.csv
MUJOCO_GL=egl .venv/bin/python tools/render_clips.py --ckpt logs/<run name>/ckpt_latest.pt --tag boxobs \
    --best-per-category 3 --npz-dir data/npz/omniretarget_box --out-dir videos   # videos of the 3 successes per category with the lowest box error
```

## Deployment: ONNX export and sim2sim

The real robot cannot see the box pose, so deployment uses a policy **without box observation** (the static-start version). Its ONNX inputs are exactly those of the motion-only version, so `motion_tracking_controller` needs no changes. The export script refuses checkpoints whose policy sees the box. Place the box where the reference trajectory starts.

```bash
bash run/deploy_venvs.sh                # small venvs: onnxruntime 1.20.1 + MuJoCo 3.2.7 / 3.16
bash run/fetch_unitree_description.sh   # deploy-side G1 model (ros-jazzy-unitree-description, unpacked only)

C=sub10_largebox_060_original_static    # example: one carry clip (static-start clip name = <original name>_static)
VARIANT=nobox_static START_STEP=0 bash run/export_onnx.sh logs/<static-start nobox run>/ckpt_latest.pt \
    data/npz/omniretarget_box_static_all deploy/onnx $C
SIM2SIM_ARGS="--box-mass 0.6" bash run/sim2sim.sh deploy data/npz/omniretarget_box_static_all deploy/sim2sim deploy/onnx/nobox_static_$C.onnx
SIM2SIM_ARGS="--box-mass 0.6" bash run/sim2sim.sh mesh   data/npz/omniretarget_box_static_all deploy/sim2sim deploy/onnx/nobox_static_$C.onnx
SIM2SIM_ARGS="--box-mass 0.6" bash run/sim2sim.sh train  data/npz/omniretarget_box_static_all deploy/sim2sim deploy/onnx/nobox_static_$C.onnx
```

The three sim2sim modes:
- `deploy`: the deploy-side `unitree_description` g1.xml.
- `mesh`: the same model with the hand and wrist collision geoms replaced by the training mesh-hand capsules.
- `train`: the training model, for comparison.

The box defaults to 0.6 kg (our real box). Success is judged as in the training benchmark. `--no-box-term` ends a run only on a fall.

**Known issue: the deploy model's hands are too fat.** The hand collision capsules in unitree's g1.xml have a 5 cm radius and do not reach the fingertips. The training rubber-hand capsules are 3.5 cm; the real rubber hand is 6.7 cm wide and up to 17 cm long. As a result, with the deploy model the box often slips or sits crooked in the hands when carried. We ran sim2sim on 219 carry and push clips that succeed in the training sim: `deploy` succeeded on 101, `mesh` on 212, `train` on 211, and no run fell. So carry failures come from the deploy model's hand geometry, not from the export or the timing. On the real robot, try pushing before carrying.

## Known issues

- **The return-based sampler favors clip ends.** `bin_return` is a discounted return-to-go with no bootstrap at the clip end, so the last bin of every clip always has the lowest return. "Hard" sampling therefore puts over 90% of the start points in the last 1 s of the clips. OmniRetarget clips average only 5.9 s, so the effect is large: about half the start points are wasted and clip beginnings are under-trained. A possible fix is to normalize the return by the maximum reachable for the remaining steps, but that changes the shared upstream sampler and is **not fixed**.
- **The static-start transition can sweep through the box.** The arm path in the transition does not avoid the box; in about 11% of the clips the wrist enters the box by more than 1 cm during the transition.

## Verification status

The results above come from this code, trained, evaluated, exported and run in sim2sim on a SLURM cluster. The scripts in `run/` are those cluster job scripts rewritten as plain shell, with the same commands and arguments, but **the rewritten scripts have not yet been run end to end on a fresh machine**. If a step fails, run the commands inside that script one at a time to find the problem.

## Layout

```
run/      reproduction scripts (env / setup / data / training / deploy venvs / deploy model download / export / sim2sim)
tools/    export_deploy.py, ort_check.py, sim2sim.py (with the box), render_clips.py (with the box and its reference ghost)
src/      the supermimic package (upstream + anchor_ori_only + resume fix + box)
scripts/  data conversion (including omniretarget_to_npz.py / static_start_prefix.py), evaluation, replay
tests/    upstream tests + test_box.py
```

Data license: the OmniRetarget dataset ([HF omniretarget/OmniRetarget_Dataset](https://huggingface.co/datasets/omniretarget/OmniRetarget_Dataset)) is marked MIT on its dataset card; its robot-object trajectories derive from OMOMO. This repository contains no data and no checkpoints.
