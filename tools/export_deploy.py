"""Export a SuperMimic checkpoint to motion_tracking_controller (legged_control2) ONNX files, one per clip, and record
a parity rollout that ort_check.py replays with the deploy-side onnxruntime.

ONNX interface (same names and metadata keys as BeyondMimic's whole_body_tracking exporter):
  inputs : obs [1, 930 + A] (A = 9 anchor pos+ori, 6 ori-only), time_step [1, 1]
  outputs: actions [1, 29], joint_pos / joint_vel [1, 29],
           body_pos_w / body_quat_w / body_lin_vel_w / body_ang_vel_w [1, B, 3|4] (reference at frame time_step)
obs layout (legged_control2 ObservationManager: terms in metadata order, each term's history oldest -> newest):
  joint_pos x10, joint_vel x10, base_ang_vel x10, projected_gravity x10, actions x10, [motion_anchor_pos_b], motion_anchor_ori_b
The policy goal is the 68-d features of 10 reference frames starting at time_step - 1: legged_control2 builds the
observation before forward(), from the reference the ONNX returned on the previous call (frame time_step - 1), and in
training the anchor observation and the goal window share one frame.

Run on a GPU node with the supermimic venv and PYTHONPATH=<anchor-ori-only worktree>/src.

Box checkpoints (object branch, policy without box observation): the scene and parity rollout include
the box; the ONNX interface is unchanged. --start-step 0 for static-start clips (they begin in the standby pose).
The export also saves the training MjModel (train_model.mjb, one per out dir) for the CPU-MuJoCo control run of sim2sim.py.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import mujoco
import numpy as np
import onnx
import torch
import torch.nn as nn
import tyro
from mjlab.envs import ManagerBasedRlEnv
from scipy.spatial.transform import Rotation

from supermimic.envs import SuperMimicEnvCfgArgs
from supermimic.envs.mdp.motion_library import discover_npz
from supermimic.envs.tracking_env_cfg import make_env_cfg
from supermimic.eval.benchmark import load_policy_from_checkpoint
from supermimic.policy.nets import Policy

POLICY_TERMS = ("joint_pos", "joint_vel", "base_ang_vel", "projected_gravity", "actions")
ANCHOR_SIZES = {"motion_anchor_pos_b": 3, "motion_anchor_ori_b": 6}
# Deploy-side onnxruntime is 1.20 (ros-jazzy-onnxruntime-ament): it loads IR <= 10 and opset <= 21.
OPSET = 17
IR_VERSION = 8
OUTPUT_NAMES = ["actions", "joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w"]


class DeployModel(nn.Module):
    def __init__(self, policy: Policy, tables: dict[str, torch.Tensor], n_policy_obs: int, future_frames: int):
        super().__init__()
        self.policy = policy
        for name, t in tables.items():
            self.register_buffer(name, t)
        self.register_buffer("offsets", torch.arange(future_frames))
        self.n_policy_obs = n_policy_obs
        self.last = int(tables["goal_feat"].shape[0]) - 1

    def forward(self, obs: torch.Tensor, time_step: torch.Tensor):
        t = time_step.long().squeeze(-1)
        idx = torch.clamp(torch.clamp(t - 1, min=0)[:, None] + self.offsets[None, :], max=self.last)
        goal = torch.cat([self.goal_feat[idx].reshape(obs.shape[0], -1), obs[:, self.n_policy_obs :]], dim=-1)
        actions = self.policy(obs[:, : self.n_policy_obs], goal)
        tc = torch.clamp(t, max=self.last)
        return (
            actions,
            self.joint_pos[tc],
            self.joint_vel[tc],
            self.body_pos_w[tc],
            self.body_quat_w[tc],
            self.body_lin_vel_w[tc],
            self.body_ang_vel_w[tc],
        )


class CppTerm:
    """legged_control2 ObservationTerm history: the first value fills every slot, then FIFO, output oldest -> newest."""

    def __init__(self, n: int):
        self.n, self.h = n, None

    def __call__(self, v: np.ndarray) -> np.ndarray:
        self.h = [v] * self.n if self.h is None else self.h[1:] + [v]
        return np.concatenate(self.h)


def rot(q_wxyz) -> np.ndarray:
    q = np.asarray(q_wxyz, dtype=np.float64)
    return Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()


def yaw_only(R: np.ndarray) -> np.ndarray:
    """yawQuaternion() in motion_tracking_controller/MotionCommand.h."""
    x, y, z, w = Rotation.from_matrix(R).as_quat()
    return Rotation.from_euler("z", np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))).as_matrix()


def csv(values) -> str:
    return ",".join(v if isinstance(v, str) else f"{float(v):.9g}" for v in values)


def deploy_metadata(env: ManagerBasedRlEnv, cmd, run_path: str, obs_names: list[str], hist_lengths: list[int]) -> dict:
    """Keys read by legged_control2 OnnxPolicy + MotionOnnxPolicy; gains extracted as in mjlab's get_base_metadata."""
    robot = env.scene["robot"]
    term = env.action_manager.get_term("joint_pos")
    joint_names = list(robot.joint_names)
    assert list(term.target_names) == joint_names, (term.target_names, joint_names)
    ctrl = {a.target.split("/")[-1]: a.id for a in robot.spec.actuators}
    ids = [ctrl[j] for j in joint_names]
    scale = term._scale[0].cpu().tolist() if torch.is_tensor(term._scale) else [float(term._scale)] * len(joint_names)
    return {
        "run_path": run_path,
        "joint_names": csv(joint_names),
        "joint_stiffness": csv(env.sim.mj_model.actuator_gainprm[ids, 0]),
        "joint_damping": csv(-env.sim.mj_model.actuator_biasprm[ids, 2]),
        "default_joint_pos": csv(robot.data.default_joint_pos[0].cpu().tolist()),
        "action_scale": csv(scale),
        "command_names": "motion",
        "observation_names": csv(obs_names),
        "observation_history_lengths": csv(hist_lengths),
        "anchor_body_name": cmd.cfg.anchor_body_name,
        "body_names": csv(cmd.cfg.body_names),
    }


def export_onnx(model: DeployModel, obs_dim: int, path: Path, metadata: dict) -> None:
    dummy = (torch.zeros(1, obs_dim), torch.zeros(1, 1))
    torch.onnx.export(
        model, dummy, str(path), dynamo=False, opset_version=OPSET, input_names=["obs", "time_step"], output_names=OUTPUT_NAMES
    )
    m = onnx.load(str(path))
    m.ir_version = IR_VERSION
    for k, v in metadata.items():
        m.metadata_props.append(onnx.StringStringEntryProto(key=k, value=v))
    onnx.checker.check_model(m)
    onnx.save(m, str(path))


def parity_rollout(env, cmd, policy, model: DeployModel, start: int, steps: int, obs_names: list[str], hist: int) -> dict:
    """Closed-loop deterministic rollout from `start`; rebuild every observation the way legged_control2 would."""
    robot = env.scene["robot"]
    lib = cmd.library
    torso = robot.body_names.index(cmd.cfg.anchor_body_name)
    ai = cmd.motion_anchor_body_index
    q0 = robot.data.default_joint_pos[0].double().cpu().numpy()
    origin = env.scene.env_origins[0].double().cpu().numpy()
    n_anchor = sum(ANCHOR_SIZES[n] for n in obs_names if n in ANCHOR_SIZES)
    terms = [CppTerm(hist) for _ in POLICY_TERMS]
    last_action = np.zeros(len(q0))
    rec = {k: [] for k in ("cpp_obs", "cpp_obs_noalign", "env_obs", "time_step", "env_action")}
    cmd.forced_frame[:] = start
    obs, _ = env.reset()
    cmd.forced_frame[:] = -1
    R_w2i = t_w2i = None
    for _ in range(steps):
        f = int(cmd.frame[0])
        d = robot.data
        vals = [
            d.joint_pos[0].double().cpu().numpy() - q0,
            d.joint_vel[0].double().cpu().numpy(),
            d.root_link_ang_vel_b[0].double().cpu().numpy(),
            rot(d.root_link_quat_w[0].cpu().numpy()).T @ np.array([0.0, 0.0, -1.0]),
            last_action,
        ]
        policy_part = np.concatenate([term(v) for term, v in zip(terms, vals)])
        Ra, pa = rot(d.body_link_quat_w[0, torso].cpu().numpy()), d.body_link_pos_w[0, torso].double().cpu().numpy() - origin
        Rr, pr = rot(lib.body_quat_w[f, ai].cpu().numpy()), lib.body_pos_w[f, ai].double().cpu().numpy()
        if R_w2i is None:  # MotionCommandTerm::reset(): align the start frame to the robot in position and yaw
            R_w2i = yaw_only(Ra) @ yaw_only(Rr).T
            t_w2i = pa - R_w2i @ pr
            out_align = (float(np.degrees(np.arctan2(R_w2i[1, 0], R_w2i[0, 0]))), float(np.linalg.norm(t_w2i)))

        def anchor_terms(R_a, t_a):
            terms = []
            if "motion_anchor_pos_b" in obs_names:
                terms.append(Ra.T @ (R_a @ pr + t_a - pa))
            Rrel = Ra.T @ R_a @ Rr
            terms.append(np.array([Rrel[0, 0], Rrel[0, 1], Rrel[1, 0], Rrel[1, 1], Rrel[2, 0], Rrel[2, 1]]))
            return terms

        rec["cpp_obs"].append(np.concatenate([policy_part, *anchor_terms(R_w2i, t_w2i)]).astype(np.float32))
        # Same without the start alignment: isolates formula / frame errors from the alignment offset itself
        rec["cpp_obs_noalign"].append(np.concatenate([policy_part, *anchor_terms(np.eye(3), np.zeros(3))]).astype(np.float32))
        rec["env_obs"].append(torch.cat([obs["policy"][0], obs["goal"][0, obs["goal"].shape[1] - n_anchor :]]).cpu().numpy())
        rec["time_step"].append([float(f + 1)])  # the controller calls forward(time_step=f+1) while tracking frame f
        with torch.no_grad():
            act = policy(obs["policy"], obs["goal"])
        rec["env_action"].append(act[0].cpu().numpy())
        last_action = act[0].double().cpu().numpy()
        obs, _, terminated, _, _ = env.step(act)
        if bool(terminated[0]):
            break
    out = {k: np.asarray(v, dtype=np.float32) for k, v in rec.items()}
    with torch.no_grad():
        ref = model(torch.from_numpy(out["cpp_obs"]), torch.from_numpy(out["time_step"]))
        noalign_actions = model(torch.from_numpy(out["cpp_obs_noalign"]), torch.from_numpy(out["time_step"]))[0]
    for name, t in zip(OUTPUT_NAMES, ref):
        out[f"torch_{name}"] = t.numpy()
    out["torch_actions_noalign"] = noalign_actions.numpy()
    out["start_alignment"] = np.asarray(out_align, dtype=np.float32)  # (yaw deg, translation m)
    return out


def main(
    ckpt: Path,
    variant: str,
    clips: list[str],
    npz_dir: Path,
    out_dir: Path,
    parity_steps: int = 300,
    start_window_s: tuple[float, float] = (1.0, 6.0),
    start_step: int = -1,
    device: str = "cuda:0",
) -> None:
    policy, cfg = load_policy_from_checkpoint(ckpt, device)
    policy_cpu, _ = load_policy_from_checkpoint(ckpt, "cpu")
    env_cfg = cfg["env"]
    box = env_cfg.get("box", False)
    assert not env_cfg.get("policy_box_obs", False), "the deploy ONNX has no box observation input"
    anchor_names = []
    if env_cfg.get("anchor_obs", False):
        anchor_names = ["motion_anchor_ori_b"] if env_cfg.get("anchor_ori_only", False) else ["motion_anchor_pos_b", "motion_anchor_ori_b"]
    hist = int(cfg["policy"]["history"])
    obs_names = list(POLICY_TERMS) + anchor_names
    hist_lengths = [hist] * len(POLICY_TERMS) + [1] * len(anchor_names)
    out_dir.mkdir(parents=True, exist_ok=True)
    bench_path = ckpt.parent / f"benchmark_{ckpt.stem}.json"
    bench = {}
    if bench_path.exists():
        bench = {r["name"]: r for r in json.loads(bench_path.read_text())["clips"]}

    for clip in clips:
        args = SuperMimicEnvCfgArgs(
            npz_files=[str(f) for f in discover_npz(npz_dir, names=[clip])],
            num_envs=1,
            future_frames=env_cfg["future_frames"],
            episode_length_s=1.0e5,
            action_noise_std=0.0,
            obs_noise=False,
            reset_noise=False,
            domain_randomization=False,
            sampling_mode="start",
            record_transitions=False,
            seed=0,
            anchor_obs=env_cfg.get("anchor_obs", False),
            anchor_ori_only=env_cfg.get("anchor_ori_only", False),
            **(dict(box=True, policy_box_obs=False, box_term_start_step=0) if box else {}),
        )
        env = ManagerBasedRlEnv(make_env_cfg(args), device=device)
        try:
            cmd = env.command_manager.get_term("motion")
            lib = cmd.library
            n_frames = lib.num_frames
            frames = torch.arange(n_frames, device=lib.device)
            ref = lib.ref(frames)
            tables = {
                "goal_feat": lib.goal_feat,
                "joint_pos": ref.joint_pos,
                "joint_vel": ref.joint_vel,
                "body_pos_w": lib.body_pos_w,
                "body_quat_w": lib.body_quat_w,
                "body_lin_vel_w": lib.body_lin_vel_w,
                "body_ang_vel_w": lib.body_ang_vel_w,
            }
            tables = {k: v[:n_frames].float().cpu().contiguous() for k, v in tables.items()}
            model = DeployModel(policy_cpu, tables, n_policy_obs=hist * 93, future_frames=env_cfg["future_frames"]).eval()
            obs_dim = hist * 93 + sum(ANCHOR_SIZES[n] for n in anchor_names)

            # Start step: the frame in the window closest to the default stance (skips LAFAN1's opening T-pose).
            q0 = env.scene["robot"].data.default_joint_pos[0].cpu().numpy()
            qref = tables["joint_pos"].numpy()
            lo, hi = int(start_window_s[0] * 50), min(int(start_window_s[1] * 50), n_frames - 1)
            start = lo + int(np.argmin(np.linalg.norm(qref[lo:hi] - q0, axis=1))) if start_step < 0 else start_step

            name = f"{variant}_{clip}"
            run_path = f"supermimic/{ckpt.parent.name}/{ckpt.name}"
            metadata = deploy_metadata(env, cmd, run_path, obs_names, hist_lengths)
            export_onnx(model, obs_dim, out_dir / f"{name}.onnx", metadata)
            mjb = out_dir / "train_model.mjb"  # same scene for every clip: one copy (written atomically, shards share out_dir)
            if not mjb.exists():
                tmp = out_dir / f".train_model.{os.getpid()}.mjb"
                mujoco.mj_saveModel(env.sim.mj_model, str(tmp), None)
                os.replace(tmp, mjb)

            joint_names = list(env.scene["robot"].joint_names)
            (out_dir / f"{name}.initial_pose.json").write_text(json.dumps(dict(zip(joint_names, map(float, qref[start])))))

            par = parity_rollout(env, cmd, policy, model, start, parity_steps, obs_names, hist)
            np.savez(out_dir / f"{name}.parity.npz", **par)
            diff = np.abs(par["cpp_obs"] - par["env_obs"])
            n_pol = hist * 93
            b = bench.get(clip, {})
            info = {
                "onnx": f"{name}.onnx",
                "variant": variant,
                "clip": clip,
                "box": box,
                "checkpoint": str(ckpt),
                "frames": n_frames,
                "duration_s": n_frames / 50.0,
                "start_step": start,
                "start_time_s": start / 50.0,
                "obs_dim": obs_dim,
                "observation_names": obs_names,
                "observation_history_lengths": hist_lengths,
                "anchor_body_name": cmd.cfg.anchor_body_name,
                "body_names": list(cmd.cfg.body_names),
                "opset": OPSET,
                "ir_version": IR_VERSION,
                "benchmark_norm_return": b.get("norm_return"),
                "benchmark_failed": b.get("failed"),
                "parity_steps": int(len(par["cpp_obs"])),
                "parity_max_abs_obs_policy": float(diff[:, :n_pol].max()),
                "parity_max_abs_obs_anchor": float(diff[:, n_pol:].max()) if obs_dim > n_pol else None,
                "parity_max_abs_action_deploy_vs_env": float(np.abs(par["torch_actions"] - par["env_action"]).max()),
                "parity_start_alignment_yaw_deg": float(par["start_alignment"][0]),
                "parity_start_alignment_pos_m": float(par["start_alignment"][1]),
                "parity_max_abs_obs_anchor_noalign": float(np.abs(par["cpp_obs_noalign"] - par["env_obs"])[:, n_pol:].max())
                if obs_dim > n_pol else None,
                "parity_max_abs_action_noalign_vs_env": float(np.abs(par["torch_actions_noalign"] - par["env_action"]).max()),
            }
            (out_dir / f"{name}.deploy.json").write_text(json.dumps(info, indent=2, ensure_ascii=False))
            print(f"[export] {name}: start_step {start} ({start / 50:.2f} s), obs_dim {obs_dim}, "
                  f"obs parity policy {info['parity_max_abs_obs_policy']:.2e} anchor {info['parity_max_abs_obs_anchor']}, "
                  f"action deploy-vs-env {info['parity_max_abs_action_deploy_vs_env']:.2e} over {info['parity_steps']} steps | "
                  f"start alignment {info['parity_start_alignment_yaw_deg']:.2f} deg / {info['parity_start_alignment_pos_m'] * 1000:.1f} mm, "
                  f"without it: anchor {info['parity_max_abs_obs_anchor_noalign']}, action {info['parity_max_abs_action_noalign_vs_env']:.2e}",
                  flush=True)
        finally:
            env.close()


if __name__ == "__main__":
    tyro.cli(main)
