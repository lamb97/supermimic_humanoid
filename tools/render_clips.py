"""Render a trained SuperMimic policy tracking clips to mp4 (MuJoCo EGL offscreen, translucent reference ghost).

Deterministic play from each clip's first frame (no noise, no DR), the same settings as eval_benchmark.py.
Stops at the clip end, or early if the training termination fires (a fall, or with a box: the box more
than 1 m / 45 deg off its reference, on from the first step as in the benchmark). Run on a GPU node with
the supermimic venv and PYTHONPATH pointing at a worktree that knows the checkpoint's flags:

  MUJOCO_GL=egl python render_clips.py --ckpt <run>/ckpt_latest.pt --tag a --clips walk1_subject1 --out-dir videos

Box checkpoints (object branch): the box is drawn solid, its reference as an orange ghost, and the
frame carries a caption with the live box position error. Instead of --clips, --best-per-category K picks
the K box successes per category (carry / push / kick) with the lowest mean box error from the run's
benchmark_ckpt_latest.json.
"""

from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "egl")

import json
from pathlib import Path

import imageio
import numpy as np
import torch
import tyro
from mjlab.envs import ManagerBasedRlEnv
from PIL import Image, ImageDraw, ImageFont

from supermimic.envs import SuperMimicEnvCfgArgs
from supermimic.envs.mdp.motion_library import discover_npz
from supermimic.envs.tracking_env_cfg import make_env_cfg
from supermimic.eval.benchmark import BOX_SUCCESS_POS, load_policy_from_checkpoint


def best_box_clips(benchmark_json: Path, k: int) -> list[tuple[str, str]]:
    """(category, clip) of the k box successes per category with the lowest mean box position error."""
    clips = json.loads(benchmark_json.read_text())["clips"]
    ok = [c for c in clips if not c["failed"] and c["box_pos_err_final"] < BOX_SUCCESS_POS]
    out = []
    for cat in sorted({c["category"] for c in ok}):
        best = sorted((c for c in ok if c["category"] == cat), key=lambda c: c["box_pos_err_mean"])[:k]
        out += [(cat, c["name"]) for c in best]
    return out


def caption(frame: np.ndarray, text: str, font: ImageFont.ImageFont) -> np.ndarray:
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    draw.rectangle((0, 0, img.width, 44), fill=(0, 0, 0))
    draw.text((14, 8), text, fill=(255, 255, 255), font=font)
    return np.asarray(img)


def main(
    ckpt: Path,
    tag: str,
    npz_dir: Path,
    out_dir: Path,
    clips: list[str] = [],
    best_per_category: int = 0,
    seconds: float = 25.0,
    width: int = 1280,
    height: int = 720,
    device: str = "cuda:0",
) -> None:
    policy, cfg = load_policy_from_checkpoint(ckpt, device)
    box = cfg["env"].get("box", False)
    policy_box_obs = cfg["env"].get("policy_box_obs", False)
    jobs = [("", c) for c in clips]
    if best_per_category:
        jobs += best_box_clips(ckpt.parent / "benchmark_ckpt_latest.json", best_per_category)
    font = ImageFont.load_default(size=26)
    out_dir.mkdir(parents=True, exist_ok=True)
    for category, clip in jobs:
        args = SuperMimicEnvCfgArgs(
            npz_files=[str(f) for f in discover_npz(npz_dir, names=[clip])],
            num_envs=1,
            future_frames=cfg["env"]["future_frames"],
            episode_length_s=1.0e5,
            action_noise_std=0.0,
            obs_noise=False,
            reset_noise=False,
            domain_randomization=False,
            sampling_mode="start",
            record_transitions=False,
            seed=0,
            anchor_obs=cfg["env"].get("anchor_obs", False),
            anchor_ori_only=cfg["env"].get("anchor_ori_only", False),
            debug_vis=True,
            **(dict(box=True, policy_box_obs=policy_box_obs, box_term_start_step=0) if box else {}),
        )
        phase_end = None
        with np.load(args.npz_files[0]) as d:
            if not category and "category" in d:
                category = str(d["category"])
            if "static_prefix_frames" in d:  # static_start_prefix.py: hold, transition, ease-in lengths (frames)
                n_hold, n_tr, n_e, _ = d["static_prefix_frames"].tolist()
                phase_end = [("standby hold", n_hold), ("transition", n_hold + n_tr + 1), ("ease-in", n_hold + n_tr + 1 + n_e)]
        env_cfg = make_env_cfg(args)
        env_cfg.viewer.width, env_cfg.viewer.height = width, height
        env = ManagerBasedRlEnv(env_cfg, device=device, render_mode="rgb_array")
        cmd = env.command_manager.get_term("motion")
        fps = round(1.0 / env.step_dt)
        path = out_dir / f"{tag}_{category + '_' if category else ''}{clip}.mp4"
        writer = imageio.get_writer(path, fps=fps, codec="libx264", quality=8)
        fell_at, err = None, []
        try:
            obs, _ = env.reset()
            with torch.no_grad():
                for t in range(int(seconds * fps)):
                    frame = env.render()
                    if box:
                        e = f"{100 * err[-1]:.1f} cm" if err else "-"
                        phase = f"  |  {next((n for n, end in phase_end if t < end), 'clip')}" if phase_end else ""
                        frame = caption(frame, f"{tag}  |  {category}  |  {clip}{phase}  |  box error {e}   (orange = reference box)", font)
                    writer.append_data(frame)
                    obj = obs["object"] if policy_box_obs else None
                    obs, _, terminated, truncated, _ = env.step(policy(obs["policy"], obs["goal"], obj))
                    # mjlab resets finished envs inside step(), so stop before rendering the reset pose
                    if bool(terminated[0]):
                        fell_at = (t + 1) / fps
                        break
                    if bool(truncated[0]):
                        break
                    if box:
                        err.append(float(cmd.metrics["error_box_pos"][0]))
        finally:
            writer.close()
            env.close()
        status = f"TERMINATED at {fell_at:.2f} s" if fell_at is not None else f"played {(t + 1) / fps:.1f} s"
        if err:
            status += f", box error mean {100 * np.mean(err):.1f} cm / final {100 * err[-1]:.1f} cm"
        print(f"[render] {path}: {status}", flush=True)


if __name__ == "__main__":
    tyro.cli(main)
