"""Benchmark evaluation (paper Appendix F.1 / plan 5.1).

Every clip is played once from its first frame with the deterministic policy:
no observation/action noise, no reset noise, no domain randomization (foot friction 1.0),
deterministic physics. A clip *fails* if the training termination (pelvis height /
tilt vs reference) fires before the clip ends. Normalised return = sum of per-step
rewards / (max step reward * number of steps).

With a box the box termination is on from the first step, the reward includes the box terms
(max step reward 0.14), and every clip also reports its box tracking errors and category
(carry / push / kick, from the npz). Box success = the clip did not fail and the box ends
within 0.2 m of its reference.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from supermimic.envs import SuperMimicEnvCfgArgs, make_env
from supermimic.envs.mdp.rewards import TrackingRewardCfg
from supermimic.policy.nets import Policy, PolicyCfg

BOX_SUCCESS_POS = 0.2


@dataclass
class ClipResult:
    name: str
    frames: int
    steps: int
    norm_return: float
    failed: bool
    fail_step: int  # -1 if not failed
    terms: dict | None = None  # per-term mean reward per step (dt-scaled)
    category: str = ""
    box_pos_err_mean: float = float("nan")
    box_rot_err_mean: float = float("nan")
    box_pos_err_final: float = float("nan")


def load_policy_from_checkpoint(path: str | Path, device: str) -> tuple[Policy, dict]:
    ck = torch.load(path, map_location=device, weights_only=False)
    pc = ck["config"]["policy"]
    pcfg = PolicyCfg(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in pc.items()})
    policy = Policy(pcfg).to(device)
    policy.load_state_dict(ck["policy"])
    policy.eval()
    return policy, ck["config"]


@torch.no_grad()
def run_benchmark(policy: Policy, npz_files: list[str], device: str = "cuda:0", num_envs: int | None = None, future_frames: int = 10, verbose: bool = True, anchor_obs: bool = False, anchor_ori_only: bool = False, box: bool = False, policy_box_obs: bool = False) -> list[ClipResult]:
    num_envs = min(num_envs or 1024, len(npz_files))
    categories = []
    for f in npz_files:
        with np.load(f) as d:
            categories.append(str(d["category"]) if "category" in d else "")
    args = SuperMimicEnvCfgArgs(
        npz_files=npz_files,
        num_envs=num_envs,
        future_frames=future_frames,
        episode_length_s=1.0e5,  # clip end decides
        action_noise_std=0.0,
        obs_noise=False,
        reset_noise=False,
        domain_randomization=False,
        sampling_mode="start",
        record_transitions=False,
        seed=0,
        anchor_obs=anchor_obs,
        anchor_ori_only=anchor_ori_only,
        box=box,
        policy_box_obs=policy_box_obs,
        box_term_start_step=0,
    )
    env = make_env(args, device=device)
    try:
        cmd = env.command_manager.get_term("motion")
        lib = cmd.library
        n_clips = lib.num_clips
        max_r = TrackingRewardCfg(has_box=box).max_step_reward  # paper reward upper bound per step (dt * 5) [+ box dt * 2]
        box_pos_sum = torch.zeros(num_envs, device=device)
        box_rot_sum = torch.zeros(num_envs, device=device)
        box_pos_last = torch.zeros(num_envs, device=device)

        # Clip queue: env e plays clips e, e+N, e+2N, ...
        current = torch.arange(num_envs, device=device)
        nxt = current + num_envs
        cmd.forced_frame[:] = lib.clip_start[current]
        obs, _ = env.reset()
        cmd.forced_frame[:] = lib.clip_start[torch.clamp(nxt, max=n_clips - 1)]
        active = torch.ones(num_envs, dtype=torch.bool, device=device)
        ret_sum = torch.zeros(num_envs, device=device)
        steps = torch.zeros(num_envs, dtype=torch.long, device=device)
        term_names = list(env.reward_manager.active_terms)
        term_sum = torch.zeros(num_envs, len(term_names), device=device)
        results: dict[int, ClipResult] = {}
        remaining = n_clips
        while remaining > 0:
            act = policy(obs["policy"], obs["goal"], obs["object"] if policy_box_obs else None)
            obs, rew, term, tout, _ = env.step(act)
            ret_sum += rew
            term_sum += env.reward_manager.term_rewards
            steps += 1
            done = (term | tout) & active
            if box:
                # Envs that finished this step were already reset: keep their last pre-reset error.
                live = ~(term | tout)
                box_pos_sum += torch.where(live, cmd.metrics["error_box_pos"], box_pos_last)
                box_rot_sum += torch.where(live, cmd.metrics["error_box_rot"], 0.0)
                box_pos_last = torch.where(live, cmd.metrics["error_box_pos"], box_pos_last)
            if done.any():
                ids = done.nonzero().squeeze(-1)
                for e in ids.tolist():
                    ci = int(current[e])
                    if ci < n_clips and ci not in results:
                        st = int(steps[e])
                        results[ci] = ClipResult(
                            name=lib.clips[ci].name,
                            frames=lib.clips[ci].length,
                            steps=st,
                            norm_return=float(ret_sum[e]) / (max_r * max(st, 1)),
                            failed=bool(term[e]),
                            fail_step=st if bool(term[e]) else -1,
                            terms={n: float(term_sum[e, i]) / max(st, 1) for i, n in enumerate(term_names)},
                            category=categories[ci],
                            box_pos_err_mean=float(box_pos_sum[e]) / max(st, 1) if box else float("nan"),
                            box_rot_err_mean=float(box_rot_sum[e]) / max(st - 1, 1) if box else float("nan"),
                            box_pos_err_final=float(box_pos_last[e]) if box else float("nan"),
                        )
                        remaining -= 1
                        if verbose and (len(results) % 50 == 0 or len(results) == n_clips):
                            print(f"  evaluated {len(results)}/{n_clips}")
                # advance the queue for these envs (the auto-reset already used forced_frame)
                current[ids] = nxt[ids]
                nxt[ids] = nxt[ids] + num_envs
                finished = current[ids] >= n_clips
                active[ids[finished]] = False
                cmd.forced_frame[ids] = lib.clip_start[torch.clamp(nxt[ids], max=n_clips - 1)]
                ret_sum[ids] = 0.0
                term_sum[ids] = 0.0
                steps[ids] = 0
                box_pos_sum[ids] = 0.0
                box_rot_sum[ids] = 0.0
        return [results[i] for i in range(n_clips)]
    finally:
        env.close()


def _summary_of(results: list[ClipResult]) -> dict:
    n = len(results)
    failed = sum(r.failed for r in results)
    mean_ret = sum(r.norm_return for r in results) / max(n, 1)
    # Length-weighted return (paper Fig. 3a uses the benchmark aggregate; weighting by steps).
    tot_steps = sum(r.steps for r in results)
    w_ret = sum(r.norm_return * r.steps for r in results) / max(tot_steps, 1)
    out = {"clips": n, "failed": failed, "mean_norm_return": mean_ret, "step_weighted_norm_return": w_ret}
    if results and not np.isnan(results[0].box_pos_err_mean):
        out["box_pos_err_mean"] = float(np.mean([r.box_pos_err_mean for r in results]))
        out["box_rot_err_mean"] = float(np.mean([r.box_rot_err_mean for r in results]))
        out["box_pos_err_final"] = float(np.mean([r.box_pos_err_final for r in results]))
        out["box_success"] = sum((not r.failed) and r.box_pos_err_final < BOX_SUCCESS_POS for r in results) / max(n, 1)
    return out


def summarize(results: list[ClipResult]) -> dict:
    out = _summary_of(results)
    tot_steps = sum(r.steps for r in results)
    if results and results[0].terms:
        out["terms"] = {k: sum(r.terms[k] * r.steps for r in results) / max(tot_steps, 1) for k in results[0].terms}
    cats = sorted({r.category for r in results if r.category})
    if cats:
        out["by_category"] = {c: _summary_of([r for r in results if r.category == c]) for c in cats}
    return out


def save_results(results: list[ClipResult], path: str | Path) -> None:
    Path(path).write_text(json.dumps({"summary": summarize(results), "clips": [asdict(r) for r in results]}, indent=1))
