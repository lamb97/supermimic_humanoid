"""Paper terminations: root (pelvis) height deviation and tilt; clip end as time-out."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch
from mjlab.utils.lab_api.math import quat_apply_inverse

from supermimic.envs.mdp.commands import MultiClipMotionCommand

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def bad_root_height(env: ManagerBasedRlEnv, command_name: str, threshold: float = 0.25) -> torch.Tensor:
    cmd = cast(MultiClipMotionCommand, env.command_manager.get_term(command_name))
    return (cmd.root_pos_w[:, 2] - cmd.robot_root_pos_w[:, 2]).abs() > threshold


def bad_root_tilt(env: ManagerBasedRlEnv, command_name: str, threshold: float = 0.8) -> torch.Tensor:
    cmd = cast(MultiClipMotionCommand, env.command_manager.get_term(command_name))
    g = cmd.robot.data.gravity_vec_w
    ref = quat_apply_inverse(cmd.root_quat_w, g)[:, 2]
    rob = quat_apply_inverse(cmd.robot_root_quat_w, g)[:, 2]
    return (ref - rob).abs() > threshold


def clip_end(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    cmd = cast(MultiClipMotionCommand, env.command_manager.get_term(command_name))
    return cmd.clip_end


def bad_box(env: ManagerBasedRlEnv, command_name: str, pos_threshold: float = 1.0, ori_threshold: float = 0.785, start_step: int = 0) -> torch.Tensor:
    """Box deviates from its reference by more than pos_threshold (m) or ori_threshold (rad).

    OmniRetarget applies the object criterion only once body tracking is reasonable: it stays off for the
    first ``start_step`` vectorised env steps (holosoma multi-task lesson: without the gate episodes
    collapse to ~22 steps because the box is dropped long before the body can track)."""
    cmd = cast(MultiClipMotionCommand, env.command_manager.get_term(command_name))
    if env.common_step_counter < start_step:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    from supermimic.envs.box import box_errors

    e_pos, e_ori = box_errors(cmd.box_pos_w, cmd.box_quat_w, cmd.ref_box_pos_w, cmd.ref_box_quat_w)
    return (e_pos > pos_threshold**2) | (e_ori > ori_threshold**2)
