from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from supermimic.envs.mdp.commands import MultiClipMotionCommand

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def goal_future_frames(env: ManagerBasedRlEnv, command_name: str, with_object: bool = False) -> torch.Tensor:
    """(N, K*68) future reference-frame features (plan D5); (N, K*77) with the reference box pose per frame."""
    command = cast(MultiClipMotionCommand, env.command_manager.get_term(command_name))
    return command.goal_with_object() if with_object else command.goal


def box_pose_b(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """(N, 9) box pose in the robot pelvis frame: what the deployed policy gets from motion capture."""
    command = cast(MultiClipMotionCommand, env.command_manager.get_term(command_name))
    return command.box_pose_b
