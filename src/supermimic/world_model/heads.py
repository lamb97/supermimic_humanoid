"""Analytic heads of the world model (plan D4): observation, reward, termination.

Everything here is a pure differentiable function of the 71-d simulator state
``s = [qpos(36) | qvel(35)]`` (root position expressed relative to the env origin,
see ``StateIO``; with a box, its 13-d state appended), the previous clean action,
the sim context ``xi`` and the reference frame, so the same code serves the
simulator-side tests and the imagined rollouts.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from supermimic.envs import box as box_def
from supermimic.envs.mdp.rewards import TrackingRewardCfg, box_reward_terms, tracking_reward_terms
from supermimic.kinematics.g1_fk import G1Kinematics
from supermimic.kinematics.rotations import projected_gravity, quat_apply_inverse

OBS_DIM = 93
# Term layout of the ``policy`` observation group (order matters: term-major history).
OBS_TERMS = (("joint_pos", 29), ("joint_vel", 29), ("base_ang_vel", 3), ("projected_gravity", 3), ("actions", 29))


def split_state(s: torch.Tensor):
    """-> root_pos (…,3), root_quat (…,4), joint_pos (…,29), root_lin_vel_w (…,3), root_ang_vel_b (…,3), joint_vel (…,29)."""
    return s[..., 0:3], s[..., 3:7], s[..., 7:36], s[..., 36:39], s[..., 39:42], s[..., 42:71]


def analytic_observation(
    s: torch.Tensor, prev_action: torch.Tensor, xi: torch.Tensor, default_joint_pos: torch.Tensor
) -> torch.Tensor:
    """Noise-free 93-d proprioceptive observation (paper Table A.1) from state + context.

    xi[..., 1:30] is the encoder bias: observed joint positions are ``q + bias - q0``.
    """
    _, root_quat, q, _, ang_b, qd = split_state(s)
    bias = xi[..., 1:30]
    return torch.cat([q + bias - default_joint_pos, qd, ang_b, projected_gravity(root_quat), prev_action], dim=-1)


def anchor_relative_obs(robot_anchor_pos: torch.Tensor, robot_anchor_quat: torch.Tensor, ref_anchor_pos: torch.Tensor, ref_anchor_quat: torch.Tensor) -> torch.Tensor:
    """Reference anchor pose expressed in the robot anchor frame: [pos (3), rot6d (6)] (= mjlab's
    motion_anchor_pos_b / motion_anchor_ori_b, plan D5 alternative)."""
    from supermimic.kinematics.rotations import matrix_from_quat, quat_inv, quat_mul

    q10 = quat_inv(robot_anchor_quat)
    pos = quat_apply_inverse(robot_anchor_quat, ref_anchor_pos - robot_anchor_pos)
    q12 = quat_mul(q10, ref_anchor_quat)
    mat = matrix_from_quat(q12)
    return torch.cat([pos, mat[..., :2].reshape(*q12.shape[:-1], 6)], dim=-1)


ANCHOR_OBS_DIM = 9


def flatten_history_term_major(hist: torch.Tensor) -> torch.Tensor:
    """(H, N, 93) oldest→newest  ->  (N, 930) in mjlab's term-major layout [A_t0..A_tH-1, B_t0..]."""
    H, N, _ = hist.shape
    chunks = []
    start = 0
    for _, d in OBS_TERMS:
        chunks.append(hist[:, :, start : start + d].permute(1, 0, 2).reshape(N, H * d))
        start += d
    return torch.cat(chunks, dim=-1)


@dataclass
class ReferenceFrame:
    """Reference quantities of one target frame for a batch (N, ·)."""

    body_pos_w: torch.Tensor  # (N, B, 3)
    body_quat_w: torch.Tensor  # (N, B, 4)
    body_lin_vel_w: torch.Tensor  # (N, B, 3)
    body_ang_vel_w: torch.Tensor  # (N, B, 3)
    root_pos_w: torch.Tensor  # (N, 3)
    root_quat_w: torch.Tensor  # (N, 4)
    box_pos_w: torch.Tensor | None = None  # (N, 3) reference box
    box_quat_w: torch.Tensor | None = None  # (N, 4)


class AnalyticHeads:
    """Reward / termination features computed from the predicted state through differentiable FK."""

    def __init__(
        self,
        fk: G1Kinematics,
        tracked_body_names: tuple[str, ...],
        anchor_body_name: str,
        root_body_name: str,
        soft_limits: torch.Tensor,
        default_joint_pos: torch.Tensor,
        reward_cfg: TrackingRewardCfg | None = None,
    ):
        self.fk = fk
        self.tracked_idx = fk.body_indices(tracked_body_names)
        self.anchor_index = tracked_body_names.index(anchor_body_name)
        self.root_fk_index = fk.body_index(root_body_name)
        self.soft_limits = soft_limits
        self.default_joint_pos = default_joint_pos
        self.reward_cfg = reward_cfg or TrackingRewardCfg()
        self.has_box = self.reward_cfg.has_box
        self.contact_idx = fk.body_indices(tuple(n for n, _ in box_def.CONTACT_POINTS))
        self.contact_offsets = torch.tensor([o for _, o in box_def.CONTACT_POINTS], dtype=default_joint_pos.dtype, device=default_joint_pos.device)

    def robot_bodies(self, s: torch.Tensor) -> dict[str, torch.Tensor]:
        out = self.fk(s[..., :36], s[..., 36:71])
        idx = self.tracked_idx
        return {
            "pos": out["pos"][..., idx, :],
            "quat": out["quat"][..., idx, :],
            "lin_vel": out["lin_vel"][..., idx, :],
            "ang_vel": out["ang_vel"][..., idx, :],
            "root_pos": out["pos"][..., self.root_fk_index, :],
            "root_quat": out["quat"][..., self.root_fk_index, :],
        }

    def reward(self, s_next: torch.Tensor, action: torch.Tensor, prev_action: torch.Tensor, ref: ReferenceFrame, bodies=None):
        """Reward r_{t+1} of reaching s_next (= s_{t+1}) under target frame g_t. Returns dict with 'total'."""
        b = bodies if bodies is not None else self.robot_bodies(s_next)
        terms = self._tracking_terms(s_next, action, prev_action, ref, b)
        if self.has_box:
            box = box_reward_terms(s_next[..., 71:74], s_next[..., 74:78], ref.box_pos_w, ref.box_quat_w, self.reward_cfg)
            terms["total"] = terms["total"] + box["box_pos"] + box["box_ori"]
            terms.update(box)
        return terms

    def _tracking_terms(self, s_next, action, prev_action, ref: ReferenceFrame, b):
        return tracking_reward_terms(
            robot_body_pos_w=b["pos"],
            robot_body_quat_w=b["quat"],
            robot_body_lin_vel_w=b["lin_vel"],
            robot_body_ang_vel_w=b["ang_vel"],
            ref_body_pos_w=ref.body_pos_w,
            ref_body_quat_w=ref.body_quat_w,
            ref_body_lin_vel_w=ref.body_lin_vel_w,
            ref_body_ang_vel_w=ref.body_ang_vel_w,
            anchor_index=self.anchor_index,
            action=action,
            prev_action=prev_action,
            joint_pos=s_next[..., 7:36],
            soft_limits=self.soft_limits,
            cfg=self.reward_cfg,
        )

    @staticmethod
    def termination_features(root_pos: torch.Tensor, root_quat: torch.Tensor, ref: ReferenceFrame, box_pos=None, box_quat=None) -> torch.Tensor:
        """[height error, tilt error] (N, 2) — the quantities the paper thresholds (0.25 m, 0.8);
        with a box also [box position error (m), box orientation error (rad)] (N, 4)."""
        g = torch.tensor([0.0, 0.0, -1.0], dtype=root_quat.dtype, device=root_quat.device).expand(root_quat.shape[:-1] + (3,))
        tilt = quat_apply_inverse(ref.root_quat_w, g)[..., 2] - quat_apply_inverse(root_quat, g)[..., 2]
        height = ref.root_pos_w[..., 2] - root_pos[..., 2]
        feats = [height, tilt]
        if box_pos is not None:
            e_pos, e_ori = box_def.box_errors(box_pos, box_quat, ref.box_pos_w, ref.box_quat_w)
            feats += [torch.sqrt(e_pos + 1e-12), torch.sqrt(e_ori + 1e-12)]
        return torch.stack(feats, dim=-1)

    def state_termination_features(self, s: torch.Tensor, ref: ReferenceFrame) -> torch.Tensor:
        box = (s[..., 71:74], s[..., 74:78]) if self.has_box else (None, None)
        return self.termination_features(s[..., 0:3], s[..., 3:7], ref, *box)  # pelvis == free-joint root

    def terminated(self, s: torch.Tensor, ref: ReferenceFrame, height_thr: float = 0.25, tilt_thr: float = 0.8) -> torch.Tensor:
        f = self.termination_features(s[..., 0:3], s[..., 3:7], ref)  # pelvis == free-joint root
        return (f[..., 0].abs() > height_thr) | (f[..., 1].abs() > tilt_thr)

    # ------------------------------------------------------------------ box

    def box_obs(self, s: torch.Tensor) -> torch.Tensor:
        """(N, 9) box pose in the robot pelvis frame (the policy's ``object`` observation)."""
        return box_def.pose_in_frame(s[..., 0:3], s[..., 3:7], s[..., 71:74], s[..., 74:78])

    @staticmethod
    def box_error_features(s: torch.Tensor, ref: ReferenceFrame) -> torch.Tensor:
        """Critic-only privileged box tracking error (N, 4)."""
        return box_def.box_error_features(s[..., 71:74], s[..., 74:78], ref.box_pos_w, ref.box_quat_w, ref.root_quat_w)

    def contact_geometry(self, s: torch.Tensor) -> torch.Tensor:
        """(N, 12) hand / foot contact points relative to the box, box frame (world-model input feature)."""
        out = self.fk(s[..., :36])
        i = self.contact_idx
        return box_def.contact_geometry(out["pos"][..., i, :], out["quat"][..., i, :], self.contact_offsets, s[..., 71:74], s[..., 74:78])

    @staticmethod
    def root_error_features(s: torch.Tensor, ref: ReferenceFrame) -> torch.Tensor:
        """Critic-only privileged drift features (plan D10): root position error in the reference
        heading frame (3) and yaw error (1). (N, 4)"""
        from supermimic.kinematics.rotations import heading_yaw, quat_apply_inverse, wrap_to_pi, yaw_quat

        dp = quat_apply_inverse(yaw_quat(ref.root_quat_w), s[..., 0:3] - ref.root_pos_w)
        dyaw = wrap_to_pi(heading_yaw(s[..., 3:7]) - heading_yaw(ref.root_quat_w))
        return torch.cat([dp, dyaw[..., None]], dim=-1)
