"""State representation for the world model (plan D1–D3).

Simulator state  s = [root_pos(3), root_quat wxyz(4), joint_pos(29), root_lin_vel_w(3),
                      root_ang_vel_b(3), joint_vel(29)]            (71, MuJoCo layout, origin-free)

Canonical features (network input, heading-invariant, 71):
    [root_z(1), rot6d(heading^-1 ⊗ root_quat)(6), root_lin_vel in heading frame(3),
     root_ang_vel_b(3), joint_pos(29), joint_vel(29)]

Network output ``delta`` (70) and the integration step (dt = 0.02 s):
    dv_h(3), dw_b(3), dqd(29)            velocity increments, heading / body frame
    dp_h(3), dth_b(3), dq(29)            position corrections on top of explicit integration
    v'  = v + R_h dv_h               w' = w + dw_b          qd' = qd + dqd
    p'  = p + dt v' + R_h dp_h       q' = q ⊗ exp(dt w' + dth_b)    qj' = qj + dt qd' + dq
With zero output the model extrapolates at constant velocity (a physical prior); the
last layer of the dynamics MLP is zero-initialised so training starts from it.

Loss / error normalisation (D3): per-quantity nominal scales sigma applied to the
70-d error vector [dp(3), rotvec(3), dv(3), dw(3), dq(29), dqd(29)].

With a box (``has_box``) the state gains the box's free-joint state
    s[71:84] = [box_pos(3), box_quat wxyz(4), box_lin_vel_w(3), box_ang_vel_b(3)]   (84)
the features gain the box relative to the robot, in the robot heading frame (15):
    [box_pos - root_pos (3), rot6d(heading^-1 ⊗ box_quat)(6), box_lin_vel (3), box_ang_vel_b (3)]
the network output gains the box increments, integrated exactly like the root (12, heading frame =
the robot's, so everything stays translation and yaw invariant):
    dv_box_h(3), dw_box_b(3), dp_box_h(3), dth_box_b(3)                                   (82)
and the error vector gains [dp, rotvec, dv, dw] of the box (82).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from supermimic.kinematics.rotations import (
    heading_quat,
    quat_apply,
    quat_apply_inverse,
    quat_canonical,
    quat_from_rotvec,
    quat_mul,
    quat_normalize,
    rot6d_from_quat,
    rotvec_from_quat,
)

STATE_DIM = 71
FEAT_DIM = 71
DELTA_DIM = 70
NJ = 29
BOX_STATE_DIM = 13
BOX_FEAT_DIM = 15
BOX_DELTA_DIM = 12


@dataclass
class StateSpecCfg:
    dt: float = 0.02
    # Nominal error scales (D3) for [root pos m, root rotvec rad, root lin vel m/s, root ang vel rad/s, joint pos rad, joint vel rad/s].
    err_scale: tuple[float, ...] = (0.05, 0.1, 0.5, 1.0, 0.1, 2.0)
    # Output scales of the dynamics head for [dv_h, dw_b, dqd, dp_h, dth_b, dq].
    out_scale: tuple[float, ...] = (0.3, 0.5, 2.0, 0.01, 0.02, 0.02)
    # Input feature scales for [root_z, rot6d, lin_vel_h, ang_vel_b, joint_pos, joint_vel] (divide by these).
    in_scale: tuple[float, ...] = (1.0, 1.0, 1.0, 2.0, 1.0, 5.0)
    explicit_integration: bool = True
    # xi normalisation: (offset, scale) for friction, encoder bias, com offset (+ box mass, box friction).
    xi_offset: tuple[float, ...] = field(default_factory=lambda: (0.75,) + (0.0,) * 29 + (0.0,) * 3)
    xi_scale: tuple[float, ...] = field(default_factory=lambda: (0.45,) + (0.01,) * 29 + (0.05,) * 3)
    has_box: bool = False
    # Box scales: error [pos m, rotvec rad, lin vel m/s, ang vel rad/s], output [dv_h, dw_b, dp_h, dth_b],
    # input [rel pos, rot6d, lin vel, ang vel], xi [mass, friction] offset / scale.
    box_err_scale: tuple[float, ...] = (0.05, 0.1, 0.5, 1.0)
    box_out_scale: tuple[float, ...] = (0.3, 0.5, 0.01, 0.02)
    box_in_scale: tuple[float, ...] = (1.0, 1.0, 1.0, 2.0)
    box_xi_offset: tuple[float, ...] = (0.75, 0.8)
    box_xi_scale: tuple[float, ...] = (0.25, 0.2)


class StateSpec:
    def __init__(self, cfg: StateSpecCfg, device, dtype=torch.float32):
        self.cfg = cfg
        self.dt = cfg.dt
        self.has_box = cfg.has_box
        rep = lambda scales, sizes: torch.cat([torch.full((n,), float(s)) for s, n in zip(scales, sizes)])  # noqa: E731
        err, out, inp = [rep(cfg.err_scale, (3, 3, 3, 3, NJ, NJ))], [rep(cfg.out_scale, (3, 3, NJ, 3, 3, NJ))], [rep(cfg.in_scale, (1, 6, 3, 3, NJ, NJ))]
        xi_off, xi_sc = list(cfg.xi_offset), list(cfg.xi_scale)
        if self.has_box:
            err.append(rep(cfg.box_err_scale, (3, 3, 3, 3)))
            out.append(rep(cfg.box_out_scale, (3, 3, 3, 3)))
            inp.append(rep(cfg.box_in_scale, (3, 6, 3, 3)))
            xi_off += list(cfg.box_xi_offset)
            xi_sc += list(cfg.box_xi_scale)
        self.err_scale = torch.cat(err).to(device=device, dtype=dtype)
        self.out_scale = torch.cat(out).to(device=device, dtype=dtype)
        self.in_scale = torch.cat(inp).to(device=device, dtype=dtype)
        self.xi_offset = torch.tensor(xi_off, device=device, dtype=dtype)
        self.xi_scale = torch.tensor(xi_sc, device=device, dtype=dtype)
        self.state_dim = STATE_DIM + (BOX_STATE_DIM if self.has_box else 0)
        self.feat_dim = FEAT_DIM + (BOX_FEAT_DIM if self.has_box else 0)
        self.delta_dim = DELTA_DIM + (BOX_DELTA_DIM if self.has_box else 0)
        self.xi_dim = len(xi_off)

    # ------------------------------------------------------------------ split / features

    @staticmethod
    def split(s: torch.Tensor):
        return s[..., 0:3], s[..., 3:7], s[..., 7:36], s[..., 36:39], s[..., 39:42], s[..., 42:71]

    @staticmethod
    def split_box(s: torch.Tensor):
        """-> box_pos (…,3), box_quat (…,4), box_lin_vel_w (…,3), box_ang_vel_b (…,3)."""
        return s[..., 71:74], s[..., 74:78], s[..., 78:81], s[..., 81:84]

    def canonical(self, s: torch.Tensor) -> torch.Tensor:
        """Heading-invariant, scaled network input features (…, 71) [+ box (…, 15)]."""
        p, q, qj, v, w, qd = self.split(s)
        q = quat_normalize(q)
        hq = heading_quat(q)
        q_local = quat_canonical(quat_mul(quat_apply_inverse_quat(hq), q))
        v_h = quat_apply_inverse(hq, v)
        parts = [p[..., 2:3], rot6d_from_quat(q_local), v_h, w, qj, qd]
        if self.has_box:
            bp, bq, bv, bw = self.split_box(s)
            bq_local = quat_mul(quat_apply_inverse_quat(hq), quat_normalize(bq))
            parts += [quat_apply_inverse(hq, bp - p), rot6d_from_quat(bq_local), quat_apply_inverse(hq, bv), bw]
        return torch.cat(parts, dim=-1) / self.in_scale

    def normalize_xi(self, xi: torch.Tensor) -> torch.Tensor:
        return (xi - self.xi_offset) / self.xi_scale

    # ------------------------------------------------------------------ integration

    def integrate(self, s: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        """Apply the (unscaled) network output ``delta`` (…, 70) to state ``s`` (…, 71)."""
        d = delta * self.out_scale
        dv_h, dw_b, dqd, dp_h, dth_b, dq = d[..., 0:3], d[..., 3:6], d[..., 6:35], d[..., 35:38], d[..., 38:41], d[..., 41:70]
        p, q, qj, v, w, qd = self.split(s)
        q = quat_normalize(q)
        hq = heading_quat(q)
        v_new = v + quat_apply(hq, dv_h)
        w_new = w + dw_b
        qd_new = qd + dqd
        dt = self.dt if self.cfg.explicit_integration else 0.0
        p_new = p + dt * v_new + quat_apply(hq, dp_h)
        q_new = quat_normalize(quat_mul(q, quat_from_rotvec(dt * w_new + dth_b)))
        qj_new = qj + dt * qd_new + dq
        out = [p_new, q_new, qj_new, v_new, w_new, qd_new]
        if self.has_box:
            bdv_h, bdw_b, bdp_h, bdth_b = d[..., 70:73], d[..., 73:76], d[..., 76:79], d[..., 79:82]
            bp, bq, bv, bw = self.split_box(s)
            bv_new = bv + quat_apply(hq, bdv_h)
            bw_new = bw + bdw_b
            bp_new = bp + dt * bv_new + quat_apply(hq, bdp_h)
            bq_new = quat_normalize(quat_mul(quat_normalize(bq), quat_from_rotvec(dt * bw_new + bdth_b)))
            out += [bp_new, bq_new, bv_new, bw_new]
        return torch.cat(out, dim=-1)

    # ------------------------------------------------------------------ errors

    def error_vector(self, s_pred: torch.Tensor, s_true: torch.Tensor) -> torch.Tensor:
        """Normalised 70-d error [dp, rotvec, dv, dw, dq, dqd] / sigma."""
        p1, q1, qj1, v1, w1, qd1 = self.split(s_pred)
        p2, q2, qj2, v2, w2, qd2 = self.split(s_true)
        q1 = quat_normalize(q1)
        q2 = quat_normalize(q2)
        rot_err = rotvec_from_quat(quat_mul(quat_apply_inverse_quat(q2), q1))
        parts = [p1 - p2, rot_err, v1 - v2, w1 - w2, qj1 - qj2, qd1 - qd2]
        if self.has_box:
            bp1, bq1, bv1, bw1 = self.split_box(s_pred)
            bp2, bq2, bv2, bw2 = self.split_box(s_true)
            brot = rotvec_from_quat(quat_mul(quat_apply_inverse_quat(quat_normalize(bq2)), quat_normalize(bq1)))
            parts += [bp1 - bp2, brot, bv1 - bv2, bw1 - bw2]
        return torch.cat(parts, dim=-1) / self.err_scale

    def normalized_sq_error(self, s_pred: torch.Tensor, s_true: torch.Tensor) -> torch.Tensor:
        """Mean over dims of the squared normalised error (the reported 'prediction error')."""
        e = self.error_vector(s_pred, s_true)
        return (e * e).mean(-1)


def quat_apply_inverse_quat(q: torch.Tensor) -> torch.Tensor:
    """Conjugate of a unit quaternion (inverse)."""
    return q * torch.tensor([1.0, -1.0, -1.0, -1.0], dtype=q.dtype, device=q.device)
