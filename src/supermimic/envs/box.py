"""OmniRetarget large box (carry / push / kick) as a free rigid body, plus the box geometry helpers
shared by the environment (observations, rewards, terminations) and the world model's analytic heads.

The OMOMO ``largebox.obj`` mesh is a box that is *not* aligned with its own mesh axes: a minimum-volume
oriented bounding box fit gives the real box (32.5 x 33.8 x 36.3 cm), rotated 170.9 deg inside the mesh
frame. The data's object pose is the mesh frame, so the collision geom is that OBB placed with a fixed
local pose inside the box body (fit residual: resting boxes sit flat, 1.7 deg / -1.2 cm at the rounded corners).

Contact: the floor and the G1's non-foot collision geoms (including the rubber hands) are condim 1 /
priority 0, so the box geom gets condim 3 and priority 1: box-floor and box-hand contacts then use the
box's own (randomized) friction, and the hands can actually grip it.
"""

from __future__ import annotations

import mujoco
import torch

from supermimic.kinematics.rotations import quat_apply, quat_apply_inverse, quat_inv, quat_mul, rot6d_from_quat, yaw_quat

BOX_ENTITY = "box"
BOX_BODY = "box"
BOX_GEOM = "box_geom"
BOX_HALF_EXTENTS = (0.162291, 0.168838, 0.181696)
BOX_GEOM_POS = (0.000556, -0.002050, 0.008592)
BOX_GEOM_QUAT = (0.079186, 0.967445, 0.235360, 0.048848)  # wxyz, mesh frame <- box frame
BOX_NOMINAL_MASS = 0.75
BOX_NOMINAL_FRICTION = 0.8
BOX_OBS_DIM = 9  # position (3) + rot6d (6)
BOX_STATE_DIM = 13  # pos (3), quat wxyz (4), lin vel world (3), ang vel body (3)
BOX_XI_DIM = 2  # mass, friction

# Contact reference points (link, offset in the link frame): centre of the rubber-hand capsule and
# centre of the foot capsule cluster of the mjlab G1 model.
CONTACT_POINTS = (
    ("left_wrist_yaw_link", (0.11, -0.01, 0.0)),
    ("right_wrist_yaw_link", (0.11, 0.01, 0.0)),
    ("left_ankle_roll_link", (0.04, 0.0, -0.025)),
    ("right_ankle_roll_link", (0.04, 0.0, -0.025)),
)
CONTACT_FEAT_DIM = 3 * len(CONTACT_POINTS)


def get_box_spec() -> mujoco.MjSpec:
    spec = mujoco.MjSpec()
    body = spec.worldbody.add_body(name=BOX_BODY)
    body.add_freejoint(name="box_joint")
    body.add_geom(
        name=BOX_GEOM,
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=BOX_HALF_EXTENTS,
        pos=BOX_GEOM_POS,
        quat=BOX_GEOM_QUAT,
        mass=BOX_NOMINAL_MASS,
        friction=(BOX_NOMINAL_FRICTION, 0.005, 0.0001),
        condim=3,
        priority=1,
        contype=1,
        conaffinity=1,
        rgba=(0.85, 0.65, 0.35, 1.0),
    )
    return spec


def pose_in_frame(frame_pos: torch.Tensor, frame_quat: torch.Tensor, pos: torch.Tensor, quat: torch.Tensor) -> torch.Tensor:
    """Pose (pos, quat) expressed in a frame: [position (3), rot6d (6)] (..., 9)."""
    p = quat_apply_inverse(frame_quat, pos - frame_pos)
    return torch.cat([p, rot6d_from_quat(quat_mul(quat_inv(frame_quat), quat))], dim=-1)


def box_errors(box_pos: torch.Tensor, box_quat: torch.Tensor, ref_pos: torch.Tensor, ref_quat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(squared position error, squared geodesic angle) between box and reference, both (...,)."""
    from mjlab.utils.lab_api.math import quat_box_minus

    v = quat_box_minus(ref_quat, box_quat)
    return ((ref_pos - box_pos) ** 2).sum(-1), (v * v).sum(-1)


def box_error_features(box_pos: torch.Tensor, box_quat: torch.Tensor, ref_pos: torch.Tensor, ref_quat: torch.Tensor, ref_root_quat: torch.Tensor) -> torch.Tensor:
    """Critic-only privileged box tracking error: position error in the reference heading frame (3) and
    orientation error angle (1). (..., 4)"""
    dp = quat_apply_inverse(yaw_quat(ref_root_quat), box_pos - ref_pos)
    _, ang_sq = box_errors(box_pos, box_quat, ref_pos, ref_quat)
    return torch.cat([dp, torch.sqrt(ang_sq + 1e-12)[..., None]], dim=-1)


def contact_geometry(link_pos: torch.Tensor, link_quat: torch.Tensor, offsets: torch.Tensor, box_pos: torch.Tensor, box_quat: torch.Tensor) -> torch.Tensor:
    """Hand / foot contact points relative to the box, in the box frame.

    link_pos (..., P, 3), link_quat (..., P, 4), offsets (P, 3), box (..., 3/4) -> (..., 3P)."""
    pts = link_pos + quat_apply(link_quat, offsets.expand_as(link_pos))
    rel = quat_apply_inverse(box_quat[..., None, :].expand_as(link_quat), pts - box_pos[..., None, :])
    return rel.reshape(*rel.shape[:-2], -1)
