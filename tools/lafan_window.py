"""Cut a short window out of a LAFAN1 clip for the real robot, starting and ending at rest in the standby pose.

On the real G1 the tracking policy takes over from the standby controller (joint PD holding standby_controller's
default_position) at rest, and when the reference ends it holds the last frame until the operator switches back to
standby. LAFAN1 dances never stand still, so the window is wrapped the way static_start_prefix.py wraps box clips:

  hold        standby pose at rest, soles flat, under the window's first-frame feet (frame k0's heading)
  transition  min-jerk blend standby -> frame k0, legs by IK (each foot moves straight to its k0 placement)
  ease-in     the clip from k0, time-warped so its speed ramps 0 -> 1 (covers ease_s / 2 of clip time)
  window      k0 .. k1 at full speed
  ease-out    speed ramps 1 -> 0, stopping exactly at frame k1
  transition  min-jerk blend frame k1 -> standby pose under the frame-k1 feet
  hold        standby pose at rest (the frame the reference stays on after the clip)

k0 / k1 are picked inside the given search ranges and window-length bounds: feet down, lowest body speed
(static_start_prefix.choose_frame's cost). Velocities and bodies come from the same finite differences / FK as the
original conversion.

  python tools/lafan_window.py --src data/npz/lafan1/dance1_subject1.npz \
      --name dance1_cartwheel --start-range 103.3 105.6 --end-range 108.4 110.6 --out-dir out
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import tyro
from scipy.spatial.transform import Rotation, Slerp

from static_start_prefix import FEET, STANDBY, Kin, R, heading, min_jerk, sample, sample_quat, wxyz
from supermimic.data.motion_convert import G1_JOINT_NAMES, forward_kinematics, resample_motion


STANDBY_STANCE = np.array([0.0, 0.237])  # left minus right ankle in the feet heading frame, standby pose


def frame_cost(d: dict, names: list[str]) -> np.ndarray:
    """Per frame: pelvis speed + 0.05 max joint speed + foot speeds + 10 x foot height above 3 cm + 4 x the foot slide
    of a transition from the standby stance; inf unless both ankles are within 8 cm of the ground (LAFAN1's retargeted
    stance ankles wander a few cm)."""
    fi = [names.index(f) for f in FEET]
    z = d["body_pos_w"][:, fi, 2]
    vf = np.linalg.norm(d["body_lin_vel_w"][:, fi], axis=-1)
    h = z - np.percentile(z.min(1), 5)
    yaw = np.stack([np.array([heading(R(q)) for q in d["body_quat_w"][:, i]]) for i in fi], 1)
    psi = np.arctan2(np.sin(yaw).mean(1), np.cos(yaw).mean(1))
    dv = d["body_pos_w"][:, fi[0], :2] - d["body_pos_w"][:, fi[1], :2]
    stance = np.stack([np.cos(psi) * dv[:, 0] + np.sin(psi) * dv[:, 1], -np.sin(psi) * dv[:, 0] + np.cos(psi) * dv[:, 1]], 1)
    slide = 0.5 * np.linalg.norm(stance - STANDBY_STANCE, axis=1)
    cost = (np.linalg.norm(d["body_lin_vel_w"][:, names.index("pelvis")], axis=-1) + 0.05 * np.abs(d["joint_vel"]).max(1)
            + vf.sum(1) + 10 * np.clip(h - 0.03, 0, None).sum(1) + 4 * slide)
    return np.where((h < 0.08).all(1), cost, np.inf)


def choose_window(cost: np.ndarray, fps: float, start_range, end_range, length_range) -> tuple[int, int]:
    best, pair = np.inf, None
    for k0 in range(int(start_range[0] * fps), int(start_range[1] * fps) + 1):
        for k1 in range(int(end_range[0] * fps), int(end_range[1] * fps) + 1):
            if length_range[0] * fps <= k1 - k0 <= length_range[1] * fps and cost[k0] + cost[k1] < best:
                best, pair = cost[k0] + cost[k1], (k0, k1)
    assert pair is not None, "no window with both feet down at both ends in the search ranges"
    return pair


def standby_under(kin: Kin, qk: np.ndarray) -> np.ndarray:
    """Standby pose, soles flat, feet midpoint on frame k's foot midpoint, mean foot heading = frame k's."""
    feet_k = kin.foot_poses(qk)
    psi = np.arctan2(*np.mean([[np.sin(heading(R(q))), np.cos(heading(R(q)))] for _, q in feet_k], 0))
    ankle_z = float(np.mean([p[2] for p, _ in feet_k]))
    qs_j = np.array([STANDBY.get(n, 0.0) for n in G1_JOINT_NAMES])
    feet0 = kin.foot_poses(np.concatenate([np.zeros(3), [1.0, 0, 0, 0], qs_j]))
    z_up = np.mean([R(q).apply([0, 0, 1.0]) for _, q in feet0], 0)
    tilt = Rotation.align_vectors([[0, 0, 1.0]], [z_up])[0]
    yaw0 = np.arctan2(*np.mean([[np.sin(heading(tilt * R(q))), np.cos(heading(tilt * R(q)))] for _, q in feet0], 0))
    r_s = Rotation.from_euler("z", psi - yaw0) * tilt
    mid0 = np.mean([p for p, _ in feet0], 0)
    mid_k = np.mean([p for p, _ in feet_k], 0)
    return np.concatenate([np.array([mid_k[0], mid_k[1], ankle_z]) - r_s.apply(mid0), wxyz(r_s), qs_j])


def blend(kin: Kin, qa: np.ndarray, qb: np.ndarray, n: int) -> tuple[list[np.ndarray], dict]:
    """n + 1 rows from qa to qb (min-jerk): pelvis / waist / arms interpolated, legs by IK on straight foot paths."""
    fa, fb = kin.foot_poses(qa), kin.foot_poses(qb)
    root_slerp = Slerp([0, 1], R(np.stack([qa[3:7], qb[3:7]])))
    foot_slerps = [Slerp([0, 1], R(np.stack([a[1], b[1]]))) for a, b in zip(fa, fb)]
    rows, q_prev, ik_p = [], qa.copy(), 0.0
    for i in range(n + 1):
        s = float(min_jerk(np.array(i / n)))
        q = np.empty_like(qa)
        q[0:3] = (1 - s) * qa[0:3] + s * qb[0:3]
        q[3:7] = wxyz(root_slerp(s))
        q[7:] = (1 - s) * qa[7:] + s * qb[7:]
        q[7:19] = q_prev[7:19]  # legs: warm start from the previous frame
        targets = [((1 - s) * a[0] + s * b[0], wxyz(sl(s))) for a, b, sl in zip(fa, fb, foot_slerps)]
        q, rp, _ = kin.leg_ik(q, targets)
        ik_p = max(ik_p, rp)
        rows.append(q)
        q_prev = q
    gap = float(np.abs(rows[-1][7:19] - qb[7:19]).max())
    rows[0], rows[-1] = qa.copy(), qb.copy()
    slide = [round(100 * float(np.linalg.norm((a[0] - b[0])[:2])), 1) for a, b in zip(fa, fb)]
    turn = [round(float(np.degrees(np.linalg.norm((R(b[1]) * R(a[1]).inv()).as_rotvec()))), 1) for a, b in zip(fa, fb)]
    return rows, dict(foot_slide_cm=slide, foot_turn_deg=turn, ik_res_mm=1000 * ik_p, ik_leg_gap_rad=gap)


def main(src: Path, name: str, start_range: tuple[float, float], end_range: tuple[float, float], out_dir: Path,
         length_range: tuple[float, float] = (4.8, 5.5), hold_s: float = 1.0, transition_s: float = 1.5, ease_s: float = 1.0) -> None:
    kin = Kin()
    d = dict(np.load(src))
    fps = float(d["fps"][0])
    names = [str(n) for n in d["body_names"]]
    assert tuple(str(n) for n in d["joint_names"]) == G1_JOINT_NAMES
    p = names.index("pelvis")
    qpos_src = np.concatenate([d["body_pos_w"][:, p], d["body_quat_w"][:, p], d["joint_pos"]], 1).astype(np.float64)
    k0, k1 = choose_window(frame_cost(d, names), fps, start_range, end_range, length_range)

    n_hold, n_tr, n_e = (int(round(s * fps)) for s in (hold_s, transition_s, ease_s))
    qs0, qs1 = standby_under(kin, qpos_src[k0]), standby_under(kin, qpos_src[k1])
    pre, rep_in = blend(kin, qs0, qpos_src[k0], n_tr)
    post, rep_out = blend(kin, qpos_src[k1], qs1, n_tr)
    # Ease-in: clip time k0 + E (x^3 - x^4 / 2), x = t / E (speed smoothstep(x)); ease-out mirrored, ending at k1.
    x = np.arange(1, n_e + 1) / n_e
    f_in = k0 + ease_s * (x**3 - x**4 / 2) * fps
    f_out = k1 - ease_s * ((1 - x) ** 3 - (1 - x) ** 4 / 2) * fps
    f_mid = np.arange(np.ceil(f_in[-1] + 1e-6), np.floor(f_out[0] - 1e-6) + 1)
    f_clip = np.concatenate([f_in, f_mid, f_out])
    clip_rows = np.concatenate([sample(qpos_src[:, 0:3], f_clip), sample_quat(qpos_src[:, 3:7], f_clip), sample(qpos_src[:, 7:], f_clip)], 1)
    qpos = np.concatenate([np.repeat(qs0[None], n_hold, 0), np.stack(pre), clip_rows, np.stack(post[1:]), np.repeat(qs1[None], n_hold, 0)])

    raw = np.concatenate([qpos[:, 0:3], qpos[:, 3:7][:, [1, 2, 3, 0]], qpos[:, 7:]], 1)
    motion = resample_motion(raw, fps, fps)
    assert motion.num_frames == len(qpos), (motion.num_frames, len(qpos))
    bodies = forward_kinematics(kin.model, motion)
    assert list(bodies.body_names) == names

    ends = np.cumsum([n_hold, n_tr + 1, n_e, len(f_mid), n_e, n_tr, n_hold])
    phase_names = ["standby hold", "transition", "ease-in", "clip", "ease-out", "transition back", "standby hold (end)"]
    out = dict(
        fps=d["fps"], input_fps=d["input_fps"], joint_names=d["joint_names"], body_names=np.array(bodies.body_names),
        joint_pos=motion.joint_pos.astype(np.float32),
        joint_vel=motion.joint_vel.astype(np.float32),
        body_pos_w=bodies.pos_w.astype(np.float32),
        body_quat_w=bodies.quat_w.astype(np.float32),
        body_lin_vel_w=bodies.lin_vel_w.astype(np.float32),
        body_ang_vel_w=bodies.ang_vel_w.astype(np.float32),
        source=np.array(f"{src.name} frames {k0}-{k1} ({k0 / fps:.2f}-{k1 / fps:.2f} s), standby hold {hold_s} s + transition "
                        f"{transition_s} s + ease {ease_s} s at both ends"),
        phase_names=np.array(phase_names),
        phase_ends=ends.astype(np.int64),
        window_frames=np.array([k0, k1]),
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / f"{name}.npz", **out)
    report = dict(
        name=name, src=src.stem, k0=k0, k1=k1, k0_s=k0 / fps, k1_s=k1 / fps, window_s=(k1 - k0) / fps,
        frames=len(qpos), duration_s=len(qpos) / fps, phase_ends_s=[round(e / fps, 2) for e in ends.tolist()],
        pelvis_speed_k0=float(np.linalg.norm(d["body_lin_vel_w"][k0, p])), pelvis_speed_k1=float(np.linalg.norm(d["body_lin_vel_w"][k1, p])),
        transition_in=rep_in, transition_out=rep_out,
        max_joint_speed=float(np.abs(motion.joint_vel).max()), max_pelvis_speed=float(np.linalg.norm(motion.root_lin_vel_w, axis=-1).max()),
    )
    (out_dir / f"{name}.window.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    tyro.cli(main)
