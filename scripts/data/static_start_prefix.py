"""Give box clips a static start in the real robot's standby pose.

On the real G1 the tracking policy takes over from the standby controller (joint PD holding the
``default_position`` of motion_tracking_controller's config/g1/controllers.yaml) at rest and plays the
reference from frame 0, but every OmniRetarget clip starts mid-motion (pelvis up to 1.7 m/s). New reference:

  hold        standby pose at rest for ``hold_s``; pelvis tilted so both soles are flat, placed so the two
              feet sit on the clip's frame-k foot midpoint, facing the frame-k mean foot heading (stepped
              straight back from the box if that placement comes closer than ``clear_m`` to it);
  transition  min-jerk blend (zero velocity at both ends) from the standby pose to clip frame k, the first
              double-support frame of the clip's first ``search_s``: pelvis / waist / arms interpolated,
              legs by IK so each foot moves straight from its standby to its frame-k placement (sliding
              by the stance difference, reported);
  ease-in     the clip from frame k time-warped so its speed ramps 0 -> 1 (smoothstep) over ``ease_s``
              (covers ease_s / 2 of clip time), then the rest of the clip unchanged.

The box rests at its frame-k pose during hold and transition, then follows the (warped) clip. Velocities
come from the same finite differences / FK as the original conversion. Reports per clip: chosen frame,
IK residual, foot slide, and the minimum robot-box distance over hold + transition (negative = penetration)
next to the clip's own value at frame k.

  python scripts/data/static_start_prefix.py --src-dir data/npz/omniretarget_box --out-dir data/npz/omniretarget_box_static_all --workers 32
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import mujoco
import numpy as np
import tyro
from scipy.spatial.transform import Rotation, Slerp

from supermimic.data.motion_convert import G1_JOINT_NAMES, forward_kinematics, load_g1_model, resample_motion
from supermimic.envs.box import BOX_GEOM_POS, BOX_GEOM_QUAT, BOX_HALF_EXTENTS

# motion_tracking_controller config/g1/controllers.yaml, standby_controller.default_position
STANDBY = {
    "left_hip_pitch_joint": -0.312, "left_knee_joint": 0.669, "left_ankle_pitch_joint": -0.33,
    "right_hip_pitch_joint": -0.312, "right_knee_joint": 0.669, "right_ankle_pitch_joint": -0.33,
    "left_shoulder_pitch_joint": 0.2, "left_shoulder_roll_joint": 0.2, "left_elbow_joint": 0.6,
    "right_shoulder_pitch_joint": 0.2, "right_shoulder_roll_joint": -0.2, "right_elbow_joint": 0.6,
}
FEET = ("left_ankle_roll_link", "right_ankle_roll_link")
LEGS = (np.arange(0, 6), np.arange(6, 12))  # joint indices of the left / right leg in G1_JOINT_NAMES


def R(q_wxyz: np.ndarray) -> Rotation:
    return Rotation.from_quat(np.asarray(q_wxyz)[..., [1, 2, 3, 0]])


def wxyz(r: Rotation) -> np.ndarray:
    return r.as_quat()[..., [3, 0, 1, 2]]


def heading(r: Rotation) -> float:
    x = r.apply([1.0, 0.0, 0.0])
    return float(np.arctan2(x[1], x[0]))


def min_jerk(x: np.ndarray) -> np.ndarray:
    return 10 * x**3 - 15 * x**4 + 6 * x**5


def sample(arr: np.ndarray, f: np.ndarray) -> np.ndarray:
    """Linear interpolation of per-frame rows at fractional frame indices."""
    i0 = np.clip(np.floor(f).astype(int), 0, len(arr) - 1)
    i1 = np.minimum(i0 + 1, len(arr) - 1)
    w = (f - i0).reshape(-1, *([1] * (arr.ndim - 1)))
    return arr[i0] * (1 - w) + arr[i1] * w


def sample_quat(q: np.ndarray, f: np.ndarray) -> np.ndarray:
    return wxyz(Slerp(np.arange(len(q)), R(q))(np.clip(f, 0, len(q) - 1)))


class Kin:
    """CPU MuJoCo G1 for FK / leg IK, plus the box as a mocap body for robot-box distances."""

    def __init__(self) -> None:
        self.model = load_g1_model()
        spec = mujoco.MjSpec.from_file(str(_g1_xml()))
        box = spec.worldbody.add_body(name="box_probe", mocap=True)
        box.add_geom(name="box_probe_geom", type=mujoco.mjtGeom.mjGEOM_BOX, size=BOX_HALF_EXTENTS, pos=BOX_GEOM_POS, quat=BOX_GEOM_QUAT,
                     contype=0, conaffinity=0)
        self.cmodel = spec.compile()
        self.cdata = mujoco.MjData(self.cmodel)
        self.box_geom = mujoco.mj_name2id(self.cmodel, mujoco.mjtObj.mjOBJ_GEOM, "box_probe_geom")
        # Robot collision geoms: everything not in the visual class (group 2 in the mjlab G1 xml).
        self.robot_geoms = [g for g in range(self.cmodel.ngeom) if g != self.box_geom and self.cmodel.geom_bodyid[g] > 0 and self.cmodel.geom_group[g] != 2]
        self.data = mujoco.MjData(self.model)
        self.feet = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, n) for n in FEET]

    def fk(self, qpos: np.ndarray) -> None:
        self.data.qpos[:] = qpos
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)  # cdof, needed by mj_jacBody

    def foot_poses(self, qpos: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
        self.fk(qpos)
        return [(self.data.xpos[b].copy(), self.data.xquat[b].copy()) for b in self.feet]

    def leg_ik(self, qpos: np.ndarray, targets: list[tuple[np.ndarray, np.ndarray]], iters: int = 50) -> tuple[np.ndarray, float, float]:
        """Solve both legs (6 joints each) for 6-DoF foot targets; returns qpos, max pos / rot residual."""
        q = qpos.copy()
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        for _ in range(iters):
            self.fk(q)
            worst = 0.0
            for b, (tp, tq), leg in zip(self.feet, targets, LEGS):
                ep = tp - self.data.xpos[b]
                er = (R(tq) * R(self.data.xquat[b]).inv()).as_rotvec()
                e = np.concatenate([ep, er])
                worst = max(worst, float(np.abs(e).max()))
                mujoco.mj_jacBody(self.model, self.data, jacp, jacr, b)
                J = np.concatenate([jacp, jacr])[:, 6 + leg]
                q[7 + leg] += J.T @ np.linalg.solve(J @ J.T + 1e-6 * np.eye(6), e)
            if worst < 1e-7:
                break
        self.fk(q)
        res_p = max(float(np.linalg.norm(tp - self.data.xpos[b])) for b, (tp, _) in zip(self.feet, targets))
        res_r = max(float(np.linalg.norm((R(tq) * R(self.data.xquat[b]).inv()).as_rotvec())) for b, (_, tq) in zip(self.feet, targets))
        return q, res_p, res_r

    def box_distance(self, qpos: np.ndarray, box_pos: np.ndarray, box_quat: np.ndarray) -> tuple[float, str]:
        """Minimum signed distance between the robot's collision geoms and the box (negative = penetration)."""
        d = self.cdata
        d.qpos[:] = qpos
        d.mocap_pos[0] = box_pos
        d.mocap_quat[0] = box_quat
        mujoco.mj_kinematics(self.cmodel, d)
        fromto = np.zeros(6)
        best, who = np.inf, ""
        for g in self.robot_geoms:
            dist = mujoco.mj_geomDistance(self.cmodel, d, g, self.box_geom, 0.3, fromto)
            if dist < best:
                best, who = dist, mujoco.mj_id2name(self.cmodel, mujoco.mjtObj.mjOBJ_BODY, self.cmodel.geom_bodyid[g])
        return float(best), who


def _g1_xml() -> Path:
    from supermimic.data.motion_convert import g1_xml_path

    return g1_xml_path()


def choose_frame(d, fps: float, search_s: float) -> int:
    """First-``search_s`` frame with both feet down and the lowest body speed."""
    names = [str(n) for n in d["body_names"]]
    z = d["body_pos_w"][:, [names.index(f) for f in FEET], 2]
    vf = np.linalg.norm(d["body_lin_vel_w"][:, [names.index(f) for f in FEET]], axis=-1)
    ground = np.percentile(z.min(1), 5)
    n = min(len(z), int(round(search_s * fps)))
    down = (z[:n] < ground + 0.012).all(1) & (vf[:n] < 0.3).all(1)
    p = names.index("pelvis")
    cost = np.linalg.norm(d["body_lin_vel_w"][:n, p], axis=-1) + 0.05 * np.abs(d["joint_vel"][:n]).max(1)
    if not down.any():
        down = z[:n].max(1) <= z[:n].max(1).min() + 1e-6  # fall back to the lowest swing foot
    return int(np.argmin(np.where(down, cost, np.inf)))


def build(kin: Kin, src: Path, hold_s: float, transition_s: float, ease_s: float, search_s: float, clear_m: float) -> tuple[dict, dict]:
    d = dict(np.load(src))
    fps = float(d["fps"][0])
    names = [str(n) for n in d["body_names"]]
    assert tuple(str(n) for n in d["joint_names"]) == G1_JOINT_NAMES
    p = names.index("pelvis")
    root_pos, root_quat, jq = d["body_pos_w"][:, p].astype(np.float64), d["body_quat_w"][:, p].astype(np.float64), d["joint_pos"].astype(np.float64)
    box_pos, box_quat = d["object_pos_w"].astype(np.float64), d["object_quat_w"].astype(np.float64)
    k = choose_frame(d, fps, search_s)

    # Clip frame k: feet placement and heading.
    qk = np.concatenate([root_pos[k], root_quat[k], jq[k]])
    feet_k = kin.foot_poses(qk)
    psi = np.arctan2(*np.mean([[np.sin(heading(R(q))), np.cos(heading(R(q)))] for _, q in feet_k], 0))
    ankle_z = float(np.mean([pp[2] for pp, _ in feet_k]))

    # Standby pose: soles flat, feet midpoint on the frame-k midpoint, mean foot heading = frame-k heading.
    qs_j = np.array([STANDBY.get(n, 0.0) for n in G1_JOINT_NAMES])
    q0 = np.concatenate([np.zeros(3), [1.0, 0, 0, 0], qs_j])
    feet0 = kin.foot_poses(q0)
    z_up = np.mean([R(q).apply([0, 0, 1.0]) for _, q in feet0], 0)
    tilt = Rotation.align_vectors([[0, 0, 1.0]], [z_up])[0]
    yaw0 = np.arctan2(*np.mean([[np.sin(heading(tilt * R(q))), np.cos(heading(tilt * R(q)))] for _, q in feet0], 0))
    r_s = Rotation.from_euler("z", psi - yaw0) * tilt
    mid0 = np.mean([pp for pp, _ in feet0], 0)
    mid_k = np.mean([pp for pp, _ in feet_k], 0)
    pos_s = np.array([mid_k[0], mid_k[1], ankle_z]) - r_s.apply(mid0)
    qs = np.concatenate([pos_s, wxyz(r_s), qs_j])
    # Retargeted clips can start with the body touching the box: step the standby placement straight
    # back from the box until it clears it by clear_m (a reset inside the box would launch it).
    shift = 0.0
    for _ in range(20):
        dist, _ = kin.box_distance(qs, box_pos[k], box_quat[k])
        if dist >= clear_m or shift > 0.3:
            break
        away = qs[0:2] - box_pos[k, 0:2]
        step = clear_m - dist + 0.005
        qs[0:2] += step * away / (np.linalg.norm(away) + 1e-9)
        shift += step
    feet_s = kin.foot_poses(qs)

    # Hold + transition (min-jerk, legs by IK on linearly moving foot targets).
    n_hold, n_tr = int(round(hold_s * fps)), int(round(transition_s * fps))
    rows = [qs.copy() for _ in range(n_hold)]
    root_slerp = Slerp([0, 1], R(np.stack([qs[3:7], qk[3:7]])))
    foot_slerps = [Slerp([0, 1], R(np.stack([a[1], b[1]]))) for a, b in zip(feet_s, feet_k)]
    ik_p, ik_r = 0.0, 0.0
    q_prev = qs.copy()
    for i in range(n_tr + 1):
        s = float(min_jerk(np.array(i / n_tr)))
        q = np.empty_like(qs)
        q[0:3] = (1 - s) * qs[0:3] + s * qk[0:3]
        q[3:7] = wxyz(root_slerp(s))
        q[7:] = (1 - s) * qs[7:] + s * qk[7:]
        q[7:19] = q_prev[7:19]  # legs: warm start from the previous frame
        targets = [((1 - s) * a[0] + s * b[0], wxyz(sl(s))) for a, b, sl in zip(feet_s, feet_k, foot_slerps)]
        q, rp, rr = kin.leg_ik(q, targets)
        ik_p, ik_r = max(ik_p, rp), max(ik_r, rr)
        rows.append(q)
        q_prev = q
    leg_gap = float(np.abs(rows[-1][7:19] - qk[7:19]).max())  # IK end vs the clip's own legs at frame k
    rows[-1] = qk.copy()
    n_pre = len(rows)

    # Ease-in: clip time tau(t) = ease_s * (x^3 - x^4 / 2), x = t / ease_s -> speed smoothstep(x), tau(ease_s) = ease_s / 2.
    n_e = int(round(ease_s * fps))
    x = np.arange(1, n_e + 1) / n_e
    f_ease = k + ease_s * (x**3 - x**4 / 2) * fps
    f_rest = np.arange(np.ceil(f_ease[-1] + 1e-6), len(jq))
    f_clip = np.concatenate([f_ease, f_rest])
    clip_rows = np.concatenate([sample(root_pos, f_clip), sample_quat(root_quat, f_clip), sample(jq, f_clip)], 1)
    qpos = np.concatenate([np.stack(rows), clip_rows])
    bpos = np.concatenate([np.repeat(box_pos[k][None], n_pre, 0), sample(box_pos, f_clip)])
    bquat = np.concatenate([np.repeat(box_quat[k][None], n_pre, 0), sample_quat(box_quat, f_clip)])

    # Velocities / bodies exactly as in omniretarget_to_npz.py (identity resample = finite differences).
    to_raw = lambda pos, quat, joints: np.concatenate([pos, quat[:, [1, 2, 3, 0]], joints], 1)  # noqa: E731
    motion = resample_motion(to_raw(qpos[:, 0:3], qpos[:, 3:7], qpos[:, 7:]), fps, fps)
    box = resample_motion(to_raw(bpos, bquat, np.zeros((len(bpos), len(G1_JOINT_NAMES)))), fps, fps)
    assert motion.num_frames == len(qpos), (motion.num_frames, len(qpos))
    bodies = forward_kinematics(kin.model, motion)

    # Checks: robot-box clearance over hold + transition vs the clip's own clearance at frame k.
    dists = [kin.box_distance(q, box_pos[k], box_quat[k]) for q in qpos[:n_pre]]
    i_min = int(np.argmin([dd for dd, _ in dists]))
    clear_k = kin.box_distance(qk, box_pos[k], box_quat[k])
    slide = [float(np.linalg.norm((a[0] - b[0])[:2])) for a, b in zip(feet_s, feet_k)]
    turn = [float(np.degrees(np.linalg.norm((R(b[1]) * R(a[1]).inv()).as_rotvec()))) for a, b in zip(feet_s, feet_k)]
    out = dict(
        d,
        joint_pos=motion.joint_pos.astype(np.float32),
        joint_vel=motion.joint_vel.astype(np.float32),
        body_pos_w=bodies.pos_w.astype(np.float32),
        body_quat_w=bodies.quat_w.astype(np.float32),
        body_lin_vel_w=bodies.lin_vel_w.astype(np.float32),
        body_ang_vel_w=bodies.ang_vel_w.astype(np.float32),
        object_pos_w=box.root_pos.astype(np.float32),
        object_quat_w=box.root_quat.astype(np.float32),
        object_lin_vel_w=box.root_lin_vel_w.astype(np.float32),
        object_ang_vel_w=box.root_ang_vel_w.astype(np.float32),
        body_names=np.array(bodies.body_names),
        source=np.array(f"{src.name} (static start: hold {hold_s} s + transition {transition_s} s to frame {k} + ease-in {ease_s} s)"),
        static_prefix_frames=np.array([n_hold, n_tr, n_e, k]),
    )
    assert out["body_names"].tolist() == names
    report = dict(
        clip=src.stem, category=str(d["category"]), frame_k=k, frame_k_s=k / fps, frames_in=len(jq), frames_out=len(qpos),
        pelvis_speed_k=float(np.linalg.norm(d["body_lin_vel_w"][k, p])),
        foot_slide_cm=[round(100 * s, 1) for s in slide], foot_turn_deg=[round(t, 1) for t in turn],
        standby_pelvis_z=float(pos_s[2]), clip_pelvis_z_k=float(root_pos[k, 2]), standby_shift_cm=100 * shift,
        ik_res_mm=1000 * ik_p, ik_res_deg=float(np.degrees(ik_r)), ik_leg_gap_rad=leg_gap,
        box_clear_min_cm=100 * dists[i_min][0], box_clear_min_body=dists[i_min][1], box_clear_min_t_s=i_min / fps,
        box_clear_frame_k_cm=100 * clear_k[0], box_clear_frame_k_body=clear_k[1],
        max_joint_speed=float(np.abs(motion.joint_vel[: n_pre + n_e]).max()), max_pelvis_speed=float(np.linalg.norm(motion.root_lin_vel_w[: n_pre + n_e], axis=-1).max()),
    )
    return out, report


_KIN: Kin | None = None


def _init_worker() -> None:
    global _KIN
    _KIN = Kin()


def _convert_one(job: tuple[Path, Path, dict]) -> dict:
    src, dst, kw = job
    try:
        out, rep = build(_KIN, src, **kw)
        np.savez(dst, **out)
        return rep
    except Exception as e:  # noqa: BLE001
        return {"clip": src.stem, "error": repr(e)}


def main(src_dir: Path, out_dir: Path, clips: list[str] = [], hold_s: float = 1.0, transition_s: float = 1.5, ease_s: float = 1.0,
         search_s: float = 0.5, clear_m: float = 0.02, workers: int = 1) -> None:
    """``clips`` empty = every npz in src_dir. Writes <clip>_static.npz, static_report.json, manifest.csv and
    manifest_original.csv (the un-augmented ``*_original`` clips, for evaluation)."""
    import multiprocessing as mp
    import time

    names = clips or sorted(p.stem for p in src_dir.glob("*.npz"))
    out_dir.mkdir(parents=True, exist_ok=True)
    kw = dict(hold_s=hold_s, transition_s=transition_s, ease_s=ease_s, search_s=search_s, clear_m=clear_m)
    jobs = [(src_dir / f"{c}.npz", out_dir / f"{c}_static.npz", kw) for c in names]
    t0 = time.time()
    if workers > 1:
        with mp.Pool(workers, initializer=_init_worker) as pool:
            reports = []
            for i, r in enumerate(pool.imap_unordered(_convert_one, jobs, chunksize=2), 1):
                reports.append(r)
                if i % 200 == 0 or i == len(jobs):
                    print(f"  {i}/{len(jobs)} done, {time.time() - t0:.0f}s", flush=True)
    else:
        _init_worker()
        reports = [_convert_one(j) for j in jobs]
    reports.sort(key=lambda r: r["clip"])
    bad = [r for r in reports if "error" in r]
    ok = [r for r in reports if "error" not in r]
    for r in bad[:20]:
        print("FAILED", r["clip"], r["error"])
    if len(ok) <= 40:
        for rep in ok:
            print(f"{rep['clip']:28s} {rep['category']:5s} k={rep['frame_k']:2d} ({rep['pelvis_speed_k']:.2f} m/s) | slide L/R {rep['foot_slide_cm']} cm turn {rep['foot_turn_deg']} deg"
                  f" | shift {rep['standby_shift_cm']:.1f} cm | pelvis z {rep['standby_pelvis_z']:.3f}->{rep['clip_pelvis_z_k']:.3f} | IK {rep['ik_res_mm']:.2f} mm gap {rep['ik_leg_gap_rad']:.3f}"
                  f" | box clear min {rep['box_clear_min_cm']:5.1f} cm ({rep['box_clear_min_body']} @{rep['box_clear_min_t_s']:.2f}s) vs frame k {rep['box_clear_frame_k_cm']:5.1f}"
                  f" | max qd {rep['max_joint_speed']:.1f} v {rep['max_pelvis_speed']:.2f}")
    if ok:
        slide = np.array([max(r["foot_slide_cm"]) for r in ok])
        turn = np.array([max(r["foot_turn_deg"]) for r in ok])
        shift = np.array([r["standby_shift_cm"] for r in ok])
        clear = np.array([r["box_clear_min_cm"] for r in ok])
        pct = lambda a: f"p50 {np.percentile(a, 50):.1f} p90 {np.percentile(a, 90):.1f} max {a.max():.1f}"  # noqa: E731
        print(f"{len(ok)} clips ok, {len(bad)} failed, {time.time() - t0:.0f}s | foot slide cm {pct(slide)} | foot turn deg {pct(turn)}")
        print(f"  standby shifted back from the box: {(shift > 0).sum()} clips ({pct(shift)} cm) | prefix box clearance < 0: {(clear < 0).sum()}, < -2 cm: {(clear < -2).sum()}"
              f" | max IK residual {max(r['ik_res_mm'] for r in ok):.3f} mm | max joint speed {max(r['max_joint_speed'] for r in ok):.1f} rad/s")
    (out_dir / "static_report.json").write_text(json.dumps(reports, indent=1))
    fields = ["name", "category", "frames", "keep"]
    for fname, keep in (("manifest.csv", lambda r: True), ("manifest_original.csv", lambda r: r["clip"].endswith("_original"))):
        with open(out_dir / fname, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows([{"name": f"{r['clip']}_static", "category": r["category"], "frames": r["frames_out"], "keep": int(keep(r))} for r in ok])


if __name__ == "__main__":
    tyro.cli(main)
