"""Convert the OmniRetarget ``robot-object`` subset (G1 + OMOMO large box) to 50 Hz motion npz files
with the box trajectory, and label every clip carry / push / kick.

Source: ``robot-object.zip`` from huggingface.co/datasets/omniretarget/OmniRetarget_Dataset; each npz holds
``qpos`` (T, 43) at 30 fps = [robot quat wxyz (4), robot pos (3), 29 joints, box quat wxyz (4), box pos (3)].

Example:
  python scripts/data/omniretarget_to_npz.py --zip-path .../robot-object.zip --output-dir data/npz/omniretarget_box

Writes <output_dir>/<clip>.npz (the motion_convert keys plus object_pos_w / object_quat_w /
object_lin_vel_w / object_ang_vel_w, world frame, wxyz) and two manifests: ``manifest.csv`` (all clips,
keep=1) and ``manifest_original.csv`` (keep only the 312 un-augmented clips; used for evaluation).

Category rule (box height above its resting height z0 = 5th percentile of the clip):
  carry: lifted > 10 cm for more than 20 % of the clip;
  push / kick: otherwise, the box slides on the floor (< 3 cm up, > 0.15 m/s) for more than 0.3 m,
               push if the hands are closer to the box than the feet while it slides, else kick;
  other: everything else.
"""

from __future__ import annotations

import csv
import io
import multiprocessing as mp
import os
import re
import time
import zipfile
from pathlib import Path

import numpy as np
import tyro

_MODEL = None
_ZIP = None


def _init_worker(zip_path: str) -> None:
    global _MODEL, _ZIP
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    from supermimic.data.motion_convert import load_g1_model

    _MODEL = load_g1_model()
    _ZIP = zipfile.ZipFile(zip_path)


def _point_positions(bodies, names: list[str], link: str, offset: tuple[float, float, float]) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    i = names.index(link)
    q = bodies.quat_w[:, i]
    return bodies.pos_w[:, i] + Rotation.from_quat(q[:, [1, 2, 3, 0]]).apply(np.asarray(offset))


def _category(obj_pos: np.ndarray, hands: np.ndarray, feet: np.ndarray, dt: float) -> str:
    z = obj_pos[:, 2]
    z0 = np.percentile(z, 5)
    if (z - z0 > 0.10).mean() > 0.2:
        return "carry"
    speed = np.linalg.norm(np.gradient(obj_pos[:, :2], dt, axis=0), axis=1)
    sliding = (z - z0 < 0.03) & (speed > 0.15)
    if speed[sliding].sum() * dt <= 0.3:
        return "other"
    d_hand = np.linalg.norm(hands - obj_pos[:, None], axis=-1).min(1)[sliding].mean()
    d_foot = np.linalg.norm(feet - obj_pos[:, None], axis=-1).min(1)[sliding].mean()
    return "push" if d_hand < d_foot else "kick"


def _convert_one(args: tuple[str, str, float]) -> dict:
    from supermimic.data.motion_convert import G1_JOINT_NAMES, forward_kinematics, resample_motion
    from supermimic.envs.box import CONTACT_POINTS

    member, dst, out_fps = args
    name = Path(member).stem
    try:
        with np.load(io.BytesIO(_ZIP.read(member))) as d:
            q = d["qpos"].astype(np.float64)
            in_fps = float(d["fps"])
        assert q.shape[1] == 43, q.shape
        zeros = np.zeros((q.shape[0], len(G1_JOINT_NAMES)))
        # resample_motion takes the CSV layout [pos, quat xyzw, joints]; reuse it for the box with zero joints.
        robot_raw = np.concatenate([q[:, 4:7], q[:, [1, 2, 3, 0]], q[:, 7:36]], axis=1)
        box_raw = np.concatenate([q[:, 40:43], q[:, [37, 38, 39, 36]], zeros], axis=1)
        motion = resample_motion(robot_raw, in_fps, out_fps)
        box = resample_motion(box_raw, in_fps, out_fps)
        bodies = forward_kinematics(_MODEL, motion)
        hands = np.stack([_point_positions(bodies, bodies.body_names, l, o) for l, o in CONTACT_POINTS[:2]], 1)
        feet = np.stack([_point_positions(bodies, bodies.body_names, l, o) for l, o in CONTACT_POINTS[2:]], 1)
        cat = _category(box.root_pos, hands, feet, 1.0 / out_fps)
        np.savez(
            dst,
            fps=np.array([out_fps], dtype=np.float32),
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
            joint_names=np.array(G1_JOINT_NAMES),
            body_names=np.array(bodies.body_names),
            category=np.array(cat),
            source=np.array(member),
            input_fps=np.array([in_fps], dtype=np.float32),
        )
        return {"name": name, "ok": True, "error": "", "category": cat, "frames": motion.num_frames}
    except Exception as e:  # noqa: BLE001
        return {"name": name, "ok": False, "error": repr(e), "category": "", "frames": 0}


def _aug(name: str) -> tuple[str, str]:
    m = re.match(r"(.+?)_(original|rot_\d+|trans_\d+)$", name)
    return (m.group(1), m.group(2)) if m else (name, "original")


def main(zip_path: Path, output_dir: Path, output_fps: float = 50.0, workers: int = 16, overwrite: bool = False) -> None:
    members = sorted(m for m in zipfile.ZipFile(zip_path).namelist() if m.endswith(".npz"))
    output_dir.mkdir(parents=True, exist_ok=True)
    jobs = [(m, str(output_dir / f"{Path(m).stem}.npz"), output_fps) for m in members]
    if not overwrite:
        jobs = [j for j in jobs if not Path(j[1]).exists()]
    print(f"{len(members)} clips in {zip_path.name}, {len(jobs)} to convert, {workers} workers")
    t0 = time.time()
    results = []
    if jobs:
        with mp.Pool(workers, initializer=_init_worker, initargs=(str(zip_path),)) as pool:
            for i, info in enumerate(pool.imap_unordered(_convert_one, jobs, chunksize=4), 1):
                results.append(info)
                if i % 200 == 0 or i == len(jobs):
                    print(f"  {i}/{len(jobs)} done, {time.time() - t0:.0f}s")
    for r in [r for r in results if not r["ok"]][:20]:
        print("FAILED", r["name"], r["error"])

    rows = []
    for npz in sorted(output_dir.glob("*.npz")):
        with np.load(npz) as d:
            n = int(d["joint_pos"].shape[0])
            fps = float(d["fps"][0])
            cat = str(d["category"])
        base, aug = _aug(npz.stem)
        rows.append({"name": npz.stem, "base": base, "aug": aug, "category": cat, "frames": n, "duration_s": (n - 1) / fps})
    fields = ["name", "base", "aug", "category", "frames", "duration_s", "keep"]
    for fname, keep in (("manifest.csv", lambda r: True), ("manifest_original.csv", lambda r: r["aug"] == "original")):
        with open(output_dir / fname, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows([{**r, "keep": int(keep(r))} for r in rows])
    from collections import Counter

    orig = [r for r in rows if r["aug"] == "original"]
    print(f"{len(rows)} clips, {sum(r['duration_s'] for r in rows) / 3600:.3f} h; categories (all) {dict(Counter(r['category'] for r in rows))}; "
          f"(original {len(orig)}) {dict(Counter(r['category'] for r in orig))}; {time.time() - t0:.0f}s")


if __name__ == "__main__":
    tyro.cli(main)
