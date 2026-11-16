"""Sim2sim for the SuperMimic box policies: run exported deploy ONNX files the way motion_tracking_controller runs
them, in CPU MuJoCo with the box, and score them with eval_benchmark.py's criteria.

Robot models (--model):
  deploy  unitree_description mjcf/g1.xml, the model the workstation Docker sim2sim loads (MuJoCo 3.2.7), at its 2 ms
          step. Every step applies the mujoco_ros2_control (legged_control branch) joint law
          tau = clamp(kp (q* - q) - kd qdot, +-effort limit) as qfrc_applied, from the state at the start of the step;
          the policy runs every 10 steps (50 Hz).
  train   the training MjModel saved by export_deploy.py (mjlab G1, position actuators, 4 x 5 ms) in CPU MuJoCo, with
          the same policy loop: deploy vs train separates the robot-model gap from the simulator / observation pipeline.
Observations follow legged_control2 (see export_deploy.py): per-term history filled with the first value, oldest
first, raw last action; the anchor term uses the reference torso that the previous forward() returned, with
MotionCommandTerm::reset()'s start alignment. The robot starts at the reference start frame (static-start clips: the
standby pose at rest), the box at its reference pose. Termination and success as in the benchmark: pelvis height off
by > 0.25 m, gravity tilt off by > 0.8, box off by > 1 m / 45 deg; success = no termination and final box error < 0.2 m.
Clips without a box (lafan_window.py windows) run the robot alone; success = no fall.

  MUJOCO_GL=egl python sim2sim.py --onnx A.onnx B.onnx --npz-dir <clips> --deploy-xml <unitree_description>/mjcf/g1.xml \
      --out-dir out --model deploy
"""

from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "egl")

import argparse
import json
from pathlib import Path

import imageio
import mujoco
import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw, ImageFont

# supermimic/envs/box.py: the data's object pose is the OMOMO mesh frame; the collision box sits inside it.
BOX_HALF_EXTENTS = (0.162291, 0.168838, 0.181696)
BOX_GEOM_POS = (0.000556, -0.002050, 0.008592)
BOX_GEOM_QUAT = (0.079186, 0.967445, 0.235360, 0.048848)  # wxyz
ROOT_HEIGHT_THRESHOLD, ROOT_TILT_THRESHOLD = 0.25, 0.8
BOX_TERM_POS, BOX_TERM_ORI = 1.0, np.pi / 4
BOX_SUCCESS_POS = 0.2
POLICY_DT = 0.02
TERM_SIZES = {"joint_pos": 29, "joint_vel": 29, "base_ang_vel": 3, "projected_gravity": 3, "actions": 29,
              "motion_anchor_pos_b": 3, "motion_anchor_ori_b": 6}


def mat(q) -> np.ndarray:
    m = np.zeros(9)
    mujoco.mju_quat2Mat(m, np.asarray(q, dtype=np.float64))
    return m.reshape(3, 3)


def rot_z(yaw: float) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def yaw_of(R: np.ndarray) -> float:
    return float(np.arctan2(R[1, 0], R[0, 0]))


def quat_angle(q1, q2) -> float:
    return float(2.0 * np.arccos(min(1.0, abs(float(np.dot(q1, q2))))))


class CppTerm:
    """legged_control2 ObservationTerm history: the first value fills every slot, then FIFO, output oldest -> newest."""

    def __init__(self, n: int):
        self.n, self.h = n, None

    def __call__(self, v: np.ndarray) -> np.ndarray:
        self.h = [v] * self.n if self.h is None else self.h[1:] + [v]
        return np.concatenate(self.h)


def box_xml(mass: float, friction: float, pos=(0.0, 0.0, 0.2), quat=(1.0, 0.0, 0.0, 0.0)) -> str:
    f = lambda v: " ".join(f"{x:.6g}" for x in v)  # noqa: E731
    return (f'\n        <body name="box" pos="{f(pos)}" quat="{f(quat)}">\n'
            f'            <freejoint name="box_joint"/>\n'
            f'            <geom name="box_geom" type="box" size="{f(BOX_HALF_EXTENTS)}" pos="{f(BOX_GEOM_POS)}" '
            f'quat="{f(BOX_GEOM_QUAT)}" mass="{mass:.6g}" friction="{friction:.6g} 0.005 0.0001" condim="3" priority="1" '
            f'contype="1" conaffinity="1" rgba="0.85 0.65 0.35 1"/>\n'
            f'        </body>\n    ')


# hands="mesh": unitree_description's hand collision is one fat short capsule (r 5 cm, x 0-15 cm) while the rubber-hand
# mesh spans x 4-17 cm, 6.7 cm wide; swap in the training model's (mjlab) wrist + hand capsules, which fit the mesh.
MESH_HANDS = {
    '<geom name="{s}_hand_collision" class="collision" type="capsule" size="0.05" fromto="0.05 0 0 0.1 0 0"/>':
        '<geom name="{s}_hand_collision" class="collision" type="capsule" size="0.035 0.0412" pos="0.11 {y} 0" quat="0.7071 {qx} -0.686 0"/>',
    '<geom class="visual" mesh="{s}_wrist_pitch_link"/>':
        '<geom class="visual" mesh="{s}_wrist_pitch_link"/><geom name="{s}_wrist_collision" class="collision" type="capsule" '
        'size="0.035 0.035" pos="0.025 0 0" quat="0.7071 0 -0.7071 0"/>',
}


def deploy_scene_xml(xml: Path, mass: float | None = None, friction: float = 0.8, hands: str = "unitree", **pose) -> str:
    """unitree_description g1.xml (absolute meshdir), plus the box unless mass is None (box geom: condim 3 / priority 1,
    as in training); hands="mesh" swaps in mesh-fitting hand collision capsules (MESH_HANDS)."""
    s = xml.read_text().replace('meshdir="../meshes/g1"', f'meshdir="{(xml.parent.parent / "meshes" / "g1").resolve()}"')
    if hands == "mesh":
        for side, y, qx in (("left", "-0.01", "-0.1715"), ("right", "0.01", "0.1715")):
            for old, new in MESH_HANDS.items():
                old, new = old.format(s=side), new.format(s=side, y=y, qx=qx)
                assert s.count(old) == 1, old
                s = s.replace(old, new)
        # the added wrist capsule overlaps the elbow capsule (g1.xml keeps robot self-collision on): exclude that pair
        excl = "".join(f'\n        <exclude body1="{side}_elbow_link" body2="{side}_wrist_{b}_link"/>'
                       for side in ("left", "right") for b in ("roll", "pitch", "yaw"))
        assert s.count("<contact>") == 1
        s = s.replace("<contact>>", "<contact>" + excl, 1) if "<contact>>" in s else s.replace("<contact>", "<contact>" + excl, 1)
    if mass is None:
        return s
    i = s.rfind("</worldbody>")
    return s[:i] + box_xml(mass, friction, **pose) + s[i:]


def find_joint(m: mujoco.MjModel, name: str) -> int:
    for j in range(m.njnt):
        n = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) or ""
        if n == name or n.endswith("/" + name):
            return j
    raise KeyError(name)


def find_body(m: mujoco.MjModel, name: str) -> int:
    for b in range(m.nbody):
        n = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or ""
        if n == name or n.endswith("/" + name):
            return b
    raise KeyError(name)


def free_joint_of(m: mujoco.MjModel, body: int) -> int:
    j = int(m.body_jntadr[body])
    assert j >= 0 and m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE, body
    return j


class Scene:
    """Robot + box in CPU MuJoCo with the deploy (qfrc_applied PD) or training (position actuators) joint drive."""

    def __init__(self, kind: str, deploy_xml: Path, train_mjb: Path | None, joint_names: list[str], kp, kd,
                 box_mass: float | None, box_friction: float, hands: str = "unitree"):
        self.kind = kind
        if kind == "deploy":
            self.m = mujoco.MjModel.from_xml_string(deploy_scene_xml(deploy_xml, box_mass, box_friction, hands=hands))
        else:
            self.m = mujoco.MjModel.from_binary_path(str(train_mjb))
        m = self.m
        self.d = mujoco.MjData(m)
        jids = [find_joint(m, n) for n in joint_names]
        self.qadr = np.array([m.jnt_qposadr[j] for j in jids])
        self.vadr = np.array([m.jnt_dofadr[j] for j in jids])
        self.pelvis, self.torso = find_body(m, "pelvis"), find_body(m, "torso_link")
        self.root_q = m.jnt_qposadr[free_joint_of(m, self.pelvis)]
        self.root_v = m.jnt_dofadr[free_joint_of(m, self.pelvis)]
        self.box = find_body(m, "box") if box_mass is not None else None  # box_mass None: robot only
        if self.box is not None:
            self.box_q = m.jnt_qposadr[free_joint_of(m, self.box)]
            self.box_geom = int(m.body_geomadr[self.box])
        if kind == "train":
            if self.box is not None:  # training model carries the nominal box: set mass (inertia scaled) and friction
                s = box_mass / float(m.body_mass[self.box])
                m.body_mass[self.box] *= s
                m.body_inertia[self.box] *= s
                m.geom_friction[self.box_geom, 0] = box_friction
            act_of = {int(m.actuator_trnid[a, 0]): a for a in range(m.nu)}
            self.act = np.array([act_of[j] for j in jids])
        self.kp, self.kd = np.asarray(kp), np.asarray(kd)
        self.tau_max = np.array([m.jnt_actfrcrange[j, 1] if m.jnt_actfrclimited[j] else np.inf for j in jids])
        self.n_sub = int(round(POLICY_DT / m.opt.timestep))
        self.foot_geoms = [g for g in range(m.ngeom) if "foot" in (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "")
                           and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_CAPSULE and (m.geom_contype[g] or m.geom_conaffinity[g])]

    def reset(self, root_pos, root_quat, q, box_pos=None, box_quat=None) -> float:
        """Set the start state (zero velocity); shift the robot vertically so the lowest sole capsule touches the floor."""
        d = self.d
        mujoco.mj_resetData(self.m, d)
        d.qpos[self.root_q:self.root_q + 3] = root_pos
        d.qpos[self.root_q + 3:self.root_q + 7] = root_quat
        d.qpos[self.qadr] = q
        if self.box is not None:
            d.qpos[self.box_q:self.box_q + 3] = box_pos
            d.qpos[self.box_q + 3:self.box_q + 7] = box_quat
        mujoco.mj_kinematics(self.m, d)
        low = min(float(d.geom_xpos[g, 2] - self.m.geom_size[g, 0]) for g in self.foot_geoms)
        d.qpos[self.root_q + 2] -= low
        mujoco.mj_forward(self.m, d)
        return -low

    def step(self, q_target: np.ndarray) -> None:
        d = self.d
        for _ in range(self.n_sub):
            if self.kind == "deploy":
                tau = self.kp * (q_target - d.qpos[self.qadr]) - self.kd * d.qvel[self.vadr]
                d.qfrc_applied[self.vadr] = np.clip(tau, -self.tau_max, self.tau_max)
            else:
                d.ctrl[self.act] = q_target
            mujoco.mj_step(self.m, d)

    # robot / box state (the pelvis IMU site in g1.xml is unrotated: its gyro equals the free joint's body-frame rate)
    def joint_pos(self): return self.d.qpos[self.qadr].copy()
    def joint_vel(self): return self.d.qvel[self.vadr].copy()
    def base_ang_vel(self): return self.d.qvel[self.root_v + 3:self.root_v + 6].copy()
    def pelvis_R(self): return self.d.xmat[self.pelvis].reshape(3, 3).copy()
    def pelvis_pos(self): return self.d.xpos[self.pelvis].copy()
    def torso_R(self): return self.d.xmat[self.torso].reshape(3, 3).copy()
    def torso_pos(self): return self.d.xpos[self.torso].copy()
    def box_pose(self): return self.d.qpos[self.box_q:self.box_q + 3].copy(), self.d.qpos[self.box_q + 3:self.box_q + 7].copy()


class Clip:
    def __init__(self, path: Path, joint_names: list[str]):
        with np.load(path) as d:
            names = [str(n) for n in d["body_names"]]
            jn = [str(n) for n in d["joint_names"]]
            ji = [jn.index(n) for n in joint_names]
            self.n = int(d["joint_pos"].shape[0])
            self.q = d["joint_pos"][:, ji].astype(np.float64)
            pi = names.index("pelvis")
            self.root_pos, self.root_quat = d["body_pos_w"][:, pi].astype(np.float64), d["body_quat_w"][:, pi].astype(np.float64)
            self.feet = d["body_pos_w"][:, [names.index("left_ankle_roll_link"), names.index("right_ankle_roll_link")]].astype(np.float64)
            self.box_pos = self.box_quat = None
            if "object_pos_w" in d:
                self.box_pos, self.box_quat = d["object_pos_w"].astype(np.float64), d["object_quat_w"].astype(np.float64)
            self.category = str(d["category"]) if "category" in d else ""
            self.phases = None
            if "phase_ends" in d:  # lafan_window.py: named phases, end frame of each
                self.phases = list(zip((str(n) for n in d["phase_names"]), d["phase_ends"].tolist()))
            elif "static_prefix_frames" in d:  # static_start_prefix.py: hold, transition, ease-in lengths (frames)
                n_hold, n_tr, n_e, _ = d["static_prefix_frames"].tolist()
                self.phases = [("standby hold", n_hold), ("transition", n_hold + n_tr + 1), ("ease-in", n_hold + n_tr + 1 + n_e)]

    def phase(self, f: int) -> str:
        return next((n for n, end in self.phases if f < end), "clip") if self.phases else ""


def placement(clip: Clip, f: int) -> dict:
    """Where the box stands relative to the robot at frame f: origin = midpoint between the ankles on the floor,
    x = robot heading (pelvis yaw), y = left. Box size along the robot's x / y / vertical, and its yaw in (-45, 45]."""
    yaw = yaw_of(mat(clip.root_quat[f]))
    Rh = rot_z(yaw)
    mid = clip.feet[f].mean(0)
    mid[2] = 0.0
    Rb = mat(clip.box_quat[f]) @ mat(BOX_GEOM_QUAT)
    center = clip.box_pos[f] + mat(clip.box_quat[f]) @ np.asarray(BOX_GEOM_POS)
    rel = Rh.T @ (center - mid)
    A = Rh.T @ Rb  # box axes in the robot heading frame (columns)
    vert = int(np.argmax(np.abs(A[2])))
    horiz = [i for i in range(3) if i != vert]
    fwd = max(horiz, key=lambda i: abs(A[0, i]))
    side = [i for i in horiz if i != fwd][0]
    box_yaw = np.degrees(np.arctan2(A[1, fwd], A[0, fwd]))
    box_yaw = (box_yaw + 45.0) % 90.0 - 45.0
    h = np.asarray(BOX_HALF_EXTENTS)
    return {
        "box_center_forward_m": round(float(rel[0]), 3),
        "box_center_left_m": round(float(rel[1]), 3),
        "box_center_height_m": round(float(center[2]), 3),
        "box_yaw_deg": round(float(box_yaw), 1),
        "box_size_along_forward_m": round(2 * float(h[fwd]), 3),
        "box_size_along_left_m": round(2 * float(h[side]), 3),
        "box_size_vertical_m": round(2 * float(h[vert]), 3),
        "near_face_distance_m": round(float(rel[0]) - float(h[fwd]), 3),
        "box_tilt_deg": round(float(np.degrees(np.arccos(min(1.0, abs(A[2, vert]))))), 1),
    }


class Video:
    def __init__(self, path: Path, scene: Scene, ghost: mujoco.MjModel, robot_joint_names: list[str], width: int, height: int, yaw0: float,
                 cheap: bool = False):
        m = scene.m
        m.vis.global_.offwidth, m.vis.global_.offheight = max(width, m.vis.global_.offwidth), max(height, m.vis.global_.offheight)
        self.r = mujoco.Renderer(m, height=height, width=width)
        if cheap:  # software (llvmpipe) rendering: shadows and floor reflection cost most of the frame time
            self.r.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
            self.r.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 0
        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.cam.distance, self.cam.elevation, self.cam.azimuth = 3.4, -18.0, np.degrees(yaw0) + 115.0
        self.lookat = None
        self.ghost, self.gd = ghost, mujoco.MjData(ghost)
        gj = [find_joint(ghost, n) for n in robot_joint_names]
        self.g_qadr = np.array([ghost.jnt_qposadr[j] for j in gj])
        self.g_root = ghost.jnt_qposadr[free_joint_of(ghost, find_body(ghost, "pelvis"))]
        self.opt, self.pert = mujoco.MjvOption(), mujoco.MjvPerturb()
        self.font = ImageFont.load_default(size=24)
        self.w = imageio.get_writer(path, fps=round(1 / POLICY_DT), codec="libx264", quality=8)

    def frame(self, scene: Scene, clip: Clip, f: int, text: str) -> None:
        target = scene.pelvis_pos() if scene.box is None else 0.5 * (scene.pelvis_pos() + scene.box_pose()[0])
        target[2] = 0.55
        self.lookat = target if self.lookat is None else 0.9 * self.lookat + 0.1 * target
        self.cam.lookat[:] = self.lookat
        self.r.update_scene(scene.d, camera=self.cam)
        scn = self.r.scene
        gd = self.gd
        gd.qpos[self.g_root:self.g_root + 3] = clip.root_pos[f]
        gd.qpos[self.g_root + 3:self.g_root + 7] = clip.root_quat[f]
        gd.qpos[self.g_qadr] = clip.q[f]
        mujoco.mj_kinematics(self.ghost, gd)
        n0 = scn.ngeom
        mujoco.mjv_addGeoms(self.ghost, gd, self.opt, self.pert, mujoco.mjtCatBit.mjCAT_DYNAMIC, scn)
        for i in range(n0, scn.ngeom):
            scn.geoms[i].rgba[:] = (0.35, 0.65, 1.0, 0.22)
        if clip.box_pos is not None and scn.ngeom < scn.maxgeom:  # reference box: translucent orange
            Rm = mat(clip.box_quat[f])
            mujoco.mjv_initGeom(scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_BOX, np.asarray(BOX_HALF_EXTENTS),
                                clip.box_pos[f] + Rm @ np.asarray(BOX_GEOM_POS), (Rm @ mat(BOX_GEOM_QUAT)).flatten(),
                                np.array([1.0, 0.55, 0.1, 0.35], dtype=np.float32))
            scn.ngeom += 1
        img = Image.fromarray(self.r.render())
        draw = ImageDraw.Draw(img)
        draw.rectangle((0, 0, img.width, 40), fill=(0, 0, 0))
        draw.text((12, 7), text, fill=(255, 255, 255), font=self.font)
        self.w.append_data(np.asarray(img))

    def close(self):
        self.w.close()
        self.r.close()


def run(onnx_path: Path, npz_dir: Path, kind: str, deploy_xml: Path, box_mass: float, box_friction: float,
        out_dir: Path, tag: str, render: bool, width: int, height: int, box_term: bool = True, cheap_render: bool = False,
        hands: str = "unitree") -> dict:
    info = json.loads(onnx_path.with_suffix(".deploy.json").read_text())
    clip_name, start = info["clip"], int(info["start_step"])
    so = ort.SessionOptions()
    so.intra_op_num_threads = so.inter_op_num_threads = 1  # single-sample inference; avoids slurm cpuset affinity errors
    sess = ort.InferenceSession(str(onnx_path), sess_options=so, providers=["CPUExecutionProvider"])
    meta = sess.get_modelmeta().custom_metadata_map
    floats = lambda k: np.array([float(x) for x in meta[k].split(",")])  # noqa: E731
    joint_names = meta["joint_names"].split(",")
    kp, kd, q0, scale = floats("joint_stiffness"), floats("joint_damping"), floats("default_joint_pos"), floats("action_scale")
    obs_names = meta["observation_names"].split(",")
    hists = [int(float(h)) for h in meta["observation_history_lengths"].split(",")]
    bodies = meta["body_names"].split(",")
    ai = bodies.index(meta["anchor_body_name"])
    assert "motion_anchor_pos_b" not in obs_names, "position-anchor policies need a state estimator; not simulated here"

    clip = Clip(npz_dir / f"{clip_name}.npz", joint_names)
    has_box = clip.box_pos is not None
    mjb = onnx_path.with_suffix(".train_model.mjb")  # older exports: one per clip; now one per directory
    scene = Scene(kind, deploy_xml, mjb if mjb.exists() else onnx_path.parent / "train_model.mjb", joint_names, kp, kd,
                  box_mass if has_box else None, box_friction, hands)
    z_shift = scene.reset(clip.root_pos[start], clip.root_quat[start], clip.q[start],
                          *((clip.box_pos[start], clip.box_quat[start]) if has_box else ()))

    def forward(obs, t):
        return sess.run(None, {"obs": obs[None].astype(np.float32), "time_step": np.array([[float(t)]], dtype=np.float32)})

    obs_dim = sum(TERM_SIZES[n] * h for n, h in zip(obs_names, hists))
    ref = forward(np.zeros(obs_dim), start)  # reference at the start frame, as the controller has it after reset()
    Rr, pr = mat(ref[4][0, ai]), ref[3][0, ai].astype(np.float64)
    Ra, pa = scene.torso_R(), scene.torso_pos()
    R_w2i = rot_z(yaw_of(Ra)) @ rot_z(yaw_of(Rr)).T  # MotionCommandTerm::reset(): start frame -> robot position + yaw
    t_w2i = pa - R_w2i @ pr
    terms = {n: CppTerm(h) for n, h in zip(obs_names, hists)}
    last_action = np.zeros(len(joint_names))
    video = None
    if render:
        ghost = mujoco.MjModel.from_xml_string(deploy_scene_xml(deploy_xml))  # robot only
        video = Video(out_dir / f"{tag}_{kind}_{clip.category}_{clip_name}.mp4", scene, ghost, joint_names, width, height,
                      yaw_of(mat(clip.root_quat[start])), cheap_render)
    box_err, box_rot, q_err, dz_all, tilt_all = [], [], [], [], []
    fail, f = None, start
    try:
        for f in range(start, clip.n - 1):
            if video is not None and has_box:
                e = f"{100 * box_err[-1]:.1f} cm" if box_err else "-"
                video.frame(scene, clip, f, f"{tag} [{kind} model, box {box_mass:g} kg]  |  {clip.category}  |  {clip_name}  |  "
                                            f"{clip.phase(f)}  |  box error {e}   (blue = reference, orange = reference box)")
            elif video is not None:
                video.frame(scene, clip, f, f"{tag} [{kind} model]  |  {clip_name}  |  {clip.phase(f)}  |  "
                                            f"t {(f - start) * POLICY_DT:.1f} s   (blue = reference)")
            Rp = scene.pelvis_R()
            vals = {"joint_pos": scene.joint_pos() - q0, "joint_vel": scene.joint_vel(), "base_ang_vel": scene.base_ang_vel(),
                    "projected_gravity": Rp.T @ np.array([0.0, 0.0, -1.0]), "actions": last_action}
            Ra = scene.torso_R()
            Rrel = Ra.T @ R_w2i @ Rr
            vals["motion_anchor_ori_b"] = np.array([Rrel[0, 0], Rrel[0, 1], Rrel[1, 0], Rrel[1, 1], Rrel[2, 0], Rrel[2, 1]])
            obs = np.concatenate([terms[n](vals[n]) for n in obs_names])
            out = forward(obs, f + 1)  # the controller calls forward(time_step = f + 1) while tracking frame f
            action = out[0][0].astype(np.float64)
            scene.step(q0 + scale * action)
            last_action = action
            Rr = mat(out[4][0, ai])
            # score against the reference frame the env now tracks (f + 1)
            g = f + 1
            if has_box:
                bp, bq = scene.box_pose()
                box_err.append(float(np.linalg.norm(bp - clip.box_pos[g])))
                box_rot.append(quat_angle(bq, clip.box_quat[g]))
            q_err.append(float(np.abs(scene.joint_pos() - clip.q[g]).mean()))
            dz = abs(float(clip.root_pos[g, 2] - scene.pelvis_pos()[2]))
            tilt = abs(float((mat(clip.root_quat[g]).T @ [0, 0, -1])[2] - (scene.pelvis_R().T @ [0, 0, -1])[2]))
            dz_all.append(dz)
            tilt_all.append(tilt)
            if dz > ROOT_HEIGHT_THRESHOLD or tilt > ROOT_TILT_THRESHOLD:
                fail = ("fall (pelvis height)" if dz > ROOT_HEIGHT_THRESHOLD else "fall (tilt)", g)
            elif box_term and has_box and (box_err[-1] > BOX_TERM_POS or box_rot[-1] > BOX_TERM_ORI):
                fail = ("box off reference", g)
            if fail:
                break
    finally:
        if video is not None:
            video.close()
    root_drift = float(np.linalg.norm(scene.pelvis_pos()[:2] - clip.root_pos[f + 1, :2]))
    res = {
        "onnx": onnx_path.name, "clip": clip_name, "category": clip.category, "model": kind, "hands": hands if kind == "deploy" else "mjlab",
        "frames": clip.n, "start_step": start, "steps": len(q_err), "failed": fail is not None,
        "fail_reason": fail[0] if fail else None, "fail_time_s": round((fail[1] - start) * POLICY_DT, 2) if fail else None,
        "fail_phase": clip.phase(fail[1]) if fail else None,
        "joint_err_mean_rad": float(np.mean(q_err)), "pelvis_xy_drift_final_m": root_drift,
        "pelvis_z_err_max_m": max(dz_all), "gravity_tilt_err_max": max(tilt_all),
        "success": fail is None and (not has_box or box_err[-1] < BOX_SUCCESS_POS),
        "start_alignment_yaw_deg": float(np.degrees(yaw_of(R_w2i))), "start_alignment_pos_m": float(np.linalg.norm(t_w2i)),
        "start_z_shift_m": z_shift,
    }
    status = f"FAIL {fail[0]} at {res['fail_time_s']} s ({res['fail_phase']})" if fail else "played"
    if has_box:
        res.update({
            "box_mass": box_mass, "box_friction": box_friction, "box_term": box_term,
            "box_pos_err_mean": float(np.mean(box_err)), "box_pos_err_final": box_err[-1],
            "box_rot_err_mean": float(np.mean(box_rot)), "box_rot_err_final": box_rot[-1], "box_rot_err_max": max(box_rot),
            "box_success": res["success"], "placement": placement(clip, start),
        })
        status += (f" | box err mean {100 * res['box_pos_err_mean']:.1f} cm final {100 * res['box_pos_err_final']:.1f} cm | "
                   f"rot max {np.degrees(res['box_rot_err_max']):.0f} deg final {np.degrees(res['box_rot_err_final']):.0f} deg")
    print(f"[sim2sim {kind}] {clip_name}: {status} | success {res['success']} | joint err {res['joint_err_mean_rad']:.3f} rad | "
          f"pelvis z err max {100 * res['pelvis_z_err_max_m']:.1f} cm, tilt err max {res['gravity_tilt_err_max']:.2f} | "
          f"pelvis drift {100 * root_drift:.1f} cm | start z shift {1000 * z_shift:.1f} mm", flush=True)
    return res


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--onnx", type=Path, nargs="+", required=True)
    p.add_argument("--npz-dir", type=Path, required=True)
    p.add_argument("--deploy-xml", type=Path, required=True)
    p.add_argument("--model", choices=["deploy", "train"], default="deploy")
    p.add_argument("--box-mass", type=float, default=0.6)
    p.add_argument("--box-friction", type=float, default=0.8)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--tag", default="sim2sim")
    p.add_argument("--results", type=Path, default=None, help="results json (default <out-dir>/<tag>_<model>_results.json)")
    p.add_argument("--no-render", action="store_true")
    p.add_argument("--hands", choices=["unitree", "mesh"], default="unitree", help="deploy model hand collision (see MESH_HANDS)")
    p.add_argument("--cheap-render", action="store_true", help="no shadows / reflection (CPU software rendering)")
    p.add_argument("--no-box-term", action="store_true", help="only falls end the run (the box may stay off its reference)")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    a = p.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for o in a.onnx:
        try:
            results.append(run(o, a.npz_dir, a.model, a.deploy_xml, a.box_mass, a.box_friction, a.out_dir, a.tag, not a.no_render,
                               a.width, a.height, not a.no_box_term, a.cheap_render, a.hands))
        except Exception as e:  # keep going through a batch; the error is recorded
            print(f"[sim2sim {a.model}] {o.name}: ERROR {e!r}", flush=True)
            results.append({"onnx": o.name, "model": a.model, "error": repr(e), "success": False, "box_success": False})
    out = a.results or a.out_dir / f"{a.tag}_{a.model}_results.json"
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    ok = sum(r["success"] for r in results)
    print(f"[sim2sim {a.model}] success {ok}/{len(results)} -> {out}", flush=True)


if __name__ == "__main__":
    main()
