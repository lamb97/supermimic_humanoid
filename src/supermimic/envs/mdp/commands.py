"""Multi-clip motion command with future-frame goals and return-based reset sampling.

Exposes the same attribute API as mjlab's ``MotionCommand`` (``anchor_pos_w``,
``robot_body_pos_w``, ``body_pos_relative_w`` ...) so mjlab's tracking reward
functions work unchanged, plus:

* ``goal``        (N, K*68) future-frame goal features (plan D5)
* ``root_*``      reference pelvis quantities for the paper's root terminations
* ``clip_end``    (N,) True once the clip's last frame has been reached (time-out)
* return-based reset sampling over 1 s bins of the whole library (plan D14)
* with ``object_entity_name``: the object (box) is reset to its reference state together with the
  robot, and ``ref_box_*`` / ``box_*`` / ``box_pose_b`` expose the reference and simulated box
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import torch
from mjlab.managers import CommandTerm, CommandTermCfg
from mjlab.utils.lab_api.math import (
    quat_apply,
    quat_error_magnitude,
    quat_from_euler_xyz,
    quat_inv,
    quat_mul,
    sample_uniform,
    yaw_quat,
)

from supermimic.envs.mdp.motion_library import MotionLibrary

if TYPE_CHECKING:
    from mjlab.entity import Entity
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.viewer.debug_visualizer import DebugVisualizer


class MultiClipMotionCommand(CommandTerm):
    cfg: MultiClipMotionCommandCfg
    _env: ManagerBasedRlEnv

    def __init__(self, cfg: MultiClipMotionCommandCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        self.robot: Entity = env.scene[cfg.entity_name]
        self.robot_anchor_body_index = self.robot.body_names.index(cfg.anchor_body_name)
        self.motion_anchor_body_index = cfg.body_names.index(cfg.anchor_body_name)
        self.body_indexes = torch.tensor(
            self.robot.find_bodies(cfg.body_names, preserve_order=True)[0], dtype=torch.long, device=self.device
        )
        self.robot_root_body_index = self.robot.body_names.index(cfg.root_body_name)

        default_joint_pos = self.robot.data.default_joint_pos[0].detach()
        self.library = MotionLibrary(
            cfg.npz_files, cfg.body_names, default_joint_pos, device=self.device, bin_frames=cfg.bin_frames,
            compact=cfg.compact_library, store_dtype=torch.float16 if cfg.library_fp16 else torch.float32,
        )
        print(f"[INFO] {self.library.summary()}")
        lib = self.library
        self.has_box = cfg.object_entity_name is not None
        if self.has_box:
            assert lib.has_object, "object_entity_name is set but the clips carry no object trajectory"
            self.box: Entity = env.scene[cfg.object_entity_name]

        n = self.num_envs
        self.frame = torch.zeros(n, dtype=torch.long, device=self.device)
        self.clip_end = torch.zeros(n, dtype=torch.bool, device=self.device)
        self._refresh_ref()
        self.body_pos_relative_w = torch.zeros(n, len(cfg.body_names), 3, device=self.device)
        self.body_quat_relative_w = torch.zeros(n, len(cfg.body_names), 4, device=self.device)
        self.body_quat_relative_w[:, :, 0] = 1.0

        # Return-based reset sampling state.
        self.bin_return = torch.full((lib.num_bins,), cfg.g_max, device=self.device)
        self.bin_memory = torch.full((lib.num_bins,), 1.0 / lib.num_bins, device=self.device)
        self.max_ep_steps = min(int(math.ceil(env.cfg.episode_length_s / env.step_dt)) + 2, cfg.max_logged_steps)
        # Evaluation hook: when >= 0, the next reset of that env starts exactly at this global frame.
        self.forced_frame = torch.full((n,), -1, dtype=torch.long, device=self.device)
        self.ep_rewards = torch.zeros(n, self.max_ep_steps, device=self.device)
        self.ep_bins = torch.zeros(n, self.max_ep_steps, dtype=torch.long, device=self.device)
        self.ep_len = torch.zeros(n, dtype=torch.long, device=self.device)
        self._gamma_pow = cfg.gamma ** torch.arange(self.max_ep_steps, device=self.device, dtype=torch.float64)
        self.bootstrap_fn = None  # optional: env_ids -> value estimate for truncated episodes

        metric_names = ("error_anchor_pos", "error_anchor_rot", "error_body_pos", "error_body_rot",
                        "error_joint_pos", "error_joint_vel", "sampling_entropy", "episode_return_to_go")
        for k in metric_names + (("error_box_pos", "error_box_rot") if self.has_box else ()):
            self.metrics[k] = torch.zeros(n, device=self.device)

    # ------------------------------------------------------------------ reference (mjlab MotionCommand API)

    def _refresh_ref(self) -> None:
        """Cache the reference state / tracked-body states of the current target frames."""
        self._ref = self.library.ref(self.frame)
        self._bodies = self.library.bodies(self.frame)
        if self.has_box:
            self._obj = self.library.obj(self.frame)

    @property
    def command(self) -> torch.Tensor:
        return torch.cat([self.joint_pos, self.joint_vel], dim=1)

    @property
    def goal(self) -> torch.Tensor:
        return self.library.goal_window(self.frame, self.cfg.future_frames)

    def goal_with_object(self) -> torch.Tensor:
        """(N, K*77) future-frame goal features including the reference box pose."""
        return self.library.goal_window(self.frame, self.cfg.future_frames, with_object=True)

    # ------------------------------------------------------------------ box

    @property
    def ref_box_pos_w(self) -> torch.Tensor:
        return self._obj.pos_w + self._env.scene.env_origins

    @property
    def ref_box_quat_w(self) -> torch.Tensor:
        return self._obj.quat_w

    @property
    def box_pos_w(self) -> torch.Tensor:
        return self.box.data.root_link_pos_w

    @property
    def box_quat_w(self) -> torch.Tensor:
        return self.box.data.root_link_quat_w

    @property
    def box_pose_b(self) -> torch.Tensor:
        """(N, 9) simulated box pose in the robot root (pelvis) frame: [position, rot6d]."""
        from supermimic.envs.box import pose_in_frame

        return pose_in_frame(self.robot_root_pos_w, self.robot_root_quat_w, self.box_pos_w, self.box_quat_w)

    @property
    def joint_pos(self) -> torch.Tensor:
        return self._ref.joint_pos

    @property
    def joint_vel(self) -> torch.Tensor:
        return self._ref.joint_vel

    @property
    def body_pos_w(self) -> torch.Tensor:
        return self._bodies.pos_w + self._env.scene.env_origins[:, None, :]

    @property
    def body_quat_w(self) -> torch.Tensor:
        return self._bodies.quat_w

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        return self._bodies.lin_vel_w

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        return self._bodies.ang_vel_w

    @property
    def anchor_pos_w(self) -> torch.Tensor:
        return self._bodies.pos_w[:, self.motion_anchor_body_index] + self._env.scene.env_origins

    @property
    def anchor_quat_w(self) -> torch.Tensor:
        return self._bodies.quat_w[:, self.motion_anchor_body_index]

    @property
    def root_pos_w(self) -> torch.Tensor:
        return self._ref.root_pos_w + self._env.scene.env_origins

    @property
    def root_quat_w(self) -> torch.Tensor:
        return self._ref.root_quat_w

    # ------------------------------------------------------------------ robot

    @property
    def robot_joint_pos(self) -> torch.Tensor:
        return self.robot.data.joint_pos

    @property
    def robot_joint_vel(self) -> torch.Tensor:
        return self.robot.data.joint_vel

    @property
    def robot_body_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_link_pos_w[:, self.body_indexes]

    @property
    def robot_body_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_link_quat_w[:, self.body_indexes]

    @property
    def robot_body_lin_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_link_lin_vel_w[:, self.body_indexes]

    @property
    def robot_body_ang_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_link_ang_vel_w[:, self.body_indexes]

    @property
    def robot_anchor_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_link_pos_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_link_quat_w[:, self.robot_anchor_body_index]

    @property
    def robot_root_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_link_pos_w[:, self.robot_root_body_index]

    @property
    def robot_root_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_link_quat_w[:, self.robot_root_body_index]

    # ------------------------------------------------------------------ metrics

    def _update_metrics(self):
        self.metrics["error_anchor_pos"] = torch.norm(self.anchor_pos_w - self.robot_anchor_pos_w, dim=-1)
        self.metrics["error_anchor_rot"] = quat_error_magnitude(self.anchor_quat_w, self.robot_anchor_quat_w)
        self.metrics["error_body_pos"] = torch.norm(self.body_pos_relative_w - self.robot_body_pos_w, dim=-1).mean(-1)
        self.metrics["error_body_rot"] = quat_error_magnitude(self.body_quat_relative_w, self.robot_body_quat_w).mean(-1)
        self.metrics["error_joint_pos"] = torch.norm(self.joint_pos - self.robot_joint_pos, dim=-1)
        self.metrics["error_joint_vel"] = torch.norm(self.joint_vel - self.robot_joint_vel, dim=-1)
        if self.has_box:
            self.metrics["error_box_pos"] = torch.norm(self.ref_box_pos_w - self.box_pos_w, dim=-1)
            self.metrics["error_box_rot"] = quat_error_magnitude(self.ref_box_quat_w, self.box_quat_w)

    # ------------------------------------------------------------------ reward bookkeeping for the sampler

    def _log_step_rewards(self, env_mask: torch.Tensor | None = None) -> None:
        """Append this step's simulator reward (and the bin it was earned in) to each env's episode log."""
        reward = getattr(self._env, "reward_buf", None)
        if reward is None:
            return
        ids = torch.arange(self.num_envs, device=self.device) if env_mask is None else env_mask.nonzero().squeeze(-1)
        if ids.numel() == 0:
            return
        pos = self.ep_len[ids].clamp_max(self.max_ep_steps - 1)
        self.ep_rewards[ids, pos] = reward[ids]
        self.ep_bins[ids, pos] = self.library.bin_of_frame(self.frame[ids])
        self.ep_len[ids] = torch.minimum(self.ep_len[ids] + 1, torch.full_like(pos, self.max_ep_steps))

    def _finalize_episodes(self, env_ids: torch.Tensor) -> None:
        """Fold the finished episodes' discounted returns-to-go into the per-bin estimates."""
        cfg = self.cfg
        lens = self.ep_len[env_ids]
        if lens.sum() == 0:
            return
        T = int(lens.max().item())
        r = self.ep_rewards[env_ids, :T].double()
        bins = self.ep_bins[env_ids, :T]
        tmask = torch.arange(T, device=self.device)[None, :] < lens[:, None]
        r = r * tmask
        gp = self._gamma_pow[:T]
        # Return-to-go R_t = sum_{s>=t} gamma^{s-t} r_s via a reverse cumulative sum.
        disc = r * gp[None, :]
        rtg = torch.flip(torch.cumsum(torch.flip(disc, [1]), 1), [1]) / gp[None, :]
        # Truncated (timed-out) episodes: bootstrap with the critic if available, otherwise
        # drop the steps too close to the truncation point (plan D14).
        terminated = self._env.termination_manager.terminated[env_ids] if hasattr(self._env, "termination_manager") else torch.ones_like(lens, dtype=torch.bool)
        truncated = ~terminated & ~self.clip_end[env_ids]
        if truncated.any():
            if self.bootstrap_fn is not None:
                v = self.bootstrap_fn(env_ids).double()  # (n,)
                remaining = (lens[:, None] - torch.arange(T, device=self.device)[None, :]).clamp_min(0).double()
                rtg = rtg + truncated[:, None] * (cfg.gamma ** remaining) * v[:, None]
            else:
                remaining = lens[:, None] - torch.arange(T, device=self.device)[None, :]
                tmask = tmask & ~(truncated[:, None] & (remaining < cfg.min_remaining_steps))
        valid = tmask.reshape(-1)
        b = bins.reshape(-1)[valid]
        g = rtg.reshape(-1)[valid].float()
        if b.numel() == 0:
            return
        sums = torch.zeros(self.library.num_bins, device=self.device).index_add_(0, b, g)
        cnts = torch.zeros(self.library.num_bins, device=self.device).index_add_(0, b, torch.ones_like(g))
        touched = cnts > 0
        mean = sums[touched] / cnts[touched]
        self.bin_return[touched] = (1 - cfg.ema_alpha) * self.bin_return[touched] + cfg.ema_alpha * mean
        self.metrics["episode_return_to_go"][env_ids] = rtg[:, 0].float()
        self.ep_len[env_ids] = 0

    # ------------------------------------------------------------------ sampling

    def _sample_frames(self, n: int) -> torch.Tensor:
        lib = self.library
        cfg = self.cfg
        if cfg.sampling_mode == "start":
            clips = torch.randint(0, lib.num_clips, (n,), device=self.device)
            return lib.clip_start[clips]
        if cfg.sampling_mode == "uniform":
            # Uniform over frames (long clips get proportionally more starts), leaving >= 2 steps.
            f = torch.randint(0, lib.num_frames, (n,), device=self.device)
            return torch.minimum(f, lib.last_frame_of(f) - 2).clamp_min(0)
        assert cfg.sampling_mode == "return"
        hard = torch.softmax(-cfg.softmax_temp * self.bin_return / cfg.g_max, dim=0)
        self.bin_memory = (1 - cfg.memory_alpha) * self.bin_memory + cfg.memory_alpha * hard
        p = (
            cfg.uniform_ratio / lib.num_bins
            + cfg.hard_ratio * hard
            + cfg.memory_ratio * self.bin_memory / self.bin_memory.sum()
        )
        p = p / p.sum()
        b = torch.multinomial(p, n, replacement=True)
        u = torch.rand(n, device=self.device)
        f = lib.bin_start[b] + (u * (lib.bin_end[b] - lib.bin_start[b]).float()).long()
        f = torch.minimum(f, lib.last_frame_of(f) - 2).clamp_min(lib.clip_start[lib.frame_clip[f]])
        ent = -(p * (p + 1e-12).log()).sum() / math.log(max(lib.num_bins, 2))
        self.metrics["sampling_entropy"][:] = ent
        return f

    def _write_reference_state_to_sim(self, env_ids, root_pos, root_ori, root_lin_vel, root_ang_vel, joint_pos, joint_vel):
        soft_limits = self.robot.data.soft_joint_pos_limits[env_ids]
        joint_pos = torch.clip(joint_pos, soft_limits[:, :, 0], soft_limits[:, :, 1])
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        root_state = torch.cat([root_pos, root_ori, root_lin_vel, root_ang_vel], dim=-1)
        self.robot.write_root_state_to_sim(root_state, env_ids=env_ids)
        self.robot.reset(env_ids=env_ids)

    def _write_box_reference_to_sim(self, env_ids: torch.Tensor, frame: torch.Tensor) -> None:
        """Put the box at its reference state of ``frame`` (velocities in world frame)."""
        o = self.library.obj(frame)
        state = torch.cat([o.pos_w + self._env.scene.env_origins[env_ids], o.quat_w, o.lin_vel_w, o.ang_vel_w], dim=-1)
        self.box.write_root_state_to_sim(state, env_ids=env_ids)
        self.box.reset(env_ids=env_ids)

    def _resample_command(self, env_ids: torch.Tensor):
        # Episode bookkeeping for the envs being reset (terminal reward is already in reward_buf).
        mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        mask[env_ids] = True
        if self.ep_len.sum() > 0 or hasattr(self._env, "reward_buf"):
            self._log_step_rewards(mask)
            self._finalize_episodes(env_ids)
        self.ep_len[env_ids] = 0

        sampled = self._sample_frames(len(env_ids))
        forced = self.forced_frame[env_ids]
        self.frame[env_ids] = torch.where(forced >= 0, forced, sampled)
        self.clip_end[env_ids] = False

        r = self.library.ref(self.frame[env_ids])
        root_pos = r.root_pos_w + self._env.scene.env_origins[env_ids]
        root_ori = r.root_quat_w.clone()
        root_lin_vel = r.root_lin_vel_w.clone()
        root_ang_vel = r.root_ang_vel_w.clone()
        joint_pos = r.joint_pos.clone()
        joint_vel = r.joint_vel

        keys = ["x", "y", "z", "roll", "pitch", "yaw"]
        ranges = torch.tensor([self.cfg.pose_range.get(k, (0.0, 0.0)) for k in keys], device=self.device)
        rs = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device)
        root_pos = root_pos + rs[:, 0:3]
        root_ori = quat_mul(quat_from_euler_xyz(rs[:, 3], rs[:, 4], rs[:, 5]), root_ori)
        ranges = torch.tensor([self.cfg.velocity_range.get(k, (0.0, 0.0)) for k in keys], device=self.device)
        rs = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device)
        root_lin_vel = root_lin_vel + rs[:, :3]
        root_ang_vel = root_ang_vel + rs[:, 3:]
        lo, hi = self.cfg.joint_position_range
        if hi > lo:
            joint_pos = joint_pos + sample_uniform(lo, hi, joint_pos.shape, device=self.device)

        self._write_reference_state_to_sim(env_ids, root_pos, root_ori, root_lin_vel, root_ang_vel, joint_pos, joint_vel)
        if self.has_box:
            self._write_box_reference_to_sim(env_ids, self.frame[env_ids])

    def update_relative_body_poses(self) -> None:
        nb = len(self.cfg.body_names)
        anchor_pos = self.anchor_pos_w[:, None, :].repeat(1, nb, 1)
        anchor_quat = self.anchor_quat_w[:, None, :].repeat(1, nb, 1)
        robot_anchor_pos = self.robot_anchor_pos_w[:, None, :].repeat(1, nb, 1)
        robot_anchor_quat = self.robot_anchor_quat_w[:, None, :].repeat(1, nb, 1)
        delta_pos_w = robot_anchor_pos
        delta_pos_w[..., 2] = anchor_pos[..., 2]
        delta_ori_w = yaw_quat(quat_mul(robot_anchor_quat, quat_inv(anchor_quat)))
        self.body_quat_relative_w = quat_mul(delta_ori_w, self.body_quat_w)
        self.body_pos_relative_w = delta_pos_w + quat_apply(delta_ori_w, self.body_pos_w - anchor_pos)

    def _update_command(self, env_ids: torch.Tensor | None = None):
        # Clip end must be judged against the clip of the *current* frame (frame+1 may already be
        # the first frame of the next clip in the concatenated library).
        last = self.library.last_frame_of(self.frame)
        if env_ids is None:
            # Per-step path: log the reward earned against the current target frame for the envs
            # that were NOT just reset (their terminal reward was logged in _resample_command).
            reset_buf = getattr(self._env, "reset_buf", None)
            self._log_step_rewards(None if reset_buf is None else ~reset_buf)
            self.frame += 1
        else:
            self.frame[env_ids] += 1
        over = self.frame > last
        if over.any():
            self.clip_end |= over
            self.frame = torch.where(over, last, self.frame)
        self._refresh_ref()
        self.update_relative_body_poses()

    def reset_to_frame(self, env_ids: torch.Tensor, frame: torch.Tensor) -> None:
        """Deterministic reset to exact reference frames (no noise); used by evaluation."""
        self.ep_len[env_ids] = 0
        self.frame[env_ids] = frame
        self.clip_end[env_ids] = False
        r = self.library.ref(frame)
        self._write_reference_state_to_sim(
            env_ids, r.root_pos_w + self._env.scene.env_origins[env_ids], r.root_quat_w, r.root_lin_vel_w, r.root_ang_vel_w, r.joint_pos, r.joint_vel
        )
        if self.has_box:
            self._write_box_reference_to_sim(env_ids, frame)

    # ------------------------------------------------------------------ debug visualization (viewer only)

    _ghost_model = None

    def _debug_vis_impl(self, visualizer: DebugVisualizer) -> None:
        """Draw the reference pose as a translucent ghost robot (same approach as mjlab's MotionCommand)."""
        env_indices = visualizer.get_env_indices(self.num_envs)
        if not env_indices:
            return
        if self._ghost_model is None:
            import copy

            m = copy.deepcopy(self._env.sim.mj_model)
            # The box has no visual mesh: its collision geom is the one to draw as the reference box.
            box_geoms = set(self.box.indexing.geom_ids.tolist()) if self.has_box else set()
            for gi in range(m.ngeom):
                if gi in box_geoms:
                    m.geom_rgba[gi] = (1.0, 0.55, 0.1, 0.35)
                elif m.geom_contype[gi] != 0 or m.geom_conaffinity[gi] != 0:
                    m.geom_rgba[gi, 3] = 0  # hide collision geoms
                else:
                    m.geom_rgba[gi] = (0.2, 0.6, 1.0, 0.45)
            self._ghost_model = m
        indexing = self.robot.indexing
        free_q = indexing.free_joint_q_adr.cpu().numpy()
        joint_q = indexing.joint_q_adr.cpu().numpy()
        root_pos = self.root_pos_w.cpu().numpy()
        root_quat = self.root_quat_w.cpu().numpy()
        joint_pos = self.joint_pos.cpu().numpy()
        import numpy as np

        if self.has_box:
            box_q = self.box.indexing.free_joint_q_adr.cpu().numpy()
            box_pos = self.ref_box_pos_w.cpu().numpy()
            box_quat = self.ref_box_quat_w.cpu().numpy()
        for e in env_indices:
            qpos = np.zeros(self._ghost_model.nq)
            qpos[free_q[0:3]] = root_pos[e]
            qpos[free_q[3:7]] = root_quat[e]
            qpos[joint_q] = joint_pos[e]
            if self.has_box:
                qpos[box_q[0:3]] = box_pos[e]
                qpos[box_q[3:7]] = box_quat[e]
            visualizer.add_ghost_mesh(qpos, model=self._ghost_model, label=f"ghost_{e}")


@dataclass(kw_only=True)
class MultiClipMotionCommandCfg(CommandTermCfg):
    npz_files: list[str]
    anchor_body_name: str
    root_body_name: str = "pelvis"
    body_names: tuple[str, ...]
    entity_name: str = "robot"
    future_frames: int = 10
    bin_frames: int = 50
    pose_range: dict[str, tuple[float, float]] = field(default_factory=dict)
    velocity_range: dict[str, tuple[float, float]] = field(default_factory=dict)
    joint_position_range: tuple[float, float] = (0.0, 0.0)
    sampling_mode: Literal["return", "uniform", "start"] = "return"
    # Return-based sampling (plan D14).
    uniform_ratio: float = 0.5
    hard_ratio: float = 0.4
    memory_ratio: float = 0.1
    softmax_temp: float = 25.0
    g_max: float = 10.0
    gamma: float = 0.99
    ema_alpha: float = 0.05
    memory_alpha: float = 0.01
    min_remaining_steps: int = 100
    max_logged_steps: int = 1024
    compact_library: bool = False
    library_fp16: bool = True
    object_entity_name: str | None = None
    resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9)

    def build(self, env: ManagerBasedRlEnv) -> MultiClipMotionCommand:
        return MultiClipMotionCommand(self, env)
