"""Read / write the full simulator state as one tensor: robot [qpos | qvel] (71) and, when the scene
has a box, the box's free-joint state appended (13) -> 84."""

from __future__ import annotations

import torch

STATE_DIM = 36 + 35
NQ = 36
NV = 35


class StateIO:
    def __init__(self, env, object_entity: str | None = None):
        self.env = env
        self.sim = env.sim
        # States are stored relative to each env's origin so that they are directly comparable
        # with the (origin-free) motion library and translation-invariant across envs.
        self.origins = env.scene.env_origins.clone()  # (N, 3)
        robot = env.scene["robot"].indexing
        self.robot_q = torch.cat([robot.free_joint_q_adr, robot.joint_q_adr]).long()
        self.robot_v = torch.cat([robot.free_joint_v_adr, robot.joint_v_adr]).long()
        assert self.robot_q.numel() == NQ and self.robot_v.numel() == NV, (self.robot_q.numel(), self.robot_v.numel())
        self.has_object = object_entity is not None
        if self.has_object:
            obj = env.scene[object_entity].indexing
            self.obj_q = obj.free_joint_q_adr.long()
            self.obj_v = obj.free_joint_v_adr.long()
            assert self.obj_q.numel() == 7 and self.obj_v.numel() == 6
        else:
            assert self.sim.data.qpos.shape[-1] == NQ and self.sim.data.qvel.shape[-1] == NV, (
                self.sim.data.qpos.shape,
                self.sim.data.qvel.shape,
            )
        self.state_dim = STATE_DIM + (13 if self.has_object else 0)

    def read(self) -> torch.Tensor:
        """(N, 71) copy of the robot [qpos | qvel], plus the box [pos, quat, lin vel w, ang vel b] -> (N, 84)."""
        # Index with [:] to get plain tensors out of mjlab's TorchArray proxies (torch.cat on the
        # proxies recurses inside their __torch_function__).
        qpos, qvel = self.sim.data.qpos[:], self.sim.data.qvel[:]
        parts = [qpos[:, self.robot_q], qvel[:, self.robot_v]]
        if self.has_object:
            parts += [qpos[:, self.obj_q], qvel[:, self.obj_v]]
        s = torch.cat(parts, dim=-1).clone()
        s[:, 0:3] -= self.origins
        if self.has_object:
            s[:, 71:74] -= self.origins
        return s

    @torch.no_grad()
    def write(self, env_ids: torch.Tensor, state: torch.Tensor, forward: bool = True) -> None:
        """Overwrite qpos/qvel of ``env_ids`` and refresh derived quantities."""
        state = state.to(self.sim.data.qpos.dtype).clone()
        state[:, 0:3] += self.origins[env_ids]
        ids = env_ids[:, None]
        self.sim.data.qpos[ids, self.robot_q[None, :]] = state[:, :NQ]
        self.sim.data.qvel[ids, self.robot_v[None, :]] = state[:, NQ:STATE_DIM]
        if self.has_object:
            state[:, 71:74] += self.origins[env_ids]
            self.sim.data.qpos[ids, self.obj_q[None, :]] = state[:, 71:78]
            self.sim.data.qvel[ids, self.obj_v[None, :]] = state[:, 78:84]
        if hasattr(self.sim.data, "qacc_warmstart"):
            self.sim.data.qacc_warmstart[env_ids] = 0.0
        if forward:
            self.sim.forward()
