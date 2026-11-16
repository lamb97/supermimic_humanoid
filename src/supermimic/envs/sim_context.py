"""Per-environment simulation context xi (plan: 33-d = foot friction 1 + encoder bias 29 + torso COM offset 3;
with a box, + box mass 1 + box friction 1 -> 35)."""

from __future__ import annotations

import numpy as np
import torch

XI_DIM = 33


class SimContext:
    def __init__(self, env, foot_geom_regex: str, com_body_name: str = "torso_link", entity_name: str = "robot", object_entity: str | None = None):
        self.env = env
        self.robot = env.scene[entity_name]
        local_ids, names = self.robot.find_geoms(foot_geom_regex)
        if len(local_ids) == 0:
            raise ValueError(f"no geoms match {foot_geom_regex}")
        self.foot_geom_global = int(self.robot.indexing.geom_ids[local_ids[0]].item())
        self.com_body_global = int(self.robot.indexing.body_ids[self.robot.body_names.index(com_body_name)].item())
        self.com_nominal = torch.as_tensor(
            np.array(env.sim.mj_model.body_ipos[self.com_body_global]), dtype=torch.float32, device=env.device
        )
        self.has_object = object_entity is not None
        if self.has_object:
            obj = env.scene[object_entity]
            self.obj_body_global = int(obj.indexing.body_ids[0].item())
            self.obj_geom_global = int(obj.indexing.geom_ids[0].item())
        self.dim = XI_DIM + (2 if self.has_object else 0)

    def read(self) -> torch.Tensor:
        model = self.env.sim.model
        n = self.env.num_envs
        fr = model.geom_friction
        if fr.dim() == 3:  # expanded per world
            friction = fr[:, self.foot_geom_global, 0]
        else:
            friction = fr[self.foot_geom_global, 0].expand(n)
        ipos = model.body_ipos
        if ipos.dim() == 3:
            com = ipos[:, self.com_body_global] - self.com_nominal
        else:
            com = (ipos[self.com_body_global] - self.com_nominal).expand(n, 3)
        bias = self.robot.data.encoder_bias
        parts = [friction.reshape(n, 1), bias, com]
        if self.has_object:
            mass = model.body_mass
            obj_mass = mass[:, self.obj_body_global] if mass.dim() == 2 else mass[self.obj_body_global].expand(n)
            obj_fr = fr[:, self.obj_geom_global, 0] if fr.dim() == 3 else fr[self.obj_geom_global, 0].expand(n)
            parts += [obj_mass.reshape(n, 1), obj_fr.reshape(n, 1)]
        xi = torch.cat(parts, dim=-1)
        assert xi.shape == (n, self.dim)
        return xi.clone()
