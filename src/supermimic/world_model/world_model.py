"""World model: learned dynamics + learned termination head + analytic observation/reward heads."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from supermimic.world_model.dynamics import MLP, DynamicsMLP
from supermimic.world_model.heads import AnalyticHeads, ReferenceFrame
from supermimic.world_model.state_spec import StateSpec, StateSpecCfg


@dataclass
class WorldModelCfg:
    hidden: tuple[int, ...] = (2048, 2048, 2048)
    term_hidden: tuple[int, ...] = (64, 64)
    state_spec: StateSpecCfg = field(default_factory=StateSpecCfg)
    term_loss_weight: float = 1.0
    height_threshold: float = 0.25
    tilt_threshold: float = 0.8
    box_pos_threshold: float = 1.0
    box_ori_threshold: float = 0.785
    contact_feat_scale: float = 0.5  # divide the hand/foot-to-box vectors (m) by this


class TerminationHead(nn.Module):
    """P(terminal | state, reference). Input: analytic termination features and their magnitudes."""

    def __init__(self, hidden=(64, 64), thresholds: tuple[float, ...] = (0.25, 0.8)):
        super().__init__()
        self.mlp = MLP(2 * len(thresholds), tuple(hidden), 1)
        self.register_buffer("scale", 1.0 / torch.tensor(thresholds))

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        """feats (…, 2) = [height error, tilt error] (+ [box pos error, box ori error]) -> logits (…,)"""
        x = feats * self.scale
        return self.mlp(torch.cat([x, x.abs()], dim=-1)).squeeze(-1)


class WorldModel(nn.Module):
    def __init__(self, cfg: WorldModelCfg, heads: AnalyticHeads, device):
        super().__init__()
        self.cfg = cfg
        self.spec = StateSpec(cfg.state_spec, device)
        self.has_box = self.spec.has_box
        assert self.has_box == heads.has_box, "state spec and analytic heads disagree about the box"
        from supermimic.envs.box import CONTACT_FEAT_DIM

        self.feat_dim = self.spec.feat_dim + (CONTACT_FEAT_DIM if self.has_box else 0)
        self.dynamics = DynamicsMLP(feat_dim=self.feat_dim, xi_dim=self.spec.xi_dim, hidden=cfg.hidden, out_dim=self.spec.delta_dim)
        thresholds = (cfg.height_threshold, cfg.tilt_threshold) + ((cfg.box_pos_threshold, cfg.box_ori_threshold) if self.has_box else ())
        self.term_head = TerminationHead(cfg.term_hidden, thresholds)
        self.heads = heads  # analytic observation / reward / FK (no parameters)
        self.to(device)

    # ------------------------------------------------------------------ one step

    def features(self, s: torch.Tensor) -> torch.Tensor:
        """Network input features of a state: heading-invariant canonical state (+ hand/foot-to-box geometry)."""
        feat = self.spec.canonical(s)
        if self.has_box:
            feat = torch.cat([feat, self.heads.contact_geometry(s) / self.cfg.contact_feat_scale], dim=-1)
        return feat

    def step(self, s: torch.Tensor, action: torch.Tensor, xi: torch.Tensor) -> torch.Tensor:
        """s_{t+1} = integrate(s_t, f(s_t, a_t, xi))."""
        delta = self.dynamics(self.features(s), action, self.spec.normalize_xi(xi))
        return self.spec.integrate(s, delta)

    def termination_logits(self, s_next: torch.Tensor, ref: ReferenceFrame) -> torch.Tensor:
        # The pelvis is the free-joint body, so its pose is the state's root pose: no FK needed.
        return self.term_head(self.heads.state_termination_features(s_next, ref))

    # ------------------------------------------------------------------ Eq. 1

    def _loss_step(self, s, a, xi, s_true, done, ref_root_pos, ref_root_quat, ref_box_pos=None, ref_box_quat=None):
        """One autoregressive step of Eq. 1: returns (s_next, sq_err (n,70|82), term_bce (n,), logits (n,))."""
        s_next = self.step(s, a, xi)
        e = self.spec.error_vector(s_next, s_true)
        ref = ReferenceFrame(None, None, None, None, ref_root_pos, ref_root_quat, ref_box_pos, ref_box_quat)
        logits = self.term_head(self.heads.state_termination_features(s_next, ref))
        bce = F.binary_cross_entropy_with_logits(logits, done, reduction="none")
        return s_next, e * e, bce, logits

    def enable_compile(self, mode: str | None = "default") -> None:
        self._loss_step_fn = torch.compile(self._loss_step, mode=None if mode == "default" else mode, dynamic=False) if mode else self._loss_step

    def window_loss(self, states: torch.Tensor, a_exec: torch.Tensor, xi: torch.Tensor, done: torch.Tensor, refs: list[ReferenceFrame]):
        """k-step autoregressive loss.

        states: (k+1, n, 71|84) logged states; a_exec: (k, n, 29); xi: (n, 33|35);
        done: (k, n) termination label of states 1..k; refs[j]: reference of transition j (for the
        termination features of state j+1).
        """
        k = a_exec.shape[0]
        s = states[0]
        total_state = 0.0
        total_term = 0.0
        metrics = {}
        err_first = None
        err_last = None
        acc_num = 0.0
        step_fn = getattr(self, "_loss_step_fn", None) or self._loss_step
        for j in range(k):
            s, sq, bce, logits = step_fn(s, a_exec[j], xi, states[j + 1], done[j], refs[j].root_pos_w, refs[j].root_quat_w, refs[j].box_pos_w, refs[j].box_quat_w)
            total_state = total_state + sq.sum(-1).mean()
            total_term = total_term + bce.mean()
            with torch.no_grad():
                nse = sq.mean(-1).mean()
                if j == 0:
                    err_first = nse
                err_last = nse
                acc_num += ((logits > 0).float() == done[j]).float().mean()
        loss = total_state / k + self.cfg.term_loss_weight * total_term / k
        # Metrics stay on-device (detached tensors): a float() here would stall the CPU on the
        # forward pass every minibatch, which in data-parallel runs turns into lockstep waiting.
        metrics.update(
            state_loss=(total_state / k).detach(),
            term_bce=(total_term / k).detach(),
            nse_1step=err_first.detach(),
            nse_kstep=err_last.detach(),
            term_acc=(acc_num / k).detach(),
        )
        return loss, metrics

    # ------------------------------------------------------------------ diagnostics

    @torch.no_grad()
    def rollout_error(self, states: torch.Tensor, a_exec: torch.Tensor, xi: torch.Tensor) -> torch.Tensor:
        """Open-loop k-step normalised squared error per step (k,), averaged over the batch."""
        k = a_exec.shape[0]
        s = states[0]
        out = []
        for j in range(k):
            s = self.step(s, a_exec[j], xi)
            out.append(self.spec.normalized_sq_error(s, states[j + 1]).mean())
        return torch.stack(out)
