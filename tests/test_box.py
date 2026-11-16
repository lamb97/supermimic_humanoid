"""OmniRetarget box: state I/O, reset, analytic box reward / observation / termination features and the
sim context must match the environment; world model and training loop must run with the box."""

from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
NPZ = ROOT / "data/npz/omniretarget_box"
CLIPS = ["sub10_largebox_000_original", "sub10_largebox_014_original", "sub10_largebox_061_original"]  # push, carry, kick
pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA"),
    pytest.mark.skipif(not NPZ.exists(), reason="OmniRetarget box npz not built"),
]


@pytest.fixture(scope="module")
def setup():
    from supermimic.data.motion_convert import load_g1_model
    from supermimic.envs import SuperMimicEnvCfgArgs, make_env
    from supermimic.envs.mdp.motion_library import discover_npz
    from supermimic.envs.mdp.rewards import TrackingRewardCfg
    from supermimic.envs.state_io import StateIO
    from supermimic.envs.tracking_env_cfg import ANCHOR_BODY, ROOT_BODY, TRACKED_BODIES
    from supermimic.kinematics.g1_fk import G1Kinematics
    from supermimic.world_model.heads import AnalyticHeads

    files = discover_npz(NPZ, names=CLIPS)
    assert len(files) == 3
    args = SuperMimicEnvCfgArgs(npz_files=[str(f) for f in files], num_envs=16, replay_capacity=64, obs_noise=False, seed=5,
                                box=True, policy_box_obs=True, box_term_start_step=0)
    env = make_env(args, device="cuda:0")
    torch.backends.cuda.matmul.allow_tf32 = False
    robot = env.scene["robot"]
    heads = AnalyticHeads(G1Kinematics(load_g1_model(), device=env.device), TRACKED_BODIES, ANCHOR_BODY, ROOT_BODY,
                          robot.data.soft_joint_pos_limits[0].clone(), robot.data.default_joint_pos[0].clone(), reward_cfg=TrackingRewardCfg(has_box=True))
    yield dict(env=env, heads=heads, io=StateIO(env, object_entity="box"))
    env.close()


def _ref(cmd):
    from supermimic.world_model.heads import ReferenceFrame

    lib, f = cmd.library, cmd.frame
    o = lib.obj(f)
    return ReferenceFrame(lib.body_pos_w[f], lib.body_quat_w[f], lib.body_lin_vel_w[f], lib.body_ang_vel_w[f],
                          lib.root_pos_w[f], lib.root_quat_w[f], o.pos_w, o.quat_w)


def test_state_io_box_round_trip(setup):
    env, io = setup["env"], setup["io"]
    env.reset()
    s = io.read()
    assert s.shape == (16, 84)
    ids = torch.arange(16, device=env.device)
    s2 = s.clone()
    s2[:, 71:74] += torch.tensor([0.1, -0.2, 0.05], device=env.device)
    io.write(ids, s2)
    np.testing.assert_allclose(io.read().cpu().numpy(), s2.cpu().numpy(), atol=1e-5)
    box = env.scene["box"]
    np.testing.assert_allclose((box.data.root_link_pos_w - env.scene.env_origins).cpu().numpy(), s2[:, 71:74].cpu().numpy(), atol=1e-5)
    io.write(ids, s)


def test_box_reset_to_reference(setup):
    env, io = setup["env"], setup["io"]
    cmd = env.command_manager.get_term("motion")
    lib = cmd.library
    start = lib.clip_start[torch.arange(16, device=env.device) % 3]
    cmd.forced_frame[:] = start
    env.reset()
    cmd.forced_frame[:] = -1
    # env.reset() runs command.compute(dt=0) after the reference write, which advances the target frame by one:
    # the state sits at frame `start` while cmd.frame == start + 1 (same convention as the robot).
    assert torch.equal(cmd.frame, start + 1)
    s = io.read()
    o = lib.obj(start)
    np.testing.assert_allclose(s[:, 71:74].cpu().numpy(), o.pos_w.cpu().numpy(), atol=1e-5)
    dq = torch.minimum((s[:, 74:78] - o.quat_w).abs().amax(-1), (s[:, 74:78] + o.quat_w).abs().amax(-1))
    assert dq.max() < 1e-5


def test_box_reward_obs_termination_match_env(setup):
    env, heads, io = setup["env"], setup["heads"], setup["io"]
    from supermimic.envs.mdp import rewards as sm_rew
    from supermimic.envs.mdp.motion_library import GOAL_FEAT_DIM, OBJ_GOAL_FEAT_DIM

    obs, _ = env.reset()
    cmd = env.command_manager.get_term("motion")
    rec = env.recorder_manager.get_term("replay")
    for _ in range(15):
        frame_before = cmd.frame.clone()
        obs, *_ = env.step(0.3 * torch.randn(16, 29, device=env.device))
        s = io.read()
        ref = _ref(cmd)
        ours = heads.reward(s, env.action_manager.action, env.action_manager.prev_action, ref)
        for name, fn, std in (("box_pos", sm_rew.box_position_error_exp, 0.3), ("box_ori", sm_rew.box_orientation_error_exp, 0.4)):
            np.testing.assert_allclose((ours[name] / 0.02).cpu().numpy(), fn(env, "motion", std).cpu().numpy(), atol=2e-5, err_msg=name)
        np.testing.assert_allclose(heads.box_obs(s).cpu().numpy(), obs["object"].cpu().numpy(), atol=2e-5)
        goal = cmd.library.goal_window(cmd.frame, 10, with_object=True)
        assert obs["goal"].shape[1] == 10 * (GOAL_FEAT_DIM + OBJ_GOAL_FEAT_DIM)
        np.testing.assert_allclose(goal.cpu().numpy(), obs["goal"].cpu().numpy(), atol=1e-6)
        # CommandTerm.compute refreshes the metrics *before* advancing the frame: they compare the
        # post-step state with the pre-step target frame (envs that auto-reset this step are skipped).
        from supermimic.world_model.heads import ReferenceFrame

        live = ~env.reset_buf
        o_prev = cmd.library.obj(frame_before)
        r_prev = cmd.library.ref(frame_before)
        ref_prev = ReferenceFrame(None, None, None, None, r_prev.root_pos_w, r_prev.root_quat_w, o_prev.pos_w, o_prev.quat_w)
        f = heads.state_termination_features(s, ref_prev)
        np.testing.assert_allclose(f[live, 2].cpu().numpy(), cmd.metrics["error_box_pos"][live].cpu().numpy(), atol=2e-5)
        np.testing.assert_allclose(f[live, 3].cpu().numpy(), cmd.metrics["error_box_rot"][live].cpu().numpy(), atol=2e-4)
        xi = rec.ctx.read()
        assert xi.shape == (16, 35)
        assert 0.4 < float(xi[:, 33].min()) and float(xi[:, 33].max()) < 1.1  # box mass (DR on: 0.5..1.0 kg)
        assert 0.55 < float(xi[:, 34].min()) and float(xi[:, 34].max()) < 1.05  # box friction


def test_world_model_box_step_and_window_loss(setup):
    env, heads = setup["env"], setup["heads"]
    from supermimic.world_model.world_model import WorldModel, WorldModelCfg

    cfg = WorldModelCfg(hidden=(128, 128))
    cfg.state_spec.has_box = True
    wm = WorldModel(cfg, heads, env.device)
    assert wm.feat_dim == 71 + 15 + 12 and wm.spec.delta_dim == 82 and wm.spec.xi_dim == 35
    rec = env.recorder_manager.get_term("replay")
    for _ in range(40):
        env.step(0.3 * torch.randn(16, 29, device=env.device))
    w = rec.buffer.sample_windows(64, 8)
    from supermimic.world_model.wm_trainer import WMTrainerCfg, WorldModelTrainer

    tr = WorldModelTrainer(wm, env.command_manager.get_term("motion").library, WMTrainerCfg(batch_size=64, batches_per_update=1, compile=None))
    loss, m = tr.loss_on(w)
    assert torch.isfinite(loss)
    loss.backward()
    # Zero-initialised dynamics = constant velocity: one step from s moves the box by dt * v.
    s = w.states[0]
    s1 = wm.step(s, w.a_exec[0], w.xi)
    np.testing.assert_allclose((s1[:, 71:74] - s[:, 71:74]).detach().cpu().numpy(), (0.02 * s[:, 78:81]).cpu().numpy(), atol=1e-6)


@pytest.mark.parametrize("policy_box_obs", [False, True])
def test_training_smoke_box(tmp_path, policy_box_obs):
    from supermimic.config import TrainConfig
    from supermimic.training.loop import Trainer

    cfg = TrainConfig(run_name=f"box_{int(policy_box_obs)}", log_root=tmp_path, seed=1)
    cfg.data.npz_dir = NPZ
    cfg.data.clips = CLIPS
    cfg.env.box = True
    cfg.env.policy_box_obs = policy_box_obs
    cfg.env.anchor_obs = True  # the real runs: orientation-only anchor (b)
    cfg.env.anchor_ori_only = True
    cfg.env.box_term_start_step = 30
    cfg.env.num_envs = 128
    cfg.env.replay_capacity = 200
    cfg.world_model.hidden = (256, 256)
    cfg.policy.encoder_hidden = (256, 128)
    cfg.policy.decoder_hidden = (256, 128)
    cfg.wm_trainer.batch_size = 256
    cfg.wm_trainer.batches_per_update = 2
    cfg.policy_trainer.batch_size = 128
    cfg.policy_trainer.horizon = 8
    cfg.schedule.total_iterations = 60
    cfg.schedule.warmup_iterations = 20
    cfg.schedule.log_interval = 20
    cfg.schedule.checkpoint_interval = 60
    tr = Trainer(cfg)
    try:
        expect_goal = 10 * (77 if policy_box_obs else 68) + 6
        assert tr.policy.cfg.goal_dim == expect_goal and tr.policy.cfg.obj_dim == (9 if policy_box_obs else 0)
        tr.run()
        assert tr.num_updates > 0
        assert (tmp_path / cfg.run_name / "ckpt_latest.pt").exists()
    finally:
        tr.env.close()
