"""Stage-A training plumbing: launchers, observations, policy export."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from angrybirds.env import AngryBirdsEnv, observe  # noqa: E402
from angrybirds.launchers import ArmLauncher, make_launcher  # noqa: E402


def test_arm_launcher_matches_measured_launch():
    arm = ArmLauncher()
    L = arm.launch(45.0, 0.9)
    # measured by the MuJoCo co-simulation (tests/test_cosim.py scenario)
    assert abs(L.speed - 16.7) < 0.3 and abs(L.angle_deg - 40.1) < 1.0
    for a, d in [(-10, 0.45), (30, 0.7), (80, 1.0)]:
        a2, d2 = arm.action_to_shot(arm.shot_to_action(a, d))
        assert abs(a2 - a) < 1e-9 and abs(d2 - d) < 1e-9


def test_env_trains_with_arm_launcher():
    env = AngryBirdsEnv(levels=["L3_reel_tower"], slingshot=make_launcher("arm"))
    obs, _ = env.reset(seed=0)
    assert np.array_equal(obs, observe(env.gw))
    _, r, term, _, info = env.step(np.array(ArmLauncher().shot_to_action(24.0, 0.951), dtype=np.float32))
    assert info["shot_score"] > 0


def test_set_shaping_reaches_env():
    env = AngryBirdsEnv(levels=["L1_open_pig"], shaping=0.1)
    env.set_shaping(0.0)
    assert env.shaping == 0.0


def test_exported_actor_matches_sb3(tmp_path):
    pytest.importorskip("stable_baselines3")
    from stable_baselines3 import PPO

    from angrybirds.policy import NumpyPolicy
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "train"))
    from aim_ppo import export_actor

    env = AngryBirdsEnv(levels=["L2_wood_shelter"], slingshot=make_launcher("arm"))
    model = PPO("MlpPolicy", env, n_steps=16, batch_size=16, device="cpu",
                policy_kwargs=dict(net_arch=dict(pi=[32, 32], vf=[32, 32])))

    class A:
        launcher = "arm"
    export_actor(model, tmp_path / "policy.npz", A())
    pol = NumpyPolicy(tmp_path / "policy.npz")
    obs, _ = env.reset(seed=1)
    a_sb3, _ = model.predict(obs, deterministic=True)
    assert np.allclose(pol.act(obs), np.clip(a_sb3, -1, 1), atol=1e-5)
