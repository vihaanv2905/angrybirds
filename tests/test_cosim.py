"""SO-101 + slingshot co-simulation checks (MuJoCo, headless)."""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
mujoco = pytest.importorskip("mujoco")

from angrybirds.cosim import CoSim, ShotPlan  # noqa: E402
from angrybirds.levels import STARTER_LEVELS  # noqa: E402
from angrybirds.tabletop import SQRT_LAMBDA, TabletopSlingshot  # noqa: E402


@pytest.fixture(scope="module")
def l1_shot():
    sim = CoSim(STARTER_LEVELS["L1_open_pig"]())
    sim.fire(ShotPlan(45.0, 0.9))
    while sim.busy and sim.d.time < 20:
        sim.step()
        assert np.isfinite(sim.d.qacc).all()
    return sim


def test_scene_compiles_with_so101():
    sim = CoSim(STARTER_LEVELS["L3_reel_tower"]())
    names = [sim.m.joint(i).name for i in range(sim.m.njnt)]
    assert names[:6] == ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
    assert sim.m.nmocap == len(sim.gw.entities)


def test_ik_reaches_every_pull():
    sim = CoSim(STARTER_LEVELS["L1_open_pig"]())
    c = sim.ctrl
    for th in range(-10, 81, 10):
        q = c.q_grasp.copy()
        u = np.array([math.cos(math.radians(th)), 0, math.sin(math.radians(th))])
        for f in np.linspace(0, 1, 30):
            tgt = sim.rest3 - f * sim.sling.d_max * u
            q = c.ik_pouch(tgt, f * math.radians(th), q, iters=8)
        assert c.reach_error(q, tgt) < 1e-3


def test_full_shot_runs_and_hands_off(l1_shot):
    lg = l1_shot.logs[-1]
    assert lg.actual_game_v is not None and lg.score is not None
    assert l1_shot.ctrl.phase == "done"


def test_mujoco_launch_matches_analytic(l1_shot):
    """MuJoCo launch vs the analytic tabletop model, compared at the hand-off
    point by propagating the analytic launch ballistically."""
    lg = l1_shot.logs[-1]
    s = TabletopSlingshot()
    L = s.launch(45.0, 0.9)
    gx, gy = lg.actual_game_pos
    x_handoff = gx * 0.05
    vx, vz = L.velocity
    t = (x_handoff - L.position[0]) / vx
    v_pred = math.hypot(vx, vz - 9.81 * t) / SQRT_LAMBDA
    v_sim = math.hypot(*lg.actual_game_v)
    assert abs(v_sim - v_pred) / v_pred < 0.06


def test_so101_holds_full_draw_within_torque_limits(l1_shot):
    hold = l1_shot.logs[-1].peak_joint_torque
    assert hold and max(hold.values()) < 2.94 * 0.6
