"""Environment checks from Part 1 of the plan (run: pytest -q)."""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from angrybirds.config import RED_BIRD, SimConfig  # noqa: E402
from angrybirds.env import AngryBirdsEnv  # noqa: E402
from angrybirds.generator import random_level  # noqa: E402
from angrybirds.levels import STARTER_LEVELS, Builder, Level  # noqa: E402
from angrybirds.slingshot import Slingshot  # noqa: E402
from angrybirds.world import GameWorld  # noqa: E402

G = 9.81
SKIN = 0.01


# ------------------------------------------------------------------ slingshot
def test_slingshot_matches_reel():
    s = Slingshot()
    L = s.launch(0.0, 1.0)
    assert abs(L.stored_energy - 213) < 5          # reel: U = 213 J
    assert 0.74 < L.efficiency < 0.82              # reel: eta = 79 %
    assert 20.0 < L.speed < 22.0                   # reel: v0 = 21.2 m/s
    assert 450 < L.hold_force < 550                # reel: F = 495 N
    assert abs(s.d_max - 0.67) < 0.01


def test_launch_speed_drops_with_angle():
    s = Slingshot()
    assert s.launch(0, 1).speed > s.launch(45, 1).speed > s.launch(80, 1).speed


def test_action_mapping_roundtrip():
    s = Slingshot()
    for a, d in [(-10, 0.05), (35, 0.5), (80, 1.0)]:
        a2, d2 = s.action_to_shot(s.shot_to_action(a, d))
        assert abs(a2 - a) < 1e-9 and abs(d2 - d) < 1e-9


# ----------------------------------------------------------------- projectile
def _flight(level, drag=False, speed=20.0, angle=40.0):
    gw = GameWorld(level, SimConfig(drag=drag))
    th = math.radians(angle)
    r = gw.shoot((0.0, 4.5), (speed * math.cos(th), speed * math.sin(th)))
    # x of the bird's first ground contact, from the contact listener
    assert r.first_impact is not None, "bird never landed"
    return r.first_impact[0]


def test_projectile_matches_closed_form():
    v, ang, h = 20.0, 40.0, 4.5 - RED_BIRD.radius - SKIN
    th = math.radians(ang)
    vx, vy = v * math.cos(th), v * math.sin(th)
    t = (vy + math.sqrt(vy ** 2 + 2 * G * h)) / G
    x_true = vx * t
    x_sim = _flight(Level("empty"), speed=v, angle=ang)
    assert abs(x_sim - x_true) / x_true < 0.01


def test_drag_shortens_range():
    x0 = _flight(Level("empty"))
    x1 = _flight(Level("empty"), drag=True)
    assert 0.80 * x0 < x1 < 0.97 * x0


# ------------------------------------------------------------------ stability
@pytest.mark.parametrize("name", list(STARTER_LEVELS))
def test_starter_levels_stand_still(name):
    rep = GameWorld(STARTER_LEVELS[name]()).stability_report(5.0)
    assert rep["broken"] == 0 and rep["score"] == 0
    assert rep["max_move"] < 0.02


def test_generated_levels_valid():
    rng = np.random.default_rng(123)
    for _ in range(5):
        lvl = random_level(rng)
        assert lvl.pigs
        rep = GameWorld(lvl).stability_report(3.0)
        assert rep["broken"] == 0


# --------------------------------------------------------------------- damage
def _hit(material, speed):
    b = Builder("t")
    b.block("RectMedium", material, 10.0, 0.0, vertical=True)
    gw = GameWorld(b.level)
    gw.shoot((7.0, 1.0), (speed, 0.0))
    return gw.entities[0]


def test_ice_breaks_stone_survives():
    assert not _hit("ice", 15).alive
    assert not _hit("wood", 21).alive
    assert _hit("wood", 6).alive
    stone = _hit("stone", 21)
    assert stone.alive and stone.hp < stone.max_hp


@pytest.mark.parametrize("h,survives", [(1.0, True), (4.0, False)])
def test_pig_fall(h, survives):
    b = Builder("t")
    b.pig("BasicMedium", 10.0, h)
    gw = GameWorld(b.level, settle=False)
    for _ in range(int(3 / gw.dt)):
        gw._step()
    assert (gw.pigs_left == 1) == survives


# ---------------------------------------------------------- scoring, env API
def test_determinism():
    s = Slingshot()
    L = s.launch(30, 0.9)
    lvl = STARTER_LEVELS["L3_reel_tower"]()
    poses = GameWorld.settled_poses(lvl)
    a = GameWorld(lvl, poses=poses).shoot(L.position, L.velocity)
    b = GameWorld(lvl, poses=poses).shoot(L.position, L.velocity)
    assert a.score == b.score and a.trajectory[-1] == b.trajectory[-1]


def test_level_clear_bonus():
    env = AngryBirdsEnv(levels=["L1_open_pig"])
    env.reset(seed=0)
    # a shot known (from the sweep) to kill the L1 pig: 30 deg, ~66 % draw
    s = env.sling
    hit = None
    for d in np.linspace(0.55, 0.8, 26):
        env.reset()
        _, r, term, _, info = env.step(np.array(s.shot_to_action(30, d), dtype=np.float32))
        if info["cleared"]:
            hit = info
            break
    assert hit is not None
    assert hit["shot_score"] == 5000 + 2 * 10000       # pig + 2 unused birds
    assert term


def test_gymnasium_api():
    from gymnasium.utils.env_checker import check_env
    check_env(AngryBirdsEnv(), skip_render_check=True)


def test_xml_roundtrip(tmp_path):
    lvl = STARTER_LEVELS["L4_fortress"]()
    lvl.save(tmp_path / "l.xml")
    back = Level.load(tmp_path / "l.xml")
    assert back.to_xml() == lvl.to_xml()
