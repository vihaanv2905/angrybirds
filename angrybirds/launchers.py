"""Launchers: turn a shot (angle, draw) into a game-scale bird release.

Two interchangeable launchers share the `Slingshot` interface
(`action_to_shot`, `shot_to_action`, `launch`, `min_angle`, ...), so
`AngryBirdsEnv(slingshot=...)` can train against either:

- `Slingshot` (slingshot.py): the reel's full-size slingshot, analytic.
- `ArmLauncher` (here): what the **SO-101 really does** in MuJoCo, read from
  the measured launch table (scripts/calibrate_launch.py). A policy trained
  with it outputs (angle, draw) that the robot can execute directly as a
  `cosim.ShotPlan`. Numpy only: no MuJoCo needed on CPU training nodes.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from .slingshot import Launch

TABLE_PATH = Path(__file__).resolve().parents[1] / "robot" / "so101" / "launch_table.npz"


class LaunchTable:
    """(angle, draw) -> measured game release (x, y, vx, vy), bilinear."""

    def __init__(self, path=TABLE_PATH):
        z = np.load(path)
        self.angles, self.draws, self.data = z["angles"], z["draws"], z["data"]

    def __call__(self, angle: float, draw: float) -> tuple[tuple, tuple]:
        a = np.clip(angle, self.angles[0], self.angles[-1])
        r = np.clip(draw, self.draws[0], self.draws[-1])
        i = min(np.searchsorted(self.angles, a, side="right") - 1, len(self.angles) - 2)
        j = min(np.searchsorted(self.draws, r, side="right") - 1, len(self.draws) - 2)
        ta = (a - self.angles[i]) / (self.angles[i + 1] - self.angles[i])
        tr = (r - self.draws[j]) / (self.draws[j + 1] - self.draws[j])
        v = ((1 - ta) * (1 - tr) * self.data[i, j] + ta * (1 - tr) * self.data[i + 1, j]
             + (1 - ta) * tr * self.data[i, j + 1] + ta * tr * self.data[i + 1, j + 1])
        return (float(v[0]), float(v[1])), (float(v[2]), float(v[3]))


class ArmLauncher:
    """The SO-101's measured launches, behind the `Slingshot` interface.
    Draws below ~0.42 leave the bands slack, so the usable range starts at 0.45."""

    min_angle = -10.0
    max_angle = 80.0
    min_draw_frac = 0.45

    def __init__(self, path=TABLE_PATH):
        self.table = LaunchTable(path)

    def action_to_shot(self, action) -> tuple[float, float]:
        a = min(max(float(action[0]), -1.0), 1.0)
        b = min(max(float(action[1]), -1.0), 1.0)
        angle = self.min_angle + (a + 1.0) / 2.0 * (self.max_angle - self.min_angle)
        draw = self.min_draw_frac + (b + 1.0) / 2.0 * (1.0 - self.min_draw_frac)
        return angle, draw

    def shot_to_action(self, angle_deg: float, draw_frac: float) -> tuple[float, float]:
        a = 2.0 * (angle_deg - self.min_angle) / (self.max_angle - self.min_angle) - 1.0
        b = 2.0 * (draw_frac - self.min_draw_frac) / (1.0 - self.min_draw_frac) - 1.0
        return a, b

    def launch(self, angle_deg: float, draw_frac: float) -> Launch:
        (x, y), (vx, vy) = self.table(angle_deg, draw_frac)
        speed = math.hypot(vx, vy)
        nan = float("nan")
        return Launch(position=(x, y), velocity=(vx, vy), speed=speed,
                      angle_deg=math.degrees(math.atan2(vy, vx)), draw=draw_frac,
                      pull_point=(nan, nan), stored_energy=nan, bird_energy=nan,
                      efficiency=nan, hold_force=nan, peak_accel_g=nan, duration=nan)


def make_launcher(name: str):
    """'reel' -> analytic full-size slingshot; 'arm' -> measured SO-101."""
    if name == "reel":
        from .slingshot import Slingshot
        return Slingshot()
    if name == "arm":
        return ArmLauncher()
    raise ValueError(f"unknown launcher {name!r} (use 'reel' or 'arm')")
