"""The 1:20 tabletop slingshot the SO-101 pulls, and Froude scaling to the game.

Why a tabletop version: the reel's slingshot needs ~500 N held at full draw.
The SO-101's STS3215 servos give at most 2.94 N*m (forcerange in the
Menagerie model), so the arm can hold roughly 10 N at its reach. We therefore
build the whole scene at scale LAMBDA = 1/20 (bird 3 cm across, pouch 22.5 cm
off the ground, structures 1.1-2.2 m away) and size the bands so a full draw
needs ~6.5 N.

Froude scaling: under gravity, a model at length scale LAMBDA reproduces the
full-size motion exactly if velocities scale by sqrt(LAMBDA) and times by
sqrt(LAMBDA). So a tabletop launch (x, z, vx, vz) maps to the game launch
(x / LAMBDA, z / LAMBDA, vx / sqrt(LAMBDA), vz / sqrt(LAMBDA)), and the game's
trajectory, shrunk by LAMBDA and replayed sqrt(LAMBDA) faster, is exactly
what the tabletop bird would do.

Geometry (tabletop metres; x downrange, z up, y sideways):
  pouch rest centre (0, 0, 0.225); fork tips at y = +/-0.030.
  The pouch is a cradle whose side flaps wrap forward around the bird; each
  band attaches at a flap tip, a = 0.016 m ahead of the pouch centre and
  0.021 m to the side, so it spans w = 0.009 m sideways.
  Each side: one latex band of rest length L0 = 0.040 m.
  Pulling the pouch centre back by d along -u (u = aim direction) stretches
  each band to L = sqrt((d - a)^2 + w^2); lambda = L / L0;
  tension T = A0 * P_ogden(lambda).
The bands are slack while |d - a| < s = sqrt(L0^2 - w^2) = 0.039 m. On release
the pouch and bird coast through the fork until the bands catch the pouch at
d = a - s (0.023 m beyond rest), and the bird flies free there.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from .config import GRAVITY
from .slingshot import Launch, ogden_nominal_stress

LAMBDA = 1.0 / 20.0          # tabletop length / game length
SQRT_LAMBDA = math.sqrt(LAMBDA)
GAME_REST = (0.0, 4.5)       # game-scale pouch rest point (matches Slingshot)


@dataclass
class TabletopSlingshot:
    rest: tuple[float, float] = (0.0, 0.225)   # (x, z), metres
    fork_half_width: float = 0.030
    site_half_width: float = 0.021   # band anchors on the pouch flaps
    site_ahead: float = 0.016        # ... this far ahead of the pouch centre
    bird_offset: float = 0.019       # bird centre ahead of the pouch centre
    L0: float = 0.040
    A0: float = 3.7e-6        # m^2 per band (~1.9 mm square latex)
    n_sides: int = 2
    bird_mass: float = 0.020  # kg, 3 cm rubber ball
    bird_radius: float = 0.015
    pouch_mass: float = 0.003
    hysteresis: float = 0.90
    d_max: float = 0.130      # m: lambda = 2.86 at full draw, as in the reel

    @property
    def w(self) -> float:
        return self.fork_half_width - self.site_half_width

    @property
    def s_slack(self) -> float:
        return math.sqrt(max(self.L0 ** 2 - self.w ** 2, 0.0))

    @property
    def d_slack(self) -> float:
        """Draw below which the bands are slack."""
        return self.site_ahead + self.s_slack

    @property
    def d_sep(self) -> float:
        """Signed draw where the bird separates (negative = beyond rest)."""
        return self.site_ahead - self.s_slack

    def stretch(self, d: float) -> float:
        return math.hypot(d - self.site_ahead, self.w) / self.L0

    def band_tension(self, d: float) -> float:
        lam = self.stretch(d)
        return self.A0 * ogden_nominal_stress(lam) if lam > 1.0 else 0.0

    def pull_force(self, d: float) -> float:
        """Total force along the draw line the arm must hold at draw d."""
        L = math.hypot(d - self.site_ahead, self.w)
        return self.n_sides * self.band_tension(d) * (d - self.site_ahead) / L if L > 0 else 0.0

    def stored_energy(self, d: float, n: int = 400) -> float:
        if d <= self.d_slack:
            return 0.0
        h = (d - self.d_slack) / n
        s = self.pull_force(self.d_slack) + self.pull_force(d)
        for i in range(1, n):
            s += (4 if i % 2 else 2) * self.pull_force(self.d_slack + i * h)
        return s * h / 3.0

    def launch(self, angle_deg: float, draw_frac: float, g: float = GRAVITY,
               dt: float = 2e-5) -> Launch:
        """Tabletop launch, integrated in the x-z plane (the MuJoCo sim agrees
        to a few %). Pouch + bird move together under the bands and gravity;
        the pouch lines up with the band pull, so each band runs from a point
        `site_ahead` in front of the pouch centre (toward the fork) to its fork
        tip, `w` to the side. The bird leaves when the bands go slack."""
        d0 = min(max(draw_frac, 0.0), 1.0) * self.d_max
        th = math.radians(angle_deg)
        ux, uz = math.cos(th), math.sin(th)
        m = self.bird_mass + self.pouch_mass
        rx, rz = -d0 * ux, -d0 * uz            # pouch centre relative to rest
        vx = vz = 0.0
        hx, hz = ux, uz                        # pouch heading
        t, peak = 0.0, 0.0
        while t < 0.5:
            r = math.hypot(rx, rz)
            if r > 1e-9:
                hx, hz = -rx / r, -rz / r          # heading: toward the fork
            sx, sz = rx + self.site_ahead * hx, rz + self.site_ahead * hz
            L = math.sqrt(sx * sx + sz * sz + self.w ** 2)
            lam = L / self.L0
            if lam <= 1.0 and t > 0.0:
                break                              # slack: bird flies free
            T = self.hysteresis * self.A0 * ogden_nominal_stress(lam) if lam > 1.0 else 0.0
            ax = -self.n_sides * T * sx / L / m
            az = -self.n_sides * T * sz / L / m - g
            peak = max(peak, math.hypot(ax, az + g))
            vx += ax * dt
            vz += az * dt
            rx += vx * dt
            rz += vz * dt
            t += dt
        speed = math.hypot(vx, vz)
        U = self.stored_energy(d0)
        eb = 0.5 * self.bird_mass * speed ** 2
        bx, bz = self.rest[0] + rx + self.bird_offset * hx, self.rest[1] + rz + self.bird_offset * hz
        return Launch(position=(bx, bz), velocity=(vx, vz), speed=speed,
                      angle_deg=math.degrees(math.atan2(vz, vx)), draw=d0,
                      pull_point=(self.rest[0] - d0 * ux, self.rest[1] - d0 * uz),
                      stored_energy=U, bird_energy=eb, efficiency=eb / U if U > 0 else 0.0,
                      hold_force=self.pull_force(d0), peak_accel_g=peak / g, duration=t)

    def draw_for_speed(self, angle_deg: float, speed: float) -> float | None:
        """Draw fraction that gives `speed` (tabletop m/s) at `angle_deg`."""
        lo, hi = 0.0, 1.0
        if self.launch(angle_deg, hi).speed < speed:
            return None
        for _ in range(50):
            mid = 0.5 * (lo + hi)
            if self.launch(angle_deg, mid).speed < speed:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)


# ------------------------------------------------------------ Froude scaling
def table_to_game_pos(x: float, z: float) -> tuple[float, float]:
    return x / LAMBDA, z / LAMBDA


def table_to_game_vel(vx: float, vz: float) -> tuple[float, float]:
    return vx / SQRT_LAMBDA, vz / SQRT_LAMBDA


def game_to_table_pos(x: float, y: float) -> tuple[float, float]:
    return x * LAMBDA, y * LAMBDA


def game_speed_to_table(v: float) -> float:
    return v * SQRT_LAMBDA


def game_dt_to_table(dt: float) -> float:
    return dt * SQRT_LAMBDA
