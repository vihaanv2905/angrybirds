"""Slingshot model from the reel: 4 latex tubes, Ogden fit to Treloar's rubber.

The pouch is pulled back a distance d along the launch line (opposite to the
aim direction). Band stretch is lambda = 1 + d / L0, so the reel's
lambda_max = 2.86 with L0 = 0.36 m gives d_max = 0.67 m (the reel shows ~0.65 m).

On release, bird + pouch + (1/3 of the band mass, the classic spring-mass
correction) are accelerated by the bands back to the rest point, where the
bands go slack and the bird separates. Rubber hysteresis returns only a
fraction of the stored energy. Gravity acts along the launch line, so steep
shots come out slightly slower: the opposite of the original game, where
Rhett Allain measured the launch speed to be the same at every angle.

Stage A of the RL plan calls `launch(angle, draw_frac)` directly. In Part 2 the
MuJoCo arm replaces the "pull to d and let go" step but hands over the same
`Launch` (release position and velocity).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from .config import GRAVITY, RED_BIRD

# Ogden 3-term fit to Treloar's (1944) natural-rubber data: (mu [Pa], alpha)
OGDEN_TRELOAR = ((0.63e6, 1.3), (0.0012e6, 5.0), (-0.01e6, -2.0))


def ogden_nominal_stress(lam: float, params=OGDEN_TRELOAR) -> float:
    """Uniaxial nominal (engineering) stress, Pa, incompressible Ogden solid."""
    return sum(mu * (lam ** (a - 1.0) - lam ** (-a / 2.0 - 1.0)) for mu, a in params)


def ogden_energy_density(lam: float, params=OGDEN_TRELOAR) -> float:
    """Strain energy per unit reference volume, J/m^3."""
    return sum(mu / a * (lam ** a + 2.0 * lam ** (-a / 2.0) - 3.0) for mu, a in params)


@dataclass
class Launch:
    position: tuple[float, float]   # release point (m)
    velocity: tuple[float, float]   # bird velocity at release (m/s)
    speed: float
    angle_deg: float
    draw: float                     # m
    pull_point: tuple[float, float]  # where the arm must hold the pouch
    stored_energy: float            # J in the bands at full draw
    bird_energy: float              # J of bird kinetic energy at release
    efficiency: float               # bird_energy / stored_energy
    hold_force: float               # N the arm must hold at full draw
    peak_accel_g: float
    duration: float                 # s from release to separation


@dataclass
class Slingshot:
    rest_point: tuple[float, float] = (0.0, 4.5)
    n_tubes: int = 4
    L0: float = 0.36                # m, unstretched tube length
    A0: float = 148e-6              # m^2, tube cross-section
    lambda_max: float = 2.86
    bird_mass: float = RED_BIRD.mass
    pouch_mass: float = 0.05
    rubber_density: float = 930.0   # kg/m^3
    hysteresis: float = 0.90        # fraction of stored energy given back
    min_angle: float = -10.0        # deg
    max_angle: float = 80.0         # deg
    min_draw_frac: float = 0.05

    # ---------------------------------------------------------- properties
    @property
    def d_max(self) -> float:
        return (self.lambda_max - 1.0) * self.L0

    @property
    def band_mass(self) -> float:
        return self.n_tubes * self.A0 * self.L0 * self.rubber_density

    @property
    def m_eff(self) -> float:
        return self.bird_mass + self.pouch_mass + self.band_mass / 3.0

    def band_force(self, d: float) -> float:
        lam = 1.0 + max(d, 0.0) / self.L0
        return self.n_tubes * self.A0 * ogden_nominal_stress(lam)

    def stored_energy(self, d: float) -> float:
        lam = 1.0 + max(d, 0.0) / self.L0
        return self.n_tubes * self.A0 * self.L0 * ogden_energy_density(lam)

    # -------------------------------------------------------------- launch
    def action_to_shot(self, action) -> tuple[float, float]:
        """Map an RL action in [-1, 1]^2 to (angle_deg, draw_frac)."""
        a = min(max(float(action[0]), -1.0), 1.0)
        b = min(max(float(action[1]), -1.0), 1.0)
        angle = self.min_angle + (a + 1.0) / 2.0 * (self.max_angle - self.min_angle)
        draw = self.min_draw_frac + (b + 1.0) / 2.0 * (1.0 - self.min_draw_frac)
        return angle, draw

    def shot_to_action(self, angle_deg: float, draw_frac: float) -> tuple[float, float]:
        a = 2.0 * (angle_deg - self.min_angle) / (self.max_angle - self.min_angle) - 1.0
        b = 2.0 * (draw_frac - self.min_draw_frac) / (1.0 - self.min_draw_frac) - 1.0
        return a, b

    def launch(self, angle_deg: float, draw_frac: float, g: float = GRAVITY,
               ode_dt: float = 2e-5) -> Launch:
        draw_frac = min(max(draw_frac, 0.0), 1.0)
        d0 = draw_frac * self.d_max
        th = math.radians(angle_deg)
        ux, uy = math.cos(th), math.sin(th)
        m = self.m_eff
        h = self.hysteresis
        g_along = g * math.sin(th)

        # Integrate the pouch along the launch line: x is remaining draw.
        x, v, t = d0, 0.0, 0.0
        peak = 0.0
        while x > 0.0 and t < 1.0:
            a = h * self.band_force(x) / m - g_along
            peak = max(peak, a)
            v += a * ode_dt
            if v <= 0.0 and a <= 0.0:   # too weak to lift the bird
                v = 0.0
                break
            x -= v * ode_dt
            t += ode_dt
        # Exact speed from energy balance (the ODE gives timing and peak accel).
        ke_total = h * self.stored_energy(d0) - m * g_along * d0
        speed = math.sqrt(max(2.0 * ke_total / m, 0.0))
        e_bird = 0.5 * self.bird_mass * speed ** 2
        U = self.stored_energy(d0)
        rx, ry = self.rest_point
        return Launch(
            position=(rx, ry),
            velocity=(speed * ux, speed * uy),
            speed=speed,
            angle_deg=angle_deg,
            draw=d0,
            pull_point=(rx - d0 * ux, ry - d0 * uy),
            stored_energy=U,
            bird_energy=e_bird,
            efficiency=e_bird / U if U > 0 else 0.0,
            hold_force=self.band_force(d0),
            peak_accel_g=peak / g,
            duration=t,
        )
