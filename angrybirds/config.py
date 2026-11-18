"""World constants: scale, materials, block/pig catalogue, scoring.

Scale
-----
Geometry follows Science Birds (the open-source Angry Birds clone used by the
AIBIRDS competition), converted at SB_UNIT metres per Science Birds unit. The
value is chosen so that the reel's slingshot (max launch ~21 m/s, g = 9.81)
gives Angry-Birds proportions: full power at 45 deg just overshoots a level
whose structures sit 25-45 m from the slingshot.

Mass
----
Rigid-body dynamics under gravity is unchanged if every mass is multiplied by
the same factor, so absolute masses are free. We pin the bird to the reel's
0.75 kg so the slingshot/robot hardware from the reel works unchanged, and set
the 2-D densities (kg/m^2) of everything else relative to it.
"""
from __future__ import annotations

from dataclasses import dataclass

GRAVITY = 9.81
SB_UNIT = 1.4  # metres per Science Birds unit

# ---------------------------------------------------------------- materials


@dataclass(frozen=True)
class Material:
    name: str
    density: float      # 2-D density, kg/m^2
    friction: float
    restitution: float
    hp: float           # hit points, in N*s of impulse above the threshold
    threshold: float    # contact force (N) below which no damage is taken


MATERIALS: dict[str, Material] = {
    # ice: light-ish, slippery, shatters easily (also when it falls)
    "ice": Material("ice", density=1.2, friction=0.30, restitution=0.05, hp=3.0, threshold=40.0),
    # wood: the default building material
    "wood": Material("wood", density=1.3, friction=0.60, restitution=0.10, hp=7.0, threshold=60.0),
    # stone: heavy and tough; a single red-bird hit only chips it
    "stone": Material("stone", density=3.3, friction=0.80, restitution=0.05, hp=24.0, threshold=150.0),
}

# ------------------------------------------------------------------- blocks
# Science Birds block footprints in SB units (width, height) at rotation 0.
# Use rotation=90 for pillars. "SquareHole" is approximated as solid.
SB_BLOCKS: dict[str, tuple[float, float]] = {
    "SquareHole": (0.84, 0.84),
    "RectFat": (0.85, 0.43),
    "SquareSmall": (0.43, 0.43),
    "SquareTiny": (0.22, 0.22),
    "RectTiny": (0.43, 0.22),
    "RectSmall": (0.85, 0.22),
    "RectMedium": (1.68, 0.22),
    "RectBig": (2.06, 0.22),
}
BLOCK_TYPES: dict[str, tuple[float, float]] = {
    k: (w * SB_UNIT, h * SB_UNIT) for k, (w, h) in SB_BLOCKS.items()
}
# circles (radius) and right triangles (leg lengths), metres
CIRCLE_BLOCKS = {"Circle": 0.375 * SB_UNIT, "CircleSmall": 0.22 * SB_UNIT}
TRIANGLE_BLOCKS = {"Triangle": (0.82 * SB_UNIT, 0.82 * SB_UNIT)}

# --------------------------------------------------------------- pigs, bird


@dataclass(frozen=True)
class PigType:
    """Pigs take damage as velocity change (impulse / own mass, m/s), so a
    big pig does not die from its own weight landing. hp is in m/s of
    accumulated delta-v; threshold_accel (m/s^2) ignores resting loads."""
    name: str
    radius: float
    hp: float
    threshold_accel: float = 100.0
    density: float = 1.0
    friction: float = 0.6
    restitution: float = 0.2


PIG_TYPES: dict[str, PigType] = {
    "BasicSmall": PigType("BasicSmall", radius=0.22, hp=6.0),
    "BasicMedium": PigType("BasicMedium", radius=0.32, hp=7.0),
    "BasicBig": PigType("BasicBig", radius=0.45, hp=9.0),
}


@dataclass(frozen=True)
class BirdType:
    name: str = "BirdRed"
    radius: float = 0.30          # ~ a SquareSmall block across
    mass: float = 0.75            # kg, from the reel
    friction: float = 0.5
    restitution: float = 0.3
    angular_damping: float = 2.0  # stops endless rolling on flat ground


RED_BIRD = BirdType()

# ------------------------------------------------------------------ scoring
# Classic Angry Birds values (fan-documented): pig 5,000; destroyed block 500;
# each unused bird 10,000 when the level is cleared.
PIG_POINTS = 5000
BLOCK_POINTS = 500
UNUSED_BIRD_POINTS = 10000

# ------------------------------------------------------------------- world
GROUND_FRICTION = 0.8
WORLD_X_MIN, WORLD_X_MAX = -30.0, 140.0
WORLD_Y_MIN = -5.0


@dataclass
class SimConfig:
    hz: float = 120.0             # physics rate
    velocity_iters: int = 10
    position_iters: int = 8
    settle_time: float = 2.0      # s of settling before the first shot
    max_shot_time: float = 10.0   # s cap per shot
    quiet_speed: float = 0.05     # m/s; below this a body counts as at rest
    quiet_time: float = 0.5       # s everything must be quiet to end a shot
    drag: bool = False            # quadratic air drag on the bird
    # Drag is set by a ballistic coefficient k = rho*Cd*A/(2m) (1/m) rather
    # than the game-scale bird size, because aerodynamics does not survive the
    # mass rescaling above. Default = the reel's 0.75 kg, ~13 cm bird.
    drag_k: float = 0.5 * 1.2 * 0.47 * 0.0133 / 0.75
