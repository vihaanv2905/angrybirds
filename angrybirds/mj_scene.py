"""Build the MuJoCo model: SO-101 + tabletop slingshot (static XML) plus the
current Angry Birds level as mocap bodies at 1:20 scale.

The level's bodies are *driven* by the Box2D game (mocap = positioned by us
every frame); Box2D stays the single source of truth for the game physics.
"""
from __future__ import annotations

import math
from pathlib import Path

import mujoco
import numpy as np

from .levels import Level
from .tabletop import LAMBDA

SCENE_XML = Path(__file__).resolve().parents[1] / "robot" / "so101" / "angrybirds_scene.xml"
SLING_X = 0.39            # MuJoCo x of the pouch rest point (game x = 0)
BLOCK_DEPTH = 0.045       # visual depth (y) of tabletop blocks, m
HIDDEN_Z = -0.3           # where destroyed bodies are parked (under the floor)

RGBA = {
    "wood": (0.85, 0.68, 0.42, 1),
    "ice": (0.70, 0.88, 0.97, 0.75),
    "stone": (0.55, 0.55, 0.57, 1),
    "pig": (0.49, 0.78, 0.27, 1),
    "snout": (0.62, 0.88, 0.40, 1),
    "hill": (0.72, 0.62, 0.42, 1),
}


def game_to_mj(x: float, y: float) -> tuple[float, float, float]:
    """Game (x, y) metres -> MuJoCo world (x, y, z)."""
    return SLING_X + x * LAMBDA, 0.0, y * LAMBDA


def angle_quat(a: float) -> np.ndarray:
    """Box2D angle (CCW in the x-y game plane) -> MuJoCo quat (about -y)."""
    return np.array([math.cos(a / 2), 0.0, -math.sin(a / 2), 0.0])


def _extruded(verts2d, depth):
    v = []
    for y in (-depth / 2, depth / 2):
        for (x, z) in verts2d:
            v += [x, y, z]
    return v


def build_model(level: Level | None = None, entities=None):
    """Return (spec, model). `entities` is GameWorld.entities (for shapes);
    one mocap body `ent<eid>` is created per entity."""
    spec = mujoco.MjSpec.from_file(str(SCENE_XML))
    wb = spec.worldbody
    G = mujoco.mjtGeom
    if level is not None:
        for i, hill in enumerate(level.hills):
            verts = [(SLING_X + x * LAMBDA, y * LAMBDA) for x, y in hill.vertices]
            spec.add_mesh(name=f"hill{i}", uservert=_extruded(verts, 0.25))
            wb.add_geom(type=G.mjGEOM_MESH, meshname=f"hill{i}", rgba=RGBA["hill"],
                        contype=0, conaffinity=0)
    for e in entities or []:
        # start at the real pose (keeps MuJoCo's automatic scene centre sane)
        pos = game_to_mj(e.body.position.x, e.body.position.y) if e.alive else (0, 0, HIDDEN_Z)
        b = wb.add_body(name=f"ent{e.eid}", mocap=True, pos=list(pos))
        kw = dict(contype=0, conaffinity=0, group=1)
        if e.kind == "pig":
            r = e.shape[1] * LAMBDA
            b.add_geom(type=G.mjGEOM_SPHERE, size=[r, 0, 0], rgba=RGBA["pig"], **kw)
            # snout, eyes, ears: purely cosmetic
            b.add_geom(type=G.mjGEOM_CYLINDER, size=[0.35 * r, 0.12 * r, 0], pos=[-0.9 * r, 0, -0.1 * r],
                       quat=[0.7071, 0, 0.7071, 0], rgba=RGBA["snout"], **kw)
            for s in (-1, 1):
                b.add_geom(type=G.mjGEOM_SPHERE, size=[0.18 * r, 0, 0], pos=[-0.75 * r, 0.35 * s * r, 0.4 * r],
                           rgba=(1, 1, 1, 1), **kw)
                b.add_geom(type=G.mjGEOM_SPHERE, size=[0.08 * r, 0, 0], pos=[-0.9 * r, 0.35 * s * r, 0.4 * r],
                           rgba=(0, 0, 0, 1), **kw)
                b.add_geom(type=G.mjGEOM_SPHERE, size=[0.22 * r, 0, 0], pos=[0, 0.55 * s * r, 0.8 * r],
                           rgba=RGBA["pig"], **kw)
        elif e.shape[0] == "box":
            _, w, h = e.shape
            b.add_geom(type=G.mjGEOM_BOX, size=[w / 2 * LAMBDA, BLOCK_DEPTH / 2, h / 2 * LAMBDA],
                       rgba=RGBA[e.material], **kw)
        elif e.shape[0] == "circle":
            b.add_geom(type=G.mjGEOM_CYLINDER, size=[e.shape[1] * LAMBDA, BLOCK_DEPTH / 2, 0],
                       quat=[0.7071, 0.7071, 0, 0], rgba=RGBA[e.material], **kw)
        else:
            verts = [(x * LAMBDA, y * LAMBDA) for x, y in e.shape[1]]
            spec.add_mesh(name=f"tri{e.eid}", uservert=_extruded(verts, BLOCK_DEPTH))
            b.add_geom(type=G.mjGEOM_MESH, meshname=f"tri{e.eid}", rgba=RGBA[e.material], **kw)
    model = spec.compile()
    return spec, model
