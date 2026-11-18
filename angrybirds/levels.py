"""Level description, XML save/load, and the hand-built starter levels.

The XML mirrors Science Birds' tag names (Block / Pig with type, material,
x, y, rotation) but coordinates are metres in this world: the slingshot rest
point is at (0, 4.5) and the ground is y = 0. Block (x, y) is the centre.
"""
from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

from .config import BLOCK_TYPES, CIRCLE_BLOCKS, PIG_TYPES, TRIANGLE_BLOCKS


@dataclass
class BlockSpec:
    type: str
    material: str
    x: float
    y: float
    rotation: float = 0.0   # degrees


@dataclass
class PigSpec:
    type: str
    x: float
    y: float


@dataclass
class HillSpec:
    vertices: list[tuple[float, float]]   # convex polygon, static


@dataclass
class Level:
    name: str
    blocks: list[BlockSpec] = field(default_factory=list)
    pigs: list[PigSpec] = field(default_factory=list)
    hills: list[HillSpec] = field(default_factory=list)
    birds: int = 3

    # ------------------------------------------------------------- scoring
    def max_score(self) -> int:
        from .config import BLOCK_POINTS, PIG_POINTS, UNUSED_BIRD_POINTS
        return (PIG_POINTS * len(self.pigs) + BLOCK_POINTS * len(self.blocks)
                + UNUSED_BIRD_POINTS * max(self.birds - 1, 0))

    # ----------------------------------------------------------------- XML
    def to_xml(self) -> str:
        root = ET.Element("Level", name=self.name, units="m")
        birds = ET.SubElement(root, "Birds")
        for _ in range(self.birds):
            ET.SubElement(birds, "Bird", type="BirdRed")
        objs = ET.SubElement(root, "GameObjects")
        for h in self.hills:
            ET.SubElement(objs, "Hill", vertices=" ".join(f"{x:.3f},{y:.3f}" for x, y in h.vertices))
        for b in self.blocks:
            ET.SubElement(objs, "Block", type=b.type, material=b.material,
                          x=f"{b.x:.4f}", y=f"{b.y:.4f}", rotation=f"{b.rotation:g}")
        for p in self.pigs:
            ET.SubElement(objs, "Pig", type=p.type, x=f"{p.x:.4f}", y=f"{p.y:.4f}")
        ET.indent(root)
        return ET.tostring(root, encoding="unicode")

    def save(self, path) -> None:
        Path(path).write_text(self.to_xml() + "\n")

    @classmethod
    def from_xml(cls, text: str) -> "Level":
        root = ET.fromstring(text)
        lvl = cls(name=root.get("name", "level"))
        birds = root.find("Birds")
        lvl.birds = len(birds) if birds is not None else 3
        for el in root.find("GameObjects"):
            if el.tag == "Block":
                lvl.blocks.append(BlockSpec(el.get("type"), el.get("material"),
                                            float(el.get("x")), float(el.get("y")),
                                            float(el.get("rotation", 0))))
            elif el.tag == "Pig":
                lvl.pigs.append(PigSpec(el.get("type"), float(el.get("x")), float(el.get("y"))))
            elif el.tag == "Hill":
                verts = [tuple(map(float, v.split(","))) for v in el.get("vertices").split()]
                lvl.hills.append(HillSpec(verts))
        return lvl

    @classmethod
    def load(cls, path) -> "Level":
        return cls.from_xml(Path(path).read_text())


# ------------------------------------------------------------ build helpers
GAP = 0.002  # small vertical gap so stacked blocks start just touching


def block_extent(type_: str, rotation: float) -> tuple[float, float]:
    """Axis-aligned (width, height) of a block at 0/90 degree rotation."""
    if type_ in BLOCK_TYPES:
        w, h = BLOCK_TYPES[type_]
    elif type_ in CIRCLE_BLOCKS:
        w = h = 2 * CIRCLE_BLOCKS[type_]
    elif type_ in TRIANGLE_BLOCKS:
        w, h = TRIANGLE_BLOCKS[type_]
    else:
        raise KeyError(type_)
    if abs(math.sin(math.radians(rotation))) > 0.5:
        w, h = h, w
    return w, h


class Builder:
    """Stack blocks bottom-up without doing geometry by hand."""

    def __init__(self, name: str, birds: int = 3):
        self.level = Level(name=name, birds=birds)

    def block(self, type_: str, material: str, x: float, bottom: float,
              vertical: bool = False) -> float:
        """Place a block with its bottom edge at `bottom`; returns its top y."""
        rot = 90.0 if vertical else 0.0
        _, h = block_extent(type_, rot)
        y = bottom + GAP + h / 2
        self.level.blocks.append(BlockSpec(type_, material, x, y, rot))
        return y + h / 2

    def pig(self, type_: str, x: float, bottom: float) -> float:
        r = PIG_TYPES[type_].radius
        self.level.pigs.append(PigSpec(type_, x, bottom + GAP + r))
        return bottom + 2 * r

    def frame(self, x: float, bottom: float, pillar: str, beam: str,
              pillar_mat: str, beam_mat: str) -> float:
        """Two pillars with a beam across. Returns the beam's top y."""
        bw, _ = block_extent(beam, 0)
        pw, _ = block_extent(pillar, 90)
        half = bw / 2 - pw / 2 - 0.02
        top = self.block(pillar, pillar_mat, x - half, bottom, vertical=True)
        self.block(pillar, pillar_mat, x + half, bottom, vertical=True)
        return self.block(beam, beam_mat, x, top)

    def hill(self, vertices) -> None:
        self.level.hills.append(HillSpec(list(vertices)))


# ---------------------------------------------------------- starter levels

def level_open_pig() -> Level:
    """L1: one pig on open ground. Closed-form answer exists: sanity check."""
    b = Builder("L1_open_pig")
    b.pig("BasicMedium", 30.0, 0.0)
    return b.level


def level_wood_shelter() -> Level:
    """L2: a pig under a wooden frame, a second pig on the roof."""
    b = Builder("L2_wood_shelter")
    top = b.frame(30.0, 0.0, "RectMedium", "RectBig", "wood", "wood")
    b.pig("BasicSmall", 30.0, 0.0)
    b.pig("BasicMedium", 30.0, top)
    return b.level


def level_reel_tower() -> Level:
    """L3: the tower from the end of the reel. Wood ground floor, ice upper
    floor, stone cap with a pig, a stone side stack with a pig, 4 pigs."""
    b = Builder("L3_reel_tower")
    x = 33.0
    f1 = b.frame(x, 0.0, "RectMedium", "RectBig", "wood", "wood")
    b.pig("BasicMedium", x, 0.0)
    f2 = b.frame(x, f1, "RectSmall", "RectBig", "ice", "wood")
    b.pig("BasicSmall", x, f1)
    cap = b.block("SquareSmall", "stone", x, f2)
    b.pig("BasicSmall", x, cap)
    # side stack of stone, right of the tower
    sx = x + BLOCK_TYPES["RectBig"][0] / 2 + 0.6
    s = b.block("SquareSmall", "stone", sx, 0.0)
    s = b.block("SquareSmall", "stone", sx, s)
    b.pig("BasicMedium", sx, s)
    return b.level


def level_fortress() -> Level:
    """L4: a hill hides a two-room fort. Flat shots hit the hill, so the agent
    must find the high-angle (lob) solution."""
    b = Builder("L4_fortress")
    b.hill([(24.0, 0.0), (30.0, 0.0), (29.0, 4.0), (25.5, 4.0)])
    x1, x2 = 36.0, 39.2
    t1 = b.frame(x1, 0.0, "RectSmall", "RectBig", "stone", "wood")
    b.pig("BasicMedium", x1, 0.0)
    t2 = b.frame(x2, 0.0, "RectSmall", "RectBig", "wood", "ice")
    b.pig("BasicSmall", x2, 0.0)
    b.block("SquareSmall", "wood", x1, t1)
    b.pig("BasicBig", x2, t2)
    return b.level


STARTER_LEVELS = {
    "L1_open_pig": level_open_pig,
    "L2_wood_shelter": level_wood_shelter,
    "L3_reel_tower": level_reel_tower,
    "L4_fortress": level_fortress,
}
