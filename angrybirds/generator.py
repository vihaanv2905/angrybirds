"""Random level generator for training on unseen pig placements.

Builds 1-3 structures from Science Birds blocks (frame towers of 1-3
storeys, stone/wood stacks, walls), optionally a hill in front, places pigs
inside and on top, then simulates the level with no bird and rejects it
unless it stands still and nothing breaks.
"""
from __future__ import annotations

import numpy as np

from .config import BLOCK_TYPES
from .levels import Builder, Level, block_extent
from .world import GameWorld

MATERIALS = ("wood", "ice", "stone")
MAT_P = (0.5, 0.3, 0.2)
PIGS = ("BasicSmall", "BasicMedium", "BasicBig")
PIG_P = (0.45, 0.4, 0.15)


def _mat(rng) -> str:
    return str(rng.choice(MATERIALS, p=MAT_P))


def _pig(rng, max_r: float | None = None) -> str:
    from .config import PIG_TYPES
    choices = [p for p in PIGS if max_r is None or PIG_TYPES[p].radius <= max_r]
    probs = np.array([PIG_P[PIGS.index(p)] for p in choices])
    return str(rng.choice(choices, p=probs / probs.sum()))


def _tower(b: Builder, rng, x: float, budget: dict) -> None:
    beam = str(rng.choice(["RectMedium", "RectBig"]))
    floor = 0.0
    for story in range(int(rng.integers(1, 4))):
        pillar = str(rng.choice(["RectSmall", "RectMedium"]))
        if budget["blocks"] < 3:
            break
        ph = block_extent(pillar, 90)[1]
        if budget["pigs"] > 0 and rng.random() < 0.7:
            from .config import PIG_TYPES
            t = _pig(rng, max_r=ph / 2 - 0.05)
            if PIG_TYPES[t].radius * 2 < ph - 0.05:
                b.pig(t, x + rng.uniform(-0.3, 0.3), floor)
                budget["pigs"] -= 1
        floor = b.frame(x, floor, pillar, beam, _mat(rng), _mat(rng))
        budget["blocks"] -= 3
    if budget["blocks"] > 0 and rng.random() < 0.5:
        floor = b.block("SquareSmall", _mat(rng), x, floor)
        budget["blocks"] -= 1
    if budget["pigs"] > 0 and rng.random() < 0.6:
        b.pig(_pig(rng), x, floor)
        budget["pigs"] -= 1


def _stack(b: Builder, rng, x: float, budget: dict) -> None:
    typ = str(rng.choice(["SquareSmall", "RectFat", "SquareHole"]))
    top = 0.0
    for _ in range(int(rng.integers(1, 4))):
        if budget["blocks"] <= 0:
            break
        top = b.block(typ, _mat(rng), x, top)
        budget["blocks"] -= 1
    if budget["pigs"] > 0 and rng.random() < 0.8:
        b.pig(_pig(rng), x, top)
        budget["pigs"] -= 1


def _wall(b: Builder, rng, x: float, budget: dict) -> None:
    if budget["blocks"] > 0:
        b.block(str(rng.choice(["RectSmall", "RectMedium"])), _mat(rng), x, 0.0, vertical=True)
        budget["blocks"] -= 1


def _footprint(kind: str) -> float:
    return {"tower": BLOCK_TYPES["RectBig"][0], "stack": BLOCK_TYPES["SquareHole"][0],
            "wall": 0.4}[kind] + 0.8


def random_level(rng: np.random.Generator, max_pigs: int = 6, max_blocks: int = 36,
                 x_range: tuple[float, float] = (22.0, 44.0), hill_prob: float = 0.25,
                 birds: int = 3, validate: bool = True, max_tries: int = 50,
                 name: str | None = None) -> Level:
    for attempt in range(max_tries):
        b = Builder(name or f"gen_{int(rng.integers(1 << 30)):09d}", birds=birds)
        budget = {"pigs": int(rng.integers(1, max_pigs + 1)), "blocks": max_blocks}
        x_lo, x_hi = x_range
        if rng.random() < hill_prob:
            hx = rng.uniform(x_lo - 6, x_lo - 1)
            w, h = rng.uniform(3, 6), rng.uniform(1.5, 4.5)
            b.hill([(hx - w / 2, 0), (hx + w / 2, 0), (hx + w / 2 - 0.8, h), (hx - w / 2 + 0.8, h)])
            x_lo = max(x_lo, hx + w / 2 + 1.0)
        kinds = [str(k) for k in rng.choice(["tower", "stack", "wall"], p=(0.6, 0.3, 0.1),
                                            size=int(rng.integers(1, 4)))]
        gaps = [rng.uniform(0.3, 3.0) for _ in kinds]
        span = sum(_footprint(k) for k in kinds) + sum(gaps[:-1])
        x = x_lo + rng.uniform(0, max(x_hi - x_lo - span, 0.0))   # spread over the range
        for kind, gap in zip(kinds, gaps):
            fp = _footprint(kind)
            x += fp / 2
            if x + fp / 2 > x_hi:
                break
            {"tower": _tower, "stack": _stack, "wall": _wall}[kind](b, rng, x, budget)
            x += fp / 2 + gap
        lvl = b.level
        if not lvl.pigs:                       # always at least one pig
            lvl_x = rng.uniform(x_lo, x_hi)
            b.pig(_pig(rng), lvl_x, 0.0)
        if not validate or is_stable(lvl):
            return lvl
    raise RuntimeError("could not generate a stable level")


def is_stable(level: Level, seconds: float = 3.0, max_move: float = 0.1) -> bool:
    """Reject levels that collapse, break or overlap on their own."""
    rep = GameWorld(level, settle=False).stability_report(seconds)
    if rep["broken"] or rep["max_move"] > max_move:
        return False
    rep2 = GameWorld(level).stability_report(2.0)
    return rep2["broken"] == 0 and rep2["max_move"] < 0.02
