#!/usr/bin/env python3
"""Write level XMLs, level previews, generated-level samples, and a video of
the best shot on each starter level (from results/sweeps/*.npz if present).

    python scripts/make_figures.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from angrybirds import render  # noqa: E402
from angrybirds.generator import random_level  # noqa: E402
from angrybirds.levels import STARTER_LEVELS  # noqa: E402
from angrybirds.slingshot import Slingshot  # noqa: E402
from angrybirds.world import GameWorld  # noqa: E402

IMG = ROOT / "docs" / "img"
IMG.mkdir(parents=True, exist_ok=True)
(ROOT / "levels").mkdir(exist_ok=True)

# level files + previews
fig, axes = plt.subplots(len(STARTER_LEVELS), 1, figsize=(12, 3.2 * len(STARTER_LEVELS)))
for ax, (name, fn) in zip(axes, STARTER_LEVELS.items()):
    lvl = fn()
    lvl.save(ROOT / "levels" / f"{name}.xml")
    render.draw(GameWorld(lvl), ax=ax, xlim=(-2, 46), ylim=(-0.3, 9),
                title=f"{name}: {len(lvl.pigs)} pigs, {len(lvl.blocks)} blocks, max score {lvl.max_score()}")
fig.tight_layout()
fig.savefig(IMG / "starter_levels.png", dpi=90)
plt.close(fig)

# generated samples
rng = np.random.default_rng(7)
fig, axes = plt.subplots(4, 2, figsize=(16, 10))
for ax in axes.flat:
    lvl = random_level(rng)
    render.draw(GameWorld(lvl), ax=ax, xlim=(-2, 48), ylim=(-0.3, 10),
                title=f"{lvl.name}: {len(lvl.pigs)} pigs, {len(lvl.blocks)} blocks")
fig.tight_layout()
fig.savefig(IMG / "generated_levels.png", dpi=80)
plt.close(fig)

# best-shot videos
s = Slingshot()
for name, fn in STARTER_LEVELS.items():
    npz = ROOT / "results" / "sweeps" / f"{name}.npz"
    if not npz.exists():
        continue
    d = np.load(npz)
    i, j = np.unravel_index(np.argmax(d["score"]), d["score"].shape)
    angle, draw = float(d["angles"][i]), float(d["draws"][j])
    lvl = fn()
    poses = GameWorld.settled_poses(lvl)
    res = render.save_shot_video(lambda: GameWorld(lvl, poses=poses), s.launch(angle, draw),
                                 IMG / f"best_{name}.mp4",
                                 title=f"{name}  best shot: {angle:.1f} deg, {draw*100:.0f}% draw")
    print(f"{name}: best {angle:.1f} deg / {draw:.3f} -> {res.score} points")
print("wrote", IMG)
