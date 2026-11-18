#!/usr/bin/env python3
"""Brute-force every (angle, power) shot on a level: ground truth for RL.

The Stage-A action is only 2-D, so we can try every shot on a grid and know
the true best first shot. Use the resulting best score to measure how close a
trained policy gets ("regret"). Parallel over CPU cores; on Great Lakes use
`--workers $SLURM_CPUS_PER_TASK`.

    python scripts/sweep_level.py --level L3_reel_tower --angles 181 --draws 80
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from angrybirds.config import SimConfig  # noqa: E402
from angrybirds.levels import STARTER_LEVELS, Level  # noqa: E402
from angrybirds.slingshot import Slingshot  # noqa: E402
from angrybirds.world import GameWorld  # noqa: E402

_G = {}


def _init(level_xml: str, drag: bool):
    lvl = Level.from_xml(level_xml)
    cfg = SimConfig(drag=drag)
    _G.update(level=lvl, cfg=cfg, poses=GameWorld.settled_poses(lvl, cfg), sling=Slingshot())


def _shot(args):
    i, j, angle, draw = args
    L = _G["sling"].launch(angle, draw)
    gw = GameWorld(_G["level"], _G["cfg"], poses=_G["poses"])
    r = gw.shoot(L.position, L.velocity)
    return i, j, r.score, r.pigs_killed, r.min_pig_distance


def sweep(level: Level, n_angles=181, n_draws=80, workers=None, drag=False):
    s = Slingshot()
    angles = np.linspace(s.min_angle, s.max_angle, n_angles)
    draws = np.linspace(s.min_draw_frac, 1.0, n_draws)
    jobs = [(i, j, a, d) for i, a in enumerate(angles) for j, d in enumerate(draws)]
    score = np.zeros((n_angles, n_draws))
    pigs = np.zeros((n_angles, n_draws), dtype=int)
    mind = np.zeros((n_angles, n_draws))
    with Pool(workers or os.cpu_count(), initializer=_init, initargs=(level.to_xml(), drag)) as pool:
        for i, j, sc, pk, md in pool.imap_unordered(_shot, jobs, chunksize=64):
            score[i, j], pigs[i, j], mind[i, j] = sc, pk, md
    return angles, draws, score, pigs, mind


def plot(level, angles, draws, score, pigs, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    ext = (draws[0] * 100, draws[-1] * 100, angles[0], angles[-1])
    im = axes[0].imshow(score, origin="lower", aspect="auto", extent=ext, cmap="viridis")
    fig.colorbar(im, ax=axes[0], label="first-shot score")
    i, j = np.unravel_index(np.argmax(score), score.shape)
    for ax in axes:
        ax.plot(draws[j] * 100, angles[i], "r*", ms=14, mec="w")
        ax.set_xlabel("draw (% of 0.67 m)")
        ax.set_ylabel("launch angle (deg)")
    axes[0].set_title(f"{level.name}: best {score[i, j]:.0f} at {angles[i]:.1f} deg, {draws[j]*100:.0f}% draw")
    im2 = axes[1].imshow(pigs, origin="lower", aspect="auto", extent=ext, cmap="magma",
                         vmin=0, vmax=max(len(level.pigs), 1))
    fig.colorbar(im2, ax=axes[1], label="pigs killed")
    axes[1].set_title("pigs killed by the first shot")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--level", default="L3_reel_tower", help="starter level name or path to .xml")
    ap.add_argument("--angles", type=int, default=181)
    ap.add_argument("--draws", type=int, default=80)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--drag", action="store_true")
    ap.add_argument("--out", default="results/sweeps")
    args = ap.parse_args()
    level = STARTER_LEVELS[args.level]() if args.level in STARTER_LEVELS else Level.load(args.level)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t = time.time()
    angles, draws, score, pigs, mind = sweep(level, args.angles, args.draws, args.workers, args.drag)
    n = score.size
    dt = time.time() - t
    tag = level.name + ("_drag" if args.drag else "")
    np.savez(out / f"{tag}.npz", angles=angles, draws=draws, score=score, pigs=pigs, min_pig_distance=mind)
    plot(level, angles, draws, score, pigs, out / f"{tag}.png")
    i, j = np.unravel_index(np.argmax(score), score.shape)
    print(f"{tag}: {n} shots in {dt:.1f}s ({n/dt:.0f}/s). best {score[i, j]:.0f} "
          f"at angle {angles[i]:.1f} deg, draw {draws[j]:.3f}; "
          f"{(score > 0).mean()*100:.1f}% of shots score, "
          f"{(score >= 0.9 * score.max()).mean()*100:.2f}% within 10% of best")


if __name__ == "__main__":
    main()
