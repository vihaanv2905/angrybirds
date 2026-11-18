#!/usr/bin/env python3
"""System identification: fire the SO-101 in MuJoCo over a grid of (angle,
draw) and save the measured game-scale releases to robot/so101/launch_table.npz.
The shot planner then aims with what the arm really does, not the analytic
model. Rerun after changing the arm controller, bands or pouch.

    python scripts/calibrate_launch.py --workers 2   # ~3-4 min
"""
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from angrybirds.cosim import TABLE_PATH, measure_launch  # noqa: E402
from angrybirds.tabletop import SQRT_LAMBDA, TabletopSlingshot  # noqa: E402


def job(args):
    a, d = args
    return measure_launch(a, d)


def args_workers() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    return ap.parse_args().workers


def main():
    angles = np.arange(-10.0, 80.1, 5.0)
    draws = np.round(np.arange(0.3, 1.001, 0.1), 3)
    grid = [(float(a), float(d)) for a in angles for d in draws]
    t = time.time()
    # each MuJoCo scene is ~400 MB: few workers, recycled often
    with ProcessPoolExecutor(max_workers=args_workers(), max_tasks_per_child=8) as ex:
        res = list(ex.map(job, grid, chunksize=4))
    data = np.array([[*p, *v] for p, v in res]).reshape(len(angles), len(draws), 4)
    np.savez(TABLE_PATH, angles=angles, draws=draws, data=data)
    s = TabletopSlingshot()
    errs = []
    for (a, d), (p, v) in zip(grid, res):
        if d < 0.5:                      # below ~0.42 the bands are still slack
            continue
        L = s.launch(a, d)
        errs.append(np.hypot(*v) / (L.speed / SQRT_LAMBDA) - 1)
    print(f"{len(grid)} launches in {time.time() - t:.0f}s -> {TABLE_PATH}")
    print(f"measured vs analytic speed (draw >= 0.5): mean {np.mean(errs)*100:+.1f}%, "
          f"range {np.min(errs)*100:+.1f}% .. {np.max(errs)*100:+.1f}%")
    sp = np.hypot(data[..., 2], data[..., 3])
    print(f"game launch speed range {sp.min():.1f}-{sp.max():.1f} m/s")


if __name__ == "__main__":
    main()
