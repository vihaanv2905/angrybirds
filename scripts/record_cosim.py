#!/usr/bin/env python3
"""Render the live co-simulation offscreen to an MP4 (no display needed;
works on a cluster node with MUJOCO_GL=egl). Same physics as scripts/live.py.

    MUJOCO_GL=egl python scripts/record_cosim.py --level L3_reel_tower --out docs/img/so101_L3.mp4
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from angrybirds.cosim import ShotPlan  # noqa: E402
from angrybirds.levels import STARTER_LEVELS, Level  # noqa: E402
from angrybirds.show import Show  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", default="L3_reel_tower")
    ap.add_argument("--shot", action="append", default=[])
    ap.add_argument("--out", default="docs/img/so101_live.mp4")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--slowmo", type=float, default=2.0, help="video plays this much slower than real time")
    ap.add_argument("--max-time", type=float, default=40.0, help="sim seconds cap")
    args = ap.parse_args()
    level = STARTER_LEVELS[args.level]() if args.level in STARTER_LEVELS else Level.load(args.level)
    show = Show(level, plans=[ShotPlan(*map(float, s.split(","))) for s in args.shot])
    sim = show.sim
    W, H = 960, 540
    r_main = mujoco.Renderer(sim.m, H, W)
    r_pip = mujoco.Renderer(sim.m, H // 2, W // 2)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = (0.95, 0.0, 0.12)
    cam.distance, cam.azimuth, cam.elevation = 1.9, 105, -12
    steps = max(1, round(1.0 / args.fps / args.slowmo / sim.m.opt.timestep))
    tail = None
    with tempfile.TemporaryDirectory() as tmp:
        i = 0
        while sim.d.time < args.max_time:
            for _ in range(steps):
                show.tick()
            sim.sync_mocap()
            r_main.update_scene(sim.d, camera=cam)
            img = r_main.render().copy()
            r_pip.update_scene(sim.d, camera="closeup")
            pip = r_pip.render()
            img[8:8 + H // 2 // 1, W - W // 2 - 8:W - 8] = pip      # picture-in-picture
            left, right = show.hud()
            from PIL import Image
            Image.fromarray(_hud(img, left, right)).save(f"{tmp}/f_{i:05d}.png")
            i += 1
            if show.finished:
                tail = i if tail is None else tail
                if i - tail > args.fps * 2:
                    break
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run([shutil.which("ffmpeg") or "ffmpeg", "-v", "error", "-y", "-framerate", str(args.fps),
                        "-i", f"{tmp}/f_%05d.png", "-pix_fmt", "yuv420p", str(out)], check=True)
    for k, lg in enumerate(sim.logs, 1):
        print(f"bird {k}: plan {lg.plan.angle_deg:.1f} deg / {lg.plan.draw_frac:.2f} -> {lg.score} pts, "
              f"{lg.pigs_killed} pigs; launch {np.hypot(*lg.actual_game_v):.1f} m/s (game)")
    print("final score", sim.gw.score, "->", args.out)


def _hud(img, left, right):
    """Draw the score panel onto the frame (PIL: fast)."""
    from PIL import Image, ImageDraw
    im = Image.fromarray(img)
    dr = ImageDraw.Draw(im, "RGBA")
    lines_l, lines_r = left.split("\n"), right.split("\n")
    dr.rectangle([4, 4, 340, 10 + 16 * len(lines_l)], fill=(0, 0, 0, 115))
    for k, (a, b) in enumerate(zip(lines_l, lines_r)):
        dr.text((10, 8 + 16 * k), a, fill=(255, 255, 255))
        dr.text((130, 8 + 16 * k), b, fill=(255, 255, 255))
    return np.asarray(im)


if __name__ == "__main__":
    main()
