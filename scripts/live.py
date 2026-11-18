#!/usr/bin/env python3
"""Watch the SO-101 play Angry Birds live in the MuJoCo viewer.

    python scripts/live.py --level L3_reel_tower        # Linux / Windows
    mjpython scripts/live.py --level L3_reel_tower      # macOS (needs mjpython)

Options:
    --level NAME|path.xml   starter level or a level XML (default L3_reel_tower)
    --shot ANGLE,DRAW       fire this shot instead of planning (repeatable)
    --policy runs/NAME      aim with a trained π_aim (train/aim_ppo.py, --launcher arm)
    --slowmo X              play X times slower than real time (default 1)
    --workers N             planner processes (default: all cores)

In the viewer: Space pauses; Tab / Shift+Tab show the side panels, where
"Rendering > Camera" switches between free / overview / closeup / tower /
follow_bird (or press [ and ] to cycle cameras); drag to orbit, scroll to zoom. Everything is at 1:20 scale and real time, so the game part is
fast: try --slowmo 3.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import mujoco  # noqa: E402
import mujoco.viewer  # noqa: E402

from angrybirds.cosim import ShotPlan  # noqa: E402
from angrybirds.levels import STARTER_LEVELS, Level  # noqa: E402
from angrybirds.show import Show  # noqa: E402

FPS = 60


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--level", default="L3_reel_tower")
    ap.add_argument("--policy", default=None, help="trained π_aim: runs/<name> or a policy .npz")
    ap.add_argument("--shot", action="append", default=[], help="ANGLE,DRAW e.g. 45,0.9")
    ap.add_argument("--slowmo", type=float, default=1.0)
    ap.add_argument("--workers", type=int, default=None)
    return ap.parse_args()


def load_level(name: str) -> Level:
    return STARTER_LEVELS[name]() if name in STARTER_LEVELS else Level.load(name)


def main():
    args = parse_args()
    plans = [ShotPlan(*map(float, s.split(","))) for s in args.shot]
    policy = None
    if args.policy:
        from angrybirds.policy import NumpyPolicy
        policy = NumpyPolicy(args.policy)
        print("aiming with", policy.path)
    show = Show(load_level(args.level), plans=plans, workers=args.workers, policy=policy)
    sim = show.sim
    paused = {"on": False}

    def key(keycode):
        if keycode == 32:          # space
            paused["on"] = not paused["on"]

    steps_per_frame = max(1, round(1.0 / FPS / sim.m.opt.timestep))
    last = None
    with mujoco.viewer.launch_passive(sim.m, sim.d, key_callback=key,
                                      show_left_ui=False, show_right_ui=False) as viewer:
        viewer.sync()
        with viewer.lock():          # after the first frame, or the viewer resets it
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            viewer.cam.lookat[:] = (0.95, 0.0, 0.12)
            viewer.cam.distance = 1.9
            viewer.cam.azimuth = 105
            viewer.cam.elevation = -12
        while viewer.is_running():
            t0 = time.perf_counter()
            if not paused["on"]:
                with viewer.lock():
                    for _ in range(steps_per_frame):
                        show.tick()
                    sim.sync_mocap()
            if sim.status != last:          # log phase changes to the terminal
                last = sim.status
                print(f"[t={sim.d.time:6.2f}s] {last}  score={sim.gw.score}", flush=True)
            left, right = show.hud()
            viewer.set_texts((mujoco.mjtFontScale.mjFONTSCALE_150, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                              left, right))
            viewer.sync()
            dt = steps_per_frame * sim.m.opt.timestep * args.slowmo - (time.perf_counter() - t0)
            if dt > 0:
                time.sleep(dt)
    for i, lg in enumerate(sim.logs, 1):
        print(f"bird {i}: plan {lg.plan.angle_deg:.1f} deg / {lg.plan.draw_frac:.2f} -> "
              f"{lg.score} points, {lg.pigs_killed} pigs")
    print("final score", sim.gw.score)


if __name__ == "__main__":
    main()
