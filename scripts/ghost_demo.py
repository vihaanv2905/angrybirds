#!/usr/bin/env python3
""""Ghost race" demo video: every training run plays the same level many
times at once, overlaid in the MuJoCo scene, and the best attempt is the one
the SO-101 actually fires (gold trail).

For each level:
  1. Each run in --runs ("random" = untrained) plays --samples attempts in
     Box2D with the SO-101's measured launches. Attempt 0 is the policy's mean
     action; the rest sample its Gaussian (learned PPO std, read from the
     SB3 checkpoint) plus the training launch noise, i.e. what PPO tries.
  2. The top --candidates attempts are replayed on the robot (full MuJoCo
     co-simulation, headless); the one scoring most there is the hero.
  3. Render: the hero on the SO-101, every other attempt as a translucent
     ghost bird + trail coloured by run, a live leaderboard per run.

    MUJOCO_GL=egl python scripts/ghost_demo.py --levels L3_reel_tower --out-dir docs/img
Writes <out-dir>/ghosts_<level>.mp4 and ghosts_<level>.json (all scores).
"""
from __future__ import annotations

import argparse
import io
import json
import math
import os
import shutil
import subprocess
import sys
import time
import zipfile
import zlib
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from angrybirds.config import UNUSED_BIRD_POINTS  # noqa: E402
from angrybirds.env import observe  # noqa: E402
from angrybirds.launchers import ArmLauncher  # noqa: E402
from angrybirds.levels import STARTER_LEVELS, Level  # noqa: E402
from angrybirds.policy import NumpyPolicy  # noqa: E402
from angrybirds.world import GameWorld  # noqa: E402

SQ20 = math.sqrt(20.0)              # game time = tabletop time x sqrt(20)
NOISE_ANGLE, NOISE_DRAW = 0.3, 0.01  # training launch noise
GOLD = (1.0, 0.80, 0.10)
DEFAULT_RUNS = ["random", "a1_starter_1bird", "a2_starter_3birds", "a3_mix", "t1_lr1e4", "t2_ent005",
                "t3_lr1e4ent_s1", "t4_lr1e4ent_s2", "f_lr1e4ent_s0", "f_lr1e4ent_s1", "f_lr1e4ent_s2"]
LABELS = {"random": "untrained (random)", "a1_starter_1bird": "a1  1 bird, starters",
          "a2_starter_3birds": "a2  3 birds, starters", "a3_mix": "a3  + random layouts",
          "t1_lr1e4": "t1  lr 1e-4", "t2_ent005": "t2  entropy 0.005",
          "t3_lr1e4ent_s1": "t3  lr+ent, seed 1", "t4_lr1e4ent_s2": "t4  lr+ent, seed 2",
          "f_lr1e4ent_s0": "f   seed 0 (partial)", "f_lr1e4ent_s1": "f   seed 1 (partial)",
          "f_lr1e4ent_s2": "f   seed 2 (partial)"}


# ------------------------------------------------------------------ inputs
def load_level(key: str) -> Level:
    if key.startswith("heldout"):                    # same stream as train/aim_ppo.py --eval-seed 0
        from angrybirds.generator import random_level
        rng = np.random.default_rng(10_000)
        for _ in range(int(key[len("heldout"):]) + 1):
            level = random_level(rng)
        level.name = key
        return level
    return STARTER_LEVELS[key]() if key in STARTER_LEVELS else Level.load(key)


def policy_std(run: str) -> np.ndarray:
    """The PPO policy's learned action std (cached in runs/<run>/policy_std.json)."""
    cache = ROOT / "runs" / run / "policy_std.json"
    ck_root = Path(os.environ.get("AB_CKPT_DIR", Path.home() / "angrybirds_ckpt"))
    cks = [ck_root / run / n for n in ("best_model.zip", "final_model.zip") if (ck_root / run / n).exists()]
    if cache.exists() and (not cks or cache.stat().st_mtime >= cks[0].stat().st_mtime):
        return np.array(json.loads(cache.read_text())["std"])   # still matches the checkpoint
    for p in cks:
        import torch
        with zipfile.ZipFile(p) as z:
            sd = torch.load(io.BytesIO(z.read("policy.pth")), map_location="cpu")
        std = sd["log_std"].exp().numpy().astype(float)
        cache.write_text(json.dumps({"std": std.tolist(), "from": p.name}) + "\n")
        return std
    return np.array([0.1, 0.2])


def run_colors(n_trained: int) -> list[tuple]:
    """Untrained grey, then trained runs from violet (early) to teal (late)."""
    import colorsys
    cols = [(0.62, 0.62, 0.62)]
    for k in range(n_trained):
        h = 0.80 - 0.33 * k / max(n_trained - 1, 1)       # violet -> blue -> teal
        cols.append(colorsys.hsv_to_rgb(h % 1.0, 0.65, 1.0))
    return cols


# ------------------------------------------------------------- Box2D trials
def play_trial(level: Level, policy, std, sample: int, seed: int) -> dict:
    """One full episode in Box2D. Returns plans, per-step bird xy and score."""
    rng = np.random.default_rng(seed)
    launcher = ArmLauncher()
    gw = GameWorld(level)
    shots = []
    while gw.pigs_left > 0 and gw.birds_left > 0:
        if policy is None:
            a = rng.uniform(-1, 1, 2)
        else:
            a = policy.act(observe(gw))
            if sample > 0:
                a = a + std * rng.standard_normal(2)
        angle, draw = launcher.action_to_shot(np.clip(a, -1, 1))
        if sample > 0 or policy is None:
            angle += rng.normal(0, NOISE_ANGLE)
            draw = float(np.clip(draw + rng.normal(0, NOISE_DRAW), launcher.min_draw_frac, 1.0))
        L = launcher.launch(angle, draw)
        gw.begin_shot(L.position, L.velocity)
        xy, sc = [], []
        done = False
        while not done:
            done = gw.step_shot()
            b = gw._bird
            xy.append((b.body.position.x, b.body.position.y) if b is not None else (np.nan, np.nan))
            sc.append(gw.score)
        gw.end_shot()
        if gw.pigs_left == 0:
            gw.score += UNUSED_BIRD_POINTS * gw.birds_left
            sc[-1] = gw.score
        shots.append({"angle": float(angle), "draw": float(draw),
                      "xy": np.asarray(xy, np.float32), "score": np.asarray(sc, np.int64)})
    out = {"shots": shots, "score": int(gw.score), "cleared": gw.pigs_left == 0, "dt": gw.dt}
    gw.close()
    return out


# ------------------------------------------------------------ robot replay
def make_show(level, trial, policy):
    from angrybirds.cosim import ShotPlan
    from angrybirds.show import Show
    plans = [ShotPlan(s["angle"], s["draw"]) for s in trial["shots"]]
    # if the robot's world diverges and needs another bird, the policy aims it
    return Show(level, plans=plans, policy=policy)


def robot_run(level, trial, policy, max_time=60.0) -> dict:
    show = make_show(level, trial, policy)
    sim = show.sim
    rel, end, prev_mode, prev_active = [], [], sim.bird_mode, False
    while sim.d.time < max_time and not show.finished:
        show.tick()
        if sim.bird_mode == "game" and prev_mode != "game":
            rel.append(sim.d.time)
        if prev_active and not sim.game_active:
            end.append(sim.d.time)
        prev_mode, prev_active = sim.bird_mode, sim.game_active
    return {"score": int(sim.gw.score), "cleared": sim.gw.pigs_left == 0, "release": rel, "end": end,
            "shots": len(sim.logs)}


# ----------------------------------------------------------------- render
def schedule(trial, t0, gap):
    """Tabletop start/end time of each of a ghost's shots."""
    t, out = t0, []
    for s in trial["shots"]:
        dur = len(s["xy"]) * trial["dt"] / SQ20
        out.append((t, t + dur))
        t += dur + gap
    return out


def ghost_state(trial, sched, t):
    """(shot index, step index, score) of a ghost at tabletop time t."""
    if t < sched[0][0]:
        return None, 0, 0
    for k, (a, b) in enumerate(sched):
        if t <= b:
            if t < a:                     # between shots
                return None, 0, int(trial["shots"][k - 1]["score"][-1])
            i = min(int((t - a) * SQ20 / trial["dt"]), len(trial["shots"][k]["xy"]) - 1)
            return k, i, int(trial["shots"][k]["score"][i])
    return len(sched) - 1, 10 ** 9, trial["score"]


def add_geom(scn, gtype, size, pos, rgba):
    import mujoco
    if scn.ngeom >= scn.maxgeom:
        return None
    g = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(g, gtype, np.asarray(size, float), np.asarray(pos, float), np.eye(3).ravel(),
                        np.asarray(rgba, np.float32))
    scn.ngeom += 1
    return g


def add_trail(scn, pts, width, rgba):
    import mujoco
    for p, q in zip(pts[:-1], pts[1:]):
        g = add_geom(scn, mujoco.mjtGeom.mjGEOM_CAPSULE, (width, 0, 0), p, rgba)
        if g is None:
            return
        mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, width, np.asarray(p, float), np.asarray(q, float))
        g.rgba[:] = rgba


def to_mj(xy):
    from angrybirds.mj_scene import game_to_mj
    return [game_to_mj(float(x), float(y)) for x, y in xy]


def render_level(args, level, key, runs, colors, trials, hero, info):
    import mujoco
    from PIL import Image, ImageDraw, ImageFont

    hero_trial = trials[hero]
    show = make_show(level, hero_trial, hero_trial["policy"])
    sim = show.sim
    W, H = 1280, 720
    r = mujoco.Renderer(sim.m, H, W, max_geom=30000)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = (1.27, 0.0, 0.27)        # side-on, like the game: sling left, level right
    cam.distance, cam.azimuth, cam.elevation = 1.66, 91, -7
    steps = max(1, round(1.0 / args.fps / args.slowmo / sim.m.opt.timestep))
    t0, gap = info["t0"], info["gap"]
    ghosts = [i for i in range(len(trials)) if i != hero]
    scheds = {i: schedule(trials[i], t0, gap) for i in ghosts}
    t_last = max([max(b for _, b in s) for s in scheds.values()] + [info["hero_end"]])
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 17)
        font_b = ImageFont.truetype("DejaVuSans-Bold.ttf", 22)
        font_big = ImageFont.truetype("DejaVuSans-Bold.ttf", 30)
    except OSError:
        font = font_b = font_big = ImageFont.load_default()
    hero_pts, prev_mode = [], sim.bird_mode
    n_runs = len(runs)
    stride = 4                                  # trail sampled every 4 game steps
    out = Path(args.out_dir) / f"ghosts_{key}.mp4"
    ff = subprocess.Popen([shutil.which("ffmpeg") or "ffmpeg", "-v", "error", "-y", "-f", "rawvideo",
                           "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(args.fps), "-i", "-",
                           "-c:v", "libx264", "-crf", "22", "-preset", "medium", "-pix_fmt", "yuv420p", str(out)],
                          stdin=subprocess.PIPE)
    while sim.d.time < t_last + args.tail:
        for _ in range(steps):
            show.tick()
        sim.sync_mocap()
        t = sim.d.time
        if sim.bird_mode == "game" and prev_mode != "game":
            hero_pts = []
        prev_mode = sim.bird_mode
        b = sim.gw._bird
        if sim.bird_mode == "game" and b is not None:
            hero_pts.append(tuple(b.body.position))
        r.update_scene(sim.d, camera=cam)
        scn = r.scene
        best_live = [0] * n_runs
        cleared = [0] * n_runs
        for i in ghosts:
            tr = trials[i]
            k, idx, sc = ghost_state(tr, scheds[i], t)
            best_live[tr["run"]] = max(best_live[tr["run"]], sc)
            if t > scheds[i][-1][1] and tr["cleared"]:
                cleared[tr["run"]] += 1
            if k is None:
                continue
            a, b_end = scheds[i][k]
            fade = 1.0 if t <= b_end else max(0.0, 1.0 - (t - b_end) / 1.0)
            if fade <= 0:
                continue
            col = colors[tr["run"]]
            xy = tr["shots"][k]["xy"][:idx + 1]
            pts = xy[::stride]
            pts = pts[~np.isnan(pts[:, 0])]
            if len(pts) > 1:
                add_trail(scn, to_mj(pts), 0.0028, (*col, 0.40 * fade))
            if t <= b_end and not np.isnan(xy[-1, 0]):
                add_geom(scn, mujoco.mjtGeom.mjGEOM_SPHERE, (0.015, 0, 0), to_mj(xy[-1:])[0],
                         (*col, 0.45))
        if len(hero_pts) > 1:
            add_trail(scn, to_mj(hero_pts), 0.005, (*GOLD, 1.0))
        hr = trials[hero]["run"]
        best_live[hr] = max(best_live[hr], sim.gw.score)
        if show.finished and sim.gw.pigs_left == 0:
            cleared[hr] += 1
        img = Image.fromarray(r.render())
        dr = ImageDraw.Draw(img, "RGBA")
        # title
        dr.rectangle([0, 0, W, 44], fill=(0, 0, 0, 150))
        dr.text((14, 10), f"{key}   ·   {len(trials)} attempts from {n_runs} training runs, all at once",
                font=font_b, fill=(255, 255, 255))
        # leaderboard, one row per run (chronological), best score so far
        x0, y0, rh = W - 372, 56, 25
        dr.rectangle([x0 - 10, y0 - 8, W - 8, y0 + rh * (n_runs + 1) + 6], fill=(0, 0, 0, 140))
        dr.text((x0, y0 - 2), "run", font=font, fill=(200, 200, 200))
        dr.text((x0 + 214, y0 - 2), "best", font=font, fill=(200, 200, 200))
        dr.text((x0 + 292, y0 - 2), "clears", font=font, fill=(200, 200, 200))
        for j, run in enumerate(runs):
            y = y0 + rh * (j + 1)
            c = tuple(int(255 * v) for v in colors[j])
            if j == hr:
                dr.rectangle([x0 - 6, y - 2, W - 12, y + rh - 4], outline=tuple(int(255 * v) for v in GOLD),
                             width=2)
            dr.ellipse([x0, y + 4, x0 + 12, y + 16], fill=c)
            dr.text((x0 + 20, y), LABELS.get(run, run)[:21], font=font, fill=(255, 255, 255))
            dr.text((x0 + 214, y), f"{best_live[j]:>6,}", font=font, fill=(255, 255, 255))
            dr.text((x0 + 300, y), f"{cleared[j]}/{args.samples}", font=font, fill=(255, 255, 255))
        # hero panel
        dr.rectangle([8, H - 74, 620, H - 8], fill=(0, 0, 0, 140))
        dr.ellipse([18, H - 64, 32, H - 50], fill=tuple(int(255 * v) for v in GOLD))
        dr.text((40, H - 68), f"On the robot: best attempt  ({LABELS.get(runs[hr], runs[hr])}, "
                              f"try #{trials[hero]['sample']})", font=font, fill=(255, 255, 255))
        dr.text((40, H - 42), f"score {sim.gw.score:,}   birds left {sim.gw.birds_left}   "
                              f"pigs left {sim.gw.pigs_left}   {sim.status}", font=font, fill=(255, 255, 255))
        if show.finished and t > t_last:
            msg = (f"BEST: {LABELS.get(runs[hr], runs[hr])}  -  {sim.gw.score:,} pts"
                   + ("  -  LEVEL CLEARED" if sim.gw.pigs_left == 0 else ""))
            tw = dr.textlength(msg, font=font_big)
            dr.rectangle([(W - tw) / 2 - 20, H / 2 - 34, (W + tw) / 2 + 20, H / 2 + 20], fill=(0, 0, 0, 170),
                         outline=tuple(int(255 * v) for v in GOLD), width=3)
            dr.text(((W - tw) / 2, H / 2 - 26), msg, font=font_big, fill=tuple(int(255 * v) for v in GOLD))
        ff.stdin.write(np.asarray(img.convert("RGB")).tobytes())
    ff.stdin.close()
    if ff.wait() != 0:
        raise RuntimeError("ffmpeg failed")
    r.close()
    return out, int(sim.gw.score)


# ------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", nargs="+", default=["L1_open_pig", "L2_wood_shelter", "L3_reel_tower",
                                                     "L4_fortress", "heldout0", "heldout1"])
    ap.add_argument("--runs", nargs="+", default=DEFAULT_RUNS, help="run names under runs/ (or 'random')")
    ap.add_argument("--samples", type=int, default=8, help="attempts per run per level")
    ap.add_argument("--candidates", type=int, default=3, help="top attempts replayed on the robot")
    ap.add_argument("--out-dir", default="docs/img")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--slowmo", type=float, default=2.0)
    ap.add_argument("--tail", type=float, default=2.5, help="tabletop seconds after everything ends")
    ap.add_argument("--no-render", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    runs = [r for r in args.runs if r == "random" or (ROOT / "runs" / r).is_dir()]
    colors = run_colors(len(runs) - 1) if runs[0] == "random" else run_colors(len(runs))[1:]
    pols = {r: (None if r == "random" else NumpyPolicy(ROOT / "runs" / r)) for r in runs}
    stds = {r: (None if r == "random" else policy_std(r)) for r in runs}
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    for key in args.levels:
        tic = time.time()
        level = load_level(key)
        trials = []
        for j, run in enumerate(runs):
            for s in range(args.samples):
                tr = play_trial(level, pols[run], stds[run], s, seed=zlib.crc32(f"{args.seed}/{key}/{run}/{s}".encode()))
                tr.update(run=j, sample=s, policy=pols[run])
                trials.append(tr)
        # ties: fewer birds, then the later (more trained) run
        order = sorted(range(len(trials)),
                       key=lambda i: (-trials[i]["score"], len(trials[i]["shots"]), -trials[i]["run"]))
        print(f"[{key}] {len(trials)} Box2D attempts in {time.time() - tic:.0f}s; top:",
              [(runs[trials[i]['run']], trials[i]['sample'], trials[i]['score']) for i in order[:5]], flush=True)
        best, best_rr = None, None
        for i in order[:args.candidates]:
            rr = robot_run(level, trials[i], trials[i]["policy"])
            print(f"  robot replay {runs[trials[i]['run']]} #{trials[i]['sample']}: Box2D {trials[i]['score']} "
                  f"-> robot {rr['score']}", flush=True)
            if best is None or rr["score"] > best_rr["score"]:
                best, best_rr = i, rr
        gaps = [a - b for a, b in zip(best_rr["release"][1:], best_rr["end"])]
        info = {"t0": best_rr["release"][0] if best_rr["release"] else 1.0,
                "gap": float(np.mean(gaps)) if gaps else 2.5,
                "hero_end": best_rr["end"][-1] if best_rr["end"] else 10.0}
        summary = {"level": key, "hero": {"run": runs[trials[best]["run"]], "sample": trials[best]["sample"],
                                          "box2d_score": trials[best]["score"], "robot_score": best_rr["score"],
                                          "cleared": best_rr["cleared"]},
                   "runs": {run: [t["score"] for t in trials if t["run"] == j] for j, run in enumerate(runs)},
                   "samples": args.samples, "timing": info}
        (Path(args.out_dir) / f"ghosts_{key}.json").write_text(json.dumps(summary, indent=1) + "\n")
        if args.no_render:
            continue
        out, sc = render_level(args, level, key, runs, colors, trials, best, info)
        print(f"[{key}] wrote {out} (robot score {sc}) in {time.time() - tic:.0f}s", flush=True)


if __name__ == "__main__":
    main()
