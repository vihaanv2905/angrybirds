# Project notes for Claude (read first)

Goal: train an **SO-101 robot arm** to play Angry Birds with reinforcement
learning: choose the launch angle and power (and later the arm motion itself)
to maximise the Angry Birds score for a given layout of pigs and blocks.
Owner trains on the **UMich Great Lakes** cluster. Owner's choices so far:
**PPO** (SAC only as a fallback), **SO-101 from MuJoCo Menagerie**, and
watching runs **live in the MuJoCo viewer**, not in matplotlib.

## Architecture (decided, with reasons)

- **Game = Box2D** (`angrybirds/world.py`), 2-D, 120 Hz. Box2D is the engine
  the real game uses; it does breakable blocks and tall stacks natively.
  Damage = contact impulse above a threshold (blocks) or delta-v (pigs).
  Classic scoring: pig 5000, block 500, unused bird 10000.
- **Robot = MuJoCo** (`robot/so101/angrybirds_scene.xml`, `angrybirds/cosim.py`).
- The only link between them is the bird's **release state** (x, y, vx, vy).
  The arm side hands it to `GameWorld.begin_shot`.
- **1:20 tabletop scale with Froude scaling.** The reel's slingshot needs
  ~500 N; the SO-101 (STS3215, 2.94 N*m) holds ~10 N. So the robot scene is
  1:20 (bird 3 cm / 20 g, bands 3.7 mm^2, ~6 N at full draw). Tabletop ->
  game: lengths x20, velocities x sqrt(20), times x sqrt(20)
  (`angrybirds/tabletop.py`, `LAMBDA = 1/20`). Policies and the planner work
  in **game units**; the robot works in **tabletop units**.
- Pouch rest point: game (0, 4.5) = MuJoCo (0.39, 0, 0.225). Robot base at
  the MuJoCo origin, facing +x.
- **Band model:** Ogden 3-term fit to Treloar's rubber, hysteresis 0.9, two
  spatial tendons driven by `general` actuators with ctrl = -tension. Band
  forces must be set between `mj_step1` and `mj_step2`; a one-step lag pumps
  energy in and blows the pouch up.
- **Grip:** `connect` (pivot) constraint, not a weld. The SO-101 cannot point
  its gripper steeply up at the low pull points, so the pouch pivots and the
  bands align it (band anchors sit 16 mm ahead of the pouch centre).
  Weld/connect anchors must sit on the welded body (an anchor at the world
  origin lets a soft weld sag).
- **Bird stays nocked** (weld to pouch) until both bands are slack; then it
  flies free. A soft pouch-bird contact loses energy.
- **System identification:** `scripts/calibrate_launch.py` fires the arm over
  an (angle, draw) grid and stores the measured releases in
  `robot/so101/launch_table.npz`; the planner aims with it. Rerun it after
  any change to the arm, bands or pouch.
- The level in the MuJoCo viewer is drawn as mocap bodies mirrored from Box2D;
  MuJoCo does not simulate the blocks (by design).

## Status

- Part 1 (environment) done: `angrybirds/env.py` = Gymnasium Stage-A env
  (action = (aim, power) in [-1,1]^2). `scripts/sweep_level.py` gives the
  ground-truth best shot per level.
- SO-101 co-simulation done with a **scripted** IK controller
  (`cosim.Controller`). It clears L3 with 2 birds (33,000 pts).
- Stage A trainer: `train/aim_ppo.py` + `slurm/aim_ppo.sbatch` (account
  `engin1` (shared CoE account; jobs can queue on its group CPU limit), partition `standard`). Default `--launcher arm`: policies output
  SO-101 (angle, draw) and can be watched on the robot with
  `scripts/live.py --policy runs/<name>`.
- **Next:** read the owner's first Great Lakes runs in `runs/`, tune Stage A,
  then build Stage B/C (π_arm in MuJoCo Playground on `spgpu`). See
  `docs/PART2_GREATLAKES.md`.

## Conventions

- Run `pytest -q` (26 tests) before committing. Physics changes need a
  stability/energy check, not just passing tests.
- Each MuJoCo `CoSim` is ~400 MB. Use <= 2 parallel workers on small machines.
- Training runs write a small summary to `runs/<run_name>/` (config.yaml,
  metrics.csv, eval.json, baseline_random.json, policy.npz ~0.7 MB). Commit
  those; full SB3 checkpoints stay in scratch (`$AB_CKPT_DIR`).
- To watch a run on the robot, record it offscreen:
  `MUJOCO_GL=egl python scripts/record_cosim.py --policy runs/<name>`.
- Work on branches; open pull requests into `main`.
