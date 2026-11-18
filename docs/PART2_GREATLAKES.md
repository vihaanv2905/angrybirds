# Part 2: training on Great Lakes

Two policies, trained separately, then joined:

| Policy | Learns | Simulator | Great Lakes partition | Algorithm |
|---|---|---|---|---|
| **π_aim** | level layout → target launch (angle, power) | Box2D game (CPU) | `standard` (36-core nodes) | PPO (Stable-Baselines3) |
| **π_arm** | target launch → SO-101 joint commands that pull, hold and release so the bird leaves with that velocity | MuJoCo Playground (GPU, MJX or MuJoCo Warp) | `spgpu` (A40 48 GB) | PPO (Brax or RSL-RL) |

```
level ──► π_aim ──► target (θ, v) ──► π_arm ──► SO-101 in MuJoCo ──► release (x, y, vx, vy) ──► Box2D ──► score
```

The game runs on CPU only, so a GPU would not speed up π_aim. π_arm needs
thousands of parallel arm simulations, which is what Playground on a GPU is for.

## 0. Access (once)

1. Slurm account: **`vihaanv`** (already filled into every `slurm/*.sbatch`).
   Check it with `sacctmgr show assoc user=$USER format=account%20,partition%20`.
2. Log in: `ssh <uniqname>@greatlakes.arc-ts.umich.edu` (UMich network or VPN,
   Duo). Open OnDemand (https://greatlakes.arc-ts.umich.edu) gives a browser
   terminal, file browser and remote desktop.
3. Keep the repo in scratch (home is small):
   `/scratch/vihaanv_root/vihaanv/<uniqname>/`. Run `ls /scratch` to confirm
   the path for your account; `slurm/common.sh` falls back to `$HOME` if it
   does not exist.

## 1. Setup (once, on the login node)

```bash
cd /scratch/vihaanv_root/vihaanv/$USER
git clone https://github.com/vihaanv2905/angrybirds && cd angrybirds
git checkout part2-aim-training            # until it is merged into main
bash slurm/setup_greatlakes.sh             # venv + requirements (~5 min)
sbatch slurm/test.sbatch                   # tests + a 3,000-shot training run
squeue -u $USER                            # watch it; output in logs/
```

If no Python module loads, run `module avail python` and pin one:
`export AB_PYTHON_MODULE=python/3.11.5` (add it to `~/.bashrc`).

## 2. Stage A: π_aim on CPU

```bash
sbatch slurm/aim_ppo.sbatch a1_starter_1bird  --birds 1 --total-steps 300000
sbatch slurm/aim_ppo.sbatch a2_starter_3birds --birds 3 --total-steps 1000000
sbatch --time=08:00:00 slurm/aim_ppo.sbatch a3_mix --level-mix 0.8 --total-steps 5000000
```

- `--launcher arm` (default): the shot is (angle, draw) for the SO-101, and the
  launch comes from its measured table. So the trained policy plays on the
  robot directly: `python scripts/live.py --level L3_reel_tower --policy runs/a2_starter_3birds`.
  `--launcher reel` trains on the full-size analytic slingshot instead.
- Each evaluation plays the 4 starter levels (and, with `--level-mix`, 50
  held-out random layouts never seen in training) with the deterministic
  policy, logging to `runs/<name>/metrics.csv`. With `--birds 1`,
  `frac_best_*` compares against the best shot from trying every shot
  (`levels/sweep_best_<launcher>.json`).
- Launch noise (0.3°, 1 % draw) is on by default so the policy cannot rely on
  knife-edge shots; a near-miss penalty helps early learning and fades to 0
  over the first 30 % of training.
- Throughput: about 85 shots/s per core, so ~3,000/s on a 36-core node.
  1M shots ≈ 6 min of simulation plus PPO updates.
- Checkpoints and TensorBoard logs go to scratch (`$AB_CKPT_DIR`). View them
  with `tensorboard --logdir $AB_CKPT_DIR` in an OnDemand session.

## 3. Stage B/C: π_arm in MuJoCo Playground on GPU

- Port the arm scene to a Playground `MjxEnv`. The band force is computed in
  JAX from the site positions; arm collision geometry is dropped (it never
  touches anything); the grip and nock are equality constraints toggled per
  phase.
- Observation: joint positions and velocities, pouch position and velocity,
  target (θ, v) in game units, phase clock.
- Action at 50 Hz: joint position targets (deltas) for 4 joints plus a
  release logit.
- Reward: −‖release velocity − target‖ (game units) at release, plus small
  penalties for torque near the 2.94 N·m limit, joint limits and time.
- Randomise servo gain/latency, band area ±10 %, bird mass ±10 % so it
  transfers.
- Files to add: `train/arm_env_mjx.py`, `train/arm_ppo.py`,
  `slurm/arm_ppo.sbatch`.

## 4. Join and watch

1. Load π_arm in `cosim.py` (an `ArmPolicyController` replacing the scripted
   `Controller`).
2. Rerun `scripts/calibrate_launch.py` with the learned arm.
3. Fine-tune π_aim with that table.
4. `python scripts/live.py --level L3_reel_tower` on your laptop, or in an
   OnDemand desktop on Great Lakes, to watch the trained arm play.

## Working loop with Claude

Claude cannot log in to Great Lakes from its cloud session (Duo). The loop:

1. Claude writes or changes training code and pushes it to a branch.
2. On Great Lakes: `git pull && sbatch slurm/<job>.sbatch <args>`.
3. The job writes `runs/<run_name>/{config.yaml, metrics.csv, eval.json, eval.mp4}`.
4. `git add runs/<run_name> && git commit -m "run <name>" && git push`, or
   paste the end of `logs/<job>-<id>.out` into the chat.
5. Claude reads the results, diagnoses, and proposes the next change.
