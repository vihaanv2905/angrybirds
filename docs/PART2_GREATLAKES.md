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

1. You need a Slurm **account** to charge jobs to: your PI's (or a course or
   UMRCP allocation). Ask your advisor for the account name; below it is
   `<acct>`.
2. Log in: `ssh <uniqname>@greatlakes.arc-ts.umich.edu` (UMich VPN or campus
   network; Duo 2FA). The Open OnDemand web portal gives a browser terminal,
   file browser and a remote desktop (useful for the live viewer).
3. Storage: home is small. Put venvs and checkpoints in scratch:
   `/scratch/<acct>_root/<acct>/<uniqname>/`.

Module names and versions change; check with `module avail python` and adjust
below.

## 1. Environment (once, on a login node: light work only)

```bash
cd /scratch/<acct>_root/<acct>/$USER
git clone https://github.com/vihaanv2905/angrybirds && cd angrybirds
module load python/3.11            # or what `module avail python` lists
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install "stable-baselines3[extra]" tensorboard pyyaml
pytest -q                           # 22 tests should pass
```

GPU stack (π_arm), checked from a GPU job, not the login node:

```bash
pip install "jax[cuda12]" playground
srun --account=<acct> --partition=spgpu --gpus=1 --time=00:10:00 --pty \
     python -c "import jax; print(jax.devices())"     # expect [CudaDevice(id=0)]
```

## 2. Stage A: π_aim on CPU (first, quick)

- Env: `AngryBirdsEnv(levels=[...], launch_noise=(0.3, 0.01))`. Later
  `generator=random_level` for unseen layouts.
- 32–36 `SubprocVecEnv` workers on one `standard` node, ~3,000 shots/s.
  A few million shots fit in well under an hour.
- Measure against `scripts/sweep_level.py` (true best shot) and CMA-ES.
- Then aim with the arm's measured launch table instead of the analytic
  slingshot, so π_aim only asks for shots the arm can make.
- Files to add: `train/aim_ppo.py`, `slurm/aim_ppo.sbatch`.

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
