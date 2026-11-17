# Angry Birds RL: a robot arm that learns to shoot

A physics-faithful Angry Birds environment (Box2D, the engine the original
game uses) driven by the slingshot from the "Angry Birds, with real physics"
reel: 4 latex tubes modelled with an Ogden fit to Treloar's rubber data,
212 J stored, ~21 m/s launch. Part 1 (this repo, done) is the game
environment. Part 2 trains a robot arm in MuJoCo Playground to fire it.

![starter levels](docs/img/starter_levels.png)

## How it all runs together

```
                ROBOT SIDE (Part 2: MuJoCo Playground, GPU)        GAME SIDE (Part 1: Box2D, CPU)
 level obs ──►  π_aim(level) → target shot (angle, speed)
                π_arm(joints, target) → joint torques @ 50 Hz
                grip pouch → pull back → hold → release           bird flies (gravity [+ drag])
                Ogden bands accelerate pouch + bird        ────►  hits blocks → contact impulses → damage
                                                 release x,y,vx,vy blocks/pigs break → Angry Birds score ──► reward
```

The only thing the arm hands to the game is the bird's release position and
velocity (`angrybirds.slingshot.Launch`). The two simulators can use whatever
engine and hardware suits them:

1. **Reset:** load a level (starter or generated), settle it, freeze it asleep.
2. **Aim:** a policy picks a target (angle, power).
3. **Fire:** in Stage A the slingshot model turns that straight into a
   `Launch`. In Part 2 the MuJoCo arm pulls, holds and releases, and its
   measured release state becomes the `Launch`.
4. **Simulate:** `GameWorld.shoot()` runs until everything is at rest. Contact
   impulses become damage, and broken blocks and pigs score points.
5. **Reward:** the shot's points. With 3 birds, the next shot sees the damaged
   level.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # if box2d fails to build: pip install swig first
pytest -q                                # 17 tests: physics, stability, damage, scoring, API
python scripts/sweep_level.py --level L3_reel_tower   # try every shot, write heatmap
python scripts/make_figures.py           # level previews + best-shot videos in docs/img
```

```python
from angrybirds.env import AngryBirdsEnv
from angrybirds.generator import random_level

env = AngryBirdsEnv(levels=["L3_reel_tower"], launch_noise=(0.3, 0.01))  # Stage A
# env = AngryBirdsEnv(generator=random_level)                            # unseen layouts
obs, info = env.reset(seed=0)
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
```

## The environment

| Piece | Choice | Why |
|---|---|---|
| Engine | Box2D 2.3 (pybox2d), 2-D, 120 Hz, 10/8 solver iterations | What Angry Birds uses; stable block stacking |
| Scale | Science Birds geometry × 1.4 m/unit; slingshot pouch at (0, 4.5 m); structures 22–44 m away | With the reel's ~21 m/s launch this gives the game's proportions: full power at 45° just overshoots the level |
| Bird | Red bird, r = 0.30 m, 0.75 kg (the reel's mass), bullet (continuous collision) | Masses can be scaled uniformly without changing the dynamics, so the reel's hardware fits unchanged |
| Blocks | 8 Science Birds shapes + circle + triangle; wood / ice / stone with density, friction, restitution, hit points | Same building set as the AIBIRDS research clone |
| Pigs | Small / medium / big circles | |
| Flight | Projectile (gravity only) by default; `SimConfig(drag=True)` adds quadratic drag with the reel's ballistic coefficient | Matches Angry Birds; drag shortens range ~13 % |
| Slingshot | `Slingshot.launch(angle, draw)`: Ogden bands, hysteresis 0.9, effective mass incl. ⅓ band mass, gravity along the draw | Reproduces the reel: 212 J, 21.0 m/s, 78 % efficiency, 498 N hold, ~53 g peak |
| Scoring | Pig 5,000 · destroyed block 500 · each unused bird 10,000 on clearing | Classic Angry Birds values |
| RL interface | Gymnasium; action (aim, power) ∈ [-1,1]² → angle −10…80°, draw 5–100 %; obs = birds left + up to 8 pigs × 4 + 40 blocks × 11 features | Works with Stable-Baselines3 / any Gymnasium agent |

**Damage model.** Box2D's `PostSolve` gives each contact's normal impulse.
Blocks lose `impulse − threshold·dt` hit points (hit points scale with block
size). Pigs lose the same divided by their mass (a velocity change in m/s),
so a big pig does not die from landing on its own weight. Calibrated against
the game's feel:

| Scenario | Result |
|---|---|
| Bird at full speed (21 m/s) into a wood pillar | breaks |
| … into ice at 10 m/s | breaks |
| … into stone at 21 m/s | survives with ~50 % hit points (2–3 hits) |
| Ice plank falls 2.5 m | shatters; wood and stone survive |
| Small/medium pig falls 1 m / 2 m | survives / dies |
| Plank drops 0.5 m onto a pig | pig dies |

**Verified.** Projectile range is within 0.21 % of the closed form. All starter
levels stand still for 5 s (< 2 cm drift, no damage). Same shot gives the same
result every time. Gymnasium `check_env` passes. Generated levels are rejected
unless they stand still. Speed: ~85 shots/s per CPU core on the starter
levels, so a 36-core Great Lakes node does ~3,000 shots/s.

## Ground truth: every shot on every level

The action is 2-D, so `scripts/sweep_level.py` tries all of them (181 angles ×
80 draws = 14,480 shots per level, ~1 min on 4 cores). This is the yardstick
for RL: a policy's "regret" is how far it falls short of the best shot.

| Level | Best first shot | Score | Shots that score | Within 10 % of best |
|---|---|---|---|---|
| L1 open pig | many ties (incl. a ground-skip at −10°) | 5,000 | 13.3 % | 13.3 % |
| L2 wood shelter | 15.0°, 99 % draw | 11,500 | 19.3 % | 7.1 % |
| L3 reel tower | 65.5°, 94 % draw (lob) | 24,000 (all 4 pigs) | 11.2 % | 2.7 % |
| L4 fortress | 23.5°, 94 % draw | 17,000 | 9.6 % | 0.08 % |

![L3 sweep](docs/img/sweep_L3_reel_tower.png)

Each map shows the low-trajectory and high-lob bands that every Angry Birds
player knows. Within a band the score is noisy, because how a tower collapses
is chaotic. That is why the environment has `launch_noise`: it stops the agent
from learning knife-edge shots that only work because the simulation is
exactly repeatable. Videos of the best shots are in `docs/img/best_*.mp4`.

## Part 2 plan: RL on Great Lakes

- **Stage A, aim only (CPU):** PPO from Stable-Baselines3 on `AngryBirdsEnv`,
  with `SubprocVecEnv` across a `standard` node's 36 cores. Benchmark against
  CMA-ES on fixed levels and against the sweep's best shot. Then switch to
  `generator=random_level` to learn unseen pig layouts.
- **Stage B/C, the arm (GPU):** MuJoCo Playground (MJX or MuJoCo Warp backend,
  Brax PPO or RSL-RL) on the `spgpu` partition (A40, 48 GB). π_arm learns to
  hit a requested (angle, speed) within its torque limits. The band force is
  computed in the step function using `angrybirds.slingshot`'s Ogden model.
- **Join:** measure which (angle, speed) the arm can reach and how accurately,
  then fine-tune π_aim with that arm model in the loop so it only asks for
  shots the arm can make.
- **Why not one engine:** the game needs breakable bodies and tall stacks of
  touching blocks. Box2D does both natively. MuJoCo has no breakable bodies
  and handles big stacks slowly on GPU. Isaac Lab is the industry standard,
  but its renderer needs RT-core GPUs (Great Lakes' V100s aren't supported)
  and it runs in Singularity on the cluster: worth learning later, more
  setup than this project needs.

`scripts/greatlakes_sweep.sbatch` is a CPU job template:
`sbatch --account=<acct> scripts/greatlakes_sweep.sbatch L4_fortress`.

## Layout

```
angrybirds/
  config.py      scale, materials, block/pig catalogue, scoring, SimConfig
  slingshot.py   Ogden/Treloar band model → Launch (release state)
  levels.py      Level + Science-Birds-style XML, Builder, 4 starter levels
  world.py       Box2D GameWorld: build, settle, shoot, damage, score
  generator.py   random stable levels
  env.py         Gymnasium AngryBirdsEnv (Stage A); step_launch() for Part 2
  render.py      matplotlib previews and MP4s
levels/          starter levels as XML
scripts/         sweep_level.py, make_figures.py, greatlakes_sweep.sbatch
tests/           pytest suite
```

## Known limitations

- Only the red bird; no special birds (blue split, yellow speed-up, black
  bomb) yet.
- SquareHole blocks are solid (no hole).
- Angry Birds' real damage rules are not public. The model above is
  calibrated by behaviour, not copied, and its numbers live in `config.py`.
- The scoring values come from fan wikis, not Rovio.
- L4's hill blocks all low shots, but moderate arcs (~20–30°) clear it; it is
  not strictly lob-only.
- Box2D is deterministic on one machine and build; results may differ
  slightly across platforms.
