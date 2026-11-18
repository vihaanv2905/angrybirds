"""Gymnasium environment: Stage A of the RL plan (no arm yet).

One step = one bird. Action = (aim, power) in [-1, 1]^2, mapped by the
slingshot to an angle in [-10, 80] deg and a draw of 5-100 % of 0.67 m. The
episode is one level with `birds` birds; it ends when every pig is dead
(unused birds then score 10,000 each) or the birds run out.

In Part 2 the same environment takes a `Launch` from the MuJoCo arm instead
of from `Slingshot.launch`: see `step_launch`.
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .config import TRIANGLE_BLOCKS, UNUSED_BIRD_POINTS, SimConfig
from .levels import STARTER_LEVELS, Level
from .slingshot import Launch, Slingshot
from .world import GameWorld

MATS = ("ice", "wood", "stone")
PIG_FEATS = 4      # x, y, radius, alive
BLOCK_FEATS = 11   # x, y, w, h, cos, sin, ice, wood, stone, hp_frac, present
X_SCALE, Y_SCALE = 50.0, 10.0


def _dims(ent):
    if ent.shape[0] == "box":
        return ent.shape[1], ent.shape[2]
    if ent.shape[0] == "circle":
        return 2 * ent.shape[1], 2 * ent.shape[1]
    return TRIANGLE_BLOCKS.get(ent.type, (1.0, 1.0))


def obs_size(max_pigs: int = 8, max_blocks: int = 40) -> int:
    return 1 + max_pigs * PIG_FEATS + max_blocks * BLOCK_FEATS


def observe(gw: GameWorld, max_pigs: int = 8, max_blocks: int = 40) -> np.ndarray:
    """Policy observation of a live game. Also used outside the env (e.g. the
    MuJoCo live viewer) so a trained policy sees exactly what it trained on."""
    obs = np.zeros(obs_size(max_pigs, max_blocks), dtype=np.float32)
    obs[0] = gw.birds_left / 3.0
    pigs = sorted((e for e in gw.entities if e.kind == "pig"), key=lambda e: e.eid)
    o = 1
    for i, e in enumerate(pigs[:max_pigs]):
        if e.alive:
            p = e.body.position
            obs[o + i * PIG_FEATS: o + (i + 1) * PIG_FEATS] = (p.x / X_SCALE, p.y / Y_SCALE, e.shape[1], 1.0)
    o += max_pigs * PIG_FEATS
    blocks = [e for e in gw.entities if e.kind == "block" and e.alive]
    blocks.sort(key=lambda e: (e.body.position.x, e.body.position.y))
    for i, e in enumerate(blocks[:max_blocks]):
        p, a = e.body.position, e.body.angle
        w, h = _dims(e)
        m = [1.0 if e.material == k else 0.0 for k in MATS]
        obs[o + i * BLOCK_FEATS: o + (i + 1) * BLOCK_FEATS] = (
            p.x / X_SCALE, p.y / Y_SCALE, w / 3.0, h / 3.0, np.cos(a), np.sin(a),
            *m, max(e.hp, 0.0) / e.max_hp, 1.0)
    return obs


class AngryBirdsEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 4}

    def __init__(self, levels=None, generator=None, birds: int | None = None,
                 sim: SimConfig | None = None, slingshot: Slingshot | None = None,
                 launch_noise: tuple[float, float] = (0.0, 0.0),
                 reward_mode: str = "score", shaping: float = 0.0,
                 max_pigs: int = 8, max_blocks: int = 40, render_mode: str | None = None):
        """
        levels       list of Level / starter-level names (default: all 4 starters)
        generator    callable(np.random.Generator) -> Level; overrides `levels`
        birds        birds per level (default: the level's own count)
        slingshot    launcher: `Slingshot()` (reel, default) or
                     `launchers.ArmLauncher()` (the SO-101's measured launches)
        launch_noise (angle std in deg, draw-fraction std): execution noise so
                     the agent cannot rely on knife-edge, exactly-repeatable shots
        reward_mode  "score": points / 10,000   "normalized": points / level max
        shaping      weight of a penalty on how close the bird came to a pig,
                     only on shots that kill nothing (anneal to 0 in training)
        """
        super().__init__()
        if levels is None and generator is None:
            levels = list(STARTER_LEVELS)
        self.levels = [STARTER_LEVELS[l]() if isinstance(l, str) else l for l in (levels or [])]
        self.generator = generator
        self.birds = birds
        self.sim = sim or SimConfig()
        self.sling = slingshot or Slingshot()
        self.launch_noise = launch_noise
        self.reward_mode = reward_mode
        self.shaping = shaping
        self.max_pigs, self.max_blocks = max_pigs, max_blocks
        self.render_mode = render_mode
        self._pose_cache: dict[int, list] = {}

        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        n = obs_size(max_pigs, max_blocks)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(n,), dtype=np.float32)
        self.gw: GameWorld | None = None
        self.level: Level | None = None

    def set_shaping(self, value: float) -> None:
        """Called by trainers to anneal the near-miss penalty (a method, so it
        reaches through wrappers such as Monitor)."""
        self.shaping = float(value)

    # ---------------------------------------------------------------- reset
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        if "level" in options:
            lvl = options["level"]
            self.level = STARTER_LEVELS[lvl]() if isinstance(lvl, str) else lvl
        elif self.generator is not None:
            self.level = self.generator(self.np_random)
        else:
            self.level = self.levels[self.np_random.integers(len(self.levels))]
        if self.birds is not None:
            self.level.birds = self.birds
        key = id(self.level)
        if key not in self._pose_cache:
            if self.generator is not None and "level" not in options:
                self._pose_cache.clear()        # generated levels are one-offs
            self._pose_cache[key] = GameWorld.settled_poses(self.level, self.sim)
        self.gw = GameWorld(self.level, self.sim, poses=self._pose_cache[key])
        self.total_score = 0
        return self._obs(), {"level": self.level.name, "max_score": self.level.max_score()}

    # ----------------------------------------------------------------- step
    def step(self, action):
        angle, draw = self.sling.action_to_shot(action)
        if self.launch_noise[0] or self.launch_noise[1]:
            angle += self.np_random.normal(0.0, self.launch_noise[0])
            draw = float(np.clip(draw + self.np_random.normal(0.0, self.launch_noise[1]), 0.0, 1.0))
        return self.step_launch(self.sling.launch(angle, draw))

    def step_launch(self, launch: Launch):
        """Fire a bird with an explicit release state (Part 2: from the arm)."""
        gw = self.gw
        res = gw.shoot(launch.position, launch.velocity)
        points = res.score
        cleared = gw.pigs_left == 0
        if cleared:
            points += UNUSED_BIRD_POINTS * gw.birds_left
        self.total_score += points
        terminated = cleared or gw.birds_left == 0

        if self.reward_mode == "normalized":
            reward = points / max(self.level.max_score(), 1)
        else:
            reward = points / 10000.0
        if self.shaping and res.pigs_killed == 0:
            reward -= self.shaping * min(res.min_pig_distance, 10.0) / 10.0

        info = {
            "shot_score": points, "total_score": self.total_score, "pigs_left": gw.pigs_left,
            "pigs_killed": res.pigs_killed, "blocks_destroyed": res.blocks_destroyed,
            "cleared": cleared, "min_pig_distance": res.min_pig_distance,
            "first_impact": res.first_impact, "angle_deg": launch.angle_deg,
            "speed": launch.speed, "draw": launch.draw, "sim_time": res.sim_time,
            "max_score": self.level.max_score(),
        }
        return self._obs(), float(reward), terminated, False, info

    # ---------------------------------------------------------- observation
    def _obs(self) -> np.ndarray:
        return observe(self.gw, self.max_pigs, self.max_blocks)

    # --------------------------------------------------------------- render
    def render(self):
        if self.render_mode != "rgb_array" or self.gw is None:
            return None
        import matplotlib.pyplot as plt

        from .render import draw
        fig, ax = plt.subplots(figsize=(10, 4))
        draw(self.gw, ax=ax, title=f"{self.level.name}  score {self.total_score}")
        fig.canvas.draw()
        img = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
        plt.close(fig)
        return img
