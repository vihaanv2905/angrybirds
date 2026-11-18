"""Run a trained aiming policy (runs/<name>/policy.npz) with plain numpy.

The file holds the PPO actor (tanh MLP -> mean action), exported by
train/aim_ppo.py, so no torch or Stable-Baselines3 is needed to play.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .env import observe
from .launchers import make_launcher


class NumpyPolicy:
    def __init__(self, path):
        path = Path(path)
        if path.is_dir():                       # a run directory
            best = path / "policy_best.npz"
            path = best if best.exists() else path / "policy.npz"
        z = np.load(path)
        self.path = path
        self.launcher_name = str(z["launcher"])
        n = int(z["n_hidden"])
        self.layers = [(z[f"W{k}"], z[f"b{k}"]) for k in range(n)]
        self.out = (z["W_out"], z["b_out"])
        self.launcher = make_launcher(self.launcher_name)

    def act(self, obs: np.ndarray) -> np.ndarray:
        h = np.asarray(obs, dtype=np.float64)
        for W, b in self.layers:
            h = np.tanh(W @ h + b)
        W, b = self.out
        return np.clip(W @ h + b, -1.0, 1.0)

    def shot(self, gw) -> tuple[float, float]:
        """(angle_deg, draw_frac) for the current game state."""
        return self.launcher.action_to_shot(self.act(observe(gw)))
