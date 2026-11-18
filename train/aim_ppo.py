#!/usr/bin/env python3
"""Stage A: train the aiming policy π_aim with PPO (Stable-Baselines3, CPU).

π_aim looks at the level (pigs + blocks) and outputs a shot (aim, power) in
[-1, 1]^2. With `--launcher arm` (default) the shot is (angle, draw) for the
SO-101 and the launch comes from its measured launch table, so a trained
policy can drive the robot directly (scripts/live.py --policy runs/<name>).

Writes a small, committable summary to runs/<run_name>/:
  config.yaml   every setting + git commit
  metrics.csv   one row per evaluation
  eval.json     final evaluation, shot by shot
  policy.npz    the actor network as plain numpy (~1 MB, no torch needed)
and the full SB3 checkpoint + TensorBoard logs to --ckpt-dir (scratch).

Typical runs (see slurm/aim_ppo.sbatch):
  # 1. sanity: one bird per level, the 4 starter levels
  python train/aim_ppo.py --run-name a1_starter_1bird --birds 1 --total-steps 300000
  # 2. full levels: 3 birds, clear bonus
  python train/aim_ppo.py --run-name a2_starter_3birds --birds 3 --total-steps 1000000
  # 3. generalise: 80 % random layouts, graded on 50 held-out layouts
  python train/aim_ppo.py --run-name a3_mix --level-mix 0.8 --total-steps 5000000
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from angrybirds.config import UNUSED_BIRD_POINTS  # noqa: E402
from angrybirds.env import AngryBirdsEnv, observe  # noqa: E402
from angrybirds.generator import random_level  # noqa: E402
from angrybirds.launchers import make_launcher  # noqa: E402
from angrybirds.levels import STARTER_LEVELS  # noqa: E402
from angrybirds.world import GameWorld  # noqa: E402


# ------------------------------------------------------------------ levels
class LevelMix:
    """Each episode: a random generated layout with probability `p`, else one
    of the starter levels. Picklable, so it works in SubprocVecEnv workers."""

    def __init__(self, starters: list[str], p: float):
        self.starters, self.p = starters, p

    def __call__(self, rng: np.random.Generator):
        if not self.starters or rng.random() < self.p:
            return random_level(rng)
        return STARTER_LEVELS[self.starters[int(rng.integers(len(self.starters)))]]()


def make_env(rank: int, args):
    def _init():
        from stable_baselines3.common.monitor import Monitor
        starters = [s for s in args.levels.split(",") if s]
        kw = dict(birds=args.birds, slingshot=make_launcher(args.launcher),
                  launch_noise=(args.noise_angle, args.noise_draw), shaping=args.shaping)
        if args.level_mix > 0:
            env = AngryBirdsEnv(generator=LevelMix(starters, args.level_mix), **kw)
        else:
            env = AngryBirdsEnv(levels=starters, **kw)
        env = Monitor(env)
        env.reset(seed=args.seed * 1000 + rank)
        return env
    return _init


# -------------------------------------------------------------- evaluation
def play_level(level, poses, policy_fn, launcher, birds):
    """Play one level to the end with `policy_fn(obs) -> action`. Same rules
    and scoring as AngryBirdsEnv, no launch noise."""
    level.birds = birds
    gw = GameWorld(level, poses=poses)
    total, shots, cleared = 0, [], False
    while True:
        a = np.asarray(policy_fn(observe(gw)), dtype=np.float64)
        angle, draw = launcher.action_to_shot(a)
        L = launcher.launch(angle, draw)
        res = gw.shoot(L.position, L.velocity)
        pts = res.score
        cleared = gw.pigs_left == 0
        if cleared:
            pts += UNUSED_BIRD_POINTS * gw.birds_left
        total += pts
        shots.append({"angle": round(angle, 2), "draw": round(draw, 3), "points": int(pts),
                      "pigs_killed": res.pigs_killed})
        if cleared or gw.birds_left == 0:
            break
    return {"level": level.name, "score": int(total), "cleared": cleared,
            "max_score": level.max_score(), "shots": shots}


class EvalSuite:
    def __init__(self, args):
        self.args = args
        self.launcher = make_launcher(args.launcher)
        self.starters = [(STARTER_LEVELS[n](), None) for n in args.levels.split(",") if n]
        self.starters = [(lvl, GameWorld.settled_poses(lvl)) for lvl, _ in self.starters]
        rng = np.random.default_rng(10_000 + args.eval_seed)       # fixed, never trained on
        self.heldout = []
        for _ in range(args.eval_levels if args.level_mix > 0 else 0):
            lvl = random_level(rng)
            self.heldout.append((lvl, GameWorld.settled_poses(lvl)))
        bj = ROOT / "levels" / f"sweep_best_{args.launcher}.json"
        self.sweep_best = json.loads(bj.read_text()) if bj.exists() else {}

    def run(self, policy_fn) -> dict:
        b = self.args.birds
        starter = [play_level(l, p, policy_fn, self.launcher, b) for l, p in self.starters]
        held = [play_level(l, p, policy_fn, self.launcher, b) for l, p in self.heldout]
        out = {"starter": starter, "heldout": held}
        summ = {}
        for r in starter:
            summ[f"score_{r['level']}"] = r["score"]
            if b == 1 and r["level"] in self.sweep_best:
                summ[f"frac_best_{r['level']}"] = round(r["score"] / max(self.sweep_best[r["level"]]["score"], 1), 3)
        if held:
            summ["heldout_mean_score"] = round(float(np.mean([r["score"] for r in held])), 1)
            summ["heldout_clear_rate"] = round(float(np.mean([r["cleared"] for r in held])), 3)
            summ["heldout_mean_frac_max"] = round(float(np.mean([r["score"] / r["max_score"] for r in held])), 4)
        out["summary"] = summ
        return out


def export_actor(model, path: Path, args) -> None:
    """Save the deterministic actor (mean action) as numpy arrays."""
    pol = model.policy
    layers = [m for m in pol.mlp_extractor.policy_net if hasattr(m, "weight")]
    arrays = {}
    for k, m in enumerate(layers):
        arrays[f"W{k}"] = m.weight.detach().cpu().numpy()
        arrays[f"b{k}"] = m.bias.detach().cpu().numpy()
    arrays["W_out"] = pol.action_net.weight.detach().cpu().numpy()
    arrays["b_out"] = pol.action_net.bias.detach().cpu().numpy()
    np.savez_compressed(path, activation=np.array("tanh"), launcher=np.array(args.launcher),
                        n_hidden=np.array(len(layers)), **arrays)


# --------------------------------------------------------------------- main
def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-name", default=time.strftime("aim_%m%d_%H%M%S"))
    ap.add_argument("--runs-dir", default=str(ROOT / "runs"))
    ap.add_argument("--ckpt-dir", default=os.environ.get("AB_CKPT_DIR", str(ROOT / "checkpoints")))
    ap.add_argument("--launcher", default="arm", choices=["arm", "reel"])
    ap.add_argument("--levels", default=",".join(STARTER_LEVELS), help="comma-separated starter levels")
    ap.add_argument("--level-mix", type=float, default=0.0, help="fraction of episodes on random layouts")
    ap.add_argument("--birds", type=int, default=3)
    ap.add_argument("--noise-angle", type=float, default=0.3, help="launch noise, deg (std)")
    ap.add_argument("--noise-draw", type=float, default=0.01, help="launch noise, draw fraction (std)")
    ap.add_argument("--shaping", type=float, default=0.1, help="near-miss penalty weight, annealed to 0")
    ap.add_argument("--shaping-frac", type=float, default=0.3, help="anneal over this fraction of training")
    ap.add_argument("--total-steps", type=int, default=1_000_000, help="environment steps (= shots)")
    ap.add_argument("--n-envs", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--n-steps", type=int, default=64, help="steps per env per PPO update")
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--n-epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--ent-coef", type=float, default=0.0)
    ap.add_argument("--log-std-init", type=float, default=-0.5)
    ap.add_argument("--net", default="256,256")
    ap.add_argument("--eval-every", type=int, default=50_000, help="env steps between evaluations")
    ap.add_argument("--eval-levels", type=int, default=50, help="held-out random layouts (with --level-mix)")
    ap.add_argument("--eval-seed", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--torch-threads", type=int, default=2)
    return ap.parse_args()


def main():
    args = parse_args()
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

    torch.set_num_threads(args.torch_threads)
    run_dir = Path(args.runs_dir) / args.run_name
    ckpt_dir = Path(args.ckpt_dir) / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    try:
        commit = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
                                capture_output=True, text=True).stdout.strip()
    except OSError:
        commit = "unknown"
    cfg = {**vars(args), "git_commit": commit, "host": os.uname().nodename,
           "slurm_job_id": os.environ.get("SLURM_JOB_ID")}
    import yaml
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    print("run", args.run_name, "->", run_dir, flush=True)

    fns = [make_env(i, args) for i in range(args.n_envs)]
    venv = SubprocVecEnv(fns) if args.n_envs > 1 else DummyVecEnv(fns)
    net = [int(x) for x in args.net.split(",")]
    model = PPO("MlpPolicy", venv, n_steps=args.n_steps, batch_size=args.batch_size,
                n_epochs=args.n_epochs, learning_rate=args.lr, gamma=args.gamma,
                gae_lambda=0.95, clip_range=0.2, ent_coef=args.ent_coef, seed=args.seed,
                policy_kwargs=dict(net_arch=dict(pi=net, vf=net), log_std_init=args.log_std_init),
                tensorboard_log=str(ckpt_dir / "tb"), device="cpu", verbose=0)

    suite = EvalSuite(args)
    rng = np.random.default_rng(args.seed)
    t0 = time.time()
    base = suite.run(lambda obs: rng.uniform(-1, 1, 2))["summary"]
    print("random-policy baseline:", base, f"({time.time() - t0:.0f}s)", flush=True)
    (run_dir / "baseline_random.json").write_text(json.dumps(base, indent=2) + "\n")

    metrics_path = run_dir / "metrics.csv"
    state = {"next_eval": 0, "header": None, "t0": time.time(), "best": -1e18}

    def policy_fn(obs):
        a, _ = model.predict(obs, deterministic=True)
        return a

    def evaluate(step: int, final: bool = False):
        ev = suite.run(policy_fn)
        s = ev["summary"]
        ep = list(model.ep_info_buffer)
        row = {"step": step, "wall_s": round(time.time() - state["t0"], 1),
               "shots_per_s": round(step / max(time.time() - state["t0"], 1e-9), 1),
               "train_ep_reward": round(float(np.mean([e["r"] for e in ep])), 4) if ep else None,
               "shaping": round(float(venv.get_attr("shaping")[0]), 4), **s}
        if state["header"] is None:
            state["header"] = list(row)
            with open(metrics_path, "w", newline="") as f:
                csv.DictWriter(f, fieldnames=state["header"]).writeheader()
        with open(metrics_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=state["header"], extrasaction="ignore").writerow(row)
        print(json.dumps(row), flush=True)
        key = s.get("heldout_mean_frac_max", np.mean([v for k, v in s.items() if k.startswith("score_")]))
        if key > state["best"] or final:
            if key > state["best"]:
                state["best"] = key
                model.save(ckpt_dir / "best_model")
                export_actor(model, run_dir / "policy_best.npz", args)
        if final:
            ev["config"] = {k: cfg[k] for k in ("run_name", "launcher", "birds", "level_mix", "total_steps")}
            ev["random_baseline"] = base
            (run_dir / "eval.json").write_text(json.dumps(ev, indent=2) + "\n")

    class Hooks(BaseCallback):
        def _on_step(self) -> bool:
            n = self.num_timesteps
            frac = min(n / max(args.shaping_frac * args.total_steps, 1), 1.0)
            if args.shaping:
                # a method call reaches the inner env; set_attr would only
                # set an attribute on the Monitor wrapper
                self.training_env.env_method("set_shaping", args.shaping * (1.0 - frac))
            if n >= state["next_eval"]:
                state["next_eval"] = n + args.eval_every
                evaluate(n)
            return True

    model.learn(total_timesteps=args.total_steps, callback=Hooks(), tb_log_name="ppo")
    evaluate(model.num_timesteps, final=True)
    model.save(ckpt_dir / "final_model")
    export_actor(model, run_dir / "policy.npz", args)
    venv.close()
    print(f"done in {(time.time() - state['t0']) / 60:.1f} min -> {run_dir}", flush=True)


if __name__ == "__main__":
    main()
