"""Runs a whole level in the co-simulation: plan a shot, let the SO-101 fire
it, wait for the game to settle, plan the next one from the new state.

Planning (a grid search with the analytic launch + Box2D) runs in a
background thread so a live viewer keeps rendering meanwhile. A fixed
`ShotPlan` can be given instead (e.g. from a trained policy).
"""
from __future__ import annotations

import threading

from .cosim import CoSim, ShotPlan, plan_shot
from .levels import Level


class Show:
    def __init__(self, level: Level, plans: list[ShotPlan] | None = None,
                 workers: int | None = None, pause: float = 0.8):
        self.sim = CoSim(level)
        self.fixed = list(plans or [])
        self.workers = workers
        self.pause = pause                   # sim seconds between shots
        self._plan: ShotPlan | None = None
        self._planned_score: int | None = None
        self._thread: threading.Thread | None = None
        self._idle_since: float | None = 0.0
        self.finished = False

    def _plan_async(self) -> None:
        def work():
            plan, sc = plan_shot(self.sim.gw, workers=self.workers)
            self._plan, self._planned_score = plan, sc
        self._plan, self._planned_score = None, None
        self.sim.status = "planning..."
        self._thread = threading.Thread(target=work, daemon=True)
        self._thread.start()

    def tick(self) -> None:
        """Advance the show by one 1 ms physics step."""
        sim = self.sim
        sim.step()
        if self.finished or sim.busy:
            if sim.busy:
                self._idle_since = None
            return
        gw = sim.gw
        if gw.pigs_left == 0 or gw.birds_left == 0:
            self.finished = True
            sim.status = (f"LEVEL CLEARED: {gw.score}" if gw.pigs_left == 0
                          else f"out of birds: {gw.score}")
            return
        if self._idle_since is None:
            self._idle_since = sim.d.time
        if sim.d.time - self._idle_since < self.pause:
            return
        if self.fixed:
            sim.fire(self.fixed.pop(0))
            self._idle_since = None
            return
        if self._thread is None:
            self._plan_async()
        elif not self._thread.is_alive():
            self._thread = None
            sim.fire(self._plan)
            self._idle_since = None

    def hud(self) -> tuple[str, str]:
        left, right = self.sim.hud()
        if self._planned_score is not None and self.sim.logs:
            left += "\nPlanner expected"
            right += f"\n{self._planned_score} pts"
        return left, right
