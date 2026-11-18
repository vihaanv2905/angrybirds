"""Live co-simulation: the SO-101 in MuJoCo fires the bird into the Box2D game.

One clock (MuJoCo time, 1 ms steps) drives both simulators:

  1. MuJoCo: the arm grabs the pouch (weld "grip"), pulls it back along the
     planned aim line with damped-least-squares IK, holds, and lets go. Two
     latex bands (spatial tendons) are driven every step by the Ogden model,
     so the release is a real contact launch: the pouch pushes the bird.
  2. When the bird leaves the pouch its MuJoCo position/velocity are Froude-
     scaled (x20 length, x sqrt(20) speed) into a Box2D launch.
  3. Box2D runs the shot in lockstep: each 1 ms of tabletop time is
     1/sqrt(LAMBDA) = 4.47 ms of game time. Every block, pig and the bird are
     mirrored into MuJoCo as mocap bodies, so you watch it all in one viewer.

`ShotPlan(angle_deg, draw_frac)` is what an aiming policy outputs. Replace
`Controller` with a learned arm policy in Part 2.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np

from .config import SimConfig
from .levels import BlockSpec, Level, PigSpec
from .mj_scene import HIDDEN_Z, SLING_X, angle_quat, build_model, game_to_mj
from .slingshot import ogden_nominal_stress
from .tabletop import LAMBDA, SQRT_LAMBDA, TabletopSlingshot
from .world import GameWorld

ARM_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
GRIP_OPEN, GRIP_CLOSED = 1.0, 0.25          # gripper joint targets, rad
POUCH_INSET = 0.006                          # pouch centre ahead of the jaw tips, m


@dataclass
class ShotPlan:
    angle_deg: float
    draw_frac: float


@dataclass
class ShotLog:
    plan: ShotPlan
    planned_game_v: tuple[float, float] | None = None
    actual_table_v: tuple[float, float] | None = None
    actual_game_pos: tuple[float, float] | None = None
    actual_game_v: tuple[float, float] | None = None
    peak_band_tension: float = 0.0
    peak_joint_torque: dict = field(default_factory=dict)   # holding torque at full draw
    score: int | None = None
    pigs_killed: int | None = None


def smoothstep(s: float) -> float:
    s = min(max(s, 0.0), 1.0)
    return s * s * (3 - 2 * s)


class Controller:
    """Scripted pull-and-release for the SO-101 (the baseline Part 2's RL arm
    policy replaces). Phases, in seconds of sim time after `start()`:
      approach 0-1.2 | close 1.2-1.5 | pull 1.5-3.0 | hold 3.0-3.6 |
      release 3.6 | wait 3.6-4.1 | retract 4.1-5.1 | done
    """
    T_PRE, T_APP, T_CLOSE, T_PULL, T_HOLD, T_WAIT, T_RET = 0.8, 1.2, 1.5, 3.0, 3.6, 4.1, 5.1

    def __init__(self, sim: "CoSim"):
        self.sim = sim
        m = sim.m
        self.qadr = np.array([m.joint(j).qposadr[0] for j in ARM_JOINTS])
        self.dadr = np.array([m.joint(j).dofadr[0] for j in ARM_JOINTS])
        self.lo = np.array([m.joint(j).range[0] for j in ARM_JOINTS])
        self.hi = np.array([m.joint(j).range[1] for j in ARM_JOINTS])
        self.act = np.array([m.actuator(j).id for j in ARM_JOINTS])
        self.act_grip = m.actuator("gripper").id
        self.site = m.site("gripperframe").id
        self.dk = mujoco.MjData(m)              # scratch data for IK
        self.q_home = self.ik_pouch(np.array([SLING_X - 0.14, 0.0, 0.24]), 0.0,
                                    np.array([0, -1.0, 1.0, 0.0, 0.0]))
        self.q_grasp = self.ik_pouch(sim.rest3, 0.0, self.q_home)
        self.q_pre = self.ik_pouch(sim.rest3 - np.array([0.04, 0, 0]), 0.0, self.q_grasp)
        self.phase, self.t0, self.plan = "idle", 0.0, None
        self.q_cmd = self.q_home.copy()
        self.grip_cmd = GRIP_OPEN
        self._q_ret0 = self.q_home.copy()

    # ------------------------------------------------------------------ IK
    def ik_pouch(self, pouch_target: np.ndarray, pitch: float, q0: np.ndarray, iters: int = 60) -> np.ndarray:
        """Joint angles (pan, lift, elbow, wrist_flex, roll) that put the pinch
        point (POUCH_INSET ahead of the jaw tips, where the pouch centre is held)
        at `pouch_target`. The gripper pointing along `pitch` is a secondary
        goal, solved in the null space of the position task: the SO-101 wrist
        cannot point steeply upward low down, and the pouch pivots anyway."""
        m, d = self.sim.m, self.dk
        # the SO-101 can only point ~0-25 deg upward this low: ask for what it can do
        pitch = min(max(pitch, math.radians(-10)), math.radians(15))
        u = np.array([math.cos(pitch), 0.0, math.sin(pitch)])
        q = np.array(q0, dtype=float)
        q[0], q[4] = 0.0, 0.0
        jacp = np.zeros((3, m.nv))
        jacr = np.zeros((3, m.nv))
        cols = self.dadr[1:4]
        bgrip = self.sim.b_grip
        for it in range(iters + 20):
            d.qpos[self.qadr] = q
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)
            xaxis = d.site_xmat[self.site].reshape(3, 3)[:, 0]
            pinch = d.site_xpos[self.site] + POUCH_INSET * xaxis
            ep = np.array([pouch_target[0] - pinch[0], pouch_target[2] - pinch[2]])
            # last 20 iterations: position only, so joint-limit clipping from
            # the pitch goal can never leave the pouch in the wrong place
            eo = np.cross(xaxis, u)[1] if it < iters else 0.0
            if np.linalg.norm(ep) < 1e-6 and abs(eo) < 1e-4:
                break
            mujoco.mj_jac(m, d, jacp, jacr, pinch, bgrip)
            Jp = jacp[[0, 2]][:, cols]
            Jo = jacr[1, cols]
            Jp_pinv = Jp.T @ np.linalg.inv(Jp @ Jp.T + 1e-8 * np.eye(2))
            N = np.eye(3) - Jp_pinv @ Jp
            dq = Jp_pinv @ ep + N @ (0.5 * Jo * eo)
            q[1:4] = np.clip(q[1:4] + dq, self.lo[1:4], self.hi[1:4])
        return q

    def reach_error(self, q: np.ndarray, pouch_target: np.ndarray) -> float:
        d = self.dk
        d.qpos[self.qadr] = q
        mujoco.mj_kinematics(self.sim.m, d)
        xaxis = d.site_xmat[self.site].reshape(3, 3)[:, 0]
        pinch = d.site_xpos[self.site] + POUCH_INSET * xaxis
        return float(np.linalg.norm((pinch - pouch_target)[[0, 2]]))

    # ----------------------------------------------------------- schedule
    def start(self, plan: ShotPlan, t: float) -> None:
        self.plan, self.t0, self.phase = plan, t, "approach"
        th = math.radians(plan.angle_deg)
        self.u = np.array([math.cos(th), 0.0, math.sin(th)])
        self.d0 = plan.draw_frac * self.sim.sling.d_max
        self.q_cmd = self.q_home.copy()

    def update(self, t: float) -> None:
        sim, tt = self.sim, t - self.t0
        if self.phase == "idle" or self.phase == "done":
            pass
        elif tt < self.T_PRE:
            self.q_cmd = self.q_home + smoothstep(tt / self.T_PRE) * (self.q_pre - self.q_home)
        elif tt < self.T_APP:
            self.q_cmd = self.q_pre + smoothstep((tt - self.T_PRE) / (self.T_APP - self.T_PRE)) * (self.q_grasp - self.q_pre)
        elif tt < self.T_CLOSE:
            self.phase = "grasp"
            self.grip_cmd = GRIP_OPEN + smoothstep((tt - self.T_APP) / (self.T_CLOSE - self.T_APP)) * (GRIP_CLOSED - GRIP_OPEN)
        elif tt < self.T_PULL:
            if self.phase == "grasp":
                sim.grab()
                self.phase = "pull"
            s = smoothstep((tt - self.T_CLOSE) / (self.T_PULL - self.T_CLOSE))
            target = sim.rest3 - s * self.d0 * self.u
            pitch = s * math.radians(self.plan.angle_deg)
            self.q_cmd = self.ik_pouch(target, pitch, self.q_cmd, iters=8)
        elif tt < self.T_HOLD:
            self.phase = "hold"
        elif tt < self.T_WAIT:
            if self.phase == "hold":
                sim.release()
                self.grip_cmd = GRIP_OPEN
                self.phase = "release"
                self._q_ret0 = self.q_cmd.copy()
        elif tt < self.T_RET:
            self.phase = "retract"
            s = smoothstep((tt - self.T_WAIT) / (self.T_RET - self.T_WAIT))
            self.q_cmd = self._q_ret0 + s * (self.q_home - self._q_ret0)
        else:
            self.q_cmd = self.q_home.copy()
            self.phase = "done"
        sim.d.ctrl[self.act] = self.q_cmd
        sim.d.ctrl[self.act_grip] = self.grip_cmd


class CoSim:
    def __init__(self, level: Level, sim_cfg: SimConfig | None = None,
                 sling: TabletopSlingshot | None = None):
        self.level = level
        self.cfg = sim_cfg or SimConfig()
        self.sling = sling or TabletopSlingshot()
        self.gw = GameWorld(level, self.cfg)
        self.spec, self.m = build_model(level, self.gw.entities)
        self.d = mujoco.MjData(self.m)
        m = self.m
        self.rest3 = np.array([SLING_X, 0.0, self.sling.rest[1]])
        self.b_pouch, self.b_bird, self.b_grip = m.body("pouch").id, m.body("bird").id, m.body("gripper").id
        self.q_pouch = m.joint("pouch_free").qposadr[0]
        self.q_bird = m.joint("bird_free").qposadr[0]
        self.v_pouch = m.joint("pouch_free").dofadr[0]
        self.v_bird = m.joint("bird_free").dofadr[0]
        self.g_bird = m.geom("bird").id
        self.ten = [m.tendon("band_L").id, m.tendon("band_R").id]
        self.act_band = [m.actuator("band_L").id, m.actuator("band_R").id]
        self.eq_park, self.eq_nock, self.eq_grip = (m.equality(n).id for n in ("park", "nock", "grip"))
        self.mocap = {e.eid: m.body_mocapid[m.body(f"ent{e.eid}").id] for e in self.gw.entities}
        self.ctrl = Controller(self)
        # put the arm at home before anything happens
        self.d.qpos[self.ctrl.qadr] = self.ctrl.q_home
        self.d.ctrl[self.ctrl.act] = self.ctrl.q_home
        self.d.ctrl[self.ctrl.act_grip] = GRIP_OPEN
        mujoco.mj_forward(m, self.d)
        self.bird_mode = "nocked"       # nocked | launched | game | hidden
        self.game_active = False
        self._acc = 0.0
        self.t_release = None
        self.logs: list[ShotLog] = []
        self.status = "ready"
        self.sync_mocap()

    # --------------------------------------------------------------- welds
    def _set_weld(self, eq: int, b1: int, b2: int) -> None:
        """Weld b1 to b2 in their current relative pose, anchored at b1."""
        d, m = self.d, self.m
        p1, p2 = d.xpos[b1], d.xpos[b2]
        R2 = d.xmat[b2].reshape(3, 3)
        anchor = R2.T @ (p1 - p2)                      # b1 origin in b2 frame
        q1c = np.zeros(4)
        mujoco.mju_negQuat(q1c, d.xquat[b1])
        rel = np.zeros(4)
        mujoco.mju_mulQuat(rel, q1c, d.xquat[b2])
        m.eq_data[eq, 0:3] = anchor
        m.eq_data[eq, 3:6] = 0.0
        m.eq_data[eq, 6:10] = rel
        d.eq_active[eq] = 1

    def grab(self) -> None:
        """Pinch the pouch at its centre: a ball-joint (connect) constraint, so
        the pouch can pivot and the bands line it up with the pull."""
        d, m = self.d, self.m
        p = d.xpos[self.b_pouch]
        Rg = d.xmat[self.b_grip].reshape(3, 3)
        m.eq_data[self.eq_grip, 0:3] = 0.0                                  # in pouch frame
        m.eq_data[self.eq_grip, 3:6] = Rg.T @ (p - d.xpos[self.b_grip])   # in gripper frame
        d.eq_active[self.eq_grip] = 1
        d.eq_active[self.eq_park] = 0
        self.status = "pulling"

    def release(self) -> None:
        """Fingers open. The bird rides in the pouch (nock weld stays on: a
        real pouch wraps it) until the bands go slack, then flies free."""
        self.d.eq_active[self.eq_grip] = 0
        self.t_release = self.d.time
        self.bird_mode = "launched"
        self.status = "released"

    # ------------------------------------------------------------ reload
    def reload(self) -> None:
        """Put a new bird in the parked pouch at rest."""
        d, m = self.d, self.m
        d.qpos[self.q_pouch:self.q_pouch + 7] = [*self.rest3, 1, 0, 0, 0]
        d.qpos[self.q_bird:self.q_bird + 7] = [self.rest3[0] + self.sling.bird_offset, 0, self.rest3[2], 1, 0, 0, 0]
        d.qvel[self.v_pouch:self.v_pouch + 6] = 0
        d.qvel[self.v_bird:self.v_bird + 6] = 0
        m.geom_contype[self.g_bird] = m.geom_conaffinity[self.g_bird] = 2
        d.eq_active[self.eq_park] = 1
        d.eq_active[self.eq_nock] = 1
        d.eq_active[self.eq_grip] = 0
        self.bird_mode = "nocked"
        mujoco.mj_forward(m, d)

    # -------------------------------------------------------------- bands
    def _bands(self) -> None:
        d, s = self.d, self.sling
        for t, a in zip(self.ten, self.act_band):
            lam = d.ten_length[t] / s.L0
            T = s.A0 * ogden_nominal_stress(lam) if lam > 1.0 else 0.0
            if d.ten_velocity[t] < 0.0:
                T *= s.hysteresis          # rubber gives back less on unloading
            d.ctrl[a] = -T
            if self.logs and self.bird_mode in ("nocked", "launched"):
                self.logs[-1].peak_band_tension = max(self.logs[-1].peak_band_tension, T)

    # -------------------------------------------------------------- shots
    def fire(self, plan: ShotPlan) -> None:
        if self.gw.birds_left <= 0 or self.gw.pigs_left == 0:
            raise RuntimeError("level over")
        if self.bird_mode != "nocked":
            self.reload()
        L = self.sling.launch(plan.angle_deg, plan.draw_frac)
        log = ShotLog(plan, planned_game_v=(L.velocity[0] / SQRT_LAMBDA, L.velocity[1] / SQRT_LAMBDA))
        self.logs.append(log)
        self.ctrl.start(plan, self.d.time)
        self.status = "reaching"

    @property
    def busy(self) -> bool:
        return self.ctrl.phase not in ("idle", "done") or self.game_active or self.bird_mode == "launched"

    def _handoff(self) -> None:
        """Bird has left the pouch: Froude-scale it into the Box2D game."""
        d = self.d
        p = d.xpos[self.b_bird]
        v = d.qvel[self.v_bird:self.v_bird + 3]
        gpos = ((p[0] - SLING_X) / LAMBDA, p[2] / LAMBDA)
        gvel = (v[0] / SQRT_LAMBDA, v[2] / SQRT_LAMBDA)
        log = self.logs[-1]
        log.actual_table_v = (float(v[0]), float(v[2]))
        log.actual_game_pos, log.actual_game_v = gpos, gvel
        self.gw.begin_shot(gpos, gvel)
        self.game_active = True
        self._acc = 0.0
        self.bird_mode = "game"
        self.m.geom_contype[self.g_bird] = self.m.geom_conaffinity[self.g_bird] = 0
        self.status = "flying"

    def _damp_pouch(self) -> None:
        """Air/leather damping on the pouch: always on its spin, and on its
        motion too once the bird has gone, so it flops and settles."""
        d = self.d
        R = d.xmat[self.b_pouch].reshape(3, 3)
        w = R @ d.qvel[self.v_pouch + 3:self.v_pouch + 6]        # world frame
        v = d.qvel[self.v_pouch:self.v_pouch + 3]
        c_lin = 0.02 if self.bird_mode in ("game", "hidden") else 0.0
        d.xfrc_applied[self.b_pouch, :3] = -c_lin * v
        d.xfrc_applied[self.b_pouch, 3:] = -2e-4 * w

    def step(self) -> None:
        d, m = self.d, self.m
        self.ctrl.update(d.time)
        if self.bird_mode in ("game", "hidden"):
            self._pin_bird()
        # step1 computes tendon lengths/velocities for the *current* state, so
        # the band forces are not a step late (a lag that pumps energy in)
        mujoco.mj_step1(m, d)
        self._bands()
        self._damp_pouch()
        mujoco.mj_step2(m, d)
        if self.logs and self.ctrl.phase == "hold":
            hold = self.logs[-1].peak_joint_torque     # steady holding torque, N*m
            for j, a in zip(ARM_JOINTS, self.ctrl.act):
                hold[j] = abs(float(d.actuator_force[a]))
        if self.bird_mode == "launched":
            if d.eq_active[self.eq_nock] and d.time - self.t_release > 0.002 \
                    and d.ctrl[self.act_band[0]] == 0.0 and d.ctrl[self.act_band[1]] == 0.0:
                d.eq_active[self.eq_nock] = 0          # bands slack: bird leaves the pouch
            gap = np.linalg.norm(d.xpos[self.b_bird] - d.xpos[self.b_pouch])
            if d.time - self.t_release > 0.002 and gap > self.sling.bird_offset + 0.003:
                self._handoff()
            elif d.time - self.t_release > 0.5:      # weak shot that never left
                self._handoff()
        if self.game_active:
            self._acc += m.opt.timestep / SQRT_LAMBDA
            while self._acc >= self.gw.dt and self.game_active:
                self._acc -= self.gw.dt
                if self.gw.step_shot():
                    self._finish_shot()

    def _finish_shot(self) -> None:
        res = self.gw.end_shot()
        self.game_active = False
        cleared = self.gw.pigs_left == 0
        if cleared:
            from .config import UNUSED_BIRD_POINTS
            self.gw.score += UNUSED_BIRD_POINTS * self.gw.birds_left
        log = self.logs[-1]
        log.score, log.pigs_killed = res.score, res.pigs_killed
        self.bird_mode = "hidden"
        self.status = "cleared!" if cleared else ("out of birds" if self.gw.birds_left == 0 else "shot over")

    def _pin_bird(self) -> None:
        d = self.d
        bird = self.gw._bird
        if bird is not None and self.bird_mode == "game":
            p = bird.body.position
            d.qpos[self.q_bird:self.q_bird + 3] = game_to_mj(p.x, p.y)
            d.qpos[self.q_bird + 3:self.q_bird + 7] = angle_quat(bird.body.angle)
        else:
            d.qpos[self.q_bird:self.q_bird + 3] = (0, 0, HIDDEN_Z)
        d.qvel[self.v_bird:self.v_bird + 6] = 0

    def sync_mocap(self) -> None:
        d = self.d
        for e in self.gw.entities:
            mid = self.mocap[e.eid]
            if e.alive:
                p = e.body.position
                d.mocap_pos[mid] = game_to_mj(p.x, p.y)
                d.mocap_quat[mid] = angle_quat(e.body.angle)
            else:
                d.mocap_pos[mid] = (0, 0, HIDDEN_Z)

    def hud(self) -> tuple[str, str]:
        gw = self.gw
        left = "Score\nBirds left\nPigs left\nStatus"
        right = f"{gw.score}\n{gw.birds_left}\n{gw.pigs_left}\n{self.status}"
        if self.logs:
            lg = self.logs[-1]
            left += "\nPlan\nPull force"
            right += f"\n{lg.plan.angle_deg:.1f} deg, {lg.plan.draw_frac*100:.0f}% draw"
            right += f"\n{2 * lg.peak_band_tension:.1f} N"
            if lg.actual_game_v:
                vx, vy = lg.actual_game_v
                left += "\nLaunch (game)"
                right += f"\n{math.hypot(vx, vy):.1f} m/s at {math.degrees(math.atan2(vy, vx)):.1f} deg"
        return left, right


# ------------------------------------------------------------ shot planning
def level_from_state(gw: GameWorld) -> tuple[Level, list]:
    """Snapshot the live game as a fresh Level + poses (hit points reset)."""
    lvl = Level(gw.level.name, hills=gw.level.hills, birds=1)
    poses = []
    for e in gw.entities:
        if not e.alive:
            continue
        p, a = e.body.position, e.body.angle
        if e.kind == "pig":
            lvl.pigs.append(PigSpec(e.type, p.x, p.y))
        else:
            lvl.blocks.append(BlockSpec(e.type, e.material, p.x, p.y, math.degrees(a)))
    # GameWorld builds blocks first, then pigs: keep that order for poses
    for e in [e for e in gw.entities if e.alive and e.kind == "block"] + \
             [e for e in gw.entities if e.alive and e.kind == "pig"]:
        poses.append((e.body.position.x, e.body.position.y, e.body.angle))
    return lvl, poses


# ------------------------------------------------- measured launch table
TABLE_PATH = Path(__file__).resolve().parents[1] / "robot" / "so101" / "launch_table.npz"


def measure_launch(angle_deg: float, draw_frac: float) -> tuple[tuple, tuple]:
    """Fire one bird with the SO-101 in MuJoCo and return the game-scale
    release (position, velocity) at hand-off. ~0.5 s of CPU."""
    from .levels import Builder
    b = Builder("calib", birds=1)
    b.pig("BasicSmall", 130.0, 0.0)          # far away: the shot never ends the level
    sim = CoSim(b.level)
    sim.fire(ShotPlan(angle_deg, draw_frac))
    while sim.bird_mode in ("nocked", "launched") and sim.d.time < 10:
        sim.step()
    lg = sim.logs[-1]
    return lg.actual_game_pos, lg.actual_game_v


class LaunchTable:
    """(angle, draw) -> measured game release (x, y, vx, vy), bilinear."""

    def __init__(self, path=TABLE_PATH):
        z = np.load(path)
        self.angles, self.draws, self.data = z["angles"], z["draws"], z["data"]

    def __call__(self, angle: float, draw: float) -> tuple[tuple, tuple]:
        a = np.clip(angle, self.angles[0], self.angles[-1])
        r = np.clip(draw, self.draws[0], self.draws[-1])
        i = min(np.searchsorted(self.angles, a, side="right") - 1, len(self.angles) - 2)
        j = min(np.searchsorted(self.draws, r, side="right") - 1, len(self.draws) - 2)
        ta = (a - self.angles[i]) / (self.angles[i + 1] - self.angles[i])
        tr = (r - self.draws[j]) / (self.draws[j + 1] - self.draws[j])
        v = ((1 - ta) * (1 - tr) * self.data[i, j] + ta * (1 - tr) * self.data[i + 1, j]
             + (1 - ta) * tr * self.data[i, j + 1] + ta * tr * self.data[i + 1, j + 1])
        return (float(v[0]), float(v[1])), (float(v[2]), float(v[3]))


def _launch_for(angle, draw, table):
    if table is not None:
        return table(angle, draw)
    L = TabletopSlingshot().launch(angle, draw)
    return ((L.position[0] / LAMBDA, L.position[1] / LAMBDA),
            (L.velocity[0] / SQRT_LAMBDA, L.velocity[1] / SQRT_LAMBDA))


def _score_shot(args):
    lvl_xml, poses, angle, draw, use_table = args
    lvl = Level.from_xml(lvl_xml)
    table = LaunchTable() if use_table else None
    pos, vel = _launch_for(angle, draw, table)
    gw = GameWorld(lvl, poses=poses)
    r = gw.shoot(pos, vel)
    return angle, draw, r.score, r.pigs_killed


def plan_shot(gw: GameWorld, angles=None, draws=None, workers: int | None = None) -> tuple[ShotPlan, int]:
    """Stand-in for the aiming policy: grid-search (angle, draw) on the current
    game state with Box2D. Launches come from the measured SO-101 launch table
    (scripts/calibrate_launch.py) if present, else the analytic model."""
    from concurrent.futures import ProcessPoolExecutor
    lvl, poses = level_from_state(gw)
    angles = np.arange(-5.0, 75.1, 2.5) if angles is None else angles
    draws = np.linspace(0.45, 1.0, 12) if draws is None else draws
    xml = lvl.to_xml()
    use_table = TABLE_PATH.exists()
    jobs = [(xml, poses, float(a), float(dr), use_table) for a in angles for dr in draws]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        results = list(ex.map(_score_shot, jobs, chunksize=16))
    # best score; ties -> most pigs, then the flattest shot
    a, dr, sc, pk = max(results, key=lambda r: (r[2], r[3], -abs(r[0] - 30)))
    return ShotPlan(a, dr), sc
