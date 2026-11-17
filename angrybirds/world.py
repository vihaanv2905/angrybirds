"""Box2D game world: builds a level, fires a bird, applies damage, scores.

Damage model (the original game's exact rules are not public; this copies its
behaviour): after every physics step Box2D reports each contact's normal
impulse (PostSolve). Each block or pig takes
    damage = max(0, impulse - threshold_force * dt)
so resting weight does no harm, while impacts (from the bird, falling
blocks, the ground) do. At hp <= 0 the body is removed and points are awarded.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from Box2D import b2ContactListener, b2EdgeShape, b2PolygonShape, b2World

from .config import (BLOCK_POINTS, BLOCK_TYPES, CIRCLE_BLOCKS, GRAVITY, GROUND_FRICTION,
                     MATERIALS, PIG_POINTS, PIG_TYPES, RED_BIRD, TRIANGLE_BLOCKS,
                     WORLD_X_MAX, WORLD_X_MIN, WORLD_Y_MIN, SimConfig)
from .levels import Level


@dataclass(eq=False)
class Entity:
    eid: int
    kind: str            # "block" | "pig" | "bird"
    type: str
    material: str | None
    hp: float
    max_hp: float
    threshold: float     # N
    shape: tuple         # ("box", w, h) | ("circle", r) | ("poly", verts)
    per_mass: float = 0.0  # if > 0, damage is impulse * per_mass (delta-v, m/s)
    body: object = None
    alive: bool = True
    pending: float = 0.0


@dataclass
class ScoreEvent:
    time: float
    kind: str
    type: str
    material: str | None
    points: int
    x: float
    y: float


@dataclass
class ShotResult:
    score: int                       # points scored by this shot
    events: list[ScoreEvent]
    pigs_killed: int
    blocks_destroyed: int
    pigs_left: int
    trajectory: list[tuple[float, float]]
    min_pig_distance: float          # closest bird-to-pig surface distance (m)
    first_impact: tuple[float, float] | None
    sim_time: float
    frames: list = field(default_factory=list)


class _DamageListener(b2ContactListener):
    def __init__(self, gw: "GameWorld"):
        b2ContactListener.__init__(self)
        self.gw = gw

    def PostSolve(self, contact, impulse):
        n = contact.manifold.pointCount
        if n == 0:
            return
        total = sum(impulse.normalImpulses[:n])
        if total <= 0.0:
            return
        a = contact.fixtureA.body.userData
        b = contact.fixtureB.body.userData
        gw = self.gw
        for ent, other in ((a, b), (b, a)):
            if ent is not None and ent.kind != "bird" and ent.alive:
                dmg = total - ent.threshold * gw.dt
                if dmg > 0.0:
                    ent.pending += dmg * ent.per_mass if ent.per_mass else dmg
        if gw._bird is not None and gw._first_impact is None:
            if (a is not None and a.kind == "bird") or (b is not None and b.kind == "bird"):
                p = gw._bird.body.position
                gw._first_impact = (p.x, p.y)


class GameWorld:
    def __init__(self, level: Level, cfg: SimConfig | None = None, settle: bool = True,
                 poses: list[tuple[float, float, float]] | None = None):
        """Build `level`. If `poses` (from `settled_poses()`) is given, bodies
        start at those settled poses, asleep, and no settling is simulated:
        identical start state, and much faster for repeated shots."""
        self.level = level
        self.cfg = cfg or SimConfig()
        self.dt = 1.0 / self.cfg.hz
        self.time = 0.0
        self.score = 0
        self.birds_left = level.birds
        self.world = b2World(gravity=(0.0, -GRAVITY), doSleep=True)
        self._listener = _DamageListener(self)
        self.world.contactListener = self._listener
        self.entities: list[Entity] = []
        self._bird: Entity | None = None
        self._first_impact = None
        self._build()
        if poses is not None:
            for ent, (x, y, a) in zip(self.entities, poses):
                ent.body.position = (x, y)
                ent.body.angle = a
            self.settle(0.0)
        elif settle:
            self.settle()

    @staticmethod
    def settled_poses(level: Level, cfg: SimConfig | None = None) -> list[tuple[float, float, float]]:
        gw = GameWorld(level, cfg)
        return [(e.body.position.x, e.body.position.y, e.body.angle) for e in gw.entities]

    # ---------------------------------------------------------------- build
    def _build(self) -> None:
        ground = self.world.CreateStaticBody(
            position=(0, 0), shapes=b2EdgeShape(vertices=[(WORLD_X_MIN, 0.0), (WORLD_X_MAX, 0.0)]))
        ground.fixtures[0].friction = GROUND_FRICTION
        for hill in self.level.hills:
            body = self.world.CreateStaticBody(position=(0, 0),
                                               shapes=b2PolygonShape(vertices=hill.vertices))
            body.fixtures[0].friction = GROUND_FRICTION

        for spec in self.level.blocks:
            mat = MATERIALS[spec.material]
            body = self.world.CreateDynamicBody(position=(spec.x, spec.y),
                                                angle=math.radians(spec.rotation))
            kw = dict(density=mat.density, friction=mat.friction, restitution=mat.restitution)
            if spec.type in BLOCK_TYPES:
                w, h = BLOCK_TYPES[spec.type]
                body.CreatePolygonFixture(box=(w / 2, h / 2), **kw)
                shape = ("box", w, h)
            elif spec.type in CIRCLE_BLOCKS:
                r = CIRCLE_BLOCKS[spec.type]
                body.CreateCircleFixture(radius=r, **kw)
                shape = ("circle", r)
            elif spec.type in TRIANGLE_BLOCKS:
                w, h = TRIANGLE_BLOCKS[spec.type]
                verts = [(-w / 3, -h / 3), (2 * w / 3, -h / 3), (-w / 3, 2 * h / 3)]
                body.CreatePolygonFixture(vertices=verts, **kw)
                shape = ("poly", verts)
            else:
                raise KeyError(f"unknown block type {spec.type}")
            # hit points scale with block size relative to a RectSmall-sized
            # block, so big planks are tougher than tiny squares
            area = body.mass / mat.density
            hp = mat.hp * max(0.5, min(2.0, area / 0.37))
            self._add(body, Entity(len(self.entities), "block", spec.type, spec.material,
                                   hp, hp, mat.threshold, shape))

        for spec in self.level.pigs:
            pt = PIG_TYPES[spec.type]
            body = self.world.CreateDynamicBody(position=(spec.x, spec.y), angularDamping=1.0)
            body.CreateCircleFixture(radius=pt.radius, density=pt.density,
                                     friction=pt.friction, restitution=pt.restitution)
            m = body.mass
            self._add(body, Entity(len(self.entities), "pig", spec.type, None,
                                   pt.hp, pt.hp, pt.threshold_accel * m, ("circle", pt.radius),
                                   per_mass=1.0 / m))

    def _add(self, body, ent: Entity) -> None:
        ent.body = body
        body.userData = ent
        self.entities.append(ent)

    # ------------------------------------------------------------ simulation
    def _step(self) -> list[ScoreEvent]:
        if self._bird is not None and self.cfg.drag:
            b = self._bird.body
            v = b.linearVelocity
            speed = v.length
            if speed > 0:
                k = self.cfg.drag_k * b.mass * speed
                b.ApplyForceToCenter((-k * v.x, -k * v.y), True)
        self.world.Step(self.dt, self.cfg.velocity_iters, self.cfg.position_iters)
        self.time += self.dt
        events = []
        for ent in self.entities:
            if not ent.alive:
                continue
            if ent.pending > 0.0:
                ent.hp -= ent.pending
                ent.pending = 0.0
            p = ent.body.position
            fell = p.y < WORLD_Y_MIN or p.x < WORLD_X_MIN or p.x > WORLD_X_MAX
            if ent.hp <= 0.0 or fell:
                events.append(self._destroy(ent))
        return events

    def _destroy(self, ent: Entity) -> ScoreEvent:
        p = ent.body.position
        pts = PIG_POINTS if ent.kind == "pig" else BLOCK_POINTS
        ev = ScoreEvent(self.time, ent.kind, ent.type, ent.material, pts, p.x, p.y)
        ent.alive = False
        ent.hp = min(ent.hp, 0.0)
        self.world.DestroyBody(ent.body)
        ent.body = None
        self.score += pts
        return ev

    def _all_quiet(self) -> bool:
        qs = self.cfg.quiet_speed
        for ent in self.entities:
            if ent.alive and ent.body.awake:
                b = ent.body
                if b.linearVelocity.length > qs or abs(b.angularVelocity) > 4 * qs:
                    return False
        if self._bird is not None and self._bird.body.linearVelocity.length > qs * 4:
            return False
        return True

    def settle(self, seconds: float | None = None) -> None:
        """Let the structure settle, then put it to sleep. Settling damage is
        undone (points too), so a level always starts with full hit points."""
        n = int((seconds if seconds is not None else self.cfg.settle_time) / self.dt)
        for _ in range(n):
            self.world.Step(self.dt, self.cfg.velocity_iters, self.cfg.position_iters)
        for ent in self.entities:
            ent.pending = 0.0
            if ent.body is not None:
                ent.body.linearVelocity = (0, 0)
                ent.body.angularVelocity = 0
                ent.body.awake = False
        self.time = 0.0

    # ---------------------------------------------------------------- query
    def alive(self, kind: str) -> list[Entity]:
        return [e for e in self.entities if e.alive and e.kind == kind]

    @property
    def pigs_left(self) -> int:
        return sum(1 for e in self.entities if e.alive and e.kind == "pig")

    def snapshot(self) -> list[tuple]:
        """(eid, x, y, angle) for every live body; bird has eid -1."""
        out = [(e.eid, e.body.position.x, e.body.position.y, e.body.angle)
               for e in self.entities if e.alive]
        if self._bird is not None:
            p = self._bird.body.position
            out.append((-1, p.x, p.y, self._bird.body.angle))
        return out

    # ----------------------------------------------------------- validation
    def stability_report(self, seconds: float = 5.0) -> dict:
        """Wake everything, simulate with no bird, and report how far anything
        moved, how much damage it took and whether anything broke."""
        start = {e.eid: (e.body.position.x, e.body.position.y, e.body.angle)
                 for e in self.entities if e.alive}
        hp0 = {e.eid: e.hp for e in self.entities}
        for e in self.entities:
            if e.alive:
                e.body.awake = True
        score0 = self.score
        for _ in range(int(seconds / self.dt)):
            self._step()
        moved, rot = 0.0, 0.0
        for e in self.entities:
            if e.alive:
                x0, y0, a0 = start[e.eid]
                p = e.body.position
                moved = max(moved, math.hypot(p.x - x0, p.y - y0))
                rot = max(rot, abs(e.body.angle - a0))
        return {
            "max_move": moved,
            "max_rotation_deg": math.degrees(rot),
            "broken": sum(1 for e in self.entities if not e.alive),
            "damage": sum(hp0[e.eid] - e.hp for e in self.entities if e.alive),
            "score": self.score - score0,
        }

    # ----------------------------------------------------------------- shoot
    def shoot(self, position, velocity, record_every: int = 0) -> ShotResult:
        """Fire one bird from `position` with `velocity` and simulate until
        everything comes to rest (or max_shot_time). The bird is removed after."""
        if self.birds_left <= 0:
            raise RuntimeError("no birds left")
        self.birds_left -= 1
        t0, score0 = self.time, self.score
        bt = RED_BIRD
        body = self.world.CreateDynamicBody(position=tuple(position), bullet=True,
                                            angularDamping=bt.angular_damping)
        area = math.pi * bt.radius ** 2
        body.CreateCircleFixture(radius=bt.radius, density=bt.mass / area,
                                 friction=bt.friction, restitution=bt.restitution)
        body.linearVelocity = tuple(velocity)
        self._bird = Entity(-1, "bird", bt.name, None, 1, 1, 0, ("circle", bt.radius), body=body)
        body.userData = self._bird
        self._first_impact = None

        pigs0 = [(e.body.position.x, e.body.position.y, e.shape[1]) for e in self.alive("pig")]
        min_d = math.inf
        traj, frames, events = [], [], []
        quiet = 0.0
        check_every = max(1, int(0.05 / self.dt))
        n_max = int(self.cfg.max_shot_time / self.dt)
        for i in range(n_max):
            events += self._step()
            if self._bird is not None:
                p = self._bird.body.position
                traj.append((p.x, p.y))
                for (px, py, pr) in pigs0:
                    d = math.hypot(p.x - px, p.y - py) - pr - bt.radius
                    if d < min_d:
                        min_d = d
                if p.y < WORLD_Y_MIN or p.x > WORLD_X_MAX or p.x < WORLD_X_MIN:
                    self.world.DestroyBody(self._bird.body)
                    self._bird = None
            if record_every and i % record_every == 0:
                frames.append((self.time - t0, self.snapshot()))
            if i % check_every == 0 and i > 0:
                quiet = quiet + check_every * self.dt if self._all_quiet() else 0.0
                if quiet >= self.cfg.quiet_time:
                    break
            if self.pigs_left == 0 and self._bird is None:
                break
        if record_every:
            frames.append((self.time - t0, self.snapshot()))
        if self._bird is not None:
            self.world.DestroyBody(self._bird.body)
            self._bird = None
        pk = sum(1 for e in events if e.kind == "pig")
        return ShotResult(score=self.score - score0, events=events, pigs_killed=pk,
                          blocks_destroyed=len(events) - pk, pigs_left=self.pigs_left,
                          trajectory=traj, min_pig_distance=max(min_d, 0.0) if pigs0 else 0.0,
                          first_impact=self._first_impact, sim_time=self.time - t0,
                          frames=frames)
