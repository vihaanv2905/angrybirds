"""Matplotlib rendering: level previews, shot frames, and MP4 videos."""
from __future__ import annotations

import math
import shutil
import subprocess
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Circle, Polygon  # noqa: E402

from .config import RED_BIRD  # noqa: E402

COLORS = {
    "wood": ("#d9b77e", "#8a6a3a"),
    "ice": ("#cfeaf7", "#6aa6c4"),
    "stone": ("#a9a9a9", "#5e5e5e"),
    "pig": ("#7cc644", "#3d7a1c"),
    "bird": ("#d8262e", "#7a1015"),
    "hill": ("#c9b48a", "#8c7a55"),
}
SKY, GROUND = "#eef6fb", "#9bc46b"


def _ent_patch(ent, x, y, angle):
    if ent.kind == "pig":
        fc, ec = COLORS["pig"]
    else:
        fc, ec = COLORS[ent.material]
    if ent.shape[0] == "circle":
        return Circle((x, y), ent.shape[1], fc=fc, ec=ec, lw=1.0, zorder=3)
    if ent.shape[0] == "box":
        _, w, h = ent.shape
        local = [(-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2)]
    else:
        local = ent.shape[1]
    c, s = math.cos(angle), math.sin(angle)
    pts = [(x + c * px - s * py, y + s * px + c * py) for px, py in local]
    return Polygon(pts, closed=True, fc=fc, ec=ec, lw=1.0, zorder=2)


def _draw_slingshot(ax, rest=(0.0, 4.5)):
    x, y = rest
    kw = dict(color="#6b4423", lw=5, solid_capstyle="round", zorder=1)
    ax.plot([x, x], [0, y - 1.4], **kw)
    ax.plot([x, x - 0.5], [y - 1.4, y + 0.3], **kw)
    ax.plot([x, x + 0.5], [y - 1.4, y + 0.3], **kw)


def draw(gw, snapshot=None, ax=None, trajectory=None, title=None, xlim=(-3, 50), ylim=(-0.5, 16),
         marks=None):
    """Draw a GameWorld. `snapshot` = list of (eid, x, y, angle) from
    GameWorld.snapshot(); defaults to the current state."""
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 4.6))
    ax.set_facecolor(SKY)
    ax.axhspan(ylim[0], 0, color=GROUND, zorder=0)
    for h in gw.level.hills:
        fc, ec = COLORS["hill"]
        ax.add_patch(Polygon(h.vertices, closed=True, fc=fc, ec=ec, zorder=1))
    _draw_slingshot(ax)
    snap = snapshot if snapshot is not None else gw.snapshot()
    by_id = {e.eid: e for e in gw.entities}
    for eid, x, y, ang in snap:
        if eid == -1:
            fc, ec = COLORS["bird"]
            ax.add_patch(Circle((x, y), RED_BIRD.radius, fc=fc, ec=ec, lw=1.2, zorder=4))
        else:
            ax.add_patch(_ent_patch(by_id[eid], x, y, ang))
    if trajectory:
        xs, ys = zip(*trajectory)
        ax.plot(xs, ys, ls="--", lw=1, color=COLORS["bird"][1], zorder=1)
    for (mx, my, label) in marks or []:
        ax.plot(mx, my, "x", color="k")
        ax.annotate(label, (mx, my), textcoords="offset points", xytext=(4, 4), fontsize=8)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    if title:
        ax.set_title(title, fontsize=10)
    return ax


def save_level_png(gw, path, **kw):
    fig, ax = plt.subplots(figsize=(12, 4.6))
    draw(gw, ax=ax, **kw)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def save_shot_video(gw_factory, launch, path, fps=30, title="", xlim=(-3, 50), ylim=(-0.5, 16)):
    """Re-simulate a shot on a fresh world from `gw_factory()` and write an MP4
    (needs ffmpeg) or, without ffmpeg, a strip of PNG keyframes."""
    gw = gw_factory()
    every = max(1, int(round(gw.cfg.hz / fps)))
    res = gw.shoot(launch.position, launch.velocity, record_every=every)
    if res.trajectory:                       # fit high lobs in frame
        top = max(y for _, y in res.trajectory) + 1.0
        ylim = (ylim[0], max(ylim[1], top))
    path = Path(path)
    with tempfile.TemporaryDirectory() as tmp:
        traj = []
        for i, (t, snap) in enumerate(res.frames):
            bird = [s for s in snap if s[0] == -1]
            if bird:
                traj.append((bird[0][1], bird[0][2]))
            fig, ax = plt.subplots(figsize=(12, 4.6))
            draw(gw, snapshot=snap, ax=ax, trajectory=list(traj), xlim=xlim, ylim=ylim,
                 title=f"{title}   t={t:4.2f}s")
            fig.tight_layout()
            fig.savefig(f"{tmp}/f_{i:04d}.png", dpi=90)
            plt.close(fig)
        if shutil.which("ffmpeg"):
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-framerate", str(fps), "-i",
                            f"{tmp}/f_%04d.png", "-pix_fmt", "yuv420p",
                            "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", str(path)], check=True)
        else:
            shutil.copy(f"{tmp}/f_{len(res.frames) - 1:04d}.png", path.with_suffix(".png"))
    return res
