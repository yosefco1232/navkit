"""Matplotlib visualization of maps, plans and trajectories."""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless-safe; works in CI

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from navkit.geometry import Point
from navkit.grid_map import OccupancyGrid
from navkit.simulation import SimResult


def draw_grid(ax: Axes, grid: OccupancyGrid) -> None:
    ax.imshow(
        grid.occupied,
        origin="lower",
        cmap="Greys",
        extent=(0.0, grid.width, 0.0, grid.height),
        interpolation="nearest",
    )
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")


def plot_result(grid: OccupancyGrid, result: SimResult, goal: Point, title: str = "") -> Figure:
    fig, ax = plt.subplots(figsize=(9, 6))
    draw_grid(ax, grid)
    if result.path:
        ax.plot([p.x for p in result.path], [p.y for p in result.path], "--", color="tab:blue", lw=1.5, label="A* plan")
    traj = result.trajectory
    ax.plot([p.x for p in traj], [p.y for p in traj], color="tab:orange", lw=2.5, label="robot trajectory")
    ax.plot(traj[0].x, traj[0].y, "o", color="tab:green", ms=10, label="start")
    ax.plot(goal.x, goal.y, "*", color="tab:red", ms=16, label="goal")
    ax.set_title(title or result.summary())
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8)
    fig.tight_layout()
    return fig


def save_animation(grid: OccupancyGrid, result: SimResult, goal: Point, out: str | Path, every: int = 4) -> None:
    """Write a GIF of the robot driving along its trajectory."""
    fig, ax = plt.subplots(figsize=(8, 5))
    draw_grid(ax, grid)
    if result.path:
        ax.plot([p.x for p in result.path], [p.y for p in result.path], "--", color="tab:blue", lw=1)
    ax.plot(goal.x, goal.y, "*", color="tab:red", ms=14)
    (trail,) = ax.plot([], [], color="tab:orange", lw=2)
    (body,) = ax.plot([], [], "o", color="tab:green", ms=8)
    (heading,) = ax.plot([], [], color="black", lw=1.5)
    frames = list(range(0, len(result.trajectory), every))

    def update(i: int) -> list[Artist]:
        poses = result.trajectory[: i + 1]
        p = poses[-1]
        trail.set_data([q.x for q in poses], [q.y for q in poses])
        body.set_data([p.x], [p.y])
        heading.set_data([p.x, p.x + 0.3 * math.cos(p.theta)], [p.y, p.y + 0.3 * math.sin(p.theta)])
        return [trail, body, heading]

    anim = FuncAnimation(fig, update, frames=frames, blit=True)
    anim.save(str(out), writer=PillowWriter(fps=20))
    plt.close(fig)
