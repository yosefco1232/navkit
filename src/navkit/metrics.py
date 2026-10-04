"""Quantitative evaluation of a navigation run.

Computed after the run from the trajectory alone, so the numbers are independent of
whichever controller produced them and can be compared across controllers.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from navkit.geometry import Point, Pose
from navkit.grid_map import OccupancyGrid
from navkit.simulation import SimResult


@dataclass(frozen=True)
class TrackingMetrics:
    outcome: str
    time: float  # s
    planned_length: float  # m
    driven_length: float  # m
    rms_cross_track: float  # m
    max_cross_track: float  # m
    min_clearance: float  # m, from robot center to the nearest obstacle edge (collision if < radius)
    peak_lateral_accel: float  # m/s^2, |v * omega|; what tips cargo over or makes wheels slip

    def as_row(self) -> dict[str, str]:
        return {
            "outcome": self.outcome,
            "time [s]": f"{self.time:.2f}",
            "planned [m]": f"{self.planned_length:.2f}",
            "driven [m]": f"{self.driven_length:.2f}",
            "RMS CTE [cm]": f"{self.rms_cross_track * 100:.1f}",
            "max CTE [cm]": f"{self.max_cross_track * 100:.1f}",
            "min clearance [cm]": f"{self.min_clearance * 100:.1f}",
            "peak lat. accel [m/s²]": f"{self.peak_lateral_accel:.2f}",
        }


def _xy(points: list[Point] | list[Pose]) -> NDArray[np.float64]:
    return np.array([(p.x, p.y) for p in points], dtype=np.float64).reshape(-1, 2)


def distances_to_polyline(points: NDArray[np.float64], polyline: NDArray[np.float64]) -> NDArray[np.float64]:
    """Distance from each of N points to the closest of M-1 segments, vectorized as an (N, M-1) problem."""
    if len(polyline) == 1:
        return np.asarray(np.linalg.norm(points - polyline[0], axis=1), dtype=np.float64)
    a = polyline[:-1][None, :, :]  # (1, M-1, 2)
    ab = (polyline[1:] - polyline[:-1])[None, :, :]
    ap = points[:, None, :] - a  # (N, M-1, 2)
    seg_sq = np.maximum((ab * ab).sum(axis=2), 1e-12)
    t = np.clip((ap * ab).sum(axis=2) / seg_sq, 0.0, 1.0)
    closest = a + t[..., None] * ab
    d = np.linalg.norm(points[:, None, :] - closest, axis=2)
    return np.asarray(d.min(axis=1), dtype=np.float64)


def obstacle_clearance(grid: OccupancyGrid, points: NDArray[np.float64]) -> NDArray[np.float64]:
    """Exact distance from each point to the nearest occupied cell, treating cells as squares.

    Brute force over occupied cells, O(N * K). Fine for our map sizes; a distance
    transform would be the scalable choice for large maps.
    """
    occupied = np.argwhere(grid.occupied)  # (K, 2) as (row, col)
    if len(occupied) == 0:
        return np.full(len(points), np.inf)
    centers = (occupied[:, ::-1] + 0.5) * grid.resolution  # (K, 2) as (x, y)
    gap = np.maximum(np.abs(points[:, None, :] - centers[None, :, :]) - grid.resolution / 2.0, 0.0)
    return np.asarray(np.linalg.norm(gap, axis=2).min(axis=1), dtype=np.float64)


def lateral_acceleration(trajectory: list[Pose], dt: float) -> NDArray[np.float64]:
    """|v * omega| between consecutive poses, using finite differences."""
    if len(trajectory) < 2 or dt <= 0:
        return np.zeros(1)
    xy = _xy(trajectory)
    theta = np.unwrap(np.array([p.theta for p in trajectory]))
    v = np.linalg.norm(np.diff(xy, axis=0), axis=1) / dt
    omega = np.diff(theta) / dt
    return np.asarray(np.abs(v * omega), dtype=np.float64)


def evaluate(grid: OccupancyGrid, result: SimResult) -> TrackingMetrics:
    traj = _xy(result.trajectory)
    dt = result.elapsed / (len(result.trajectory) - 1) if len(result.trajectory) > 1 else 0.0
    if result.path:
        cte = distances_to_polyline(traj, _xy(result.path))
        rms, worst = float(np.sqrt(np.mean(cte**2))), float(cte.max())
    else:
        rms = worst = float("nan")
    return TrackingMetrics(
        outcome=result.outcome.value,
        time=result.elapsed,
        planned_length=result.planned_length,
        driven_length=result.driven_length,
        rms_cross_track=rms,
        max_cross_track=worst,
        min_clearance=float(obstacle_clearance(grid, traj).min()),
        peak_lateral_accel=float(lateral_acceleration(result.trajectory, dt).max()),
    )


def markdown_table(rows: dict[str, TrackingMetrics]) -> str:
    """Render named results as a GitHub-flavored Markdown table."""
    names = list(rows)
    headers = list(rows[names[0]].as_row())
    lines = ["| config | " + " | ".join(headers) + " |", "|---" * (len(headers) + 1) + "|"]
    for name in names:
        lines.append(f"| {name} | " + " | ".join(rows[name].as_row().values()) + " |")
    return "\n".join(lines)
