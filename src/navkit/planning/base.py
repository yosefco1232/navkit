"""Planner interface. Any global planner (A*, Dijkstra, RRT, ...) implements this."""

from __future__ import annotations

from typing import Protocol

from navkit.geometry import Point
from navkit.grid_map import OccupancyGrid


class PlanningError(Exception):
    """Raised when a planning query is invalid (e.g. start inside an obstacle)."""


class Planner(Protocol):
    def plan(self, grid: OccupancyGrid, start: Point, goal: Point) -> list[Point] | None:
        """Return a collision-free path from ``start`` to ``goal``, or ``None`` if none exists."""
        ...
