"""Post-processing for grid paths.

Grid planners can only move in 8 directions, so their paths zig-zag and turn more than
needed. *Line-of-sight shortcutting* removes every waypoint that the robot can skip
by driving straight, which gives shorter paths with fewer, gentler turns.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Iterator
from dataclasses import dataclass

from navkit.geometry import Point, interpolate
from navkit.grid_map import Cell, OccupancyGrid
from navkit.planning.base import Planner


def bresenham(a: Cell, b: Cell) -> Iterator[Cell]:
    """Cells on the straight line from ``a`` to ``b`` (inclusive), using integer arithmetic only."""
    r0, c0 = a
    r1, c1 = b
    dr, dc = abs(r1 - r0), abs(c1 - c0)
    step_r = 1 if r1 > r0 else -1
    step_c = 1 if c1 > c0 else -1
    err = dc - dr
    r, c = r0, c0
    while True:
        yield r, c
        if (r, c) == (r1, c1):
            return
        e2 = 2 * err
        if e2 > -dr:
            err -= dr
            c += step_c
        if e2 < dc:
            err += dc
            r += step_r


def line_of_sight(grid: OccupancyGrid, a: Cell, b: Cell) -> bool:
    """True if the straight line from ``a`` to ``b`` crosses only free cells.

    On a diagonal step both side cells must be free too, the same "no corner cutting"
    rule that A* uses, so a line can't slip between two obstacles that touch at a corner.
    """
    prev: Cell | None = None
    for cell in bresenham(a, b):
        if not grid.is_free(cell):
            return False
        if prev is not None and _cuts_corner(grid, prev, cell):
            return False
        prev = cell
    return True


def _cuts_corner(grid: OccupancyGrid, a: Cell, b: Cell) -> bool:
    """True if the step a -> b is diagonal and either cell beside it is occupied."""
    diagonal = a[0] != b[0] and a[1] != b[1]
    return diagonal and not (grid.is_free((a[0], b[1])) and grid.is_free((b[0], a[1])))


def shortcut_indices(grid: OccupancyGrid, cells: list[Cell]) -> list[int]:
    """Greedy shortcutting: from each kept waypoint, jump to the farthest visible one.

    Returns indices into ``cells``; the first and last are always kept. Worst case is
    O(n^2) line-of-sight checks. We scan backward from the end rather than binary
    searching, because visibility along a path is not monotonic.
    """
    n = len(cells)
    if n <= 2:
        return list(range(n))
    keep = [0]
    i = 0
    while i < n - 1:
        j = n - 1
        while j > i + 1 and not line_of_sight(grid, cells[i], cells[j]):
            j -= 1
        keep.append(j)
        i = j
    return keep


def densify(points: list[Point], spacing: float) -> list[Point]:
    """Resample a polyline so consecutive points are at most ``spacing`` meters apart."""
    if spacing <= 0:
        raise ValueError("spacing must be positive")
    if len(points) < 2:
        return list(points)
    out = [points[0]]
    for a, b in itertools.pairwise(points):
        n = max(1, math.ceil(a.distance_to(b) / spacing))
        out.extend(interpolate(a, b, k / n) for k in range(1, n + 1))
    return out


@dataclass
class SmoothedPlanner:
    """Decorator: wraps any :class:`Planner` and shortcuts + resamples its output.

    Because it implements the same ``plan`` method, the simulator can't tell the
    difference, and smoothing works for every grid planner we add later.
    """

    base: Planner
    spacing: float = 0.1

    def plan(self, grid: OccupancyGrid, start: Point, goal: Point) -> list[Point] | None:
        path = self.base.plan(grid, start, goal)
        if path is None or len(path) < 3:
            return path
        cells = [grid.world_to_cell(p) for p in path]
        waypoints = [path[i] for i in shortcut_indices(grid, cells)]
        return densify(waypoints, self.spacing)
