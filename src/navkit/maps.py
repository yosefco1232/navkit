"""Loading maps from simple ASCII files.

Legend::

    #   obstacle
    .   free space
    S   start (free)
    G   goal (free)

The first line of the file is the *top* of the map.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from navkit.geometry import Point, Pose
from navkit.grid_map import OccupancyGrid

_OBSTACLE = "#"
_FREE = ".SG"


class MapFormatError(ValueError):
    pass


@dataclass(frozen=True)
class MapSpec:
    grid: OccupancyGrid
    start: Pose | None = None
    goal: Point | None = None


def parse_ascii_map(text: str, resolution: float) -> MapSpec:
    lines = [line.rstrip() for line in text.strip().splitlines() if line.strip()]
    if not lines:
        raise MapFormatError("map is empty")
    width = len(lines[0])
    if any(len(line) != width for line in lines):
        raise MapFormatError("all map rows must have the same length")

    rows = len(lines)
    occupied = np.zeros((rows, width), dtype=bool)
    start_cell: tuple[int, int] | None = None
    goal_cell: tuple[int, int] | None = None

    for i, line in enumerate(lines):
        row = rows - 1 - i  # flip: first text line is the top of the map
        for col, ch in enumerate(line):
            if ch == _OBSTACLE:
                occupied[row, col] = True
            elif ch not in _FREE:
                raise MapFormatError(f"unknown character {ch!r} at line {i + 1}, col {col + 1}")
            if ch == "S":
                start_cell = (row, col)
            elif ch == "G":
                goal_cell = (row, col)

    grid = OccupancyGrid(occupied, resolution)
    start = None
    if start_cell is not None:
        p = grid.cell_to_world(start_cell)
        start = Pose(p.x, p.y, 0.0)
    goal = grid.cell_to_world(goal_cell) if goal_cell is not None else None
    return MapSpec(grid=grid, start=start, goal=goal)


def load_ascii_map(path: str | Path, resolution: float = 0.1) -> MapSpec:
    return parse_ascii_map(Path(path).read_text(encoding="utf-8"), resolution)
