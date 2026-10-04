"""Occupancy grid map.

Convention: cell ``(row, col)`` covers the square
``[col * res, (col + 1) * res) x [row * res, (row + 1) * res)`` in world meters,
so row 0 is the bottom of the map (y = 0) and rows grow upward.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from navkit.geometry import Point

Cell = tuple[int, int]


class OccupancyGrid:
    """Binary occupancy grid. ``True`` means the cell is occupied."""

    def __init__(self, occupied: NDArray[np.bool_], resolution: float) -> None:
        if occupied.ndim != 2:
            raise ValueError("occupancy array must be 2D")
        if resolution <= 0:
            raise ValueError("resolution must be positive")
        self._occupied = occupied.astype(bool, copy=True)
        self._occupied.setflags(write=False)
        self.resolution = resolution

    @classmethod
    def empty(cls, rows: int, cols: int, resolution: float) -> OccupancyGrid:
        return cls(np.zeros((rows, cols), dtype=bool), resolution)

    # -- shape -------------------------------------------------------------
    @property
    def occupied(self) -> NDArray[np.bool_]:
        """Read-only view of the occupancy array, indexed ``[row, col]``."""
        return self._occupied

    @property
    def rows(self) -> int:
        return int(self._occupied.shape[0])

    @property
    def cols(self) -> int:
        return int(self._occupied.shape[1])

    @property
    def width(self) -> float:
        return self.cols * self.resolution

    @property
    def height(self) -> float:
        return self.rows * self.resolution

    # -- coordinate transforms ---------------------------------------------
    def world_to_cell(self, p: Point) -> Cell:
        return math.floor(p.y / self.resolution), math.floor(p.x / self.resolution)

    def cell_to_world(self, cell: Cell) -> Point:
        """Center of the cell in world coordinates."""
        row, col = cell
        return Point((col + 0.5) * self.resolution, (row + 0.5) * self.resolution)

    # -- queries -----------------------------------------------------------
    def in_bounds(self, cell: Cell) -> bool:
        row, col = cell
        return 0 <= row < self.rows and 0 <= col < self.cols

    def is_free(self, cell: Cell) -> bool:
        """Out-of-bounds cells are treated as occupied."""
        return self.in_bounds(cell) and not self._occupied[cell]

    def is_free_world(self, p: Point) -> bool:
        return self.is_free(self.world_to_cell(p))

    # -- operations --------------------------------------------------------
    def inflate(self, radius: float) -> OccupancyGrid:
        """Return a new grid with obstacles grown by ``radius`` meters.

        Planning on the inflated grid lets us treat a round robot as a point.
        """
        r = math.ceil(radius / self.resolution)
        if r <= 0:
            return OccupancyGrid(self._occupied, self.resolution)
        h, w = self._occupied.shape
        padded = np.pad(self._occupied, r, constant_values=False)
        out = self._occupied.copy()
        for dr in range(-r, r + 1):
            for dc in range(-r, r + 1):
                if dr * dr + dc * dc <= r * r:
                    out |= padded[r + dr : r + dr + h, r + dc : r + dc + w]
        return OccupancyGrid(out, self.resolution)

    def __repr__(self) -> str:
        return f"OccupancyGrid(rows={self.rows}, cols={self.cols}, resolution={self.resolution})"
