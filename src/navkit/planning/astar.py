"""A* search on an occupancy grid (4- or 8-connected)."""

from __future__ import annotations

import heapq
import itertools
import math
from dataclasses import dataclass, field

from navkit.geometry import Point
from navkit.grid_map import Cell, OccupancyGrid
from navkit.planning.base import PlanningError

_SQRT2 = math.sqrt(2.0)
_ORTHOGONAL = ((1, 0), (-1, 0), (0, 1), (0, -1))
_DIAGONAL = ((1, 1), (1, -1), (-1, 1), (-1, -1))


@dataclass
class SearchStats:
    expanded: int = 0
    cost: float = math.inf


@dataclass
class AStarPlanner:
    """Grid A* with an admissible heuristic.

    With ``allow_diagonal`` the octile distance is used and diagonal moves are only
    allowed when both adjacent orthogonal cells are free (no corner cutting).
    """

    allow_diagonal: bool = True
    last_stats: SearchStats = field(default_factory=SearchStats, init=False)

    def plan(self, grid: OccupancyGrid, start: Point, goal: Point) -> list[Point] | None:
        cells = self.plan_cells(grid, grid.world_to_cell(start), grid.world_to_cell(goal))
        if cells is None:
            return None
        path = [grid.cell_to_world(c) for c in cells]
        # Replace the cell-center endpoints with the exact requested points.
        path[0], path[-1] = start, goal
        return path

    def plan_cells(self, grid: OccupancyGrid, start: Cell, goal: Cell) -> list[Cell] | None:
        if not grid.is_free(start):
            raise PlanningError(f"start cell {start} is not free")
        if not grid.is_free(goal):
            raise PlanningError(f"goal cell {goal} is not free")

        self.last_stats = SearchStats()
        tie = itertools.count()  # tie-breaker so the heap never compares cells
        open_heap: list[tuple[float, int, Cell]] = [(self._h(start, goal), next(tie), start)]
        g: dict[Cell, float] = {start: 0.0}
        parent: dict[Cell, Cell] = {}
        closed: set[Cell] = set()

        while open_heap:
            _, _, current = heapq.heappop(open_heap)
            if current in closed:
                continue  # stale heap entry
            if current == goal:
                self.last_stats.cost = g[current]
                return self._reconstruct(parent, current)
            closed.add(current)
            self.last_stats.expanded += 1

            for nxt, step in self._neighbors(grid, current):
                if nxt in closed:
                    continue
                candidate = g[current] + step
                if candidate < g.get(nxt, math.inf):
                    g[nxt] = candidate
                    parent[nxt] = current
                    heapq.heappush(open_heap, (candidate + self._h(nxt, goal), next(tie), nxt))
        return None

    # -- helpers -----------------------------------------------------------
    def _h(self, a: Cell, b: Cell) -> float:
        dr, dc = abs(a[0] - b[0]), abs(a[1] - b[1])
        if self.allow_diagonal:
            return (dr + dc) + (_SQRT2 - 2.0) * min(dr, dc)  # octile
        return float(dr + dc)  # manhattan

    def _neighbors(self, grid: OccupancyGrid, cell: Cell) -> list[tuple[Cell, float]]:
        r, c = cell
        result = [((r + dr, c + dc), 1.0) for dr, dc in _ORTHOGONAL if grid.is_free((r + dr, c + dc))]
        if self.allow_diagonal:
            for dr, dc in _DIAGONAL:
                if grid.is_free((r + dr, c + dc)) and grid.is_free((r + dr, c)) and grid.is_free((r, c + dc)):
                    result.append(((r + dr, c + dc), _SQRT2))
        return result

    @staticmethod
    def _reconstruct(parent: dict[Cell, Cell], end: Cell) -> list[Cell]:
        path = [end]
        while path[-1] in parent:
            path.append(parent[path[-1]])
        path.reverse()
        return path
