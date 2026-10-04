import itertools
import math

import numpy as np
import pytest

from navkit.geometry import Point
from navkit.grid_map import OccupancyGrid
from navkit.planning import AStarPlanner, PlanningError


def _is_connected(cells: list[tuple[int, int]]) -> bool:
    return all(max(abs(a[0] - b[0]), abs(a[1] - b[1])) == 1 for a, b in itertools.pairwise(cells))


def test_straight_line_in_empty_grid() -> None:
    grid = OccupancyGrid.empty(10, 10, 1.0)
    planner = AStarPlanner()
    cells = planner.plan_cells(grid, (0, 0), (0, 9))
    assert cells is not None
    assert cells[0] == (0, 0) and cells[-1] == (0, 9)
    assert planner.last_stats.cost == pytest.approx(9.0)


def test_diagonal_cost_is_optimal() -> None:
    grid = OccupancyGrid.empty(10, 10, 1.0)
    planner = AStarPlanner()
    planner.plan_cells(grid, (0, 0), (4, 6))
    assert planner.last_stats.cost == pytest.approx(4 * math.sqrt(2) + 2)


def test_four_connected_uses_manhattan_moves() -> None:
    grid = OccupancyGrid.empty(10, 10, 1.0)
    planner = AStarPlanner(allow_diagonal=False)
    cells = planner.plan_cells(grid, (0, 0), (3, 3))
    assert cells is not None and len(cells) == 7
    assert planner.last_stats.cost == pytest.approx(6.0)


def test_detours_around_wall() -> None:
    occ = np.zeros((10, 10), dtype=bool)
    occ[0:8, 5] = True  # wall with a gap at the top
    grid = OccupancyGrid(occ, 1.0)
    cells = AStarPlanner().plan_cells(grid, (0, 0), (0, 9))
    assert cells is not None
    assert _is_connected(cells)
    assert all(grid.is_free(c) for c in cells)
    assert any(r >= 8 for r, _ in cells)


def test_returns_none_when_goal_unreachable() -> None:
    occ = np.zeros((10, 10), dtype=bool)
    occ[:, 5] = True
    assert AStarPlanner().plan_cells(OccupancyGrid(occ, 1.0), (0, 0), (0, 9)) is None


def test_no_corner_cutting() -> None:
    occ = np.zeros((3, 3), dtype=bool)
    occ[0, 1] = True
    occ[1, 0] = True
    # Only a diagonal squeeze between two obstacles connects (0,0) to (1,1).
    assert AStarPlanner().plan_cells(OccupancyGrid(occ, 1.0), (0, 0), (1, 1)) is None


def test_blocked_start_raises() -> None:
    occ = np.zeros((5, 5), dtype=bool)
    occ[0, 0] = True
    with pytest.raises(PlanningError):
        AStarPlanner().plan_cells(OccupancyGrid(occ, 1.0), (0, 0), (4, 4))


def test_world_plan_keeps_exact_endpoints() -> None:
    grid = OccupancyGrid.empty(20, 20, 0.1)
    start, goal = Point(0.12, 0.13), Point(1.71, 1.52)
    path = AStarPlanner().plan(grid, start, goal)
    assert path is not None
    assert path[0] == start and path[-1] == goal


def test_matches_dijkstra_on_random_maps() -> None:
    """A* with an admissible heuristic must find the same optimal cost as uniform-cost search."""
    rng = np.random.default_rng(42)
    for _ in range(20):
        occ = rng.random((25, 25)) < 0.25
        occ[0, 0] = occ[24, 24] = False
        grid = OccupancyGrid(occ, 1.0)
        astar = AStarPlanner()
        dijkstra = AStarPlanner()
        dijkstra._h = lambda a, b: 0.0  # type: ignore[method-assign]
        a = astar.plan_cells(grid, (0, 0), (24, 24))
        d = dijkstra.plan_cells(grid, (0, 0), (24, 24))
        assert (a is None) == (d is None)
        if a is not None:
            assert astar.last_stats.cost == pytest.approx(dijkstra.last_stats.cost)
            assert astar.last_stats.expanded <= dijkstra.last_stats.expanded
