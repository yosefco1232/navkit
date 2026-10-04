import itertools
from pathlib import Path

import numpy as np
import pytest

from navkit.geometry import Point, path_length
from navkit.grid_map import OccupancyGrid
from navkit.maps import load_ascii_map
from navkit.planning import AStarPlanner, SmoothedPlanner, densify, line_of_sight, shortcut_indices
from navkit.planning.smoothing import bresenham

MAPS = Path(__file__).resolve().parents[1] / "maps"


@pytest.mark.parametrize(("a", "b"), [((0, 0), (0, 5)), ((0, 0), (5, 5)), ((2, 7), (6, 1)), ((3, 3), (3, 3))])
def test_bresenham_is_connected_and_hits_endpoints(a: tuple[int, int], b: tuple[int, int]) -> None:
    cells = list(bresenham(a, b))
    assert cells[0] == a and cells[-1] == b
    for p, q in itertools.pairwise(cells):
        assert max(abs(p[0] - q[0]), abs(p[1] - q[1])) == 1
    # A Bresenham line has exactly max(|dr|, |dc|) + 1 cells.
    assert len(cells) == max(abs(a[0] - b[0]), abs(a[1] - b[1])) + 1


def test_line_of_sight_blocked_by_wall() -> None:
    occ = np.zeros((10, 10), dtype=bool)
    occ[0:6, 5] = True
    grid = OccupancyGrid(occ, 1.0)
    assert not line_of_sight(grid, (2, 0), (2, 9))
    assert line_of_sight(grid, (8, 0), (8, 9))


def test_line_of_sight_cannot_squeeze_through_corner() -> None:
    occ = np.zeros((3, 3), dtype=bool)
    occ[0, 1] = occ[1, 0] = True
    assert not line_of_sight(OccupancyGrid(occ, 1.0), (0, 0), (1, 1))


def test_shortcut_on_empty_grid_keeps_only_endpoints() -> None:
    grid = OccupancyGrid.empty(20, 20, 1.0)
    cells = AStarPlanner().plan_cells(grid, (0, 0), (15, 7))
    assert cells is not None
    assert shortcut_indices(grid, cells) == [0, len(cells) - 1]


def test_shortcut_segments_are_all_visible() -> None:
    rng = np.random.default_rng(7)
    occ = rng.random((30, 30)) < 0.2
    occ[0, 0] = occ[29, 29] = False
    grid = OccupancyGrid(occ, 1.0)
    cells = AStarPlanner().plan_cells(grid, (0, 0), (29, 29))
    if cells is None:
        pytest.skip("random map happened to be blocked")
    keep = shortcut_indices(grid, cells)
    assert keep[0] == 0 and keep[-1] == len(cells) - 1
    for i, j in itertools.pairwise(keep):
        assert line_of_sight(grid, cells[i], cells[j])


def test_densify_respects_spacing_and_keeps_vertices() -> None:
    pts = [Point(0, 0), Point(1, 0), Point(1, 0.35)]
    dense = densify(pts, 0.1)
    assert dense[0] == pts[0] and dense[-1] == pts[-1]
    assert Point(1, 0) in dense
    assert all(a.distance_to(b) <= 0.1 + 1e-9 for a, b in itertools.pairwise(dense))
    assert path_length(dense) == pytest.approx(path_length(pts))
    with pytest.raises(ValueError):
        densify(pts, 0.0)


def test_smoothed_planner_is_shorter_and_collision_free() -> None:
    spec = load_ascii_map(MAPS / "warehouse.txt")
    assert spec.start is not None and spec.goal is not None
    grid = spec.grid.inflate(0.2)
    raw = AStarPlanner().plan(grid, spec.start.position, spec.goal)
    smooth = SmoothedPlanner(AStarPlanner(), spacing=0.1).plan(grid, spec.start.position, spec.goal)
    assert raw is not None and smooth is not None
    assert smooth[0] == raw[0] and smooth[-1] == raw[-1]
    assert path_length(smooth) <= path_length(raw)
    assert all(grid.is_free_world(p) for p in smooth)
