import math

import numpy as np
import pytest

from navkit.geometry import Point, Pose, path_length, wrap_angle
from navkit.grid_map import OccupancyGrid
from navkit.maps import MapFormatError, parse_ascii_map


@pytest.mark.parametrize(
    ("angle", "expected"),
    [(0.0, 0.0), (math.pi / 2, math.pi / 2), (3 * math.pi, -math.pi), (-3 * math.pi / 2, math.pi / 2)],
)
def test_wrap_angle(angle: float, expected: float) -> None:
    assert wrap_angle(angle) == pytest.approx(expected)


def test_pose_to_local_frame() -> None:
    pose = Pose(1.0, 1.0, math.pi / 2)  # facing +y
    lx, ly = pose.to_local(Point(1.0, 2.0))
    assert (lx, ly) == pytest.approx((1.0, 0.0))
    lx, ly = pose.to_local(Point(0.0, 1.0))  # to the robot's left
    assert (lx, ly) == pytest.approx((0.0, 1.0))


def test_path_length() -> None:
    assert path_length([Point(0, 0), Point(3, 4), Point(3, 5)]) == pytest.approx(6.0)
    assert path_length([]) == 0.0


def test_world_cell_round_trip() -> None:
    grid = OccupancyGrid.empty(rows=10, cols=20, resolution=0.5)
    cell = grid.world_to_cell(Point(3.2, 1.1))
    assert cell == (2, 6)
    center = grid.cell_to_world(cell)
    assert grid.world_to_cell(center) == cell


def test_out_of_bounds_is_not_free() -> None:
    grid = OccupancyGrid.empty(5, 5, 1.0)
    assert grid.is_free((0, 0))
    assert not grid.is_free((-1, 0))
    assert not grid.is_free((0, 5))
    assert not grid.is_free_world(Point(-0.1, 2.0))


def test_grid_is_immutable() -> None:
    grid = OccupancyGrid.empty(3, 3, 1.0)
    with pytest.raises(ValueError):
        grid.occupied[0, 0] = True


def test_inflate_grows_obstacles_without_mutating_original() -> None:
    occ = np.zeros((11, 11), dtype=bool)
    occ[5, 5] = True
    grid = OccupancyGrid(occ, resolution=0.1)
    inflated = grid.inflate(0.2)  # 2 cells
    assert inflated.occupied.sum() == 13  # discrete disk of radius 2
    assert not inflated.is_free((5, 7))
    assert inflated.is_free((5, 8))
    assert not inflated.is_free((6, 6))
    assert grid.occupied.sum() == 1


def test_parse_ascii_map_orientation_and_markers() -> None:
    text = """
    ####
    #.G#
    #S.#
    ####
    """
    spec = parse_ascii_map("\n".join(line.strip() for line in text.splitlines()), resolution=1.0)
    assert spec.grid.rows == 4 and spec.grid.cols == 4
    # 'S' is on the second line from the bottom -> row 1
    assert spec.start is not None and (spec.start.x, spec.start.y) == (1.5, 1.5)
    assert spec.goal == Point(2.5, 2.5)
    assert not spec.grid.is_free((0, 0))


@pytest.mark.parametrize("bad", ["", "##\n#", "#x#"])
def test_parse_ascii_map_rejects_bad_input(bad: str) -> None:
    with pytest.raises(MapFormatError):
        parse_ascii_map(bad, 1.0)


def test_inflate_is_robust_to_floating_point_noise() -> None:
    occ = np.zeros((11, 11), dtype=bool)
    occ[5, 5] = True
    grid = OccupancyGrid(occ, resolution=0.1)
    # 0.2 + 0.1 is 0.30000000000000004 in floating point; it must still mean 3 cells.
    assert grid.inflate(0.2 + 0.1).occupied.sum() == grid.inflate(0.3).occupied.sum()
    assert not grid.inflate(0.2 + 0.1).is_free((5, 8))
    assert grid.inflate(0.2 + 0.1).is_free((5, 9))
