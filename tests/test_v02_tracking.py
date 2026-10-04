"""Tests for the v0.2 additions: geometry helpers, velocity profile, controller, motor lag, metrics."""

import itertools
import math
from pathlib import Path

import numpy as np
import pytest

from navkit.cli import baseline_setup, default_setup, main
from navkit.control import PurePursuitController
from navkit.control.velocity_profile import VelocityProfile, menger_curvature, path_curvature
from navkit.geometry import Point, Pose, circle_segment_intersections, project_onto_segment
from navkit.grid_map import OccupancyGrid
from navkit.maps import load_ascii_map
from navkit.metrics import distances_to_polyline, evaluate, lateral_acceleration, obstacle_clearance
from navkit.planning import densify
from navkit.robot import DifferentialDriveRobot, RobotLimits, Twist
from navkit.simulation import Outcome, SimResult, run_navigation

MAPS = Path(__file__).resolve().parents[1] / "maps"


# -- geometry ---------------------------------------------------------------
def test_project_onto_segment_clamps_to_ends() -> None:
    a, b = Point(0, 0), Point(2, 0)
    assert project_onto_segment(Point(1, 5), a, b) == (Point(1, 0), 0.5)
    assert project_onto_segment(Point(-3, 1), a, b) == (a, 0.0)
    assert project_onto_segment(Point(9, 1), a, b) == (b, 1.0)
    assert project_onto_segment(Point(1, 1), a, a) == (a, 0.0)


def test_circle_segment_intersections() -> None:
    a, b = Point(-2, 0), Point(2, 0)
    assert circle_segment_intersections(Point(0, 0), 1.0, a, b) == pytest.approx([0.25, 0.75])
    assert circle_segment_intersections(Point(0, 0), 1.0, Point(0, 0), b) == pytest.approx([0.5])
    assert circle_segment_intersections(Point(0, 5), 1.0, a, b) == []


# -- velocity profile -------------------------------------------------------
def test_menger_curvature_of_points_on_a_circle() -> None:
    r = 2.0
    pts = [Point(r * math.cos(t), r * math.sin(t)) for t in (0.1, 0.5, 1.2)]
    assert menger_curvature(*pts) == pytest.approx(1 / r)
    assert menger_curvature(Point(0, 0), Point(1, 0), Point(2, 0)) == 0.0


def test_profile_on_straight_line_brakes_to_zero() -> None:
    path = densify([Point(0, 0), Point(5, 0)], 0.1)
    prof = VelocityProfile.build(path, cruise_speed=1.0, max_lateral_accel=0.5, max_decel=0.5)
    assert prof.speed[0] == pytest.approx(1.0)
    assert prof.speed[-1] == 0.0
    # Braking constraint holds everywhere: v_i^2 <= v_{i+1}^2 + 2 a ds
    for i in range(len(path) - 1):
        ds = prof.s[i + 1] - prof.s[i]
        assert prof.speed[i] ** 2 <= prof.speed[i + 1] ** 2 + 2 * 0.5 * ds + 1e-9
    assert prof.at(-1) == prof.speed[0] and prof.at(99) == 0.0


def test_profile_slows_down_before_a_corner() -> None:
    path = densify([Point(0, 0), Point(4, 0), Point(4, 4)], 0.1)
    prof = VelocityProfile.build(path, cruise_speed=1.0, max_lateral_accel=0.5, max_decel=0.5)
    corner = prof.at(4.0)
    assert corner < 0.8
    # Braking from 1.0 m/s down to the corner speed at 0.5 m/s^2 starts ~0.9 m before the corner.
    assert prof.at(3.0) == pytest.approx(1.0)
    assert corner < prof.at(3.5) < 1.0
    assert max(path_curvature(path, 0.3)) > 1.0


# -- controller -------------------------------------------------------------
def test_lookahead_grows_with_speed_and_saturates() -> None:
    c = PurePursuitController(lookahead=0.3, lookahead_gain=0.5, max_lookahead=0.6)
    assert c.lookahead_for_speed(0.0) == pytest.approx(0.3)
    assert c.lookahead_for_speed(0.4) == pytest.approx(0.5)
    assert c.lookahead_for_speed(5.0) == pytest.approx(0.6)
    with pytest.raises(ValueError):
        PurePursuitController(lookahead=1.0, max_lookahead=0.5)


def test_cross_track_error_is_reported() -> None:
    c = PurePursuitController()
    c.reset(densify([Point(0, 0), Point(5, 0)], 0.1))
    c.compute(Pose(1.0, 0.25, 0.0))
    assert c.cross_track_error == pytest.approx(0.25)


def test_commands_slower_speed_near_a_corner() -> None:
    path = densify([Point(0, 0), Point(4, 0), Point(4, 4)], 0.1)
    straight, corner = PurePursuitController(), PurePursuitController()
    straight.reset(path)
    corner.reset(path)
    v_straight = straight.compute(Pose(1.0, 0.0, 0.0), speed=0.8).linear
    for x in np.arange(0.0, 3.9, 0.1):  # walk the second controller up to the corner
        cmd = corner.compute(Pose(float(x), 0.0, 0.0), speed=0.5)
    assert cmd.linear < v_straight


# -- robot ------------------------------------------------------------------
def test_motor_lag_is_first_order() -> None:
    tau, dt = 0.25, 0.05
    limits = RobotLimits(response_time=tau, max_linear_accel=100)
    robot = DifferentialDriveRobot(Pose(0, 0, 0), limits)
    robot.step(Twist(1.0, 0.0), dt)
    assert robot.velocity.linear == pytest.approx(1 - math.exp(-dt / tau))
    for _ in range(int(5 * tau / dt)):  # after 5 time constants we're within 1%
        robot.step(Twist(1.0, 0.0), dt)
    assert robot.velocity.linear == pytest.approx(1.0, abs=0.01)
    with pytest.raises(ValueError):
        RobotLimits(response_time=-1)


# -- grid clearance ---------------------------------------------------------
def test_clearance_and_footprint_collision() -> None:
    occ = np.zeros((20, 20), dtype=bool)
    occ[10, 10] = True  # square [1.0, 1.1] x [1.0, 1.1]
    grid = OccupancyGrid(occ, 0.1)
    assert grid.clearance(Point(0.7, 1.05), 1.0) == pytest.approx(0.3)
    assert grid.clearance(Point(0.7, 0.7), 1.0) == pytest.approx(math.hypot(0.3, 0.3))
    assert grid.clearance(Point(0.2, 0.2), 0.3) == pytest.approx(0.3)  # nothing within range
    assert grid.clearance(Point(-1, 0), 1.0) == 0.0
    assert grid.collides(Point(0.85, 1.05), 0.2)
    assert not grid.collides(Point(0.75, 1.05), 0.2)


# -- metrics ----------------------------------------------------------------
def test_distances_to_polyline() -> None:
    line = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 2.0]])
    pts = np.array([[1.0, 0.5], [3.0, 1.0], [-1.0, 0.0]])
    assert distances_to_polyline(pts, line) == pytest.approx([0.5, 1.0, 1.0])


def test_metrics_obstacle_clearance_matches_grid() -> None:
    spec = load_ascii_map(MAPS / "warehouse.txt")
    pts = [Point(1.5, 1.5), Point(5.0, 3.2), Point(9.1, 5.3)]
    vec = obstacle_clearance(spec.grid, np.array([(p.x, p.y) for p in pts]))
    for p, d in zip(pts, vec, strict=True):
        assert d == pytest.approx(spec.grid.clearance(p, 10.0))


def test_lateral_acceleration_on_circle_is_v2_over_r() -> None:
    v, r, dt = 0.5, 1.0, 0.01
    traj = [Pose(r * math.sin(v / r * k * dt), r - r * math.cos(v / r * k * dt), v / r * k * dt) for k in range(200)]
    assert lateral_acceleration(traj, dt) == pytest.approx(v**2 / r, rel=1e-3)


def test_evaluate_handles_no_path() -> None:
    m = evaluate(OccupancyGrid.empty(5, 5, 1.0), SimResult(Outcome.NO_PATH, [], [Pose(1, 1, 0)], 0.0))
    assert m.outcome == "no_path" and math.isnan(m.rms_cross_track)


# -- end-to-end regression: the whole point of v0.2 -------------------------
def test_v02_is_gentler_and_more_accurate_than_baseline() -> None:
    spec = load_ascii_map(MAPS / "warehouse.txt")
    assert spec.start is not None and spec.goal is not None
    results = {
        name: evaluate(spec.grid, run_navigation(spec.grid, spec.start, spec.goal, *setup()))
        for name, setup in {"baseline": baseline_setup, "v0.2": default_setup}.items()
    }
    base, new = results["baseline"], results["v0.2"]
    assert base.outcome == new.outcome == "reached"
    assert new.max_cross_track < base.max_cross_track
    assert new.peak_lateral_accel < 0.5 * base.peak_lateral_accel
    assert new.peak_lateral_accel < 0.6 * 1.1  # respects the configured limit (+10% for lag)
    assert new.planned_length < base.planned_length


def test_robot_is_stopped_when_reached() -> None:
    spec = load_ascii_map(MAPS / "warehouse.txt")
    assert spec.start is not None and spec.goal is not None
    result = run_navigation(spec.grid, spec.start, spec.goal, *default_setup())
    last = result.trajectory[-3:]
    assert all(a.position.distance_to(b.position) < 1e-3 for a, b in itertools.pairwise(last))


def test_cli_compare(tmp_path: Path) -> None:
    out = tmp_path / "cmp.png"
    assert main(["compare", str(MAPS / "warehouse.txt"), "--plot", str(out)]) == 0
    assert out.stat().st_size > 0
