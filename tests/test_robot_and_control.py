import math

import pytest

from navkit.control import PurePursuitController
from navkit.geometry import Point, Pose
from navkit.robot import DifferentialDriveRobot, RobotLimits, Twist, integrate


def test_integrate_straight() -> None:
    assert integrate(Pose(0, 0, 0), Twist(1.0, 0.0), 2.0) == Pose(2.0, 0.0, 0.0)


def test_integrate_half_circle() -> None:
    # v=1, w=1 -> radius 1; after pi seconds we are on the opposite side of the circle.
    p = integrate(Pose(0, 0, 0), Twist(1.0, 1.0), math.pi)
    assert (p.x, p.y) == pytest.approx((0.0, 2.0), abs=1e-9)
    assert abs(p.theta) == pytest.approx(math.pi)


def test_speed_limits_preserve_curvature() -> None:
    limits = RobotLimits(max_linear=1.0, max_angular=1.0, max_linear_accel=100, max_angular_accel=100)
    robot = DifferentialDriveRobot(Pose(0, 0, 0), limits)
    robot.step(Twist(2.0, 4.0), 0.1)  # curvature 2, angular is the binding limit
    assert robot.velocity.angular == pytest.approx(1.0)
    assert robot.velocity.linear == pytest.approx(0.5)


def test_acceleration_limit() -> None:
    robot = DifferentialDriveRobot(Pose(0, 0, 0), RobotLimits(max_linear_accel=1.0))
    robot.step(Twist(1.0, 0.0), 0.1)
    assert robot.velocity.linear == pytest.approx(0.1)


def test_wheel_speeds() -> None:
    robot = DifferentialDriveRobot(Pose(0, 0, 0), RobotLimits(max_angular_accel=100), wheel_base=0.5)
    robot.step(Twist(0.0, 2.0), 0.1)
    left, right = robot.wheel_speeds()
    assert left == pytest.approx(-0.5) and right == pytest.approx(0.5)


def test_invalid_dt() -> None:
    with pytest.raises(ValueError):
        DifferentialDriveRobot(Pose(0, 0, 0)).step(Twist(), 0.0)


def _drive(controller: PurePursuitController, robot: DifferentialDriveRobot, steps: int = 2000) -> int:
    for i in range(steps):
        cmd = controller.compute(robot.pose)
        if controller.done:
            return i
        robot.step(cmd, 0.05)
    return steps


def test_pure_pursuit_converges_onto_offset_line() -> None:
    path = [Point(x * 0.1, 0.0) for x in range(60)]
    controller = PurePursuitController(lookahead=0.5)
    controller.reset(path)
    robot = DifferentialDriveRobot(Pose(0.0, 0.5, 0.0))  # starts 0.5 m off the line
    steps = _drive(controller, robot)
    assert controller.done and steps < 2000
    assert robot.pose.position.distance_to(path[-1]) <= controller.goal_tolerance + 0.05


def test_pure_pursuit_turns_around_when_goal_is_behind() -> None:
    controller = PurePursuitController()
    controller.reset([Point(0, 0), Point(-2.0, 0.0)])
    cmd = controller.compute(Pose(0, 0, 0))
    assert cmd.linear == 0.0 and cmd.angular != 0.0


def test_pure_pursuit_requires_path() -> None:
    with pytest.raises(RuntimeError):
        PurePursuitController().compute(Pose(0, 0, 0))
    with pytest.raises(ValueError):
        PurePursuitController().reset([])
