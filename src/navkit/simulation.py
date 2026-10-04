"""Closed-loop navigation simulation: plan once, then track the path step by step."""

from __future__ import annotations

import enum
from dataclasses import dataclass, field

from navkit.control.pure_pursuit import PurePursuitController
from navkit.geometry import Point, Pose, path_length
from navkit.grid_map import OccupancyGrid
from navkit.planning.base import Planner
from navkit.robot import DifferentialDriveRobot, RobotLimits, Twist


class Outcome(enum.Enum):
    REACHED = "reached"
    COLLISION = "collision"
    TIMEOUT = "timeout"
    NO_PATH = "no_path"


@dataclass(frozen=True)
class SimConfig:
    dt: float = 0.05  # s
    max_time: float = 120.0  # s
    robot_radius: float = 0.2  # m, the robot's real footprint, used for collision checks
    safety_margin: float = 0.1  # m, extra inflation for planning only
    # A real drive does not reach the commanded speed instantly; 0.25 s is typical for small robots.
    limits: RobotLimits = field(default_factory=lambda: RobotLimits(response_time=0.25))


@dataclass
class SimResult:
    outcome: Outcome
    path: list[Point]
    trajectory: list[Pose]
    elapsed: float

    @property
    def planned_length(self) -> float:
        return path_length(self.path)

    @property
    def driven_length(self) -> float:
        return path_length([p.position for p in self.trajectory])

    def summary(self) -> str:
        return (
            f"outcome={self.outcome.value} time={self.elapsed:.2f}s "
            f"planned={self.planned_length:.2f}m driven={self.driven_length:.2f}m"
        )


def run_navigation(
    grid: OccupancyGrid,
    start: Pose,
    goal: Point,
    planner: Planner,
    controller: PurePursuitController,
    config: SimConfig | None = None,
) -> SimResult:
    cfg = config or SimConfig()
    # Plan with extra margin, but judge collisions against the real footprint: the gap
    # between the two is the room the controller has to deviate from the path.
    planning_grid = grid.inflate(cfg.robot_radius + cfg.safety_margin)
    path = planner.plan(planning_grid, start.position, goal)
    if path is None:
        return SimResult(Outcome.NO_PATH, [], [start], 0.0)

    robot = DifferentialDriveRobot(start, cfg.limits)
    controller.reset(path)
    trajectory = [start]
    t = 0.0
    while t < cfg.max_time:
        cmd = controller.compute(robot.pose, robot.velocity.linear)
        # "Reached" means at the goal *and* stopped: a robot that arrives at speed still
        # needs room to brake, and that braking must be collision-checked too.
        if controller.done and _is_stopped(robot.velocity):
            return SimResult(Outcome.REACHED, path, trajectory, t)
        pose = robot.step(cmd, cfg.dt)
        t += cfg.dt
        trajectory.append(pose)
        if grid.collides(pose.position, cfg.robot_radius):
            return SimResult(Outcome.COLLISION, path, trajectory, t)
    return SimResult(Outcome.TIMEOUT, path, trajectory, t)


def _is_stopped(velocity: Twist, tol: float = 0.01) -> bool:
    return abs(velocity.linear) < tol and abs(velocity.angular) < tol
