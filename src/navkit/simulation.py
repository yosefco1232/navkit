"""Closed-loop navigation simulation: plan once, then track the path step by step."""

from __future__ import annotations

import enum
from dataclasses import dataclass, field

from navkit.control.pure_pursuit import PurePursuitController
from navkit.geometry import Point, Pose, path_length
from navkit.grid_map import OccupancyGrid
from navkit.planning.base import Planner
from navkit.robot import DifferentialDriveRobot, RobotLimits


class Outcome(enum.Enum):
    REACHED = "reached"
    COLLISION = "collision"
    TIMEOUT = "timeout"
    NO_PATH = "no_path"


@dataclass(frozen=True)
class SimConfig:
    dt: float = 0.05  # s
    max_time: float = 120.0  # s
    robot_radius: float = 0.2  # m, used to inflate obstacles for planning
    limits: RobotLimits = field(default_factory=RobotLimits)


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
    planning_grid = grid.inflate(cfg.robot_radius)
    path = planner.plan(planning_grid, start.position, goal)
    if path is None:
        return SimResult(Outcome.NO_PATH, [], [start], 0.0)

    robot = DifferentialDriveRobot(start, cfg.limits)
    controller.reset(path)
    trajectory = [start]
    t = 0.0
    while t < cfg.max_time:
        cmd = controller.compute(robot.pose)
        if controller.done:
            return SimResult(Outcome.REACHED, path, trajectory, t)
        pose = robot.step(cmd, cfg.dt)
        t += cfg.dt
        trajectory.append(pose)
        if not grid.is_free_world(pose.position):
            return SimResult(Outcome.COLLISION, path, trajectory, t)
    return SimResult(Outcome.TIMEOUT, path, trajectory, t)
