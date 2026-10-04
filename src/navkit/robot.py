"""Kinematic model of a differential-drive (unicycle) robot."""

from __future__ import annotations

import math
from dataclasses import dataclass

from navkit.geometry import Pose, wrap_angle


@dataclass(frozen=True, slots=True)
class Twist:
    """Velocity command: linear [m/s] and angular [rad/s]."""

    linear: float = 0.0
    angular: float = 0.0


STOP = Twist(0.0, 0.0)


@dataclass(frozen=True, slots=True)
class RobotLimits:
    max_linear: float = 1.0  # m/s
    max_angular: float = 2.0  # rad/s
    max_linear_accel: float = 1.5  # m/s^2
    max_angular_accel: float = 6.0  # rad/s^2

    def __post_init__(self) -> None:
        if min(self.max_linear, self.max_angular, self.max_linear_accel, self.max_angular_accel) <= 0:
            raise ValueError("all limits must be positive")


class DifferentialDriveRobot:
    """Integrates velocity commands while respecting speed and acceleration limits."""

    def __init__(self, pose: Pose, limits: RobotLimits | None = None, wheel_base: float = 0.4) -> None:
        self.pose = pose
        self.limits = limits or RobotLimits()
        self.wheel_base = wheel_base
        self.velocity = STOP

    def step(self, cmd: Twist, dt: float) -> Pose:
        if dt <= 0:
            raise ValueError("dt must be positive")
        target = self._saturate(cmd)
        v = _approach(self.velocity.linear, target.linear, self.limits.max_linear_accel * dt)
        w = _approach(self.velocity.angular, target.angular, self.limits.max_angular_accel * dt)
        self.velocity = Twist(v, w)
        self.pose = integrate(self.pose, self.velocity, dt)
        return self.pose

    def wheel_speeds(self) -> tuple[float, float]:
        """(left, right) wheel linear speeds in m/s."""
        half = self.velocity.angular * self.wheel_base / 2.0
        return self.velocity.linear - half, self.velocity.linear + half

    def _saturate(self, cmd: Twist) -> Twist:
        """Clamp to speed limits by scaling both components, which preserves path curvature."""
        scale = 1.0
        if abs(cmd.linear) > self.limits.max_linear:
            scale = min(scale, self.limits.max_linear / abs(cmd.linear))
        if abs(cmd.angular) > self.limits.max_angular:
            scale = min(scale, self.limits.max_angular / abs(cmd.angular))
        return Twist(cmd.linear * scale, cmd.angular * scale)


def integrate(pose: Pose, vel: Twist, dt: float) -> Pose:
    """Exact integration of unicycle kinematics for a constant twist over ``dt``."""
    v, w, th = vel.linear, vel.angular, pose.theta
    if abs(w) < 1e-9:
        return Pose(pose.x + v * math.cos(th) * dt, pose.y + v * math.sin(th) * dt, th)
    th_new = th + w * dt
    r = v / w
    return Pose(
        pose.x + r * (math.sin(th_new) - math.sin(th)),
        pose.y - r * (math.cos(th_new) - math.cos(th)),
        wrap_angle(th_new),
    )


def _approach(current: float, target: float, max_delta: float) -> float:
    return current + max(-max_delta, min(max_delta, target - current))
