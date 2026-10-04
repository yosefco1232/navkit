"""Basic 2D geometry primitives shared across the stack."""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Point:
    """A point in the world frame, in meters."""

    x: float
    y: float

    def distance_to(self, other: Point) -> float:
        return math.hypot(other.x - self.x, other.y - self.y)


@dataclass(frozen=True, slots=True)
class Pose:
    """Robot pose in the world frame: position in meters, heading in radians."""

    x: float
    y: float
    theta: float = 0.0

    @property
    def position(self) -> Point:
        return Point(self.x, self.y)

    def to_local(self, p: Point) -> tuple[float, float]:
        """Express a world point in this pose's frame (x forward, y left)."""
        dx, dy = p.x - self.x, p.y - self.y
        c, s = math.cos(self.theta), math.sin(self.theta)
        return c * dx + s * dy, -s * dx + c * dy


def wrap_angle(angle: float) -> float:
    """Wrap an angle to the interval [-pi, pi)."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def path_length(points: list[Point]) -> float:
    return sum(a.distance_to(b) for a, b in itertools.pairwise(points))
