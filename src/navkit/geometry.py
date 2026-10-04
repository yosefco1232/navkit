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


def project_onto_segment(p: Point, a: Point, b: Point) -> tuple[Point, float]:
    """Closest point to ``p`` on segment ``ab`` and its parameter ``t`` in [0, 1]."""
    dx, dy = b.x - a.x, b.y - a.y
    seg_sq = dx * dx + dy * dy
    if seg_sq == 0.0:
        return a, 0.0
    t = ((p.x - a.x) * dx + (p.y - a.y) * dy) / seg_sq
    t = max(0.0, min(1.0, t))
    return Point(a.x + t * dx, a.y + t * dy), t


def circle_segment_intersections(center: Point, radius: float, a: Point, b: Point) -> list[float]:
    """Parameters ``t`` in [0, 1] where segment ``ab`` crosses the circle, in ascending order.

    Solves ``|a + t (b - a) - center|^2 = radius^2``, a quadratic in ``t``.
    """
    dx, dy = b.x - a.x, b.y - a.y
    fx, fy = a.x - center.x, a.y - center.y
    qa = dx * dx + dy * dy
    if qa == 0.0:
        return []
    qb = 2.0 * (fx * dx + fy * dy)
    qc = fx * fx + fy * fy - radius * radius
    disc = qb * qb - 4.0 * qa * qc
    if disc < 0.0:
        return []
    root = math.sqrt(disc)
    roots = sorted({(-qb - root) / (2.0 * qa), (-qb + root) / (2.0 * qa)})
    return [t for t in roots if 0.0 <= t <= 1.0]


def interpolate(a: Point, b: Point, t: float) -> Point:
    return Point(a.x + t * (b.x - a.x), a.y + t * (b.y - a.y))
