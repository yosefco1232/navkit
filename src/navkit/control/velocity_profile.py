"""Curvature-constrained velocity profile along a path.

A vehicle has to slow down *before* a corner, not in it. We compute, for every path
point, the fastest speed that respects:

1. the cruise speed;
2. lateral acceleration in curves: ``v^2 * kappa <= a_lat``  ->  ``v <= sqrt(a_lat / kappa)``;
3. braking: from any point the robot must be able to slow down to the speed allowed
   further along. With constant deceleration ``a``: ``v_i^2 <= v_{i+1}^2 + 2 a ds``.

(3) is enforced with a single backward pass from the goal, where the speed is zero.
This is the same kinematics as a car braking for a bend.
"""

from __future__ import annotations

import bisect
import itertools
import math
from dataclasses import dataclass

from navkit.geometry import Point


def menger_curvature(a: Point, b: Point, c: Point) -> float:
    """Curvature of the circle through three points: ``4 * area / (|ab| |bc| |ca|)``."""
    cross = (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x)  # = 2 * signed area
    denom = a.distance_to(b) * b.distance_to(c) * c.distance_to(a)
    return 0.0 if denom < 1e-12 else 2.0 * abs(cross) / denom


def cumulative_distance(path: list[Point]) -> list[float]:
    return [0.0, *itertools.accumulate(a.distance_to(b) for a, b in itertools.pairwise(path))]


def path_curvature(path: list[Point], window: float) -> list[float]:
    """Curvature at each point, from the points ``window`` meters before and after it.

    Using a window instead of immediate neighbours makes the estimate independent of
    point spacing, and roughly matches how much a tracking controller rounds a corner.
    """
    s = cumulative_distance(path)
    kappa = []
    for i, si in enumerate(s):
        j = max(0, bisect.bisect_right(s, si - window) - 1)
        k = min(len(path) - 1, bisect.bisect_left(s, si + window))
        kappa.append(menger_curvature(path[j], path[i], path[k]) if j < i < k else 0.0)
    return kappa


@dataclass(frozen=True)
class VelocityProfile:
    s: list[float]  # arc length at each path point [m]
    speed: list[float]  # max allowed speed at each path point [m/s]

    @classmethod
    def build(
        cls,
        path: list[Point],
        cruise_speed: float,
        max_lateral_accel: float,
        max_decel: float,
        curvature_window: float = 0.3,
    ) -> VelocityProfile:
        s = cumulative_distance(path)
        speed = [cruise_speed] * len(path)
        if math.isfinite(max_lateral_accel):
            for i, k in enumerate(path_curvature(path, curvature_window)):
                if k > 1e-9:
                    speed[i] = min(speed[i], math.sqrt(max_lateral_accel / k))
        speed[-1] = 0.0  # stop at the goal
        for i in range(len(path) - 2, -1, -1):  # backward pass: brake in time
            ds = s[i + 1] - s[i]
            speed[i] = min(speed[i], math.sqrt(speed[i + 1] ** 2 + 2.0 * max_decel * ds))
        return cls(s, speed)

    def at(self, distance: float) -> float:
        """Allowed speed at arc length ``distance``, linearly interpolated."""
        if distance <= self.s[0]:
            return self.speed[0]
        if distance >= self.s[-1]:
            return self.speed[-1]
        i = bisect.bisect_right(self.s, distance) - 1
        span = self.s[i + 1] - self.s[i]
        t = 0.0 if span == 0 else (distance - self.s[i]) / span
        return self.speed[i] + t * (self.speed[i + 1] - self.speed[i])
