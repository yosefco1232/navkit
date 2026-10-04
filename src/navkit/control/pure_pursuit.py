"""Pure Pursuit path-tracking controller.

Each step:

1. **Progress**: project the robot onto the path (searching only a short window ahead)
   to find where along the path we are. The distance to that projection is the
   *cross-track error*.
2. **Lookahead**: intersect a circle of radius ``L`` around the robot with the path,
   ahead of the projection. ``L`` grows with speed (adaptive lookahead).
3. **Steering**: the arc through the robot and that point has curvature
   ``k = 2 * y_local / L^2``.
4. **Speed**: read from a precomputed :class:`VelocityProfile`, which slows down
   *before* corners and brakes to a stop at the goal. The instantaneous curvature
   ``k`` is also limited (``v^2 * |k| <= a_lat``) for when the robot is off the path.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from navkit.control.velocity_profile import VelocityProfile
from navkit.geometry import Point, Pose, circle_segment_intersections, interpolate, project_onto_segment
from navkit.robot import STOP, Twist


@dataclass
class PurePursuitController:
    lookahead: float = 0.3  # minimum lookahead distance [m]
    lookahead_gain: float = 0.3  # extra lookahead per unit speed [s]; 0 disables adaptation
    max_lookahead: float = 1.2  # [m]
    cruise_speed: float = 0.8  # [m/s]
    min_speed: float = 0.1  # [m/s]
    max_lateral_accel: float = 0.6  # [m/s^2]; math.inf disables curve slow-down
    max_decel: float = 0.5  # [m/s^2] used to plan braking before corners and the goal
    curvature_window: float = 0.3  # [m] half-width used to estimate path curvature
    goal_tolerance: float = 0.1  # [m]
    turn_in_place_rate: float = 1.5  # [rad/s]
    search_window: float = 1.5  # [m] of path ahead searched when updating progress

    _path: list[Point] = field(default_factory=list, init=False, repr=False)
    _profile: VelocityProfile | None = field(default=None, init=False, repr=False)
    _seg: int = field(default=0, init=False, repr=False)
    _seg_t: float = field(default=0.0, init=False, repr=False)
    _done: bool = field(default=False, init=False, repr=False)
    _cross_track: float = field(default=0.0, init=False, repr=False)
    _current_lookahead: float = field(default=0.0, init=False, repr=False)

    def __post_init__(self) -> None:
        if not 0 < self.lookahead <= self.max_lookahead:
            raise ValueError("need 0 < lookahead <= max_lookahead")

    # -- public API --------------------------------------------------------
    def reset(self, path: list[Point]) -> None:
        if not path:
            raise ValueError("path must contain at least one point")
        # A single-point path becomes a zero-length segment so the math stays uniform.
        self._path = list(path) if len(path) > 1 else [path[0], path[0]]
        self._profile = VelocityProfile.build(
            self._path, self.cruise_speed, self.max_lateral_accel, self.max_decel, self.curvature_window
        )
        self._seg, self._seg_t = 0, 0.0
        self._done = False
        self._cross_track = 0.0

    @property
    def done(self) -> bool:
        return self._done

    @property
    def cross_track_error(self) -> float:
        """Distance from the robot to the path at the last ``compute`` call [m]."""
        return self._cross_track

    @property
    def current_lookahead(self) -> float:
        return self._current_lookahead

    def lookahead_for_speed(self, speed: float) -> float:
        return min(self.max_lookahead, self.lookahead + self.lookahead_gain * abs(speed))

    def compute(self, pose: Pose, speed: float = 0.0) -> Twist:
        """Velocity command for the robot at ``pose`` currently moving at ``speed`` m/s."""
        if not self._path:
            raise RuntimeError("call reset(path) before compute()")
        goal = self._path[-1]
        dist_to_goal = pose.position.distance_to(goal)
        if self._done or dist_to_goal <= self.goal_tolerance:
            self._done = True
            return STOP

        self._update_progress(pose.position)
        self._current_lookahead = self.lookahead_for_speed(speed)
        target = self._lookahead_point(pose.position, self._current_lookahead)
        lx, ly = pose.to_local(target)

        if lx < 0.0 and abs(ly) < abs(lx) * 2:  # target mostly behind us
            return Twist(0.0, math.copysign(self.turn_in_place_rate, ly or 1.0))

        dist_sq = lx * lx + ly * ly
        curvature = 2.0 * ly / dist_sq if dist_sq > 1e-12 else 0.0
        speed_cmd = self._speed(curvature)
        return Twist(speed_cmd, speed_cmd * curvature)

    # -- steps -------------------------------------------------------------
    def _update_progress(self, p: Point) -> None:
        """Find the closest point on the path within ``search_window`` meters ahead.

        Searching only forward (and only a window) keeps progress monotonic: the robot
        can't jump back to an earlier part of the path, or skip ahead where a path
        loops back near itself.
        """
        best_d = math.inf
        best = (self._seg, self._seg_t)
        travelled = 0.0
        for i in range(self._seg, len(self._path) - 1):
            a, b = self._path[i], self._path[i + 1]
            proj, t = project_onto_segment(p, a, b)
            if i == self._seg:
                t = max(t, self._seg_t)  # never move backward on the current segment
                proj = interpolate(a, b, t)
            d = p.distance_to(proj)
            if d < best_d:
                best_d, best = d, (i, t)
            travelled += a.distance_to(b)
            if travelled > self.search_window:
                break
        self._seg, self._seg_t = best
        self._cross_track = best_d

    def _lookahead_point(self, p: Point, radius: float) -> Point:
        """First point ahead of the projection where the path leaves the lookahead circle."""
        for i in range(self._seg, len(self._path) - 1):
            a, b = self._path[i], self._path[i + 1]
            t_min = self._seg_t if i == self._seg else 0.0
            hits = [t for t in circle_segment_intersections(p, radius, a, b) if t >= t_min]
            if hits:
                return interpolate(a, b, hits[-1])
        # No intersection: the goal is inside the circle, or we are far off the path.
        goal = self._path[-1]
        if p.distance_to(goal) <= radius:
            return goal
        return self._path[min(self._seg + 1, len(self._path) - 1)]

    def _progress_distance(self) -> float:
        assert self._profile is not None
        s = self._profile.s
        return s[self._seg] + self._seg_t * (s[self._seg + 1] - s[self._seg])

    def _speed(self, curvature: float) -> float:
        assert self._profile is not None
        v = self._profile.at(self._progress_distance())
        if abs(curvature) > 1e-9 and math.isfinite(self.max_lateral_accel):
            v = min(v, math.sqrt(self.max_lateral_accel / abs(curvature)))
        return max(v, self.min_speed)
