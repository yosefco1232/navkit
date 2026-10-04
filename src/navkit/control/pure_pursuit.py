"""Pure Pursuit path-tracking controller."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from navkit.geometry import Point, Pose
from navkit.robot import STOP, Twist


@dataclass
class PurePursuitController:
    """Steers toward a point ``lookahead`` meters ahead on the path.

    Curvature to the lookahead point is ``k = 2 * y_local / L^2`` and the command is
    ``(v, v * k)``. Speed ramps down inside ``slow_down_radius`` of the goal. If the
    target is behind the robot it rotates in place first.
    """

    lookahead: float = 0.5
    cruise_speed: float = 0.8
    min_speed: float = 0.1
    goal_tolerance: float = 0.1
    slow_down_radius: float = 0.8
    turn_in_place_rate: float = 1.5

    _path: list[Point] = field(default_factory=list, init=False, repr=False)
    _progress: int = field(default=0, init=False, repr=False)
    _done: bool = field(default=False, init=False, repr=False)

    def reset(self, path: list[Point]) -> None:
        if not path:
            raise ValueError("path must contain at least one point")
        self._path = list(path)
        self._progress = 0
        self._done = False

    @property
    def done(self) -> bool:
        return self._done

    def compute(self, pose: Pose) -> Twist:
        if not self._path:
            raise RuntimeError("call reset(path) before compute()")
        goal = self._path[-1]
        dist_to_goal = pose.position.distance_to(goal)
        if self._done or dist_to_goal <= self.goal_tolerance:
            self._done = True
            return STOP

        self._advance_progress(pose)
        target = self._lookahead_point(pose)
        lx, ly = pose.to_local(target)

        if lx < 0.0 and abs(ly) < abs(lx) * 2:  # target mostly behind us
            return Twist(0.0, math.copysign(self.turn_in_place_rate, ly or 1.0))

        dist_sq = lx * lx + ly * ly
        curvature = 2.0 * ly / dist_sq if dist_sq > 1e-12 else 0.0
        speed = self.cruise_speed * min(1.0, dist_to_goal / self.slow_down_radius)
        speed = max(speed, self.min_speed)
        return Twist(speed, speed * curvature)

    def _advance_progress(self, pose: Pose) -> None:
        """Move the progress index to the closest point ahead; never go backward."""
        p = pose.position
        best = self._progress
        best_d = p.distance_to(self._path[best])
        # Only search a window ahead, so the robot can't "jump" to a later part of a looping path.
        window_end = min(len(self._path), self._progress + 50)
        for i in range(self._progress + 1, window_end):
            d = p.distance_to(self._path[i])
            if d < best_d:
                best, best_d = i, d
        self._progress = best

    def _lookahead_point(self, pose: Pose) -> Point:
        p = pose.position
        for pt in self._path[self._progress :]:
            if p.distance_to(pt) >= self.lookahead:
                return pt
        return self._path[-1]
