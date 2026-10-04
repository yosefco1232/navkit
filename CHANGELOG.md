# Changelog

## v0.2.0 — Better paths and tracking

### Added
- `SmoothedPlanner`: line-of-sight shortcutting (Bresenham, no corner cutting) and path resampling.
- `VelocityProfile`: curvature-limited speed (`v²κ ≤ a_lat`) with a backward braking pass.
- Pure Pursuit: speed-adaptive lookahead, exact circle–segment lookahead point, cross-track error.
- Robot: first-order motor lag (`response_time`).
- `OccupancyGrid.clearance` / `collides`: exact disk-vs-cells footprint check.
- `metrics` module and `navkit compare` command with a comparison plot.

### Changed
- Simulation plans with `robot_radius + safety_margin` and checks collisions with the real footprint.
- `reached` now requires the robot to be stopped at the goal.
- Warehouse map redesigned with 1 m aisles.

### Fixed
- Collision check ignored the robot's body (center-point only).
- `inflate` rounded `0.30000000000000004 / 0.1` up to 4 cells instead of 3.

## v0.1.0 — Core pipeline
- Occupancy grid, A*, differential-drive model, Pure Pursuit, simulator, CLI, CI.
