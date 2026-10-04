# Roadmap

Each milestone is roughly 1–3 weeks at 5–8 hours per week. Open each item as a GitHub issue,
work on a branch, and merge through a pull request with tests, so the history shows real engineering process.

## v0.1 — Core pipeline ✅
- [x] Occupancy grid with inflation
- [x] A* planner
- [x] Differential-drive kinematics with limits
- [x] Pure Pursuit controller
- [x] Simulation, CLI, plots, CI

## v0.2 — Better paths and tracking
- [ ] **Path smoothing**: line-of-sight shortcutting with Bresenham ray casting (`planning/smoothing.py`)
- [ ] **Adaptive lookahead**: `L = k * v`, clamped; fixes the overshoot visible in the demo near the wall gap
- [ ] Interpolate the lookahead point on path segments instead of snapping to path vertices
- [ ] Metrics: cross-track error (RMS and max), time to goal, minimum obstacle clearance

## v0.3 — More planners and benchmarks
- [ ] Dijkstra and Theta* (any-angle) behind the same `Planner` protocol
- [ ] RRT or RRT* for continuous space
- [ ] `navkit bench`: random maps, compare planners on path length, expansions and runtime; write results to a Markdown table

## v0.4 — Dynamic world
- [ ] Moving obstacles in the simulator
- [ ] Simulated 2D lidar (ray casting) with noise
- [ ] Replanning when the current path becomes blocked
- [ ] Optional: Dynamic Window Approach (DWA) as a local planner

## v0.5 — Performance and systems
- [ ] Profile A* (`cProfile`) and document the hot spots
- [ ] Rewrite the A* core in **C++17** and expose it with **pybind11**; benchmark it against the Python version
- [ ] Add C++ unit tests (GoogleTest) and build them in CI

## v0.6 — Robotics ecosystem
- [ ] ROS 2 wrapper package: planner as a node publishing `nav_msgs/Path`, controller publishing `geometry_msgs/Twist`
- [ ] Docker image and a short demo video in the README

## Optional hardware extension
- [ ] Run the controller on a Raspberry Pi with a cheap two-wheel chassis and wheel encoders
