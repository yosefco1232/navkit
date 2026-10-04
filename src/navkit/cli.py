"""Command-line entry point.

navkit run maps/warehouse.txt --plot out.png --gif out.gif
navkit compare maps/warehouse.txt --plot compare.png
"""

from __future__ import annotations

import argparse
import math
import sys
from collections.abc import Sequence

from navkit.control import PurePursuitController
from navkit.maps import MapSpec, load_ascii_map
from navkit.metrics import evaluate, markdown_table
from navkit.planning import AStarPlanner, Planner, SmoothedPlanner
from navkit.robot import RobotLimits
from navkit.simulation import Outcome, SimConfig, SimResult, run_navigation


def baseline_setup() -> tuple[Planner, PurePursuitController]:
    """The v0.1 behaviour: raw A* path, fixed lookahead, no curve slow-down."""
    controller = PurePursuitController(lookahead=0.5, lookahead_gain=0.0, max_lookahead=0.5, max_lateral_accel=math.inf)
    return AStarPlanner(), controller


def default_setup() -> tuple[Planner, PurePursuitController]:
    """Current best: smoothed path, adaptive lookahead, curvature-aware velocity profile."""
    return SmoothedPlanner(AStarPlanner()), PurePursuitController()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="navkit", description="2D mobile-robot navigation simulator")
    sub = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("map", help="ASCII map file (# obstacle, . free, S start, G goal)")
    common.add_argument("--resolution", type=float, default=0.1, help="meters per cell (default 0.1)")
    common.add_argument("--robot-radius", type=float, default=0.2)
    common.add_argument("--motor-lag", type=float, default=0.25, help="motor time constant in seconds")
    common.add_argument("--plot", help="save a PNG of the result")

    run = sub.add_parser("run", parents=[common], help="plan and drive from S to G")
    run.add_argument("--no-smooth", action="store_true", help="use the raw A* path")
    run.add_argument("--speed", type=float, default=0.8, help="cruise speed in m/s")
    run.add_argument("--gif", help="save an animated GIF of the run")

    sub.add_parser("compare", parents=[common], help="compare the v0.1 baseline with the current defaults")
    return parser


def _load(args: argparse.Namespace) -> MapSpec | None:
    spec = load_ascii_map(args.map, args.resolution)
    if spec.start is None or spec.goal is None:
        print("error: map must contain both 'S' and 'G'", file=sys.stderr)
        return None
    return spec


def _config(args: argparse.Namespace) -> SimConfig:
    return SimConfig(robot_radius=args.robot_radius, limits=RobotLimits(response_time=args.motor_lag))


def _simulate(spec: MapSpec, planner: Planner, controller: PurePursuitController, cfg: SimConfig) -> SimResult:
    assert spec.start is not None and spec.goal is not None
    return run_navigation(spec.grid, spec.start, spec.goal, planner, controller, cfg)


def cmd_run(args: argparse.Namespace) -> int:
    spec = _load(args)
    if spec is None:
        return 2
    assert spec.goal is not None
    planner: Planner = AStarPlanner() if args.no_smooth else SmoothedPlanner(AStarPlanner())
    controller = PurePursuitController(cruise_speed=args.speed)
    result = _simulate(spec, planner, controller, _config(args))
    print(markdown_table({"run": evaluate(spec.grid, result)}))

    if args.plot or args.gif:
        from navkit import viz  # imported lazily: matplotlib is slow to import

        if args.plot:
            viz.plot_result(spec.grid, result, spec.goal).savefig(args.plot, dpi=120)
            print(f"saved {args.plot}")
        if args.gif:
            viz.save_animation(spec.grid, result, spec.goal, args.gif)
            print(f"saved {args.gif}")
    return 0 if result.outcome is Outcome.REACHED else 1


def cmd_compare(args: argparse.Namespace) -> int:
    spec = _load(args)
    if spec is None:
        return 2
    assert spec.goal is not None
    cfg = _config(args)
    results = {
        "v0.1 baseline": _simulate(spec, *baseline_setup(), cfg),
        "v0.2 default": _simulate(spec, *default_setup(), cfg),
    }
    print(markdown_table({name: evaluate(spec.grid, r) for name, r in results.items()}))
    if args.plot:
        from navkit import viz

        viz.plot_comparison(spec.grid, results, spec.goal).savefig(args.plot, dpi=120)
        print(f"saved {args.plot}")
    return 0 if all(r.outcome is Outcome.REACHED for r in results.values()) else 1


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    handlers = {"run": cmd_run, "compare": cmd_compare}
    return handlers[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
