"""Command-line entry point: ``navkit run maps/warehouse.txt``."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from navkit.control import PurePursuitController
from navkit.maps import load_ascii_map
from navkit.planning import AStarPlanner
from navkit.simulation import Outcome, SimConfig, run_navigation


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="navkit", description="2D mobile-robot navigation simulator")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="plan and drive from S to G on an ASCII map")
    run.add_argument("map", help="path to an ASCII map file (# obstacle, . free, S start, G goal)")
    run.add_argument("--resolution", type=float, default=0.1, help="meters per cell (default 0.1)")
    run.add_argument("--robot-radius", type=float, default=0.2)
    run.add_argument("--lookahead", type=float, default=0.5)
    run.add_argument("--speed", type=float, default=0.8, help="cruise speed in m/s")
    run.add_argument("--plot", help="save a PNG of the result")
    run.add_argument("--gif", help="save an animated GIF of the run")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    spec = load_ascii_map(args.map, args.resolution)
    if spec.start is None or spec.goal is None:
        print("error: map must contain both 'S' and 'G'", file=sys.stderr)
        return 2

    planner = AStarPlanner()
    controller = PurePursuitController(lookahead=args.lookahead, cruise_speed=args.speed)
    config = SimConfig(robot_radius=args.robot_radius)
    result = run_navigation(spec.grid, spec.start, spec.goal, planner, controller, config)

    print(result.summary())
    print(f"A* expanded {planner.last_stats.expanded} cells")

    if args.plot or args.gif:
        from navkit import viz  # imported lazily: matplotlib is slow to import

        if args.plot:
            viz.plot_result(spec.grid, result, spec.goal).savefig(args.plot, dpi=120)
            print(f"saved {args.plot}")
        if args.gif:
            viz.save_animation(spec.grid, result, spec.goal, args.gif)
            print(f"saved {args.gif}")

    return 0 if result.outcome is Outcome.REACHED else 1


if __name__ == "__main__":
    raise SystemExit(main())
