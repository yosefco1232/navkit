from pathlib import Path

from navkit.cli import main
from navkit.control import PurePursuitController
from navkit.geometry import Point, Pose
from navkit.maps import load_ascii_map, parse_ascii_map
from navkit.planning import AStarPlanner
from navkit.simulation import Outcome, run_navigation

MAPS = Path(__file__).resolve().parents[1] / "maps"


def test_warehouse_end_to_end() -> None:
    spec = load_ascii_map(MAPS / "warehouse.txt", resolution=0.1)
    assert spec.start is not None and spec.goal is not None
    result = run_navigation(spec.grid, spec.start, spec.goal, AStarPlanner(), PurePursuitController())
    assert result.outcome is Outcome.REACHED, result.summary()
    assert result.driven_length < 1.3 * result.planned_length
    # The robot's whole footprint (not just its center) stays clear of obstacles.
    assert not any(spec.grid.collides(p.position, 0.2) for p in result.trajectory)


def test_no_path_when_goal_is_enclosed() -> None:
    text = "\n".join(
        [
            "####################",
            "#........#.........#",
            "#........#.........#",
            "#........#.........#",
            "#....S...#....G....#",
            "#........#.........#",
            "#........#.........#",
            "#........#.........#",
            "####################",
        ]
    )
    spec = parse_ascii_map(text, resolution=0.1)
    assert spec.start is not None and spec.goal is not None
    result = run_navigation(spec.grid, spec.start, spec.goal, AStarPlanner(), PurePursuitController())
    assert result.outcome is Outcome.NO_PATH


def test_timeout_is_reported() -> None:
    from navkit.simulation import SimConfig

    spec = load_ascii_map(MAPS / "warehouse.txt")
    assert spec.start is not None and spec.goal is not None
    result = run_navigation(
        spec.grid, spec.start, spec.goal, AStarPlanner(), PurePursuitController(), SimConfig(max_time=1.0)
    )
    assert result.outcome is Outcome.TIMEOUT


def test_cli_run(tmp_path: Path, capsys: object) -> None:
    out = tmp_path / "run.png"
    code = main(["run", str(MAPS / "warehouse.txt"), "--plot", str(out)])
    assert code == 0
    assert out.exists() and out.stat().st_size > 0


def test_point_and_pose_are_hashable() -> None:
    assert len({Point(0, 0), Point(0, 0)}) == 1
    assert len({Pose(0, 0, 0), Pose(0, 0, 0)}) == 1
