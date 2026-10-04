"""Drive the robot model on its own, without a map or planner.

Run from the repository root, after `pip install -e .`:

    python examples/robot_demo.py

Commands a constant forward speed and turn rate, so the robot traces a circle of
radius v / omega = 1 m. Watch the v column: with motor lag the speed takes about
0.5 s to reach the command. Try changing V, OMEGA and LAG (0 = instant motors).
"""

import math

from navkit.geometry import Pose
from navkit.robot import DifferentialDriveRobot, RobotLimits, Twist

V, OMEGA, LAG, DT = 0.5, 0.5, 0.25, 0.05  # m/s, rad/s, s, s

robot = DifferentialDriveRobot(Pose(0.0, 0.0, 0.0), RobotLimits(response_time=LAG))
cmd = Twist(V, OMEGA)

print(f"{'t [s]':>6} {'x [m]':>7} {'y [m]':>7} {'theta [deg]':>12} {'v [m/s]':>8}")
for step in range(1, 141):
    pose = robot.step(cmd, DT)
    if step % 10 == 0:
        print(
            f"{step * DT:6.2f} {pose.x:7.3f} {pose.y:7.3f} "
            f"{math.degrees(pose.theta):12.1f} {robot.velocity.linear:8.3f}"
        )

print(f"\nexpected circle radius v/omega = {V / OMEGA:.2f} m")
