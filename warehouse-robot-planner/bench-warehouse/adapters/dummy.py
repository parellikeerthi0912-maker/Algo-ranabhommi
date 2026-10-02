"""The floor: prioritised planning in a fixed order, no repair.

Robots are planned in index order with space-time A* against everything
already committed.  First come, first served -- and whoever is planned last
gets whatever is left.

Why not the more obvious floor
------------------------------
The natural first idea is "plan every robot independently, then push the
departures of the ones that collide".  It does not work here, and the reason
is worth understanding before writing anything else: a robot that reaches its
goal *stays* there, so its goal cell is blocked from that moment until the end
of time.  If another robot's shortest path happens to cross that cell, no
departure delay whatsoever will help -- waiting longer makes it worse.  That
approach fails on essentially every instance in this suite, which is why the
floor is this instead.

This is a correct submission and it scores close to zero, because the curve is
anchored on :mod:`adapters.prioritized`.  What it is missing: it never
reconsiders the order, so when a late robot finds no path at all it simply
gives up and the whole instance scores nothing.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from adapter import Planner, Reservation, distance_map, space_time_astar  # noqa: E402
from data import Instance  # noqa: E402


class FixedOrderPlanner(Planner):
    def solve(self, instance: Instance, submit_candidate) -> object:
        reservation = Reservation(instance.horizon)
        paths = []
        for robot in range(instance.n_robots):
            heuristic = distance_map(instance, instance.goals[robot])
            path = space_time_astar(
                instance,
                instance.starts[robot],
                instance.goals[robot],
                reservation,
                heuristic,
            )
            if path is None:
                # No path left for this robot in this order. A better planner
                # would reorder and try again; this one stops.
                return None
            reservation.block_path(robot, path)
            paths.append(path)
        return paths


Planner_ = FixedOrderPlanner
