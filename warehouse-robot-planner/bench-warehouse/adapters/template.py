"""Starter template. Copy to ``adapters/<yourname>.py`` and build from here.

    python self_check.py --adapter adapters.template:MyPlanner

As written this is prioritised planning in a fixed order: valid when it works,
nothing at all when it does not. The comments mark where the points are.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from adapter import Planner, Reservation, distance_map, space_time_astar  # noqa: E402
from data import Instance  # noqa: E402


class MyPlanner(Planner):
    def solve(self, instance: Instance, submit_candidate) -> object:
        # One distance map per robot, computed from its goal. This is the
        # perfect A* heuristic (it knows the walls) and each robot's own
        # lower bound. Compute once, reuse for every replan.
        heuristics = [
            distance_map(instance, instance.goals[robot])
            for robot in range(instance.n_robots)
        ]
        own_distance = [
            heuristics[robot][instance.starts[robot]]
            for robot in range(instance.n_robots)
        ]

        # STEP 1 -- plan everyone once, longest journey first.
        # Order matters enormously: prioritised planning is incomplete, so for
        # some orders a late robot has no legal path even though the instance
        # is solvable. Every instance here IS solvable.
        order = sorted(range(instance.n_robots), key=lambda r: -own_distance[r])
        paths = self._plan(instance, order, heuristics)
        if paths is None:
            return None

        receipt = submit_candidate(paths)
        deadline = time.perf_counter() + (receipt["remaining_s"] or 0.0)

        # STEP 2 -- improve until the budget runs out.
        while time.perf_counter() < deadline - 0.25:
            # Your search goes here. Roughly in order of payoff:
            #
            #   - repair instead of rebuild: erase a dozen robots' paths and
            #     replan only those against everything else. A full rebuild on
            #     200 robots costs the same as hundreds of repairs.
            #   - choose the dozen deliberately: robots whose path is far
            #     longer than their own shortest path are being pushed around;
            #     robots sharing the busiest corridor cells are what the
            #     congestion term charges for.
            #   - when A* returns None, that is information about the priority
            #     order, not a dead end. Promote that robot and retry.
            break

        return paths

    def _plan(self, instance: Instance, order, heuristics):
        """Plan every robot in this order. None if anyone gets locked out."""
        reservation = Reservation(instance.horizon)
        paths = {}
        for robot in order:
            path = space_time_astar(
                instance,
                instance.starts[robot],
                instance.goals[robot],
                reservation,
                heuristics[robot],
            )
            if path is None:
                return None
            # block_path also reserves the goal cell for all later time steps,
            # because a finished robot never moves again. Forgetting that is
            # the most common reason a plan gets rejected.
            reservation.block_path(robot, path)
            paths[robot] = path
        return [paths[robot] for robot in range(instance.n_robots)]
