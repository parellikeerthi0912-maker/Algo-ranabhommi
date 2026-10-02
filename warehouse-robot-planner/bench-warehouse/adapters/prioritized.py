"""The published serious baseline. Match it exactly and you score 0.50.

Prioritised planning.  Order the robots, then plan each one in turn with
space-time A* against a reservation table holding everything already planned.
Each robot therefore routes *around* its predecessors instead of merely
waiting for them, which is the whole difference from the floor.

The part that actually decides the score is the priority order.  Prioritised
planning is incomplete: for some orders a late robot has no legal path at all,
even though the instance is solvable.  So this does not trust one order --
it tries several (longest path first, shortest first, most-constrained first),
and when a robot fails it promotes that robot and replans.  Then it keeps
shuffling orders until the budget runs out, keeping the best plan found.

Where the remaining points are.  Every replan here throws away the whole
solution and rebuilds it from an empty reservation table.  The organizer
reference instead keeps the plan and repairs a *subset* of robots in place --
large-neighbourhood search -- so it explores far more orderings per second and
can improve a plan that is already valid rather than only replacing it.  It
also optimises congestion and path length directly, which this ignores
entirely: it stops as soon as every robot has some path.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import List, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from adapter import (  # noqa: E402
    Planner,
    Reservation,
    distance_map,
    space_time_astar,
)
from benchkit.rng import Rng  # noqa: E402
from data import Instance, budget_for  # noqa: E402

RESERVE = 0.2


class PrioritizedPlanner(Planner):
    def solve(self, instance: Instance, submit_candidate) -> object:
        self.inst = instance
        self.submit = submit_candidate
        self.started = time.perf_counter()
        # A failed seed order does not call ``submit_candidate``, so waiting
        # for its first receipt to learn the deadline can overrun the harness.
        # The public instance budget gives a safe initial bound; the receipt
        # below remains authoritative once a valid candidate is submitted.
        self.deadline = self.started + budget_for(instance) - RESERVE
        self.rng = Rng(instance.digest() ^ 0x9A3E_0C21)
        self.best_cost: Optional[int] = None
        self.best_plan: Optional[List[List[int]]] = None

        # One distance map per robot, computed from its goal. This is both the
        # A* heuristic and the single-robot lower bound, so it pays for itself
        # immediately and is reused by every replan.
        self.heuristics = [
            distance_map(instance, instance.goals[robot])
            for robot in range(instance.n_robots)
        ]
        self.own_distance = [
            self.heuristics[robot][instance.starts[robot]]
            for robot in range(instance.n_robots)
        ]

        for order in self._seed_orders():
            if self._remaining() < RESERVE:
                break
            plan = self._plan_with(order)
            if plan is not None:
                self._offer(plan)

        self._shuffle_search()
        return self.best_plan

    # -- bookkeeping --------------------------------------------------------

    def _remaining(self) -> float:
        return self.deadline - time.perf_counter()

    def _offer(self, plan: Sequence[Sequence[int]]) -> None:
        receipt = self.submit([list(path) for path in plan])
        if receipt.get("remaining_s") is not None:
            self.deadline = min(
                self.deadline, time.perf_counter() + receipt["remaining_s"]
            )
        if receipt["accepted"]:
            cost = receipt["cost"]
            if self.best_cost is None or cost < self.best_cost:
                self.best_cost = cost
                self.best_plan = [list(path) for path in plan]

    # -- orderings ----------------------------------------------------------

    def _seed_orders(self) -> List[List[int]]:
        n = self.inst.n_robots
        longest = sorted(range(n), key=lambda r: -self.own_distance[r])
        shortest = list(reversed(longest))
        # Most constrained: robots whose goal sits in a tight spot, measured by
        # how few free neighbours it has. They are the ones that get locked out.
        tight = sorted(
            range(n),
            key=lambda r: (len(self.inst.neighbours(self.inst.goals[r])), -self.own_distance[r]),
        )
        return [longest, tight, shortest]

    # -- planning -----------------------------------------------------------

    def _plan_with(self, order: Sequence[int]) -> Optional[List[List[int]]]:
        """Plan in this order, promoting a robot that fails and retrying."""
        order = list(order)
        for _attempt in range(4):
            if self._remaining() < RESERVE:
                return None
            reservation = Reservation(self.inst.horizon)
            paths: dict = {}
            failed = None
            for robot in order:
                if self._remaining() < RESERVE:
                    return None
                path = space_time_astar(
                    self.inst,
                    self.inst.starts[robot],
                    self.inst.goals[robot],
                    reservation,
                    self.heuristics[robot],
                )
                if path is None:
                    failed = robot
                    break
                reservation.block_path(robot, path)
                paths[robot] = path

            if failed is None:
                return [paths[robot] for robot in range(self.inst.n_robots)]

            # The robot that could not move goes first next time. This is the
            # standard repair, and it is why a single fixed order is not enough.
            order.remove(failed)
            order.insert(0, failed)
        return None

    # -- search -------------------------------------------------------------

    def _shuffle_search(self) -> None:
        if self.best_plan is None and self._remaining() < RESERVE:
            return
        base = sorted(range(self.inst.n_robots), key=lambda r: -self.own_distance[r])
        while self._remaining() > RESERVE:
            order = list(base)
            # Perturb rather than fully randomise: a wholly random order is
            # usually much worse than longest-first, so the useful variations
            # live near it.
            for _ in range(max(1, self.inst.n_robots // 8)):
                i = self.rng.below(len(order))
                j = self.rng.below(len(order))
                order[i], order[j] = order[j], order[i]
            plan = self._plan_with(order)
            if plan is not None:
                self._offer(plan)


Planner_ = PrioritizedPlanner
