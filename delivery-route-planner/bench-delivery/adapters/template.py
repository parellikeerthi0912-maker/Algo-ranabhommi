"""Starter template. Copy to ``adapters/<yourname>.py`` and build from here.

    python self_check.py --adapter adapters.template:MyRouter

As written this is a valid submission that scores almost nothing: it builds
one plan and stops. The comments mark where the points are.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from adapter import Router, assign_vehicles, neighbour_lists, route_cost, simulate_route  # noqa: E402
from data import Instance  # noqa: E402


class MyRouter(Router):
    def solve(self, instance: Instance, submit_candidate) -> object:
        # k-nearest neighbours for every node. Most useful moves only ever
        # involve a customer and one of its near neighbours, so this is what
        # keeps local search from being quadratic.
        neighbours = neighbour_lists(instance, k=20)

        # STEP 1 -- get something valid submitted immediately.
        # The 1% checkpoint is a fraction of a second on large instances and
        # it is scored. Anything valid beats nothing.
        routes = self._first_plan(instance)
        if routes is None:
            return []
        plan = assign_vehicles(instance, routes)
        if plan is None:
            return []

        receipt = submit_candidate(plan)
        # The receipt is the authoritative clock: cost, current best, and how
        # much budget is actually left.
        deadline = time.perf_counter() + (receipt["remaining_s"] or 0.0)

        # STEP 2 -- improve until the budget runs out, submitting as you go.
        # The harness keeps your best valid plan at each checkpoint, so there
        # is never a reason to hold one back.
        while time.perf_counter() < deadline - 0.2:
            # Your search goes here. What separates a good score from a
            # mid-table one, roughly in order of payoff:
            #
            #   - ruin and recreate: tear out 10-40% of the customers and
            #     reinsert them, rather than nudging one at a time
            #   - regret-based insertion: reinsert whoever will suffer most
            #     if their best route is taken first
            #   - accept a slightly worse solution sometimes, or you will sit
            #     in the first local optimum you find
            #   - incremental feasibility: carry forward/backward time-window
            #     slack per route so testing a move is O(1) instead of
            #     re-simulating the whole route
            break

        return plan

    def _first_plan(self, instance: Instance):
        """Nearest feasible customer, one vehicle at a time. Replace me."""
        unvisited = set(range(1, instance.n + 1))
        xs, ys = instance.xs, instance.ys
        routes = []
        for vehicle in range(instance.num_vehicles):
            if not unvisited:
                break
            route: list = []
            current = 0
            while unvisited:
                ranked = sorted(
                    unvisited,
                    key=lambda c: (xs[current] - xs[c]) ** 2 + (ys[current] - ys[c]) ** 2,
                )
                chosen = None
                for candidate in ranked[:10]:
                    if simulate_route(instance, route + [candidate], vehicle) is not None:
                        chosen = candidate
                        break
                if chosen is None:
                    break
                route.append(chosen)
                unvisited.discard(chosen)
                current = chosen
            if route:
                routes.append(route)
        return None if unvisited else routes
