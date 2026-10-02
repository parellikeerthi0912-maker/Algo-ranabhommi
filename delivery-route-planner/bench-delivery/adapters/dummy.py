"""The floor: nearest-neighbour construction with feasibility checks.

This is what "I understood the problem statement" looks like.  It is a correct
submission -- it respects every constraint and it will usually produce a valid
plan -- and it scores close to zero, because the scoring curve is anchored on
:mod:`adapters.savings`, not on this.

Read it for the API, not for the algorithm.  Note what it does *not* do:
it never revisits a decision, it never submits a second candidate, and it
spends none of its budget, finishing in a fraction of the time it was given.
Those three omissions are most of the gap to a mid-table score.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from adapter import Router, assign_vehicles, simulate_route  # noqa: E402
from data import Instance  # noqa: E402


class NearestNeighbourRouter(Router):
    def solve(self, instance: Instance, submit_candidate) -> object:
        unvisited = set(range(1, instance.n + 1))
        xs, ys = instance.xs, instance.ys
        routes = []

        for vehicle in range(instance.num_vehicles):
            if not unvisited:
                break
            route: list = []
            current = 0
            # Customers this vehicle has already found infeasible; cleared when
            # the route moves on, because a later position may work.
            blocked: set = set()
            while unvisited:
                ranked = sorted(
                    (c for c in unvisited if c not in blocked),
                    key=lambda c: (xs[current] - xs[c]) ** 2 + (ys[current] - ys[c]) ** 2,
                )
                chosen = None
                for candidate in ranked[:12]:  # only the nearest few are worth simulating
                    if simulate_route(instance, route + [candidate], vehicle) is not None:
                        chosen = candidate
                        break
                    blocked.add(candidate)
                if chosen is None:
                    break
                route.append(chosen)
                unvisited.discard(chosen)
                current = chosen
                blocked.clear()
            if route:
                routes.append(route)

        # Nearest-neighbour strands customers whose windows closed while the
        # route was elsewhere. Sweep them into any position that still works,
        # so the floor is a valid plan rather than a zero.
        for customer in sorted(unvisited):
            for vehicle, route in enumerate(routes):
                placed = False
                for position in range(len(route) + 1):
                    trial = route[:position] + [customer] + route[position:]
                    if simulate_route(instance, trial, vehicle) is not None:
                        routes[vehicle] = trial
                        placed = True
                        break
                if placed:
                    break
            else:
                if len(routes) < instance.num_vehicles:
                    routes.append([customer])  # an idle vehicle is better than a zero

        plan = assign_vehicles(instance, routes)
        return plan if plan is not None else [list(r) for r in routes]


Router_ = NearestNeighbourRouter  # convenience alias
