"""The submission interface, plus the conveniences every solver would rewrite.

A submission is one class:

    class Router:
        def solve(self, instance, submit_candidate):
            ...
            return final_plan

``instance`` is a frozen :class:`data.Instance`.  ``submit_candidate`` may be
called as often as you like with a plan; each call is validated and costed by
the evaluator, and the best valid plan at each of the four checkpoints is what
scores.  Calling it early and often is strictly better than holding everything
back: the 1% checkpoint is scored, and a solver with nothing valid at 0.25s on
a large instance has already given points away.

A plan is a list of routes, one per vehicle, in vehicle order::

    [[3, 7, 2], [], [5, 1, 9, 4]]

Route ``v`` is driven by vehicle ``v``, so it must respect *that* vehicle's
capacity and start time.  Depot legs are implicit -- never include node 0.

The helpers below are deliberately straightforward rather than fast.  They
exist so that nobody loses points to boilerplate, not because they are good
enough to win with: ``simulate_route`` is O(len(route)) per call, which makes
any local search built directly on it an order of magnitude slower than one
that maintains incremental time-window slack. Replacing them is most of the
gap between a mid-table score and a good one.
"""

from __future__ import annotations

from math import isqrt
from abc import ABC, abstractmethod
from typing import Callable, List, Optional, Sequence, Tuple

from data import COORD_MAX, Instance

SubmitCandidate = Callable[[object], dict]


class Router(ABC):
    """Base class. Subclass it, or just match the shape -- duck typing is fine."""

    @abstractmethod
    def solve(self, instance: Instance, submit_candidate: SubmitCandidate) -> object: ...


# ---------------------------------------------------------------------------
# Provided conveniences
# ---------------------------------------------------------------------------


def neighbour_lists(instance: Instance, k: int = 20) -> List[List[int]]:
    """``k`` nearest customers for each node, depot included at index 0.

    Bucketed by a uniform grid so this stays well under the 1% checkpoint even
    at n=1000, where the naive all-pairs version would not.
    """
    n = instance.n
    xs, ys = instance.xs, instance.ys
    k = min(k, n - 1) if n > 1 else 0

    # Roughly sqrt(n) cells per axis keeps buckets small without making the
    # ring search wide.
    axis = max(1, isqrt(max(1, n)) // 2 + 1)
    cell = COORD_MAX // axis + 1
    buckets: dict = {}
    for node in range(n + 1):
        key = (xs[node] // cell, ys[node] // cell)
        buckets.setdefault(key, []).append(node)

    result: List[List[int]] = []
    for node in range(n + 1):
        cx, cy = xs[node] // cell, ys[node] // cell
        found: List[Tuple[int, int]] = []
        radius = 1
        while True:
            found.clear()
            for gx in range(cx - radius, cx + radius + 1):
                for gy in range(cy - radius, cy + radius + 1):
                    for other in buckets.get((gx, gy), ()):
                        if other == node or other == 0:
                            continue
                        dx, dy = xs[node] - xs[other], ys[node] - ys[other]
                        found.append((dx * dx + dy * dy, other))
            if len(found) >= k or radius > axis:
                break
            radius += 1
        found.sort()
        result.append([other for _, other in found[:k]])
    return result


def simulate_route(
    instance: Instance, route: Sequence[int], vehicle: int
) -> Optional[Tuple[int, int, int]]:
    """``(travel, wait, duration)`` in seconds, or ``None`` if infeasible.

    Checks capacity, every time window, the route-duration cap, and the
    horizon -- the same rules the evaluator applies.
    """
    xs, ys = instance.xs, instance.ys
    load = 0
    for customer in route:
        load += instance.demand[customer]
    if load > instance.capacities[vehicle]:
        return None

    clock = instance.starts[vehicle]
    previous = 0
    travel = wait = 0
    for customer in route:
        dx, dy = xs[previous] - xs[customer], ys[previous] - ys[customer]
        step = isqrt(dx * dx + dy * dy)
        travel += step
        arrival = clock + step
        if arrival > instance.tw_close[customer]:
            return None
        if arrival < instance.tw_open[customer]:
            wait += instance.tw_open[customer] - arrival
            clock = instance.tw_open[customer]
        else:
            clock = arrival
        clock += instance.service[customer]
        previous = customer

    dx, dy = xs[previous] - xs[0], ys[previous] - ys[0]
    step = isqrt(dx * dx + dy * dy)
    travel += step
    clock += step
    duration = clock - instance.starts[vehicle]
    if duration > instance.max_duration or clock > instance.horizon:
        return None
    return travel, wait, duration


def route_cost(instance: Instance, route: Sequence[int], vehicle: int) -> Optional[int]:
    """Cost of one route in the evaluator's units, or ``None`` if infeasible."""
    simulated = simulate_route(instance, route, vehicle)
    if simulated is None:
        return None
    travel, wait, _ = simulated
    return 100 * travel + 15 * wait


def assign_vehicles(instance: Instance, routes: Sequence[Sequence[int]]) -> Optional[List[List[int]]]:
    """Place unordered routes onto concrete vehicles, or ``None`` if they don't fit.

    Worth using even if you write everything else yourself.  Heterogeneous
    fleets mix capacities *and* start times, and a perfectly good set of routes
    scores zero if a route lands on a vehicle that starts an hour late. Vehicles
    are tried earliest-start first, then tightest-capacity first, so spare
    late-starting vehicles are held back for routes that can absorb the delay.
    """
    if len(routes) > instance.num_vehicles:
        return None

    order = sorted(
        range(instance.num_vehicles),
        key=lambda v: (instance.starts[v], instance.capacities[v], v),
    )
    by_load = sorted(
        range(len(routes)), key=lambda r: -sum(instance.demand[c] for c in routes[r])
    )

    plan: List[List[int]] = [[] for _ in range(instance.num_vehicles)]
    taken = [False] * instance.num_vehicles
    for route_index in by_load:
        route = list(routes[route_index])
        if not route:
            continue
        placed = False
        for vehicle in order:
            if taken[vehicle]:
                continue
            if simulate_route(instance, route, vehicle) is None:
                continue
            plan[vehicle] = route
            taken[vehicle] = True
            placed = True
            break
        if not placed:
            return None
    return plan
