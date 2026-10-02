"""Feasibility checking and the canonical cost.

The solver never computes its own cost.  It hands over a plan; this module
recomputes arrival times, waiting, load, and duration from the instance and
returns one integer.  That is the only number the benchmark ever ranks, which
removes a whole category of disagreement (and of cheating) by construction.

Everything is integer arithmetic, so two runs on two machines produce the same
cost bit for bit:

    cost_units = 100 * travel_seconds + 15 * waiting_seconds

which is ``travel_minutes + 0.15 * waiting_minutes`` scaled by 6000.  The
scaling exists purely so the ranking key stays an ``int``.

This file is public and contestants are expected to call it.  Speed matters:
validation time is charged to the solver's budget, so this is written to be
tight rather than pretty -- no attribute lookups in the inner loop.
"""

from __future__ import annotations

from math import isqrt
from typing import Any, List, Optional, Sequence, Tuple

from data import Instance

COST_SCALE = 6000  # cost_units / COST_SCALE == cost in minutes
_TRAVEL_WEIGHT = 100
_WAIT_WEIGHT = 15


class Infeasible(ValueError):
    """A plan that violates a hard constraint. Never softened into a penalty."""


def evaluate(instance: Instance, plan: Any) -> Tuple[Optional[int], Optional[str]]:
    """``(cost_units, None)`` when feasible, ``(None, reason)`` when not."""
    try:
        return canonical_cost(instance, plan), None
    except Infeasible as exc:
        return None, str(exc)
    except (TypeError, ValueError, IndexError, AttributeError) as exc:
        return None, f"malformed plan: {exc}"


def canonical_cost(instance: Instance, plan: Any) -> int:
    """Feasible cost, or raise ``Infeasible``."""
    routes = _as_routes(instance, plan)

    xs, ys = instance.xs, instance.ys
    demand, service = instance.demand, instance.service
    tw_open, tw_close = instance.tw_open, instance.tw_close
    capacities, starts = instance.capacities, instance.starts
    max_duration, horizon = instance.max_duration, instance.horizon

    total_travel = 0
    total_wait = 0

    for vehicle, route in enumerate(routes):
        if not route:
            continue

        capacity = capacities[vehicle]
        load = 0
        for customer in route:
            load += demand[customer]
        if load > capacity:
            raise Infeasible(
                f"vehicle {vehicle} carries {load} over capacity {capacity}"
            )

        clock = starts[vehicle]
        previous = 0
        for customer in route:
            dx = xs[previous] - xs[customer]
            dy = ys[previous] - ys[customer]
            step = isqrt(dx * dx + dy * dy)
            total_travel += step
            arrival = clock + step

            close_at = tw_close[customer]
            if arrival > close_at:
                raise Infeasible(
                    f"vehicle {vehicle} reaches customer {customer} at {arrival}, "
                    f"window closes at {close_at}"
                )

            open_at = tw_open[customer]
            if arrival < open_at:
                total_wait += open_at - arrival
                clock = open_at
            else:
                clock = arrival
            clock += service[customer]
            previous = customer

        dx = xs[previous] - xs[0]
        dy = ys[previous] - ys[0]
        step = isqrt(dx * dx + dy * dy)
        total_travel += step
        clock += step

        duration = clock - starts[vehicle]
        if duration > max_duration:
            raise Infeasible(
                f"vehicle {vehicle} route lasts {duration}s over the "
                f"{max_duration}s limit"
            )
        if clock > horizon:
            raise Infeasible(f"vehicle {vehicle} returns at {clock}, after horizon {horizon}")

    return _TRAVEL_WEIGHT * total_travel + _WAIT_WEIGHT * total_wait


def _as_routes(instance: Instance, plan: Any) -> List[Sequence[int]]:
    """Normalise and check structure before any arithmetic runs."""
    if plan is None:
        raise Infeasible("no plan submitted")
    if isinstance(plan, dict):
        plan = plan.get("routes")
        if plan is None:
            raise Infeasible("plan dict has no 'routes' key")
    if isinstance(plan, (str, bytes)):
        raise Infeasible("plan must be a sequence of routes, not a string")

    try:
        routes = [list(route) for route in plan]
    except TypeError as exc:
        raise Infeasible(f"plan is not a sequence of routes: {exc}") from exc

    if len(routes) > instance.num_vehicles:
        raise Infeasible(
            f"plan uses {len(routes)} vehicles, fleet has {instance.num_vehicles}"
        )

    n = instance.n
    seen = bytearray(n + 1)
    total = 0
    for route in routes:
        for customer in route:
            if not isinstance(customer, int) or isinstance(customer, bool):
                raise Infeasible(f"customer id {customer!r} is not an integer")
            if customer < 1 or customer > n:
                raise Infeasible(f"customer id {customer} outside 1..{n}")
            if seen[customer]:
                raise Infeasible(f"customer {customer} visited more than once")
            seen[customer] = 1
            total += 1

    if total != n:
        missing = [c for c in range(1, n + 1) if not seen[c]]
        raise Infeasible(
            f"{len(missing)} customers never visited (first few: {missing[:5]})"
        )
    return routes


# ---------------------------------------------------------------------------
# Reporting helpers -- not on the hot path
# ---------------------------------------------------------------------------


def describe(instance: Instance, plan: Any) -> dict:
    """Per-route breakdown for reports and debugging."""
    routes = _as_routes(instance, plan)
    detail = []
    for vehicle, route in enumerate(routes):
        if not route:
            continue
        clock = instance.starts[vehicle]
        previous = 0
        travel = wait = 0
        for customer in route:
            step = instance.travel(previous, customer)
            travel += step
            arrival = clock + step
            if arrival < instance.tw_open[customer]:
                wait += instance.tw_open[customer] - arrival
                clock = instance.tw_open[customer]
            else:
                clock = arrival
            clock += instance.service[customer]
            previous = customer
        back = instance.travel(previous, 0)
        travel += back
        clock += back
        detail.append(
            {
                "vehicle": vehicle,
                "stops": len(route),
                "load": sum(instance.demand[c] for c in route),
                "capacity": instance.capacities[vehicle],
                "travel_s": travel,
                "wait_s": wait,
                "duration_s": clock - instance.starts[vehicle],
                "duration_cap_s": instance.max_duration,
            }
        )
    return {
        "routes_used": len(detail),
        "cost_units": canonical_cost(instance, plan),
        "cost_minutes": round(canonical_cost(instance, plan) / COST_SCALE, 2),
        "routes": detail,
    }


def cost_minutes(cost_units: Optional[float]) -> Optional[float]:
    return None if cost_units is None else round(cost_units / COST_SCALE, 2)
