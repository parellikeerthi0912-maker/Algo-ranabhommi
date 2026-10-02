"""Collision checking and the canonical cost.

A submission is one path per robot: a list of cells starting at its start and
ending at its goal, where each step is a cardinal move to a passable adjacent
cell or a wait in place.

Padding, and why it is the goal-blocking rule
---------------------------------------------
Paths have different lengths.  The evaluator pads every one to the longest by
repeating its final cell, and *then* checks conflicts.  That single mechanism
is also the "stay at the goal" rule from the problem statement: a robot that
arrives early keeps occupying its goal cell for the rest of time, so nobody
can route through it afterwards.  A plan that only works because a finished
robot politely disappears is rejected here, which is the intended difficulty
-- parking order matters.

Collisions are never softened into a penalty. Two robots in one cell is not an
expensive plan, it is not a plan.

The cost folds three objectives against the instance's own lower bounds, in
integers so the ranking key is exact:

    cost = 750000 * makespan / lb_makespan
         + 150000 * sum_of_costs / lb_sum_of_costs
         + 100000 * congestion / congestion_scale

Makespan dominates, as the design intends.  Sum-of-costs stops a solver
parking most of the fleet to clear a lane for one robot. Congestion counts
pairs of distinct robots that use the same *corridor* cell -- the places where
robots queue -- so the cheapest way to lower it is to spread traffic rather
than to idle.
"""

from __future__ import annotations

from typing import Any, List, Mapping, Optional, Sequence, Tuple

from data import Instance

MAKESPAN_WEIGHT = 750_000
SOC_WEIGHT = 150_000
CONGESTION_WEIGHT = 100_000
COST_SCALE = 1_000_000


class Infeasible(ValueError):
    """A plan that breaks a movement or collision rule."""


def evaluate(instance: Instance, plan: Any) -> Tuple[Optional[int], Optional[str]]:
    try:
        return canonical_cost(instance, plan), None
    except Infeasible as exc:
        return None, str(exc)
    except (TypeError, ValueError, IndexError, KeyError, AttributeError) as exc:
        return None, f"malformed plan: {exc}"


def canonical_cost(instance: Instance, plan: Any) -> int:
    measured = measure(instance, plan)
    return (
        MAKESPAN_WEIGHT * measured["makespan"] // instance.lb_makespan
        + SOC_WEIGHT * measured["sum_of_costs"] // instance.lb_sum_of_costs
        + CONGESTION_WEIGHT * measured["congestion"] // instance.congestion_scale
    )


def measure(instance: Instance, plan: Any) -> dict:
    paths = _as_paths(instance, plan)

    width = instance.width
    height = instance.height
    passable = instance.passable
    n = instance.n_robots

    # -- per-robot legality, and when each one finally settles --------------
    arrival: List[int] = [0] * n
    sum_of_costs = 0
    for robot, path in enumerate(paths):
        if path[0] != instance.starts[robot]:
            raise Infeasible(
                f"robot {robot} starts at cell {path[0]}, not its start "
                f"{instance.starts[robot]}"
            )
        if path[-1] != instance.goals[robot]:
            raise Infeasible(
                f"robot {robot} ends at cell {path[-1]}, not its goal "
                f"{instance.goals[robot]}"
            )
        previous = path[0]
        for step, cell in enumerate(path):
            if cell < 0 or cell >= len(passable) or not passable[cell]:
                raise Infeasible(
                    f"robot {robot} occupies blocked or out-of-grid cell {cell} "
                    f"at time {step}"
                )
            if step:
                px, py = previous % width, previous // width
                cx, cy = cell % width, cell // width
                if abs(px - cx) + abs(py - cy) > 1:
                    raise Infeasible(
                        f"robot {robot} jumps from cell {previous} to {cell} at "
                        f"time {step}; only one cardinal step or a wait is legal"
                    )
            previous = cell

        # The robot has settled once it is at its goal and never leaves again.
        settled = len(path) - 1
        goal = instance.goals[robot]
        while settled > 0 and path[settled - 1] == goal:
            settled -= 1
        arrival[robot] = settled
        sum_of_costs += settled

    makespan = max(arrival) if n else 0
    if makespan > instance.horizon:
        raise Infeasible(
            f"plan takes {makespan} steps, over the {instance.horizon}-step horizon"
        )

    # -- conflicts on the padded common time axis ---------------------------
    span = max(len(path) for path in paths)
    occupancy = [-1] * len(passable)  # cell -> robot, reset lazily per step
    stamp = [-1] * len(passable)

    current = [path[0] for path in paths]
    for robot, cell in enumerate(current):
        if stamp[cell] == 0:
            raise Infeasible(
                f"robots {occupancy[cell]} and {robot} both start on cell {cell}"
            )
        occupancy[cell] = robot
        stamp[cell] = 0

    for time in range(1, span):
        following = [
            paths[robot][time] if time < len(paths[robot]) else paths[robot][-1]
            for robot in range(n)
        ]

        # Who was standing where a moment ago, so an exchange is recognisable.
        previously_at = {cell: robot for robot, cell in enumerate(current)}

        for robot, cell in enumerate(following):
            if stamp[cell] == time:
                raise Infeasible(
                    f"robots {occupancy[cell]} and {robot} both occupy cell "
                    f"{cell} at time {time}"
                )
            occupancy[cell] = robot
            stamp[cell] = time

        # Edge swap: two robots trading places across one edge. Both cells are
        # free by the vertex test above, so this needs its own check.
        for robot in range(n):
            source, target = current[robot], following[robot]
            if source == target:
                continue
            other = previously_at.get(target)
            if other is not None and other != robot and following[other] == source:
                raise Infeasible(
                    f"robots {robot} and {other} swap across the edge "
                    f"{source}<->{target} at time {time}"
                )

        current = following

    # -- congestion ---------------------------------------------------------
    corridor = set(instance.corridor_cells)
    visitors: dict = {}
    for robot, path in enumerate(paths):
        limit = arrival[robot]
        for step in range(limit + 1):
            cell = path[step]
            if cell in corridor:
                visitors.setdefault(cell, set()).add(robot)
    # Pairs of robots sharing a corridor cell. Quadratic, so a lane used by
    # everyone costs far more than two lanes used by half each.
    congestion = sum(len(robots) * (len(robots) - 1) // 2 for robots in visitors.values())

    return {
        "makespan": makespan,
        "sum_of_costs": sum_of_costs,
        "congestion": congestion,
        "arrival": arrival,
    }


def _as_paths(instance: Instance, plan: Any) -> List[Sequence[int]]:
    if plan is None:
        raise Infeasible("no plan submitted")
    if isinstance(plan, Mapping):
        plan = plan.get("paths")
        if plan is None:
            raise Infeasible("plan dict has no 'paths' key")
    if isinstance(plan, (str, bytes)):
        raise Infeasible("plan must be a list of paths, not a string")

    try:
        paths = [list(path) for path in plan]
    except TypeError as exc:
        raise Infeasible(f"plan is not a list of paths: {exc}") from exc

    if len(paths) != instance.n_robots:
        raise Infeasible(
            f"got {len(paths)} paths for {instance.n_robots} robots"
        )
    if any(len(path) > instance.horizon + 1 for path in paths):
        raise Infeasible("path exceeds the instance horizon")
    for robot, path in enumerate(paths):
        if not path:
            raise Infeasible(f"robot {robot} has an empty path")
        for cell in path:
            if not isinstance(cell, int) or isinstance(cell, bool):
                raise Infeasible(f"robot {robot} has non-integer cell {cell!r}")
    return paths


# ---------------------------------------------------------------------------
# Reporting -- not on the hot path
# ---------------------------------------------------------------------------


def describe(instance: Instance, plan: Any) -> dict:
    measured = measure(instance, plan)
    return {
        "makespan": measured["makespan"],
        "lb_makespan": instance.lb_makespan,
        "makespan_ratio": round(measured["makespan"] / instance.lb_makespan, 4),
        "sum_of_costs": measured["sum_of_costs"],
        "lb_sum_of_costs": instance.lb_sum_of_costs,
        "soc_ratio": round(measured["sum_of_costs"] / instance.lb_sum_of_costs, 4),
        "congestion": measured["congestion"],
        "congestion_scale": instance.congestion_scale,
        "cost_units": canonical_cost(instance, plan),
    }


def normalised_cost(cost_units: Optional[float]) -> Optional[float]:
    return None if cost_units is None else round(cost_units / COST_SCALE, 4)
