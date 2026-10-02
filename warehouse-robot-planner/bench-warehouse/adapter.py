"""The submission interface, plus the conveniences every solver would rewrite.

A submission is one class:

    class Planner:
        def solve(self, instance, submit_candidate):
            ...
            return final_plan

A plan is one path per robot, in robot order::

    [[start, cell, cell, ..., goal], ...]

Each consecutive pair must be the same cell (a wait) or one cardinal step to a
passable neighbour.  Paths may differ in length; the evaluator pads each one
by repeating its last cell, which is also how "stay at your goal" is enforced
-- an early finisher keeps blocking its goal cell forever.

The helpers below cover the parts nobody should have to rewrite: distance maps,
single-robot shortest paths, and space-time A* against a reservation table.
What they deliberately do not give you is the part that decides your score:
*which* robots to plan in what order, and what to do when one cannot find a
path.  Prioritised planning is only as good as its priority order, and the
difference between a fixed order and one that is repaired when planning fails
is most of the gap to a good result.
"""

from __future__ import annotations

from collections import deque
from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from data import Instance

SubmitCandidate = Callable[[object], dict]


class Planner(ABC):
    """Base class. Subclass it, or just match the shape -- duck typing is fine."""

    @abstractmethod
    def solve(self, instance: Instance, submit_candidate: SubmitCandidate) -> object: ...


# ---------------------------------------------------------------------------
# Distances
# ---------------------------------------------------------------------------


def distance_map(instance: Instance, source: int) -> List[int]:
    """Step distance from ``source`` to every cell; -1 where unreachable.

    Computed from the goal, this is the perfect admissible heuristic for
    space-time A* on that robot -- it already knows the walls.
    """
    width, height = instance.width, instance.height
    passable = instance.passable
    distance = [-1] * len(passable)
    distance[source] = 0
    queue = deque([source])
    while queue:
        cell = queue.popleft()
        step = distance[cell] + 1
        x, y = cell % width, cell // width
        if x > 0 and passable[cell - 1] and distance[cell - 1] < 0:
            distance[cell - 1] = step
            queue.append(cell - 1)
        if x + 1 < width and passable[cell + 1] and distance[cell + 1] < 0:
            distance[cell + 1] = step
            queue.append(cell + 1)
        if y > 0 and passable[cell - width] and distance[cell - width] < 0:
            distance[cell - width] = step
            queue.append(cell - width)
        if y + 1 < height and passable[cell + width] and distance[cell + width] < 0:
            distance[cell + width] = step
            queue.append(cell + width)
    return distance


def shortest_path(instance: Instance, start: int, goal: int) -> Optional[List[int]]:
    """Ignores every other robot. The floor, and a lower bound on one path."""
    distance = distance_map(instance, goal)
    if distance[start] < 0:
        return None
    path = [start]
    cell = start
    while cell != goal:
        step = distance[cell]
        for following in instance.neighbours(cell):
            if distance[following] == step - 1:
                cell = following
                break
        else:
            return None
        path.append(cell)
    return path


# ---------------------------------------------------------------------------
# Reservations and space-time search
# ---------------------------------------------------------------------------


class Reservation:
    """Which cells and edges are taken, at which time steps.

    ``block_path`` also reserves a robot's goal cell for *all* later time
    steps, because a parked robot never moves again. Forgetting that is the
    single most common reason a prioritised planner produces plans the
    evaluator rejects.
    """

    __slots__ = ("cells", "edges", "parked_from", "last_use", "horizon")

    def __init__(self, horizon: int) -> None:
        self.cells: Dict[Tuple[int, int], int] = {}  # (time, cell) -> robot
        self.edges: Dict[Tuple[int, int, int], int] = {}  # (time, from, to) -> robot
        # cell -> the time from which a finished robot sits there forever.
        # A dict, not a list: is_free runs inside the A* inner loop, so a scan
        # over every parked robot would dominate the whole search.
        self.parked_from: Dict[int, int] = {}
        self.last_use: Dict[int, int] = {}
        self.horizon = horizon

    def is_free(self, time: int, cell: int) -> bool:
        if (time, cell) in self.cells:
            return False
        parked = self.parked_from.get(cell)
        return parked is None or time < parked

    def edge_free(self, time: int, source: int, target: int) -> bool:
        # A robot moving target->source over the same step would be a swap.
        return (time, target, source) not in self.edges

    def block_path(self, robot: int, path: Sequence[int]) -> None:
        for time, cell in enumerate(path):
            self.cells[(time, cell)] = robot
            self.last_use[cell] = max(time, self.last_use.get(cell, -1))
            if time:
                self.edges[(time, path[time - 1], cell)] = robot
        goal, settled = path[-1], len(path) - 1
        existing = self.parked_from.get(goal)
        if existing is None or settled < existing:
            self.parked_from[goal] = settled

    def clear(self) -> None:
        self.cells.clear()
        self.edges.clear()
        self.parked_from.clear()
        self.last_use.clear()


def space_time_astar(
    instance: Instance,
    start: int,
    goal: int,
    reservation: Reservation,
    heuristic: Sequence[int],
    limit: Optional[int] = None,
) -> Optional[List[int]]:
    """Shortest path in (cell, time) that respects existing reservations.

    Returns ``None`` if no path exists within the horizon -- which happens
    routinely, and is exactly the moment a prioritised planner has to decide
    what to do next rather than give up.
    """
    import heapq

    limit = limit if limit is not None else instance.horizon
    if heuristic[start] < 0:
        return None

    start_state = (start, 0)
    open_heap = [(heuristic[start], 0, start, 0)]
    came: Dict[Tuple[int, int], Tuple[int, int]] = {}
    best: Dict[Tuple[int, int], int] = {start_state: 0}

    while open_heap:
        _, cost, cell, time = heapq.heappop(open_heap)
        if best.get((cell, time), 1 << 30) < cost:
            continue

        if cell == goal:
            # A settled robot occupies its goal forever. A fixed lookahead is
            # unsound when an existing path uses this cell later.
            if reservation.last_use.get(goal, -1) <= time:
                path = [cell]
                state = (cell, time)
                while state in came:
                    state = came[state]
                    path.append(state[0])
                path.reverse()
                return path

        if time >= limit:
            continue

        for following in instance.neighbours(cell) + [cell]:
            if not reservation.is_free(time + 1, following):
                continue
            if following != cell and not reservation.edge_free(time + 1, cell, following):
                continue
            if heuristic[following] < 0:
                continue
            state = (following, time + 1)
            step_cost = cost + 1
            if step_cost < best.get(state, 1 << 30):
                best[state] = step_cost
                came[state] = (cell, time)
                heapq.heappush(
                    open_heap, (step_cost + heuristic[following], step_cost, following, time + 1)
                )

    return None


def plan_cost(instance: Instance, paths: Sequence[Sequence[int]]) -> Optional[int]:
    """Cost in the evaluator's units, or ``None`` if the plan is illegal."""
    import validator

    cost, _reason = validator.evaluate(instance, list(paths))
    return cost
