"""Instance generation for the warehouse-robot benchmark.

A grid, some blocked cells, and a fleet of robots that each need to get from a
start cell to a goal cell.  At every time step a robot waits or steps one cell
north, south, east or west.  Two robots may never share a cell, and may never
cross the same edge in opposite directions.

Cells are integers
------------------
A cell is ``y * width + x``.  Conflict checking touches these values once per
robot per time step -- millions of times on the larger instances -- and a flat
integer index makes that a list lookup instead of tuple hashing.

Feasibility by construction
---------------------------
This is the generator decision that matters most here.  Sampling random start
and goal cells produces instances that look fine and are often *jointly*
impossible: two robots whose goals sit in the same dead-end corridor can never
both finish, and no amount of search will discover that for you.

So goals are not sampled.  Every instance begins with robots on their start
cells, and then a valid, conflict-free rollout is simulated -- each robot
drifting toward a randomly chosen attractor -- for a few hundred steps.
Wherever the robots end up *is* the goal assignment.  The rollout itself is
then a witness: a complete, legal plan that reaches every goal and parks
there.  Solvability is a property of how the instance was built, not a hope.

The witness is never published.  It is also a weak plan -- it wanders -- so it
is a feasibility proof, not a target.
"""

from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from benchkit.rng import Rng, derive_seed, digest_ints

SCHEMA_VERSION = "warehouse-1.0"

TOPOLOGIES = ("open", "aisles", "maze", "mixed")
DENSITIES = ("sparse", "packed")
GOAL_PATTERNS = ("scattered", "opposing")
GEOMETRIES = ("regular", "irregular")


@dataclass(frozen=True)
class Instance:
    name: str
    seed: int
    profile: Mapping[str, object]

    width: int
    height: int
    blocked: Tuple[int, ...]  # sorted cell indices
    passable: Tuple[bool, ...]  # indexed by cell

    starts: Tuple[int, ...]
    goals: Tuple[int, ...]
    horizon: int

    # Published lower bounds; part of the cost definition, not a solver's result.
    lb_makespan: int
    lb_sum_of_costs: int
    corridor_cells: Tuple[int, ...]
    congestion_scale: int
    _digest: int = field(default=0, repr=False)

    @property
    def n_robots(self) -> int:
        return len(self.starts)

    @property
    def size(self) -> int:
        """What the harness reports as instance size. Robots, here."""
        return len(self.starts)

    @property
    def cells(self) -> int:
        return self.width * self.height

    def xy(self, cell: int) -> Tuple[int, int]:
        return cell % self.width, cell // self.width

    def neighbours(self, cell: int) -> List[int]:
        """Passable cells one cardinal step away. Excludes waiting."""
        width = self.width
        x, y = cell % width, cell // width
        out = []
        if x > 0 and self.passable[cell - 1]:
            out.append(cell - 1)
        if x + 1 < width and self.passable[cell + 1]:
            out.append(cell + 1)
        if y > 0 and self.passable[cell - width]:
            out.append(cell - width)
        if y + 1 < self.height and self.passable[cell + width]:
            out.append(cell + width)
        return out

    # -- identity -----------------------------------------------------------

    def digest(self) -> int:
        if self._digest:
            return self._digest
        value = self.compute_digest()
        object.__setattr__(self, "_digest", value)
        return value

    def compute_digest(self) -> int:
        stream: List[int] = [self.width, self.height, self.horizon,
                             self.lb_makespan, self.lb_sum_of_costs]
        stream.extend(self.blocked)
        stream.extend(self.starts)
        stream.extend(self.goals)
        profile_bytes = json.dumps(dict(self.profile), sort_keys=True, separators=(",", ":")).encode()
        stream.extend(profile_bytes)
        return digest_ints(stream)

    def to_json(self) -> Dict[str, object]:
        return {
            "schema": SCHEMA_VERSION,
            "name": self.name,
            "seed": self.seed,
            "profile": dict(self.profile),
            "width": self.width,
            "height": self.height,
            "horizon": self.horizon,
            "blocked": list(self.blocked),
            "robots": [
                {"id": r, "start": self.starts[r], "goal": self.goals[r]}
                for r in range(self.n_robots)
            ],
            "lb_makespan": self.lb_makespan,
            "lb_sum_of_costs": self.lb_sum_of_costs,
            "digest": f"{self.digest():016x}",
        }


@dataclass(frozen=True)
class Witness:
    """The rollout the goals were read off. Organizer-only."""

    paths: Tuple[Tuple[int, ...], ...]
    makespan: int


def profile(
    robots: int,
    width: int,
    height: int,
    topology: str = "open",
    density: str = "sparse",
    goals: str = "scattered",
    geometry: str = "regular",
) -> Dict[str, object]:
    for value, allowed, label in (
        (topology, TOPOLOGIES, "topology"),
        (density, DENSITIES, "density"),
        (goals, GOAL_PATTERNS, "goals"),
        (geometry, GEOMETRIES, "geometry"),
    ):
        if value not in allowed:
            raise ValueError(f"unknown {label}: {value!r} (expected one of {allowed})")
    if not 10 <= robots <= 300:  # published fleet range
        raise ValueError("robots must be within the published 10..300 range")
    return {
        "robots": robots,
        "width": width,
        "height": height,
        "topology": topology,
        "density": density,
        "goals": goals,
        "geometry": geometry,
    }


# ---------------------------------------------------------------------------
# Maps
# ---------------------------------------------------------------------------


def _build_map(rng: Rng, width: int, height: int, topology: str, geometry: str) -> List[bool]:
    """``passable[cell]``. Border cells stay open so robots can always circulate."""
    passable = [True] * (width * height)

    def block(x: int, y: int) -> None:
        if 0 <= x < width and 0 <= y < height:
            passable[y * width + x] = False

    if topology == "open":
        for _ in range((width * height) // 18):
            block(rng.randint(1, width - 2), rng.randint(1, height - 2))

    elif topology == "aisles":
        # Shelf blocks with corridors between them: the classic warehouse.
        shelf_w = rng.randint(2, 4)
        # Three-cell aisles, not two. A two-cell aisle cannot be passed in
        # opposite directions at all, so prioritised planning fails outright
        # rather than merely finding a slow plan -- the instance stops
        # measuring anything.
        gap = 3 if geometry == "regular" else rng.randint(3, 4)
        x = 2
        while x < width - 2:
            block_h = height - 4 if geometry == "regular" else rng.randint(height // 2, height - 4)
            top = 2 if geometry == "regular" else rng.randint(1, max(1, height - block_h - 2))
            for dx in range(shelf_w):
                for dy in range(block_h):
                    block(x + dx, top + dy)
            x += shelf_w + gap
            if geometry == "irregular":
                shelf_w = rng.randint(2, 4)

    elif topology == "maze":
        # Walls with a single doorway each: long detours and real chokepoints.
        for y in range(2, height - 2, 3):
            doorway = rng.randint(1, width - 2)
            for x in range(1, width - 1):
                if abs(x - doorway) > 0:
                    block(x, y)
        for x in range(3, width - 3, 5):
            doorway = rng.randint(1, height - 2)
            for y in range(1, height - 1):
                if abs(y - doorway) > 0 and rng.chance(2, 3):
                    block(x, y)

    else:  # mixed: shelves on one side, staging area on the other
        split = width // 2
        shelf_w = 2
        x = 2
        while x < split - 1:
            for dx in range(shelf_w):
                for dy in range(2, height - 2):
                    block(x + dx, dy)
            x += shelf_w + 2
        for _ in range((width * height) // 30):
            block(rng.randint(split, width - 2), rng.randint(1, height - 2))

    return _largest_component(passable, width, height)


def _add_clutter(rng: Rng, passable: List[bool], width: int, height: int) -> List[bool]:
    """Scatter extra obstacles, then re-take the largest component.

    The ``packed`` families need robots-per-free-cell to be high enough that
    routes genuinely interfere. Adding robots alone is not the same thing:
    obstacles also create the chokepoints where interference actually bites.
    """
    free = [cell for cell in range(len(passable)) if passable[cell]]
    for cell in rng.sample(free, len(free) // 11):
        x, y = cell % width, cell // width
        if 0 < x < width - 1 and 0 < y < height - 1:
            passable[cell] = False
    return _largest_component(passable, width, height)


def _largest_component(passable: List[bool], width: int, height: int) -> List[bool]:
    """Keep only the biggest connected region.

    A map with two disconnected halves would hand out unreachable goals, which
    is an instance nobody can solve rather than a hard one.
    """
    seen = [False] * len(passable)
    best: List[int] = []
    for cell in range(len(passable)):
        if not passable[cell] or seen[cell]:
            continue
        component = []
        queue = deque([cell])
        seen[cell] = True
        while queue:
            current = queue.popleft()
            component.append(current)
            x, y = current % width, current // width
            for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
                if 0 <= nx < width and 0 <= ny < height:
                    following = ny * width + nx
                    if passable[following] and not seen[following]:
                        seen[following] = True
                        queue.append(following)
        if len(component) > len(best):
            best = component

    keep = set(best)
    return [cell in keep for cell in range(len(passable))]


def _bfs(instance_like, source: int, width: int, height: int, passable: Sequence[bool]) -> List[int]:
    """Step distance from ``source`` to every cell; -1 where unreachable."""
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


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


def generate(
    seed: int, spec: Mapping[str, object], name: Optional[str] = None
) -> Tuple[Instance, Witness]:
    width, height = int(spec["width"]), int(spec["height"])
    robots = int(spec["robots"])
    root = Rng(
        derive_seed("bench-warehouse", SCHEMA_VERSION, seed, json.dumps(dict(spec), sort_keys=True))
    )

    passable = _build_map(
        root.spawn("map"), width, height, str(spec["topology"]), str(spec["geometry"])
    )
    if str(spec["density"]) == "packed":
        passable = _add_clutter(root.spawn("clutter"), passable, width, height)
    free = [cell for cell in range(width * height) if passable[cell]]
    if len(free) < robots * 3:
        raise ValueError(
            f"{name}: map has {len(free)} free cells for {robots} robots; "
            f"too dense to be interesting"
        )

    placement = root.spawn("placement")
    starts = placement.sample(free, robots)

    # -- witness rollout ----------------------------------------------------
    rollout = root.spawn("rollout")
    steps = max(width, height) * (3 if str(spec["goals"]) == "opposing" else 2)
    paths, goals = _rollout(rollout, passable, width, height, starts, steps, str(spec["goals"]))

    # -- published bounds ---------------------------------------------------
    lb_makespan = 0
    lb_sum = 0
    for robot in range(robots):
        distance = _bfs(None, goals[robot], width, height, passable)
        own = distance[starts[robot]]
        if own < 0:
            raise ValueError(f"{name}: robot {robot} cannot reach its goal")
        lb_sum += own
        if own > lb_makespan:
            lb_makespan = own
    lb_makespan = max(1, lb_makespan)
    lb_sum = max(1, lb_sum)

    # Corridor cells are free cells with at most two free neighbours: the
    # places where robots queue behind one another. Congestion is measured
    # only there, so a solver cannot dodge the term by crossing open floor.
    corridors = []
    for cell in free:
        count = 0
        x, y = cell % width, cell // width
        if x > 0 and passable[cell - 1]:
            count += 1
        if x + 1 < width and passable[cell + 1]:
            count += 1
        if y > 0 and passable[cell - width]:
            count += 1
        if y + 1 < height and passable[cell + width]:
            count += 1
        if count <= 2:
            corridors.append(cell)

    horizon = max(lb_makespan * 4, len(paths[0]) * 2, 64)

    instance = Instance(
        name=name or f"seed{seed}-{spec['topology']}-r{robots}",
        seed=seed,
        profile=MappingProxyType(dict(spec)),
        width=width,
        height=height,
        blocked=tuple(cell for cell in range(width * height) if not passable[cell]),
        passable=tuple(passable),
        starts=tuple(starts),
        goals=tuple(goals),
        horizon=horizon,
        lb_makespan=lb_makespan,
        lb_sum_of_costs=lb_sum,
        corridor_cells=tuple(corridors),
        congestion_scale=max(1, lb_sum // 2),
    )
    witness = Witness(paths=tuple(tuple(p) for p in paths), makespan=len(paths[0]) - 1)
    return instance, witness


def _rollout(
    rng: Rng,
    passable: Sequence[bool],
    width: int,
    height: int,
    starts: Sequence[int],
    steps: int,
    pattern: str,
) -> Tuple[List[List[int]], List[int]]:
    """Simulate a legal, conflict-free wander. Its end state becomes the goals."""
    robots = len(starts)
    positions = list(starts)
    paths: List[List[int]] = [[cell] for cell in starts]

    free = [cell for cell in range(len(passable)) if passable[cell]]
    if pattern == "opposing":
        # Two crowds aimed at each other's side of the map: head-on traffic
        # rather than a diffuse shuffle.
        attractors = []
        for robot in range(robots):
            x = width - 2 if starts[robot] % width < width // 2 else 1
            attractors.append(min(free, key=lambda c, x=x: abs(c % width - x)))
    else:
        attractors = [rng.choice(free) for _ in range(robots)]

    occupied = {cell: robot for robot, cell in enumerate(positions)}
    for _ in range(steps):
        order = list(range(robots))
        rng.shuffle(order)
        previous = list(positions)
        for robot in order:
            cell = positions[robot]
            options = []
            x, y = cell % width, cell // width
            for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
                if 0 <= nx < width and 0 <= ny < height:
                    following = ny * width + nx
                    if passable[following] and following not in occupied:
                        options.append(following)
            if not options:
                continue

            # Prefer a step that closes on the attractor, but not always, or
            # every robot funnels along the same line.
            target = attractors[robot]
            tx, ty = target % width, target // width
            if rng.chance(3, 4):
                options.sort(key=lambda c: abs(c % width - tx) + abs(c // width - ty))
                choice = options[0]
            else:
                choice = rng.choice(options)

            # An edge swap is illegal even though both cells look free.
            mover = occupied.get(previous[robot])
            other = None
            for candidate in range(robots):
                if previous[candidate] == choice and positions[candidate] == cell:
                    other = candidate
                    break
            if other is not None:
                continue
            del mover  # only used for clarity above

            del occupied[cell]
            occupied[choice] = robot
            positions[robot] = choice

        for robot in range(robots):
            paths[robot].append(positions[robot])

    return paths, list(positions)


def make_instance(seed: int, spec: Mapping[str, object], name: Optional[str] = None) -> Instance:
    return generate(seed, spec, name)[0]


# ---------------------------------------------------------------------------
# Public suite
# ---------------------------------------------------------------------------

# Robot counts are set against free-cell counts, not grid area: roughly one
# robot per six or seven free cells, against eight to ten in the first suite.
# That is dense enough that routes genuinely interfere and a solver that only
# polishes a baseline plan falls behind, but not past the point where
# prioritised planning stops finding a plan at all. Packed opposing flows on
# aisle and maze maps were measured past that point and are left out.
PUBLIC_SUITE: Tuple[Dict[str, object], ...] = (
    {"name": "w01-aisles-small", "seed": 5101,
     "spec": profile(50, 24, 20, "aisles", "sparse", "scattered", "regular")},
    {"name": "w02-maze", "seed": 5102,
     "spec": profile(60, 28, 24, "maze", "sparse", "scattered", "regular")},
    {"name": "w03-open-packed", "seed": 5103,
     "spec": profile(110, 30, 26, "open", "packed", "scattered", "regular")},
    {"name": "w04-open-opposing", "seed": 5105,
     "spec": profile(100, 30, 26, "open", "packed", "opposing", "regular")},
    {"name": "w05-mixed-packed", "seed": 5106,
     "spec": profile(100, 34, 28, "mixed", "packed", "scattered", "irregular")},
    {"name": "w06-maze-irregular", "seed": 5110,
     "spec": profile(80, 40, 34, "maze", "packed", "scattered", "irregular")},
    {"name": "w07-maze-irregular-b", "seed": 5210,
     "spec": profile(80, 40, 34, "maze", "packed", "scattered", "irregular")},
    {"name": "w08-aisles-xl", "seed": 5109,
     "spec": profile(210, 52, 42, "aisles", "packed", "scattered", "irregular")},
    {"name": "w09-open-xxl", "seed": 5111,
     "spec": profile(280, 56, 46, "open", "packed", "scattered", "irregular")},
)

SELF_CHECK_NAMES = ("w01-aisles-small", "w02-maze")


def public_instances(names: Optional[Sequence[str]] = None) -> List[Instance]:
    wanted = set(names) if names else None
    return [
        make_instance(int(entry["seed"]), entry["spec"], str(entry["name"]))
        for entry in PUBLIC_SUITE
        if wanted is None or entry["name"] in wanted
    ]


def budget_for(instance: Instance) -> float:
    """Per-instance compute budget in seconds."""
    return round(4.0 + instance.n_robots / 12.0, 2)


def sample_profile(rng: Rng, shift_weight: int = 35) -> Dict[str, object]:
    def pick(easy: Sequence[str], hard: Sequence[str]) -> str:
        return rng.choice(hard if rng.below(100) < shift_weight else easy)

    robots = rng.choice((20, 35, 50, 70, 95, 120, 150, 190, 230))
    width = max(18, min(60, robots // 4 + rng.randint(14, 24)))
    height = max(16, width - rng.randint(0, 6))
    return profile(
        robots=robots,
        width=width,
        height=height,
        topology=pick(("open", "aisles"), ("maze", "mixed")),
        density=pick(("sparse",), ("packed",)),
        goals=pick(("scattered",), ("opposing",)),
        geometry=pick(("regular",), ("irregular",)),
    )


def profile_from_seed(seed: int, salt: str = "public-practice", shift_weight: int = 35):
    return sample_profile(Rng(derive_seed(salt, SCHEMA_VERSION, seed)), shift_weight)


def practice_instance(seed: int) -> Instance:
    spec = profile_from_seed(seed, "public-practice", shift_weight=30)
    return make_instance(seed, spec, f"practice-{seed}")


def write_json(instance: Instance, path: str) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(instance.to_json(), handle, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    for entry in PUBLIC_SUITE:
        inst, wit = generate(int(entry["seed"]), entry["spec"], str(entry["name"]))
        free = sum(1 for p in inst.passable if p)
        print(
            f"{inst.name:22s} robots={inst.n_robots:4d} grid={inst.width}x{inst.height} "
            f"free={free:5d} corridors={len(inst.corridor_cells):5d} "
            f"lb_mk={inst.lb_makespan:4d} lb_soc={inst.lb_sum_of_costs:6d} "
            f"horizon={inst.horizon:4d} budget={budget_for(inst):5.1f}s"
        )
