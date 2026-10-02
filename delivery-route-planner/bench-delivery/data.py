"""Instance generation for the delivery-route benchmark.

Everything here is integer arithmetic over a deterministic PRNG, so an
instance is a pure function of (seed, profile) on every machine forever.  Time
is in seconds, distance is in seconds of travel: coordinates are expressed
directly in the unit that travel time is measured in, which makes travel time
``isqrt(dx*dx + dy*dy)`` -- exact, integral, and impossible to disagree about.

Feasibility by construction
---------------------------
Random time windows over random points produce instances that are usually
infeasible and occasionally trivial, and you cannot tell which without solving
them.  So generation runs backwards: build a hidden feasible solution first
(sweep into routes, order each route, simulate it), then derive the time
windows, the fleet, and the route-duration cap *from that solution*.

Two things follow. Every instance provably admits a feasible plan, so "no
valid answer exists" is never an excuse.  And tightness becomes a dial: the
window offsets around the witness arrival times are exactly how hard the
instance is, which is what lets the private families shift difficulty without
shifting the format.

The witness is never part of the instance and never leaves this module unless
private tooling asks for it.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from benchkit.rng import Rng, derive_seed, digest_ints

SCHEMA_VERSION = "delivery-1.0"

# The map is 5000x5000 seconds-of-travel; the diagonal is about 118 minutes.
COORD_MAX = 5000
DEPOT = (COORD_MAX // 2, COORD_MAX // 2)
# An eight-hour delivery day.
HORIZON = 8 * 3600

LAYOUTS = ("uniform", "clustered", "ring", "mixed")
DEMAND_PROFILES = ("narrow", "skewed", "heavy_light")
WINDOW_PROFILES = ("broad", "tight", "waves")
FLEET_PROFILES = ("comfortable", "tight", "heterogeneous")


@dataclass(frozen=True)
class Instance:
    """One delivery day. Index 0 is the depot; customers are 1..n."""

    name: str
    seed: int
    profile: Mapping[str, object]
    xs: Tuple[int, ...]
    ys: Tuple[int, ...]
    demand: Tuple[int, ...]
    service: Tuple[int, ...]
    tw_open: Tuple[int, ...]
    tw_close: Tuple[int, ...]
    capacities: Tuple[int, ...]
    starts: Tuple[int, ...]
    max_duration: int
    horizon: int = HORIZON
    _digest: int = field(default=0, repr=False)

    # -- geometry -----------------------------------------------------------

    @property
    def n(self) -> int:
        """Customer count, excluding the depot."""
        return len(self.xs) - 1

    @property
    def size(self) -> int:
        """What the harness reports as instance size. Customers, here."""
        return self.n

    @property
    def num_vehicles(self) -> int:
        return len(self.capacities)

    def travel(self, i: int, j: int) -> int:
        """Travel time in whole seconds. Exact, symmetric, integral."""
        dx = self.xs[i] - self.xs[j]
        dy = self.ys[i] - self.ys[j]
        return math.isqrt(dx * dx + dy * dy)

    # -- identity -----------------------------------------------------------

    def digest(self) -> int:
        """Cached fingerprint of every field a solver could tamper with."""
        if self._digest:
            return self._digest
        value = self.compute_digest()
        object.__setattr__(self, "_digest", value)
        return value

    def compute_digest(self) -> int:
        """Recompute the fingerprint from scratch.

        The harness calls this after ``solve`` returns. Using the cached value
        there would defeat the entire check.
        """
        stream: List[int] = [len(self.xs), len(self.capacities), self.max_duration, self.horizon]
        for seq in (
            self.xs,
            self.ys,
            self.demand,
            self.service,
            self.tw_open,
            self.tw_close,
            self.capacities,
            self.starts,
        ):
            stream.extend(seq)
        profile_bytes = json.dumps(dict(self.profile), sort_keys=True, separators=(",", ":")).encode()
        stream.extend(profile_bytes)
        return digest_ints(stream)

    def to_json(self) -> Dict[str, object]:
        return {
            "schema": SCHEMA_VERSION,
            "name": self.name,
            "seed": self.seed,
            "profile": dict(self.profile),
            "horizon": self.horizon,
            "max_duration": self.max_duration,
            "depot": {"x": self.xs[0], "y": self.ys[0]},
            "customers": [
                {
                    "id": i,
                    "x": self.xs[i],
                    "y": self.ys[i],
                    "demand": self.demand[i],
                    "service": self.service[i],
                    "tw_open": self.tw_open[i],
                    "tw_close": self.tw_close[i],
                }
                for i in range(1, len(self.xs))
            ],
            "vehicles": [
                {"id": v, "capacity": self.capacities[v], "start": self.starts[v]}
                for v in range(len(self.capacities))
            ],
            "digest": f"{self.digest():016x}",
        }


@dataclass(frozen=True)
class Witness:
    """The hidden feasible plan an instance was constructed around.

    Organizer-only. It is a feasibility proof and a sanity anchor, never a
    target: the reference solver routinely beats it, because the witness is
    built by a sweep and never improved.
    """

    routes: Tuple[Tuple[int, ...], ...]
    arrivals: Mapping[int, int]
    cost_units: int


# ---------------------------------------------------------------------------
# Profiles
# ---------------------------------------------------------------------------


def profile(
    n: int,
    layout: str = "uniform",
    demand: str = "narrow",
    windows: str = "broad",
    fleet: str = "comfortable",
) -> Dict[str, object]:
    for value, allowed, label in (
        (layout, LAYOUTS, "layout"),
        (demand, DEMAND_PROFILES, "demand"),
        (windows, WINDOW_PROFILES, "windows"),
        (fleet, FLEET_PROFILES, "fleet"),
    ):
        if value not in allowed:
            raise ValueError(f"unknown {label}: {value!r} (expected one of {allowed})")
    if not 20 <= n <= 1000:
        raise ValueError("n must be within the published 20..1000 range")
    return {"n": n, "layout": layout, "demand": demand, "windows": windows, "fleet": fleet}


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------


def _clamp(value: int, low: int = 0, high: int = COORD_MAX) -> int:
    return low if value < low else high if value > high else value


def _on_circle(rng: Rng, radius: int) -> Tuple[int, int]:
    """A point near a circle of the given radius, without touching libm.

    Sample on the square ring, then renormalise the offset to the requested
    radius using exact integer isqrt. Uniform enough for a ring road, and
    reproducible to the bit.
    """
    side = rng.below(4)
    offset = rng.randint(-radius, radius)
    dx, dy = ((radius, offset), (-radius, offset), (offset, radius), (offset, -radius))[side]
    length = math.isqrt(dx * dx + dy * dy)
    if length == 0:
        return DEPOT
    return (
        _clamp(DEPOT[0] + dx * radius // length),
        _clamp(DEPOT[1] + dy * radius // length),
    )


def _layout_points(rng: Rng, n: int, layout: str) -> List[Tuple[int, int]]:
    margin = COORD_MAX // 20
    points: List[Tuple[int, int]] = []

    if layout == "uniform":
        for _ in range(n):
            points.append(
                (rng.randint(margin, COORD_MAX - margin), rng.randint(margin, COORD_MAX - margin))
            )
        return points

    if layout == "clustered":
        count = max(3, n // 25)
        centres = [
            (rng.randint(margin, COORD_MAX - margin), rng.randint(margin, COORD_MAX - margin))
            for _ in range(count)
        ]
        spread = COORD_MAX // 18
        for _ in range(n):
            cx, cy = rng.choice(centres)
            points.append((_clamp(rng.normal(cx, spread)), _clamp(rng.normal(cy, spread))))
        return points

    if layout == "ring":
        radius = COORD_MAX * 2 // 5
        jitter = COORD_MAX // 30
        for _ in range(n):
            if rng.chance(3, 4):
                x, y = _on_circle(rng, radius)
                points.append((_clamp(rng.normal(x, jitter)), _clamp(rng.normal(y, jitter))))
            else:  # a few deliveries inside the ring
                points.append(
                    (
                        _clamp(rng.normal(DEPOT[0], COORD_MAX // 10)),
                        _clamp(rng.normal(DEPOT[1], COORD_MAX // 10)),
                    )
                )
        return points

    # mixed: dense clusters, a ring road, and sparse outskirts in one map. The
    # point is that no single spatial assumption pays off everywhere.
    count = max(3, n // 30)
    centres = [
        (rng.randint(margin, COORD_MAX - margin), rng.randint(margin, COORD_MAX - margin))
        for _ in range(count)
    ]
    radius = COORD_MAX * 2 // 5
    for _ in range(n):
        roll = rng.below(100)
        if roll < 60:
            cx, cy = rng.choice(centres)
            points.append(
                (_clamp(rng.normal(cx, COORD_MAX // 22)), _clamp(rng.normal(cy, COORD_MAX // 22)))
            )
        elif roll < 85:
            points.append((rng.randint(0, COORD_MAX), rng.randint(0, COORD_MAX)))
        else:
            x, y = _on_circle(rng, radius)
            points.append((_clamp(rng.normal(x, COORD_MAX // 40)), _clamp(rng.normal(y, COORD_MAX // 40))))
    return points


# ---------------------------------------------------------------------------
# Demand and service
# ---------------------------------------------------------------------------


def _demands(rng: Rng, n: int, kind: str) -> List[int]:
    if kind == "narrow":
        return [rng.randint(8, 18) for _ in range(n)]
    if kind == "skewed":
        # Mostly small, occasionally large. skewed() biases toward the low end.
        return [rng.skewed(3, 40, 3) for _ in range(n)]
    # heavy_light: a bimodal mix, so capacity -- not distance -- decides how
    # many customers can share a vehicle.
    out = []
    for _ in range(n):
        if rng.chance(1, 5):
            out.append(rng.randint(45, 75))
        else:
            out.append(rng.randint(4, 12))
    return out


def _service_times(rng: Rng, n: int, demands: Sequence[int]) -> List[int]:
    # Bigger drops take longer, with noise. Keeps the instance coherent:
    # a solver that ignores service time mis-estimates every route duration.
    return [max(120, 90 * d // 10 + rng.randint(60, 420)) for d in demands]


# ---------------------------------------------------------------------------
# Witness construction
# ---------------------------------------------------------------------------


def _pseudo_angle(dx: int, dy: int) -> int:
    """Monotone integer surrogate for atan2, in [0, 40000).

    Sorting by this is identical to sorting by true angle, and it never touches
    a transcendental function.
    """
    total = abs(dx) + abs(dy)
    if total == 0:
        return 0
    if dx >= 0 and dy >= 0:
        return 10000 * dy // total
    if dx < 0 and dy >= 0:
        return 10000 + 10000 * (-dx) // total
    if dx < 0 and dy < 0:
        return 20000 + 10000 * (-dy) // total
    return 30000 + 10000 * dx // total


def _sweep_chunks(
    order: Sequence[int], demands: Sequence[int], capacity: int, max_per_route: int
) -> List[List[int]]:
    chunks: List[List[int]] = []
    current: List[int] = []
    load = 0
    for customer in order:
        d = demands[customer]
        if current and (load + d > capacity or len(current) >= max_per_route):
            chunks.append(current)
            current, load = [], 0
        current.append(customer)
        load += d
    if current:
        chunks.append(current)
    return chunks


def _order_route(inst_xs: Sequence[int], inst_ys: Sequence[int], members: Sequence[int]) -> List[int]:
    """Nearest-neighbour ordering from the depot. Cheap and good enough: the
    witness only has to be feasible, not sharp."""
    remaining = list(members)
    route: List[int] = []
    current = 0
    while remaining:
        best_index, best_cost = 0, None
        for index, candidate in enumerate(remaining):
            dx = inst_xs[current] - inst_xs[candidate]
            dy = inst_ys[current] - inst_ys[candidate]
            cost = dx * dx + dy * dy
            if best_cost is None or cost < best_cost:
                best_index, best_cost = index, cost
        current = remaining.pop(best_index)
        route.append(current)
    return route


def _window_for(rng: Rng, arrival: int, kind: str, wave_edges: Sequence[int]) -> Tuple[int, int]:
    if kind == "waves":
        # Ordered delivery waves: the window is the wave containing the hidden
        # arrival. Wide inside the wave, but the wave order is a hard global
        # sequencing constraint that a distance-only heuristic walks straight
        # into.
        for index in range(len(wave_edges) - 1):
            if wave_edges[index] <= arrival < wave_edges[index + 1]:
                start, end = wave_edges[index], wave_edges[index + 1]
                break
        else:
            start, end = wave_edges[-2], wave_edges[-1]
        # A quarter of customers get a tight window inside their wave, so
        # "just respect the waves" is not sufficient either.
        if rng.chance(1, 4):
            before = rng.randint(240, 900)
            after = rng.randint(240, 900)
            return max(start, arrival - before), min(end, arrival + after)
        return start, min(end, HORIZON)

    if kind == "tight":
        before = rng.randint(180, 1200)
        after = rng.randint(180, 1200)
    else:  # broad
        before = rng.randint(1500, 5400)
        after = rng.randint(1500, 5400)
        if rng.chance(1, 8):  # a few effectively-open customers
            before, after = 12 * 3600, 12 * 3600

    return max(0, arrival - before), min(HORIZON, arrival + after)


def _fleet_slack(kind: str) -> Tuple[int, int]:
    """(capacity numerator, denominator) over the witness requirement."""
    return {"comfortable": (140, 100), "tight": (106, 100), "heterogeneous": (122, 100)}[kind]


def generate(seed: int, spec: Mapping[str, object], name: Optional[str] = None) -> Tuple[Instance, Witness]:
    """Build one instance and the hidden feasible plan it was derived from."""
    n = int(spec["n"])
    root = Rng(derive_seed("bench-delivery", SCHEMA_VERSION, seed, json.dumps(dict(spec), sort_keys=True)))

    geo = root.spawn("geometry")
    points = _layout_points(geo, n, str(spec["layout"]))
    xs = [DEPOT[0]] + [p[0] for p in points]
    ys = [DEPOT[1]] + [p[1] for p in points]

    load_rng = root.spawn("demand")
    demand_values = _demands(load_rng, n, str(spec["demand"]))
    service_values = _service_times(load_rng, n, demand_values)
    demand = [0] + demand_values
    service = [0] + service_values

    # -- hidden feasible plan ----------------------------------------------
    fleet_rng = root.spawn("fleet")
    total_demand = sum(demand_values)
    # Route length grows with the instance so the witness never needs more
    # than ~40 routes. The published fleet range tops out at 80 vehicles, and
    # the fleet is sized at roughly twice the witness (see _spare_vehicles),
    # so the witness has to stay well inside that.
    max_per_route = min(26, max(10, -(-n // 38)))
    target_routes = max(2, min(40, -(-n // max_per_route)))
    numerator, denominator = _fleet_slack(str(spec["fleet"]))
    capacity = max(
        max(demand_values),
        -(-total_demand * numerator // (target_routes * denominator)),
    )

    angles = sorted(
        range(1, n + 1), key=lambda c: (_pseudo_angle(xs[c] - DEPOT[0], ys[c] - DEPOT[1]), c)
    )
    rotation = fleet_rng.below(max(1, n))
    angles = angles[rotation:] + angles[:rotation]

    chunks = _sweep_chunks(angles, demand, capacity, max_per_route)
    while len(chunks) > 80:  # respect the published 2..80 vehicle range
        capacity = capacity * 12 // 10
        max_per_route += 2
        chunks = _sweep_chunks(angles, demand, capacity, max_per_route)

    routes = [tuple(_order_route(xs, ys, chunk)) for chunk in chunks]

    # Simulate the witness with no waiting: arrival times become the anchors
    # the time windows are built around.
    arrivals: Dict[int, int] = {}
    durations: List[int] = []
    travel_total = 0
    for route in routes:
        clock = 0
        previous = 0
        for customer in route:
            dx, dy = xs[previous] - xs[customer], ys[previous] - ys[customer]
            step = math.isqrt(dx * dx + dy * dy)
            clock += step
            travel_total += step
            arrivals[customer] = clock
            clock += service[customer]
            previous = customer
        dx, dy = xs[previous] - DEPOT[0], ys[previous] - DEPOT[1]
        step = math.isqrt(dx * dx + dy * dy)
        clock += step
        travel_total += step
        durations.append(clock)

    # -- constraints derived from the witness -------------------------------
    window_rng = root.spawn("windows")
    wave_count = 4
    wave_edges = [i * HORIZON // wave_count for i in range(wave_count + 1)]
    tw_open = [0] * (n + 1)
    tw_close = [HORIZON] * (n + 1)
    for customer in range(1, n + 1):
        open_at, close_at = _window_for(
            window_rng, arrivals[customer], str(spec["windows"]), wave_edges
        )
        if close_at <= open_at:  # degenerate windows help nobody
            close_at = min(HORIZON, open_at + 300)
        tw_open[customer], tw_close[customer] = open_at, close_at

    # Route-duration cap. The witness never waits -- its arrival times *are*
    # the windows -- but any real solver does, and waiting counts against
    # route duration. So the slack scales with how much waiting the window
    # profile forces: wave windows make a vehicle sit idle until its wave
    # opens, and a cap derived naively from the witness would turn that into
    # an infeasibility rather than a cost.
    window_slack = {"broad": 130, "tight": 145, "waves": 160}[str(spec["windows"])]
    duration_cap = min(
        HORIZON,
        max(max(durations) * window_slack // 100, HORIZON * 55 // 100),
    )

    # -- fleet --------------------------------------------------------------
    chunk_loads = sorted((sum(demand[c] for c in route) for route in routes), reverse=True)
    vehicle_count = min(80, max(len(routes), len(routes) + _spare_vehicles(str(spec["fleet"]), len(routes))))

    if str(spec["fleet"]) == "heterogeneous":
        capacities = sorted(
            (fleet_rng.randint(capacity * 80 // 100, capacity * 125 // 100) for _ in range(vehicle_count)),
            reverse=True,
        )
        # Guarantee the witness still fits: the k largest vehicles must cover
        # the k largest routes.
        for index, load in enumerate(chunk_loads):
            if capacities[index] < load:
                capacities[index] = load
        capacities.sort(reverse=True)
    else:
        capacities = [capacity] * vehicle_count

    # Spare vehicles beyond what the witness uses may start late; the vehicles
    # the witness relies on always start at zero, so feasibility is preserved.
    starts = [0] * vehicle_count
    if str(spec["fleet"]) == "heterogeneous":
        for v in range(len(routes), vehicle_count):
            starts[v] = fleet_rng.randint(0, 3600)

    instance = Instance(
        name=name or f"seed{seed}-{spec['layout']}-{spec['windows']}-n{n}",
        seed=seed,
        profile=MappingProxyType(dict(spec)),
        xs=tuple(xs),
        ys=tuple(ys),
        demand=tuple(demand),
        service=tuple(service),
        tw_open=tuple(tw_open),
        tw_close=tuple(tw_close),
        capacities=tuple(capacities),
        starts=tuple(starts),
        max_duration=duration_cap,
        horizon=HORIZON,
    )

    witness = Witness(
        routes=tuple(routes),
        arrivals=arrivals,
        cost_units=100 * travel_total,  # the witness never waits, by construction
    )
    return instance, witness


def _spare_vehicles(fleet: str, used: int) -> int:
    """Spare vehicles beyond what the hidden witness needs.

    Counter-intuitively, the fleet is kept generous on purpose.  The objective
    already punishes using another vehicle -- an extra route pays for two more
    depot legs -- so nothing is given away by making vehicles available.  What
    a tight vehicle count *would* do is turn the instance into a feasibility
    puzzle, where most submissions score zero and the ones that score at all
    are separated by luck rather than route quality.  Difficulty belongs in
    capacity, windows, and duration, which squeeze the objective instead of
    the feasible set.
    """
    if fleet == "tight":
        return max(2, used // 2)
    if fleet == "heterogeneous":
        return max(2, used * 3 // 4)
    return max(3, used)


def make_instance(seed: int, spec: Mapping[str, object], name: Optional[str] = None) -> Instance:
    """Public entry point. The witness stays behind."""
    return generate(seed, spec, name)[0]


# ---------------------------------------------------------------------------
# Public suite
# ---------------------------------------------------------------------------

# Published, fixed, and small enough to iterate on. These teach the format and
# the scoring; they are explicitly not a preview of the private mixture, which
# weights the shifted families far more heavily.
PUBLIC_SUITE: Tuple[Dict[str, object], ...] = (
    # Every instance sits on the shifted side of at least three axes: ring and
    # mixed layouts, heavy/light demand, tight or wave windows, tight or
    # heterogeneous fleets. Sizes run to the published maximum of 1000.
    {"name": "d01-ring-tight", "seed": 7101, "spec": profile(80, "ring", "heavy_light", "tight", "tight")},
    {"name": "d02-mixed-waves", "seed": 7102, "spec": profile(120, "mixed", "heavy_light", "waves", "heterogeneous")},
    {"name": "d03-uniform-tightfleet", "seed": 7103, "spec": profile(160, "uniform", "narrow", "tight", "tight")},
    {"name": "d04-clustered-waves", "seed": 7104, "spec": profile(220, "clustered", "heavy_light", "waves", "tight")},
    {"name": "d05-ring-hetero", "seed": 7105, "spec": profile(300, "ring", "skewed", "tight", "heterogeneous")},
    {"name": "d06-mixed-tight", "seed": 7106, "spec": profile(400, "mixed", "heavy_light", "tight", "tight")},
    {"name": "d07-ring-waves-hetero", "seed": 7107, "spec": profile(520, "ring", "heavy_light", "waves", "heterogeneous")},
    {"name": "d08-uniform-waves", "seed": 7108, "spec": profile(650, "uniform", "skewed", "waves", "tight")},
    {"name": "d09-mixed-hetero-xl", "seed": 7109, "spec": profile(820, "mixed", "heavy_light", "tight", "heterogeneous")},
    {"name": "d10-clustered-xxl", "seed": 7110, "spec": profile(1000, "clustered", "heavy_light", "waves", "tight")},
)

# The self-check runs two of them, small enough to finish well under 30s.
SELF_CHECK_NAMES = ("d01-ring-tight", "d02-mixed-waves")


def sample_profile(rng: Rng, shift_weight: int = 35) -> Dict[str, object]:
    """Draw a profile. ``shift_weight`` is the percentage pull toward the
    shifted side of each axis.

    The private suite calls this with the organizer's weighting; ``run.py
    --seeds`` calls it with a gentler one so practice instances resemble the
    public set rather than leaking the private mixture.
    """
    def pick(easy: Sequence[str], hard: Sequence[str]) -> str:
        pool = hard if rng.below(100) < shift_weight else easy
        return rng.choice(pool)

    n = rng.choice((60, 90, 120, 180, 240, 320, 420, 560, 700, 880))
    return profile(
        n=n,
        layout=pick(("uniform", "clustered"), ("ring", "mixed")),
        demand=pick(("narrow", "skewed"), ("heavy_light",)),
        windows=pick(("broad",), ("tight", "waves")),
        fleet=pick(("comfortable",), ("tight", "heterogeneous")),
    )


def profile_from_seed(seed: int, salt: str = "public-practice", shift_weight: int = 35) -> Dict[str, object]:
    return sample_profile(Rng(derive_seed(salt, SCHEMA_VERSION, seed)), shift_weight)


def practice_instance(seed: int) -> Instance:
    """An unpublished instance a contestant can generate for themselves.

    Same generator, same schema, no published anchors -- ``run.py`` scores
    these provisionally against a live baseline run.
    """
    spec = profile_from_seed(seed, "public-practice", shift_weight=30)
    return make_instance(seed, spec, f"practice-{seed}")


def public_instances(names: Optional[Sequence[str]] = None) -> List[Instance]:
    wanted = set(names) if names else None
    out = []
    for entry in PUBLIC_SUITE:
        if wanted is None or entry["name"] in wanted:
            out.append(make_instance(int(entry["seed"]), entry["spec"], str(entry["name"])))
    return out


def budget_for(instance: Instance) -> float:
    """Per-instance compute budget in seconds.

    Grows with the instance so that large cases get real search time, but
    stays small enough that the 1% checkpoint (a few hundred milliseconds)
    genuinely constrains how a solver spends its first moments.
    """
    return round(4.0 + instance.n / 40.0, 2)


def write_json(instance: Instance, path: str) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(instance.to_json(), handle, indent=2)
        handle.write("\n")


if __name__ == "__main__":  # quick manual inspection
    for entry in PUBLIC_SUITE:
        inst, wit = generate(int(entry["seed"]), entry["spec"], str(entry["name"]))
        print(
            f"{inst.name:26s} n={inst.n:4d} vehicles={inst.num_vehicles:3d} "
            f"cap={inst.capacities[0]:4d} dur_cap={inst.max_duration:6d} "
            f"witness_routes={len(wit.routes):3d} witness_cost={wit.cost_units / 6000:9.1f} "
            f"digest={inst.digest():016x}"
        )
