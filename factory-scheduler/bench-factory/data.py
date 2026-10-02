"""Instance generation for the factory-scheduler benchmark.

A job is a customer order: an ordered chain of operations, each needing one
specific machine for a fixed time.  A machine does one thing at a time.  That
is the whole model, and it is the classic job shop -- with three additions
that make it a scheduling problem rather than a textbook exercise: release
times, weighted due dates, and sequence-dependent setups.

Operations are stored flat
--------------------------
Every operation has a global id.  ``op_job``, ``op_machine``, ``op_duration``
and ``op_group`` are tuples indexed by that id, and ``job_ops``/``machine_ops``
index into it.  Submissions talk in op ids.  This is not a stylistic choice:
scheduling inner loops touch these arrays millions of times, and tuple
indexing is several times faster in Python than attribute access on per-
operation objects.

Feasibility by construction
---------------------------
Due dates are derived from a hidden witness schedule rather than invented.
A random due date is either unmissable or unreachable, and you cannot tell
which without solving the instance.  Generation therefore simulates a real
(mediocre) schedule first, then sets each job's due date relative to when
that schedule happened to finish it.  Tightness becomes a dial, and "every
job is late no matter what" never happens by accident.

Unlike the routing benchmark, feasibility itself is free here: any permutation
of each machine's operations that avoids a deadlock is a valid schedule.  The
difficulty is entirely in the objective, which is where it belongs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from benchkit.rng import Rng, derive_seed, digest_ints

SCHEMA_VERSION = "factory-1.0"

SETUP_GROUPS = 4  # tool/colour families a machine changes between

WORKLOADS = ("balanced", "bottleneck")
DUE_PROFILES = ("generous", "tight")
DURATION_PROFILES = ("uniform", "heavy_tail")
ROUTINGS = ("random", "reentrant")
SETUP_PROFILES = ("none", "groups")


@dataclass(frozen=True)
class Instance:
    name: str
    seed: int
    profile: Mapping[str, object]

    n_jobs: int
    n_machines: int

    op_job: Tuple[int, ...]
    op_machine: Tuple[int, ...]
    op_duration: Tuple[int, ...]
    op_group: Tuple[int, ...]
    job_ops: Tuple[Tuple[int, ...], ...]
    machine_ops: Tuple[Tuple[int, ...], ...]

    release: Tuple[int, ...]
    due: Tuple[int, ...]
    weight: Tuple[int, ...]

    setup_machines: Tuple[int, ...]
    setup_matrix: Tuple[Tuple[int, ...], ...]

    lb_makespan: int
    twt_scale: int
    _digest: int = field(default=0, repr=False)

    @property
    def total_ops(self) -> int:
        return len(self.op_job)

    @property
    def size(self) -> int:
        """What the harness reports as instance size. Operations, here --
        jobs alone would understate a 10-operation job."""
        return len(self.op_job)

    def setup_time(self, machine: int, previous_group: int, group: int) -> int:
        """Changeover cost on ``machine`` between two operation families."""
        if previous_group < 0 or machine not in self._setup_set:
            return 0
        return self.setup_matrix[previous_group][group]

    @property
    def _setup_set(self) -> frozenset:
        cached = getattr(self, "_setup_frozen", None)
        if cached is None:
            cached = frozenset(self.setup_machines)
            object.__setattr__(self, "_setup_frozen", cached)
        return cached

    # -- identity -----------------------------------------------------------

    def digest(self) -> int:
        if self._digest:
            return self._digest
        value = self.compute_digest()
        object.__setattr__(self, "_digest", value)
        return value

    def compute_digest(self) -> int:
        stream: List[int] = [self.n_jobs, self.n_machines, self.lb_makespan, self.twt_scale]
        for seq in (
            self.op_job,
            self.op_machine,
            self.op_duration,
            self.op_group,
            self.release,
            self.due,
            self.weight,
            self.setup_machines,
        ):
            stream.extend(seq)
        for row in self.setup_matrix:
            stream.extend(row)
        profile_bytes = json.dumps(dict(self.profile), sort_keys=True, separators=(",", ":")).encode()
        stream.extend(profile_bytes)
        return digest_ints(stream)

    def to_json(self) -> Dict[str, object]:
        return {
            "schema": SCHEMA_VERSION,
            "name": self.name,
            "seed": self.seed,
            "profile": dict(self.profile),
            "n_jobs": self.n_jobs,
            "n_machines": self.n_machines,
            "lb_makespan": self.lb_makespan,
            "twt_scale": self.twt_scale,
            "setup_machines": list(self.setup_machines),
            "setup_matrix": [list(row) for row in self.setup_matrix],
            "jobs": [
                {
                    "id": j,
                    "release": self.release[j],
                    "due": self.due[j],
                    "weight": self.weight[j],
                    "operations": [
                        {
                            "op": op,
                            "machine": self.op_machine[op],
                            "duration": self.op_duration[op],
                            "group": self.op_group[op],
                        }
                        for op in self.job_ops[j]
                    ],
                }
                for j in range(self.n_jobs)
            ],
            "digest": f"{self.digest():016x}",
        }


@dataclass(frozen=True)
class Witness:
    """The hidden schedule the due dates were derived from. Organizer-only."""

    starts: Tuple[int, ...]
    completion: Tuple[int, ...]
    makespan: int


def profile(
    jobs: int,
    machines: int,
    workload: str = "balanced",
    due: str = "generous",
    durations: str = "uniform",
    routing: str = "random",
    setups: str = "none",
) -> Dict[str, object]:
    for value, allowed, label in (
        (workload, WORKLOADS, "workload"),
        (due, DUE_PROFILES, "due"),
        (durations, DURATION_PROFILES, "durations"),
        (routing, ROUTINGS, "routing"),
        (setups, SETUP_PROFILES, "setups"),
    ):
        if value not in allowed:
            raise ValueError(f"unknown {label}: {value!r} (expected one of {allowed})")
    if not 10 <= jobs <= 500:
        raise ValueError("jobs must be within the published 10..500 range")
    if not 5 <= machines <= 100:
        raise ValueError("machines must be within the published 5..100 range")
    return {
        "jobs": jobs,
        "machines": machines,
        "workload": workload,
        "due": due,
        "durations": durations,
        "routing": routing,
        "setups": setups,
    }


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


def _machine_weights(rng: Rng, machines: int, workload: str) -> List[int]:
    if workload == "balanced":
        return [100] * machines
    # A handful of machines attract far more work than the rest. Those become
    # the bottlenecks, and every good decision in the instance is really a
    # decision about what runs on them.
    weights = [100] * machines
    hot = max(1, machines // 6)
    for index in rng.sample(range(machines), hot):
        weights[index] = 420
    return weights


def _durations(rng: Rng, count: int, kind: str) -> List[int]:
    if kind == "uniform":
        return [rng.randint(15, 95) for _ in range(count)]
    # heavy_tail: mostly short, occasionally enormous. One long operation on a
    # contended machine can block a line for the rest of the horizon, which is
    # exactly the decision the benchmark wants to test.
    out = []
    for _ in range(count):
        if rng.chance(1, 12):
            out.append(rng.randint(220, 460))
        else:
            out.append(rng.randint(10, 60))
    return out


def _routing_for_job(
    rng: Rng, machines: int, length: int, routing: str, weights: Sequence[int]
) -> List[int]:
    if routing == "random" and length <= machines:
        chosen = rng.sample(range(machines), length)
        # Sampling ignores the workload weighting, so re-bias by swapping some
        # picks for weighted draws while keeping them distinct.
        if max(weights) != min(weights):
            distinct = set(chosen)
            for position in range(length):
                candidate = rng.weighted_choice(weights)
                if candidate not in distinct:
                    distinct.discard(chosen[position])
                    distinct.add(candidate)
                    chosen[position] = candidate
        return chosen

    # Re-entrant: a job may return to a machine it has already used, which is
    # what real plants with a shared oven or test rig look like. Consecutive
    # repeats are avoided because they would just be one longer operation.
    out: List[int] = []
    for _ in range(length):
        machine = rng.weighted_choice(weights)
        if out and out[-1] == machine:
            machine = (machine + 1) % machines
        out.append(machine)
    return out


def generate(
    seed: int, spec: Mapping[str, object], name: Optional[str] = None
) -> Tuple[Instance, Witness]:
    jobs = int(spec["jobs"])
    machines = int(spec["machines"])
    root = Rng(
        derive_seed("bench-factory", SCHEMA_VERSION, seed, json.dumps(dict(spec), sort_keys=True))
    )

    shape = root.spawn("shape")
    weights = _machine_weights(shape, machines, str(spec["workload"]))

    op_job: List[int] = []
    op_machine: List[int] = []
    op_group: List[int] = []
    job_ops: List[Tuple[int, ...]] = []

    routing = str(spec["routing"])
    for job in range(jobs):
        # Jobs visit most of the shop, not a handful of machines. This is the
        # difference between a hard job shop and a trivial one: when each job
        # needs only a few machines there is enough parallelism that any list
        # schedule lands within a few percent of the lower bound, and no
        # amount of search can separate submissions. When jobs contend for
        # nearly every machine, the ordering decisions actually matter.
        if routing == "random":
            length = shape.randint(max(2, machines * 2 // 3), machines)
        else:
            length = shape.randint(max(2, machines * 3 // 4), machines + machines // 3)
        route = _routing_for_job(shape, machines, length, routing, weights)
        ids = []
        for machine in route:
            ids.append(len(op_job))
            op_job.append(job)
            op_machine.append(machine)
            op_group.append(shape.below(SETUP_GROUPS))
        job_ops.append(tuple(ids))

    duration_rng = root.spawn("durations")
    op_duration = _durations(duration_rng, len(op_job), str(spec["durations"]))

    machine_ops: List[List[int]] = [[] for _ in range(machines)]
    for op, machine in enumerate(op_machine):
        machine_ops[machine].append(op)

    # -- setups -------------------------------------------------------------
    setup_rng = root.spawn("setups")
    if str(spec["setups"]) == "none":
        setup_machines: List[int] = []
        setup_matrix = [[0] * SETUP_GROUPS for _ in range(SETUP_GROUPS)]
    else:
        count = max(1, machines * 2 // 5)
        setup_machines = sorted(setup_rng.sample(range(machines), count))
        setup_matrix = []
        for a in range(SETUP_GROUPS):
            row = []
            for b in range(SETUP_GROUPS):
                row.append(0 if a == b else setup_rng.randint(15, 70))
            setup_matrix.append(row)

    # -- releases -----------------------------------------------------------
    release_rng = root.spawn("releases")
    total_work = sum(op_duration)
    spread = max(1, total_work // max(1, machines) // 2)
    release = [0 if release_rng.chance(1, 3) else release_rng.below(spread) for _ in range(jobs)]

    # -- weights ------------------------------------------------------------
    weight_rng = root.spawn("weights")
    weight = [
        weight_rng.randint(8, 12) if weight_rng.chance(1, 10) else weight_rng.randint(1, 5)
        for _ in range(jobs)
    ]

    provisional = Instance(
        name=name or f"seed{seed}-{spec['workload']}-j{jobs}",
        seed=seed,
        profile=MappingProxyType(dict(spec)),
        n_jobs=jobs,
        n_machines=machines,
        op_job=tuple(op_job),
        op_machine=tuple(op_machine),
        op_duration=tuple(op_duration),
        op_group=tuple(op_group),
        job_ops=tuple(job_ops),
        machine_ops=tuple(tuple(ops) for ops in machine_ops),
        release=tuple(release),
        due=tuple(0 for _ in range(jobs)),  # replaced below
        weight=tuple(weight),
        setup_machines=tuple(setup_machines),
        setup_matrix=tuple(tuple(row) for row in setup_matrix),
        lb_makespan=1,
        twt_scale=1,
    )

    witness = _witness_schedule(provisional)

    # -- due dates from the witness ----------------------------------------
    due_rng = root.spawn("due")
    tight = str(spec["due"]) == "tight"
    due = []
    for job in range(jobs):
        finished = witness.completion[job]
        factor = due_rng.randint(92, 116) if tight else due_rng.randint(125, 175)
        due.append(max(release[job] + 1, finished * factor // 100))

    # -- published bounds ---------------------------------------------------
    machine_load = [0] * machines
    for op, machine in enumerate(op_machine):
        machine_load[machine] += op_duration[op]
    job_span = [
        release[job] + sum(op_duration[op] for op in job_ops[job]) for job in range(jobs)
    ]
    lb_makespan = max(max(machine_load), max(job_span), 1)
    # Tardiness scale. This divisor decides how much the 35% tardiness term is
    # actually worth, and the obvious choice -- total weighted work -- is far
    # too large: real schedules produce tardiness on the order of a few percent
    # of it, so the term contributes ~0.01 and the objective collapses into
    # pure makespan, silently voiding the whole "tight due dates" family.
    #
    # A tenth of the total weighted work puts a typical tight-due-date schedule
    # at 0.10-0.35 on this term, which is the intended weight. Read it as:
    # being late by a tenth of everything the shop has to do costs the full
    # 0.35. It is a property of the instance alone -- no solver's result enters
    # it -- so the cost of a schedule never depends on how well anyone else did.
    twt_scale = max(
        1,
        sum(weight[job] * sum(op_duration[op] for op in job_ops[job]) for job in range(jobs)) // 10,
    )

    instance = Instance(
        name=provisional.name,
        seed=seed,
        profile=MappingProxyType(dict(spec)),
        n_jobs=jobs,
        n_machines=machines,
        op_job=tuple(op_job),
        op_machine=tuple(op_machine),
        op_duration=tuple(op_duration),
        op_group=tuple(op_group),
        job_ops=tuple(job_ops),
        machine_ops=tuple(tuple(ops) for ops in machine_ops),
        release=tuple(release),
        due=tuple(due),
        weight=tuple(weight),
        setup_machines=tuple(setup_machines),
        setup_matrix=tuple(tuple(row) for row in setup_matrix),
        lb_makespan=lb_makespan,
        twt_scale=twt_scale,
    )
    return instance, witness


def _witness_schedule(instance: Instance) -> Witness:
    """A plain non-delay list schedule: earliest possible start wins.

    Deliberately mediocre.  Its only jobs are to prove the instance schedules
    at all and to give the due dates something real to hang on.
    """
    machines = instance.n_machines
    machine_free = [0] * machines
    machine_group = [-1] * machines
    job_ready = list(instance.release)
    next_index = [0] * instance.n_jobs

    starts = [0] * instance.total_ops
    remaining = instance.total_ops
    completion = [0] * instance.n_jobs

    while remaining:
        best = None
        for job in range(instance.n_jobs):
            index = next_index[job]
            if index >= len(instance.job_ops[job]):
                continue
            op = instance.job_ops[job][index]
            machine = instance.op_machine[op]
            setup = instance.setup_time(machine, machine_group[machine], instance.op_group[op])
            start = max(machine_free[machine] + setup, job_ready[job])
            key = (start, instance.op_duration[op], job)
            if best is None or key < best[0]:
                best = (key, job, op, machine, start)

        _, job, op, machine, start = best
        finish = start + instance.op_duration[op]
        starts[op] = start
        machine_free[machine] = finish
        machine_group[machine] = instance.op_group[op]
        job_ready[job] = finish
        completion[job] = finish
        next_index[job] += 1
        remaining -= 1

    return Witness(
        starts=tuple(starts),
        completion=tuple(completion),
        makespan=max(completion) if completion else 0,
    )


def make_instance(seed: int, spec: Mapping[str, object], name: Optional[str] = None) -> Instance:
    return generate(seed, spec, name)[0]


# ---------------------------------------------------------------------------
# Public suite
# ---------------------------------------------------------------------------

# Shapes are deliberately close to square. A job shop with many more jobs than
# machines has so much slack that every submission scores the same.
PUBLIC_SUITE: Tuple[Dict[str, object], ...] = (
    # Every instance has tight due dates and at least four shifted axes, and
    # the largest run to 150 jobs and 5000+ operations: a move evaluated by
    # rescheduling the whole shop is too slow here to search at all.
    {"name": "f01-small-reentrant", "seed": 8101,
     "spec": profile(15, 8, "bottleneck", "tight", "heavy_tail", "reentrant", "groups")},
    {"name": "f02-small-setups", "seed": 8102,
     "spec": profile(20, 10, "bottleneck", "tight", "heavy_tail", "random", "groups")},
    {"name": "f03-reentrant-heavy", "seed": 8103,
     "spec": profile(30, 12, "balanced", "tight", "heavy_tail", "reentrant", "groups")},
    {"name": "f04-bottleneck-setups", "seed": 8104,
     "spec": profile(40, 15, "bottleneck", "tight", "uniform", "reentrant", "groups")},
    {"name": "f05-bottleneck-heavy", "seed": 8105,
     "spec": profile(50, 18, "bottleneck", "tight", "heavy_tail", "reentrant", "none")},
    {"name": "f06-large-reentrant", "seed": 8106,
     "spec": profile(70, 20, "balanced", "tight", "heavy_tail", "reentrant", "groups")},
    {"name": "f07-large-bottleneck", "seed": 8107,
     "spec": profile(90, 24, "bottleneck", "tight", "heavy_tail", "random", "groups")},
    {"name": "f08-xl-mixed", "seed": 8108,
     "spec": profile(110, 28, "bottleneck", "tight", "heavy_tail", "reentrant", "groups")},
    {"name": "f09-xl-tight", "seed": 8109,
     "spec": profile(130, 30, "balanced", "tight", "heavy_tail", "reentrant", "groups")},
    {"name": "f10-xxl-bottleneck", "seed": 8110,
     "spec": profile(150, 35, "bottleneck", "tight", "heavy_tail", "reentrant", "groups")},
)

SELF_CHECK_NAMES = ("f01-small-reentrant", "f02-small-setups")


def public_instances(names: Optional[Sequence[str]] = None) -> List[Instance]:
    wanted = set(names) if names else None
    return [
        make_instance(int(entry["seed"]), entry["spec"], str(entry["name"]))
        for entry in PUBLIC_SUITE
        if wanted is None or entry["name"] in wanted
    ]


def budget_for(instance: Instance) -> float:
    """Per-instance compute budget in seconds, scaled by operation count."""
    return round(4.0 + instance.total_ops / 150.0, 2)


def sample_profile(rng: Rng, shift_weight: int = 40) -> Dict[str, object]:
    """Draw a profile. ``shift_weight`` is the pull toward the hard side."""

    def pick(easy: Sequence[str], hard: Sequence[str]) -> str:
        return rng.choice(hard if rng.below(100) < shift_weight else easy)

    jobs = rng.choice((12, 18, 25, 32, 45, 60, 75, 95, 120))
    # Keep the shop close to square: 1 to 4 jobs per machine.
    machines = max(5, min(100, jobs // rng.randint(1, 4) + rng.randint(2, 6)))
    return profile(
        jobs=jobs,
        machines=machines,
        workload=pick(("balanced",), ("bottleneck",)),
        due=pick(("generous",), ("tight",)),
        durations=pick(("uniform",), ("heavy_tail",)),
        routing=pick(("random",), ("reentrant",)),
        setups=pick(("none",), ("groups",)),
    )


def profile_from_seed(seed: int, salt: str = "public-practice", shift_weight: int = 40):
    return sample_profile(Rng(derive_seed(salt, SCHEMA_VERSION, seed)), shift_weight)


def practice_instance(seed: int) -> Instance:
    spec = profile_from_seed(seed, "public-practice", shift_weight=35)
    return make_instance(seed, spec, f"practice-{seed}")


def write_json(instance: Instance, path: str) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(instance.to_json(), handle, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    for entry in PUBLIC_SUITE:
        inst, wit = generate(int(entry["seed"]), entry["spec"], str(entry["name"]))
        print(
            f"{inst.name:24s} jobs={inst.n_jobs:4d} mach={inst.n_machines:3d} "
            f"ops={inst.total_ops:5d} lb={inst.lb_makespan:7d} "
            f"witness_mk={wit.makespan:7d} budget={budget_for(inst):5.1f}s "
            f"digest={inst.digest():016x}"
        )
