"""Canonicalisation, feasibility checking, and the canonical cost.

A submission says only *what order* each machine works in.  This module turns
that into the unique earliest schedule consistent with that order -- pushing
every operation as early as job precedence, machine availability, setup time
and release dates allow -- and costs the result.  Two consequences worth being
explicit about:

* A solver cannot gain anything by submitting start times that idle a machine.
  Canonicalisation left-shifts everything, so only the *order* is a decision.
  Explicit start times are accepted purely as another way to express an order.
* A supplied objective value is never read.  The evaluator recomputes
  makespan and tardiness from the instance, every time.

The cost combines two normalised terms, in integers so the ranking key stays
exact:

    cost = 650000 * makespan / lb_makespan  +  350000 * TWT / twt_scale

``lb_makespan`` is a real lower bound (the busiest machine's total work, and
the longest job chain), so the first term is at least 650000 and reaches it
only for a perfectly packed schedule.  ``twt_scale`` is an instance-intrinsic
scale, not a reference result, so the cost of a schedule never depends on how
well anyone else did.

**Deadlock.** A machine order that is individually sensible can still be
globally impossible: machine A waiting on a job that machine B is holding
while B waits on a job A is holding.  The disjunctive graph then has a cycle
and no schedule exists.  This is the single most common way a first
implementation fails here, so the rejection reason names it explicitly.
"""

from __future__ import annotations

from typing import Any, List, Mapping, Optional, Sequence, Tuple

from data import Instance

MAKESPAN_WEIGHT = 650_000
TARDINESS_WEIGHT = 350_000
COST_SCALE = 1_000_000  # cost_units / COST_SCALE is the normalised cost


class Infeasible(ValueError):
    """A submission that cannot be turned into a schedule."""


def evaluate(instance: Instance, plan: Any) -> Tuple[Optional[int], Optional[str]]:
    try:
        return canonical_cost(instance, plan), None
    except Infeasible as exc:
        return None, str(exc)
    except (TypeError, ValueError, IndexError, KeyError, AttributeError) as exc:
        return None, f"malformed submission: {exc}"


def canonical_cost(instance: Instance, plan: Any) -> int:
    schedule = canonicalise(instance, plan)
    return (
        MAKESPAN_WEIGHT * schedule["makespan"] // instance.lb_makespan
        + TARDINESS_WEIGHT * schedule["twt"] // instance.twt_scale
    )


def canonicalise(instance: Instance, plan: Any) -> dict:
    """Earliest schedule consistent with the submitted machine orders."""
    sequences = _as_sequences(instance, plan)

    total = instance.total_ops
    op_job = instance.op_job
    op_machine = instance.op_machine
    op_duration = instance.op_duration
    op_group = instance.op_group
    job_ops = instance.job_ops
    setup_set = instance._setup_set
    setup_matrix = instance.setup_matrix

    # Predecessor counts and successor edges over the disjunctive graph.
    indegree = [0] * total
    job_next: List[int] = [-1] * total
    machine_next: List[int] = [-1] * total

    for ops in job_ops:
        for position in range(len(ops) - 1):
            job_next[ops[position]] = ops[position + 1]
            indegree[ops[position + 1]] += 1

    for machine, sequence in enumerate(sequences):
        for position in range(len(sequence) - 1):
            machine_next[sequence[position]] = sequence[position + 1]
            indegree[sequence[position + 1]] += 1

    earliest = [0] * total
    for job, ops in enumerate(job_ops):
        if ops:
            earliest[ops[0]] = instance.release[job]

    ready = [op for op in range(total) if indegree[op] == 0]
    finish = [0] * total
    processed = 0

    while ready:
        op = ready.pop()
        start = earliest[op]
        end = start + op_duration[op]
        finish[op] = end
        processed += 1

        following = job_next[op]
        if following >= 0:
            if end > earliest[following]:
                earliest[following] = end
            indegree[following] -= 1
            if indegree[following] == 0:
                ready.append(following)

        following = machine_next[op]
        if following >= 0:
            machine = op_machine[op]
            setup = (
                setup_matrix[op_group[op]][op_group[following]]
                if machine in setup_set
                else 0
            )
            if end + setup > earliest[following]:
                earliest[following] = end + setup
            indegree[following] -= 1
            if indegree[following] == 0:
                ready.append(following)

    if processed != total:
        stuck = [op for op in range(total) if indegree[op] > 0]
        raise Infeasible(
            f"deadlock: {len(stuck)} operations can never start because the "
            f"machine orders form a cycle (e.g. operation {stuck[0]} on machine "
            f"{op_machine[stuck[0]]}). Two machines are each waiting for a job "
            f"the other is holding."
        )

    makespan = 0
    twt = 0
    completion = [0] * instance.n_jobs
    for job, ops in enumerate(job_ops):
        done = finish[ops[-1]] if ops else instance.release[job]
        completion[job] = done
        if done > makespan:
            makespan = done
        late = done - instance.due[job]
        if late > 0:
            twt += instance.weight[job] * late

    return {
        "starts": [finish[op] - op_duration[op] for op in range(total)],
        "finish": finish,
        "completion": completion,
        "makespan": makespan,
        "twt": twt,
    }


def _as_sequences(instance: Instance, plan: Any) -> List[List[int]]:
    """Normalise either submission form into one machine order per machine."""
    if plan is None:
        raise Infeasible("no schedule submitted")

    if isinstance(plan, Mapping):
        if "sequences" in plan:
            raw = plan["sequences"]
        elif "starts" in plan:
            return _sequences_from_starts(instance, plan["starts"])
        else:
            raise Infeasible("submission dict needs a 'sequences' or 'starts' key")
    else:
        raw = plan

    if isinstance(raw, (str, bytes)):
        raise Infeasible("sequences must be a list per machine, not a string")

    try:
        sequences = [list(sequence) for sequence in raw]
    except TypeError as exc:
        raise Infeasible(f"sequences is not a list of lists: {exc}") from exc

    if len(sequences) > instance.n_machines:
        raise Infeasible(
            f"got {len(sequences)} machine sequences, instance has "
            f"{instance.n_machines} machines"
        )
    while len(sequences) < instance.n_machines:
        sequences.append([])

    total = instance.total_ops
    seen = bytearray(total)
    for machine, sequence in enumerate(sequences):
        for op in sequence:
            if not isinstance(op, int) or isinstance(op, bool):
                raise Infeasible(f"operation id {op!r} is not an integer")
            if op < 0 or op >= total:
                raise Infeasible(f"operation id {op} outside 0..{total - 1}")
            if seen[op]:
                raise Infeasible(f"operation {op} appears in more than one place")
            if instance.op_machine[op] != machine:
                raise Infeasible(
                    f"operation {op} needs machine {instance.op_machine[op]} "
                    f"but was sequenced on machine {machine}"
                )
            seen[op] = 1

    if sum(seen) != total:
        missing = [op for op in range(total) if not seen[op]]
        raise Infeasible(
            f"{len(missing)} operations were never sequenced "
            f"(first few: {missing[:5]})"
        )
    return sequences


def _sequences_from_starts(instance: Instance, starts: Any) -> List[List[int]]:
    """Derive machine orders by sorting each machine's operations by start time.

    The submitted times themselves are discarded after this: only the order
    they imply survives, and canonicalisation recomputes every time from
    scratch.
    """
    total = instance.total_ops
    if isinstance(starts, Mapping):
        times = {}
        for key, value in starts.items():
            op = int(key)
            if op < 0 or op >= total:
                raise Infeasible(f"operation id {op} outside 0..{total - 1}")
            times[op] = int(value)
    else:
        values = list(starts)
        if len(values) != total:
            raise Infeasible(
                f"got {len(values)} start times, instance has {total} operations"
            )
        times = {op: int(value) for op, value in enumerate(values)}

    if len(times) != total:
        raise Infeasible(f"got {len(times)} start times, instance has {total} operations")

    sequences: List[List[int]] = []
    for machine in range(instance.n_machines):
        ops = list(instance.machine_ops[machine])
        ops.sort(key=lambda op: (times[op], op))
        sequences.append(ops)
    return sequences


# ---------------------------------------------------------------------------
# Reporting -- not on the hot path
# ---------------------------------------------------------------------------


def describe(instance: Instance, plan: Any) -> dict:
    schedule = canonicalise(instance, plan)
    tardy = sum(
        1
        for job in range(instance.n_jobs)
        if schedule["completion"][job] > instance.due[job]
    )
    return {
        "makespan": schedule["makespan"],
        "lb_makespan": instance.lb_makespan,
        "makespan_ratio": round(schedule["makespan"] / instance.lb_makespan, 4),
        "twt": schedule["twt"],
        "twt_scale": instance.twt_scale,
        "tardy_jobs": tardy,
        "jobs": instance.n_jobs,
        "cost_units": canonical_cost(instance, plan),
    }


def normalised_cost(cost_units: Optional[float]) -> Optional[float]:
    return None if cost_units is None else round(cost_units / COST_SCALE, 4)
