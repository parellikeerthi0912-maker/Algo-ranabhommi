"""The submission interface, plus the conveniences every solver would rewrite.

A submission is one class:

    class Scheduler:
        def solve(self, instance, submit_candidate):
            ...
            return final_schedule

Submit either form:

    {"sequences": [[op, op, ...], ...]}   one list per machine, in order
    {"starts":    [t0, t1, ...]}          one time per operation

Only the *order* matters.  The evaluator recomputes every start time as early
as precedence, machine availability, setups and releases allow, so idle gaps
you build in are removed, and a start-time submission is just another way of
expressing an order.

Two representations, and the gap between them
---------------------------------------------
Machine sequences are the natural decision variables, but most orders are not
schedulable: if two machines each wait on a job the other holds, the
disjunctive graph has a cycle and nothing can run.  Searching directly in that
space means every move risks a deadlock.

The *operation-based permutation* avoids this entirely.  Write a list in which
job ``j`` appears once per operation it has; read it left to right, taking the
k-th appearance of job ``j`` to mean its k-th operation, and append that
operation to its machine's sequence.  Every such list decodes to a valid,
deadlock-free schedule, and every schedule has such a list.  ``decode`` below
does this, so a solver working in permutation space can never submit garbage.

That safety costs something.  Permutation moves are blunt -- swapping two
positions perturbs many machines at once -- whereas the sharpest known moves
for this problem reverse a pair of adjacent operations *on the critical path*,
which provably keeps the graph acyclic while directly attacking the makespan.
The organizer reference works in sequence space with critical-block moves.
Closing that gap is most of the distance between a mid-table score and a good
one.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, List, Optional, Sequence

import validator
from data import Instance

SubmitCandidate = Callable[[object], dict]


class Scheduler(ABC):
    """Base class. Subclass it, or just match the shape -- duck typing is fine."""

    @abstractmethod
    def solve(self, instance: Instance, submit_candidate: SubmitCandidate) -> object: ...


# ---------------------------------------------------------------------------
# Representations
# ---------------------------------------------------------------------------


def decode(instance: Instance, job_sequence: Sequence[int]) -> List[List[int]]:
    """Operation-based permutation -> machine sequences. Always valid.

    ``job_sequence`` must contain job ``j`` exactly ``len(job_ops[j])`` times.
    The result never deadlocks, because operations are committed to machines
    in an order that is itself a linear extension of job precedence.
    """
    sequences: List[List[int]] = [[] for _ in range(instance.n_machines)]
    cursor = [0] * instance.n_jobs
    for job in job_sequence:
        index = cursor[job]
        ops = instance.job_ops[job]
        if index >= len(ops):
            raise ValueError(f"job {job} appears more often than it has operations")
        op = ops[index]
        sequences[instance.op_machine[op]].append(op)
        cursor[job] = index + 1
    for job in range(instance.n_jobs):
        if cursor[job] != len(instance.job_ops[job]):
            raise ValueError(f"job {job} appears {cursor[job]} times, needs "
                             f"{len(instance.job_ops[job])}")
    return sequences


def flat_job_sequence(instance: Instance, job_order: Sequence[int]) -> List[int]:
    """Expand a job ordering into a decodable permutation, jobs run back to back.

    The fastest way to get *something* valid on the board. It is also a poor
    schedule -- it serialises whole jobs -- so treat it as a floor to beat
    before the 1% checkpoint, not as a starting point worth refining.
    """
    out: List[int] = []
    for job in job_order:
        out.extend([job] * len(instance.job_ops[job]))
    return out


def interleaved_job_sequence(instance: Instance, job_order: Sequence[int]) -> List[int]:
    """Round-robin the same job ordering instead of running jobs back to back.

    Still O(total operations) and still instant, but it keeps every machine
    busy from the start rather than one at a time, which is worth a large
    fraction of the makespan for free.
    """
    out: List[int] = []
    cursor = {job: 0 for job in job_order}
    remaining = sum(len(instance.job_ops[job]) for job in job_order)
    while remaining:
        for job in job_order:
            if cursor[job] < len(instance.job_ops[job]):
                out.append(job)
                cursor[job] += 1
                remaining -= 1
    return out


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def dispatch(
    instance: Instance,
    priority: Callable[[int, int, int], object],
) -> List[List[int]]:
    """Build machine sequences by simulating a dispatching rule.

    At each step every job offers its next operation; ``priority(job, op,
    earliest_start)`` scores them and the smallest key wins.  Returning
    ``earliest_start`` first makes it a non-delay schedule; returning a due-date
    term first makes it a due-date rule; and so on.

    This is O(jobs) per operation, so on the larger instances it costs a
    noticeable slice of the budget.  Get something valid submitted with
    ``decode``/``interleaved_job_sequence`` before calling it.
    """
    machine_free = [0] * instance.n_machines
    machine_group = [-1] * instance.n_machines
    job_ready = list(instance.release)
    cursor = [0] * instance.n_jobs

    sequences: List[List[int]] = [[] for _ in range(instance.n_machines)]
    active = [job for job in range(instance.n_jobs) if instance.job_ops[job]]
    setup_set = instance._setup_set
    setup_matrix = instance.setup_matrix

    while active:
        best_key = None
        best = None
        for slot, job in enumerate(active):
            op = instance.job_ops[job][cursor[job]]
            machine = instance.op_machine[op]
            previous = machine_group[machine]
            setup = (
                setup_matrix[previous][instance.op_group[op]]
                if previous >= 0 and machine in setup_set
                else 0
            )
            start = machine_free[machine] + setup
            if job_ready[job] > start:
                start = job_ready[job]
            key = priority(job, op, start)
            if best_key is None or key < best_key:
                best_key = key
                best = (slot, job, op, machine, start)

        slot, job, op, machine, start = best
        finish = start + instance.op_duration[op]
        sequences[machine].append(op)
        machine_free[machine] = finish
        machine_group[machine] = instance.op_group[op]
        job_ready[job] = finish
        cursor[job] += 1
        if cursor[job] >= len(instance.job_ops[job]):
            active.pop(slot)

    return sequences


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def schedule(instance: Instance, sequences: Sequence[Sequence[int]]) -> Optional[dict]:
    """Canonical schedule for these machine orders, or ``None`` on deadlock.

    Same routine the evaluator uses. O(total operations) per call, which is
    cheap once and expensive as the inner loop of a local search -- a move
    evaluated by rescheduling everything costs as much as the whole schedule.
    """
    try:
        return validator.canonicalise(instance, {"sequences": [list(s) for s in sequences]})
    except validator.Infeasible:
        return None


def schedule_cost(instance: Instance, sequences: Sequence[Sequence[int]]) -> Optional[int]:
    """Cost in the evaluator's units, or ``None`` if the orders deadlock."""
    result = schedule(instance, sequences)
    if result is None:
        return None
    return (
        validator.MAKESPAN_WEIGHT * result["makespan"] // instance.lb_makespan
        + validator.TARDINESS_WEIGHT * result["twt"] // instance.twt_scale
    )
