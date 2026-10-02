"""The published serious baseline. Match it exactly and you score 0.50.

Three stages:

1. **Instant valid output.** Jobs ordered by due date, round-robin interleaved,
   decoded to machine sequences.  O(total operations) and submitted within
   milliseconds, so the 1% checkpoint is never empty.
2. **A portfolio of dispatching rules** -- earliest due date, shortest
   processing time, weighted critical ratio, weighted shortest processing, and
   plain non-delay -- each simulated and kept if it beats the incumbent.  No
   single rule wins across bottlenecks, heavy tails and tight due dates, which
   is exactly why a portfolio beats picking a favourite.
3. **Adjacent-swap descent** over machine sequences: sweep every adjacent pair
   on every machine, keep a swap when the recomputed cost improves, repeat
   until nothing improves, then perturb and descend again.

Where the remaining points are.  Every move here is evaluated by rescheduling
the *entire* instance -- O(total operations) per candidate swap -- and moves
are chosen blind, without asking which operations actually determine the
makespan.  The organizer reference evaluates moves on the critical path only,
where a swap can actually change the objective, and keeps a tabu list so it
can walk through a worse solution to reach a better one.  On the large
instances that difference is worth more than an order of magnitude in moves
examined per second.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import List, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from adapter import (  # noqa: E402
    Scheduler,
    decode,
    dispatch,
    interleaved_job_sequence,
    schedule_cost,
)
from benchkit.rng import Rng  # noqa: E402
from data import Instance  # noqa: E402

SUBMIT_INTERVAL = 0.05
RESERVE = 0.15

# Descent restarts for the published baseline. Small on purpose: see
# _descend_and_perturb for why an unbounded search here breaks the scoring curve.
MAX_RESTARTS = 3


class DispatchSwapScheduler(Scheduler):
    def prepare(self, instance: Instance, deadline: float, submit=None) -> None:
        """Set up state without running anything.

        Split out so the organizer reference can reuse this whole pipeline as
        its starting point.  That reuse is deliberate: it guarantees the
        reference is never worse than the baseline it anchors against, which
        would otherwise invert the scoring curve -- and on tardiness-dominated
        instances a blind full-neighbourhood descent genuinely does find
        things a makespan-focused search misses.
        """
        self.inst = instance
        self.submit = submit if submit is not None else self._local_sink
        self.started = time.perf_counter()
        self.deadline = deadline
        self.last_submit = 0.0
        self.best_cost: Optional[int] = None
        self.best_sequences: Optional[List[List[int]]] = None
        self.rng = Rng(instance.digest() ^ 0xD15_9A77)

    def _local_sink(self, plan) -> dict:
        """Stands in for the harness when this runs as a subroutine."""
        cost = schedule_cost(self.inst, plan["sequences"])
        return {
            "accepted": cost is not None,
            "reason": None if cost is not None else "deadlock",
            "cost": cost,
            "best": self.best_cost,
            "elapsed_s": self._elapsed(),
            "remaining_s": None,
        }

    def build(self):
        """Instant output, rule portfolio, then descent -- within the deadline."""
        self._instant()
        self._portfolio()
        self._descend_and_perturb()
        return self.best_sequences, self.best_cost

    def solve(self, instance: Instance, submit_candidate) -> object:
        # 1e9 is replaced by the real deadline from the first receipt.
        self.prepare(instance, time.perf_counter() + 1e9, submit_candidate)
        self.build()
        return {"sequences": self.best_sequences} if self.best_sequences else None

    # -- bookkeeping --------------------------------------------------------

    def _remaining(self) -> float:
        return self.deadline - time.perf_counter()

    def _elapsed(self) -> float:
        return time.perf_counter() - self.started

    def _offer(self, sequences: Sequence[Sequence[int]], force: bool = True) -> None:
        now = self._elapsed()
        if not force and now - self.last_submit < SUBMIT_INTERVAL:
            return
        receipt = self.submit({"sequences": [list(s) for s in sequences]})
        self.last_submit = now
        if receipt.get("remaining_s") is not None:
            self.deadline = time.perf_counter() + receipt["remaining_s"]
        if receipt["accepted"]:
            cost = receipt["cost"]
            if self.best_cost is None or cost < self.best_cost:
                self.best_cost = cost
                self.best_sequences = [list(s) for s in sequences]

    # -- stage 1 ------------------------------------------------------------

    def _instant(self) -> None:
        inst = self.inst
        order = sorted(
            range(inst.n_jobs),
            key=lambda j: (inst.due[j], inst.release[j], j),
        )
        self._offer(decode(inst, interleaved_job_sequence(inst, order)))

    # -- stage 2 ------------------------------------------------------------

    def _portfolio(self) -> None:
        inst = self.inst
        duration = inst.op_duration
        due = inst.due
        weight = inst.weight
        op_job = inst.op_job

        # Work still ahead of each job, so critical-ratio rules can tell an
        # almost-finished job from one that has barely started.
        remaining_work = [0] * inst.n_jobs
        for job in range(inst.n_jobs):
            remaining_work[job] = sum(duration[op] for op in inst.job_ops[job])

        def edd(job, op, start):
            return (due[job], start, job)

        def spt(job, op, start):
            return (duration[op], start, job)

        def non_delay(job, op, start):
            return (start, duration[op], job)

        def weighted_critical_ratio(job, op, start):
            slack = due[job] - start - remaining_work[job]
            return (slack * 100 // max(1, weight[job]), start, job)

        def wspt(job, op, start):
            return (-(weight[job] * 1000 // max(1, duration[op])), start, job)

        rules = [weighted_critical_ratio, edd, non_delay, wspt, spt]
        # Each dispatch is O(jobs) per operation, so on a large instance one
        # rule can cost seconds. Spend at most half the budget discovering
        # which rule suits this instance; the rest belongs to search.
        cutoff = self._remaining() / 2.0
        for rule in rules:
            if self._remaining() < max(RESERVE * 2, cutoff):
                break
            sequences = dispatch(self.inst, rule)
            self._offer(sequences)

        # Track what we will improve from. The remaining_work table is reused.
        self.remaining_work = remaining_work

    # -- stage 3 ------------------------------------------------------------

    def _descend_and_perturb(self) -> None:
        """Descend to a local optimum, then a bounded number of restarts.

        The restart cap is deliberate.  Uncapped, this converges and then runs
        flat while still consuming the whole budget -- measured on one
        instance: identical cost at 6.5s, 60s and 240s.  As the *published*
        baseline it defines the 0.50 anchor, so an over-strong version
        compresses the whole scoring curve into a couple of percent of cost,
        which makes a top score easier to reach rather than harder.

        Descent plus a few restarts is also what this baseline is advertised
        to be: dispatching rules and iterative adjacent-operation swaps.
        Ruin-and-recreate, tabu memory and unbounded restarts belong to the
        organizer reference.
        """
        if self.best_sequences is None:
            return
        sequences = [list(s) for s in self.best_sequences]
        cost = self._descend(sequences, self.best_cost)
        self._offer(sequences)

        for _ in range(MAX_RESTARTS):
            if self._remaining() <= RESERVE * 2:
                break
            trial = [list(s) for s in (self.best_sequences or sequences)]
            if not self._perturb(trial):
                break
            trial_cost = self._descend(trial, schedule_cost(self.inst, trial))
            if trial_cost is not None and (cost is None or trial_cost < cost):
                sequences, cost = trial, trial_cost
                self._offer(sequences)

    def _descend(self, sequences: List[List[int]], cost: Optional[int]) -> Optional[int]:
        """First-improvement sweep over every adjacent pair on every machine."""
        if cost is None:
            cost = schedule_cost(self.inst, sequences)
            if cost is None:
                return None

        improved = True
        while improved and self._remaining() > RESERVE:
            improved = False
            for machine in range(self.inst.n_machines):
                sequence = sequences[machine]
                for position in range(len(sequence) - 1):
                    if self._remaining() <= RESERVE:
                        return cost
                    sequence[position], sequence[position + 1] = (
                        sequence[position + 1],
                        sequence[position],
                    )
                    candidate = schedule_cost(self.inst, sequences)
                    # None means this order deadlocks: two machines waiting on
                    # each other. Undo and move on.
                    if candidate is not None and candidate < cost:
                        cost = candidate
                        improved = True
                        self._offer(sequences, force=False)
                    else:
                        sequence[position], sequence[position + 1] = (
                            sequence[position + 1],
                            sequence[position],
                        )
        return cost

    def _perturb(self, sequences: List[List[int]]) -> bool:
        """A few random adjacent swaps, kept only if the result still schedules."""
        candidates = [m for m in range(self.inst.n_machines) if len(sequences[m]) > 1]
        if not candidates:
            return False
        for _ in range(self.rng.randint(2, 5)):
            machine = self.rng.choice(candidates)
            sequence = sequences[machine]
            position = self.rng.below(len(sequence) - 1)
            sequence[position], sequence[position + 1] = (
                sequence[position + 1],
                sequence[position],
            )
        return True


Scheduler_ = DispatchSwapScheduler
