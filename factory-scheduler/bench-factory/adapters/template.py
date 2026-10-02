"""Starter template. Copy to ``adapters/<yourname>.py`` and build from here.

    python self_check.py --adapter adapters.template:MyScheduler

As written this is a valid submission that scores almost nothing: one
dispatching rule, one submission. The comments mark where the points are.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from adapter import (  # noqa: E402
    Scheduler,
    decode,
    dispatch,
    interleaved_job_sequence,
    schedule_cost,
)
from data import Instance  # noqa: E402


class MyScheduler(Scheduler):
    def solve(self, instance: Instance, submit_candidate) -> object:
        # STEP 1 -- something valid, immediately.
        # decode() turns an operation-based permutation into machine
        # sequences and can never deadlock, so this is always accepted.
        order = sorted(range(instance.n_jobs), key=lambda j: instance.due[j])
        sequences = decode(instance, interleaved_job_sequence(instance, order))
        receipt = submit_candidate({"sequences": sequences})
        deadline = time.perf_counter() + (receipt["remaining_s"] or 0.0)

        # STEP 2 -- a real dispatching rule. Which rule wins depends on the
        # instance, so try several and keep the best. Each call is O(jobs) per
        # operation, which is not free on the large instances.
        work_left = [
            sum(instance.op_duration[op] for op in instance.job_ops[job])
            for job in range(instance.n_jobs)
        ]

        def critical_ratio(job, op, start):
            slack = instance.due[job] - start - work_left[job]
            return (slack * 100 // max(1, instance.weight[job]), start, job)

        best = sequences
        best_cost = schedule_cost(instance, sequences)
        for rule in (critical_ratio,):
            if time.perf_counter() > deadline - 0.5:
                break
            candidate = dispatch(instance, rule)
            cost = schedule_cost(instance, candidate)
            if cost is not None and (best_cost is None or cost < best_cost):
                best, best_cost = candidate, cost
                submit_candidate({"sequences": best})

        # STEP 3 -- search until the budget runs out.
        while time.perf_counter() < deadline - 0.2:
            # Your search goes here. Roughly in order of payoff:
            #
            #   - find the critical path and only propose swaps at the ends of
            #     its blocks; moves elsewhere cannot change the makespan, and
            #     a swap inside a block can never cause a deadlock
            #   - attack tardiness separately: pull a late job's operations
            #     forward on their machines, several positions at a time
            #   - keep a tabu list so you can pass through a worse schedule;
            #     pure descent converges early and then sits there
            #   - restart from a randomised rule when nothing improves
            #
            # schedule_cost() reschedules the whole instance, so it is the
            # thing to make cheaper if you want more moves per second.
            break

        return {"sequences": best}
