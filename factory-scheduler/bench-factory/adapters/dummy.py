"""The floor: earliest-release-first list scheduling.

One dispatching rule, one submission, no search.  It is a correct submission --
every operation runs on its machine for its duration, in a valid order -- and
it scores close to zero, because the curve is anchored on
:mod:`adapters.dispatch_swap`.

Read it for the API.  Note what it does not do: it never looks at a due date,
never reconsiders a decision, submits once, and returns having used a fraction
of its budget.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from adapter import Scheduler, dispatch  # noqa: E402
from data import Instance  # noqa: E402


class ReleaseFirstScheduler(Scheduler):
    def solve(self, instance: Instance, submit_candidate) -> object:
        sequences = dispatch(
            instance,
            # Whatever can start soonest goes next; ties to the shorter
            # operation, then the lower job id so the result is deterministic.
            lambda job, op, start: (start, instance.op_duration[op], job),
        )
        return {"sequences": sequences}


Scheduler_ = ReleaseFirstScheduler
