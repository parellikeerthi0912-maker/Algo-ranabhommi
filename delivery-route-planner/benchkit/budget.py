"""Wall-clock budgets and the checkpoint schedule.

Every benchmark in this suite scores a solver at four points in its budget
rather than once at the end.  That single decision is what stops the whole
family of "run simulated annealing for the entire budget and print at the
buzzer" submissions from scoring well, and it is why this type exists
separately from the harness.
"""

from __future__ import annotations

import time
from typing import Sequence

# 1%, 10%, 50%, 100% of the budget, per the benchmark designs.
CHECKPOINTS: Sequence[float] = (0.01, 0.10, 0.50, 1.00)

# The 1% checkpoint is deliberately brutal -- on a large instance it is a few
# hundred milliseconds, which is not enough time for a careless construction
# heuristic to finish. The early checkpoints carry most of the weight: a solver
# must build fast and improve fast, not merely converge by the buzzer.
CHECKPOINT_WEIGHTS: Sequence[float] = (0.30, 0.30, 0.20, 0.20)


class TimeBudgetExceeded(RuntimeError):
    """Raised into a solver that has run past its hard limit."""


class Budget:
    """A started stopwatch with a deadline.

    Time spent validating a submitted candidate is charged to the solver,
    because it is real work the solver asked for.  A solver that submits a
    thousand candidates per second is spending its own budget on validation,
    which is the correct incentive and needs no separate rate limit.
    """

    __slots__ = ("seconds", "checkpoints", "grace", "_start")

    def __init__(
        self,
        seconds: float,
        checkpoints: Sequence[float] = CHECKPOINTS,
        grace: float = 0.10,
    ) -> None:
        if seconds <= 0:
            raise ValueError("budget must be positive")
        self.seconds = float(seconds)
        self.checkpoints = tuple(checkpoints)
        self.grace = float(grace)
        self._start = time.perf_counter()

    def restart(self) -> "Budget":
        self._start = time.perf_counter()
        return self

    @property
    def elapsed(self) -> float:
        return time.perf_counter() - self._start

    @property
    def remaining(self) -> float:
        return self.seconds - self.elapsed

    @property
    def fraction(self) -> float:
        return self.elapsed / self.seconds

    def expired(self, with_grace: bool = True) -> bool:
        limit = self.seconds * (1.0 + self.grace) if with_grace else self.seconds
        return self.elapsed > limit

    def checkpoint_times(self) -> Sequence[float]:
        return tuple(self.seconds * f for f in self.checkpoints)

    def check(self) -> None:
        """Raise if the hard limit has passed. Solvers may call this too."""
        if self.expired():
            raise TimeBudgetExceeded(
                f"exceeded {self.seconds:.2f}s budget (+{self.grace:.0%} grace)"
            )
