"""``submit_candidate`` -- the anytime channel between solver and harness.

A solver is handed one of these and may call it as often as it likes.  Each
call is validated immediately, costed by the evaluator (never by the solver),
and timestamped.  The harness keeps the improvement history, so the score at
any checkpoint is just "the best cost whose timestamp is at or before it".

This is the piece that makes the benchmark anytime rather than one-shot, and
it is deliberately the same object in all three benchmarks: only ``evaluate``
differs.
"""

from __future__ import annotations

from typing import Any, Callable, List, Optional, Sequence, Tuple

from .budget import Budget
from .rng import digest_ints

# evaluate(candidate) -> (cost, reason). Exactly one is None.
Evaluator = Callable[[Any], Tuple[Optional[float], Optional[str]]]


class CandidateSink:
    __slots__ = (
        "_evaluate",
        "_budget",
        "_history",
        "_best",
        "_accepted",
        "_rejected",
        "_reasons",
        "_stream",
        "_first_valid_at",
        "_max_submissions",
        "_closed",
    )

    def __init__(self, evaluate: Evaluator, budget: Budget, max_submissions: int = 200_000) -> None:
        self._evaluate = evaluate
        self._budget = budget
        self._history: List[Tuple[float, float]] = []  # (elapsed, cost) improvements only
        self._best: Optional[float] = None
        self._accepted = 0
        self._rejected = 0
        self._reasons: dict = {}
        self._stream: List[int] = []
        self._first_valid_at: Optional[float] = None
        self._max_submissions = max_submissions
        self._closed = False

    # -- solver-facing ------------------------------------------------------

    def __call__(self, candidate: Any) -> dict:
        """Validate and record one candidate. Returns a small receipt.

        The receipt carries the verdict, the current best cost, and the
        authoritative clock.  That is enough to drive an acceptance criterion
        and a cooling schedule without the solver ever computing its own cost.
        Solvers may also read ``time.perf_counter()`` directly and compute
        their budget from ``data.budget_for(instance)``; the clock is not
        hidden, only the cost function is.
        """
        if self._closed:
            return self._receipt(False, "sink closed", None)

        submitted = self._accepted + self._rejected
        if submitted >= self._max_submissions:
            return self._receipt(False, "submission cap", None)

        cost, reason = self._evaluate(candidate)
        # Validation is solver-requested work.  Timestamp only after it has
        # finished, then reject anything that crossed the scored deadline.
        # This prevents lazy/hostile iterables from doing search inside the
        # validator while claiming an earlier checkpoint.
        at = self._budget.elapsed
        if at > self._budget.seconds:
            self._rejected += 1
            self._note("past budget")
            return self._receipt(False, "past budget", None)
        if cost is None:
            self._rejected += 1
            self._note(reason or "invalid")
            self._stream.append(0xFFFF_FFFF_FFFF_FFFF)
            return self._receipt(False, reason, None)

        self._accepted += 1
        self._stream.append(int(cost * 1000))
        if self._first_valid_at is None:
            self._first_valid_at = at
        if self._best is None or cost < self._best:
            self._best = cost
            self._history.append((at, cost))
        return self._receipt(True, None, cost)

    def _receipt(self, accepted: bool, reason: Optional[str], cost: Optional[float]) -> dict:
        return {
            "accepted": accepted,
            "reason": reason,
            "cost": cost,
            "best": self._best,
            "elapsed_s": round(self._budget.elapsed, 4),
            "remaining_s": round(self._budget.remaining, 4),
        }

    # -- harness-facing -----------------------------------------------------

    def _note(self, reason: str) -> None:
        self._reasons[reason] = self._reasons.get(reason, 0) + 1

    def close_with(self, final_candidate: Any) -> None:
        """The returned plan is treated as one last submission, then no more."""
        if final_candidate is not None:
            self(final_candidate)
        self._closed = True

    def best_at(self, elapsed: float) -> Optional[float]:
        best = None
        for at, cost in self._history:
            if at <= elapsed:
                best = cost
            else:
                break
        return best

    def costs_at_checkpoints(self) -> Sequence[Optional[float]]:
        return tuple(self.best_at(t) for t in self._budget.checkpoint_times())

    def stream_digest(self) -> int:
        """Fingerprint of every candidate this solver produced, in order."""
        return digest_ints(self._stream)

    def summary(self) -> dict:
        return {
            "accepted": self._accepted,
            "rejected": self._rejected,
            "rejection_reasons": dict(sorted(self._reasons.items())),
            "first_valid_at_s": (
                round(self._first_valid_at, 4) if self._first_valid_at is not None else None
            ),
            "best_cost": self._best,
            "improvements": len(self._history),
            "stream_digest": f"{self.stream_digest():016x}",
        }

    @property
    def best(self) -> Optional[float]:
        return self._best

    @property
    def rejected(self) -> int:
        return self._rejected

    @property
    def accepted(self) -> int:
        return self._accepted
