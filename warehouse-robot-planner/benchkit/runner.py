"""Running one submission against one instance, under controlled conditions.

The harness owns everything the solver must not: the clock, the cost function,
the validity verdict, and the record of what was submitted when.  A solver
receives an instance and a channel, and nothing else.

All three benchmarks run identically here.  What differs is only ``evaluate``
-- the function that turns a candidate into a cost or a rejection reason --
so that is the one thing a benchmark passes in.
"""

from __future__ import annotations

import traceback
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional, Protocol, Sequence, Tuple

from .budget import CHECKPOINTS, Budget, TimeBudgetExceeded
from .limits import hard_time_limit, memory_limit
from .sink import CandidateSink

# How far past the scored deadline a wedged solver may run before it is killed.
# Late candidates are already rejected, so this only stops a run from hanging.
HARD_LIMIT_MULTIPLIER = 3.0
DEFAULT_MEMORY_MB = 2048

_VERBOSE_ERRORS = False


class BenchInstance(Protocol):
    """What every benchmark's instance type must expose to the harness."""

    name: str

    @property
    def profile(self) -> Any: ...

    @property
    def size(self) -> int: ...

    def compute_digest(self) -> int: ...


@dataclass
class RunRecord:
    instance: str
    size: int
    profile: dict
    instance_digest: str
    budget_s: float
    costs: Sequence[Optional[float]]  # best cost at each checkpoint
    final_cost: Optional[float]
    sink: dict
    wall_s: float
    instance_untouched: bool
    within_budget: bool
    error: Optional[str] = None
    stream_digest: str = ""
    detail: dict = field(default_factory=dict)

    @property
    def produced_valid(self) -> bool:
        return self.final_cost is not None


def set_verbose_errors(value: bool) -> None:
    global _VERBOSE_ERRORS
    _VERBOSE_ERRORS = value


def run_one(
    instance: Any,
    adapter_factory: Any,
    evaluate: Callable[[Any, Any], Tuple[Optional[float], Optional[str]]],
    budget_s: float,
    memory_mb: Optional[int] = DEFAULT_MEMORY_MB,
    describe: Optional[Callable[[Any, Any], dict]] = None,
) -> RunRecord:
    """Run ``adapter_factory().solve(instance, submit_candidate)`` once."""
    fingerprint = instance.compute_digest()
    # Profile is scoring metadata, not solver-owned state.  Capture it before
    # invoking untrusted code even when an instance implementation exposes a
    # mutable mapping for backwards compatibility.
    trusted_profile = dict(instance.profile)

    budget = Budget(budget_s, CHECKPOINTS)
    sink = CandidateSink(lambda candidate: evaluate(instance, candidate), budget)

    error: Optional[str] = None
    final: Any = None
    solver = adapter_factory()

    budget.restart()
    try:
        with memory_limit(memory_mb), hard_time_limit(budget_s * HARD_LIMIT_MULTIPLIER):
            final = solver.solve(instance, sink)
    except TimeBudgetExceeded as exc:
        error = f"time: {exc}"
    except MemoryError:
        error = "memory: solver exceeded its address-space limit"
    except Exception as exc:  # noqa: BLE001 - a crashing solver is a result, not a bug
        error = f"{type(exc).__name__}: {exc}"
        if _VERBOSE_ERRORS:
            error += "\n" + traceback.format_exc(limit=6)

    wall = budget.elapsed
    sink.close_with(final)

    detail: dict = {}
    if describe is not None and sink.best is not None and final is not None:
        try:
            detail = describe(instance, final)
        except Exception:  # noqa: BLE001 - reporting must never fail a run
            detail = {}

    return RunRecord(
        instance=instance.name,
        size=instance.size,
        profile=trusted_profile,
        instance_digest=f"{fingerprint:016x}",
        budget_s=budget_s,
        costs=sink.costs_at_checkpoints(),
        final_cost=sink.best,
        sink=sink.summary(),
        wall_s=round(wall, 3),
        instance_untouched=(instance.compute_digest() == fingerprint),
        within_budget=not budget.expired(with_grace=False),
        error=error,
        stream_digest=f"{sink.stream_digest():016x}",
        detail=detail,
    )


def scored_output_stability(first: RunRecord, second: RunRecord) -> Tuple[bool, dict]:
    """Whether two replay runs of the same submission are stable enough to
    be considered reproducible -- by what actually gets scored, not by the
    raw internal candidate stream.

    Exact candidate-stream equality is not achievable for a solver that
    legitimately bounds its own work by wall-clock remaining time, which
    every published baseline in this suite does: ``CandidateSink`` hard-
    rejects any candidate submitted after the real deadline regardless of
    validity, so a few milliseconds of run-to-run scheduling jitter near
    that boundary can flip one submission's accept/reject status -- and
    with it the whole stream digest -- without the solver's *quality*
    having changed at all.

    What actually needs to reproduce is the score: the best cost at each
    checkpoint, and the final cost. Those are compared here, exactly. There
    is deliberately no secondary tolerance on accepted/rejected candidate
    counts: submissions are encouraged to resubmit often (the README says
    calling ``submit_candidate`` early and often is strictly better than
    holding back), the published baselines already resubmit every ~50ms
    near the deadline, and any fixed threshold on cross-run count drift
    would just reintroduce a milder version of the same timing-sensitivity
    problem this replaces. Validity rate already has its own, per-run,
    scored channel (``valid_output`` in ``engineering_score``).
    """
    stable = (
        tuple(first.costs) == tuple(second.costs)
        and first.final_cost == second.final_cost
    )
    return stable, {
        "checkpoint_costs_match": tuple(first.costs) == tuple(second.costs),
        "final_cost_match": first.final_cost == second.final_cost,
        "first_stream_digest": first.stream_digest,
        "second_stream_digest": second.stream_digest,
    }


def check_determinism(
    instance: Any,
    adapter_factory: Any,
    evaluate: Callable[[Any, Any], Tuple[Optional[float], Optional[str]]],
    budget_s: float,
) -> Tuple[bool, str, str]:
    """Run the same submission twice and check scored-output stability.

    A submission whose *score* cannot reproduce cannot be verified by
    anyone, so this is scored rather than merely reported. See
    :func:`scored_output_stability` for exactly what "stable" means and why
    it is not raw candidate-stream equality. The two stream digests are
    still returned for a human to inspect what (harmlessly) differed, but
    they no longer decide the verdict.
    """
    first = run_one(instance, adapter_factory, evaluate, budget_s)
    second = run_one(instance, adapter_factory, evaluate, budget_s)
    stable, _detail = scored_output_stability(first, second)
    return stable, first.stream_digest, second.stream_digest
