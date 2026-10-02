"""Configuration for the shared ``self_check.py`` / ``run.py`` runners."""

from __future__ import annotations

import _paths  # noqa: F401
import data
import harness
import metrics
from benchkit.cli import BenchCli


def _baseline():
    from adapters.dispatch_swap import DispatchSwapScheduler

    return DispatchSwapScheduler()


CLI = BenchCli(
    benchmark="factory-scheduler",
    title="factory scheduler",
    quality_label="schedule quality over time (65)",
    robustness_label="generalization (35)",
    engineering_label="engineering (x multiplier)",
    data=data,
    metrics=metrics,
    harness=harness,
    reference_path=str(_paths.PUBLIC_REFERENCE),
    baseline_factory=_baseline,
    cost_name="cost",
    validity_hint=(
        "Every operation must be sequenced exactly once, on its own machine.\n"
        "If the reason says 'deadlock', the machine orders contain a cycle:\n"
        "two machines are each waiting for a job the other is holding. Building\n"
        "sequences with adapter.decode() from an operation-based permutation\n"
        "makes that impossible by construction."
    ),
)
