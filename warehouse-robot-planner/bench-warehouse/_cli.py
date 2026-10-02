"""Configuration for the shared ``self_check.py`` / ``run.py`` runners."""

from __future__ import annotations

import _paths  # noqa: F401
import data
import harness
import metrics
from benchkit.cli import BenchCli


def _baseline():
    from adapters.prioritized import PrioritizedPlanner

    return PrioritizedPlanner()


CLI = BenchCli(
    benchmark="warehouse-robot-planner",
    title="warehouse robot planner",
    quality_label="plan quality over time (70)",
    robustness_label="robustness (30)",
    engineering_label="engineering (x multiplier)",
    data=data,
    metrics=metrics,
    harness=harness,
    reference_path=str(_paths.PUBLIC_REFERENCE),
    baseline_factory=_baseline,
    cost_name="cost",
    validity_hint=(
        "Every robot needs a path from its start to its goal. Each step must be\n"
        "a wait or one cardinal move onto a passable cell. No two robots may\n"
        "share a cell at the same time, and none may swap across an edge.\n"
        "Remember that a finished robot keeps occupying its goal cell forever:\n"
        "the evaluator pads short paths by repeating the last cell, so a plan\n"
        "that assumes finished robots disappear will be rejected."
    ),
)
