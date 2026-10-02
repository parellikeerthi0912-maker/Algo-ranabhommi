"""Configuration for the shared ``self_check.py`` / ``run.py`` runners."""

from __future__ import annotations

import _paths  # noqa: F401
import data
import harness
import metrics
from benchkit.cli import BenchCli


def _baseline():
    from adapters.savings import SavingsRouter

    return SavingsRouter()


CLI = BenchCli(
    benchmark="delivery-route-planner",
    title="delivery route planner",
    quality_label="route quality over time (70)",
    robustness_label="robustness (30)",
    engineering_label="engineering (x multiplier)",
    data=data,
    metrics=metrics,
    harness=harness,
    reference_path=str(_paths.PUBLIC_REFERENCE),
    baseline_factory=_baseline,
    cost_name="minutes",
    validity_hint=(
        "Every customer must appear exactly once across all routes, and route v\n"
        "must respect vehicle v's capacity and start time. Arriving before a\n"
        "window opens is fine -- the vehicle waits, and the wait is charged.\n"
        "Arriving after it closes invalidates the plan, as does exceeding the\n"
        "route-duration cap or returning after the horizon."
    ),
)
