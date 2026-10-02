"""Warehouse-specific wiring around the shared runner."""

from __future__ import annotations

from typing import Any, List, Optional, Sequence

import _paths  # noqa: F401
import validator
from benchkit.runner import (  # noqa: F401  (re-exported for the runners)
    DEFAULT_MEMORY_MB,
    RunRecord,
    set_verbose_errors,
)
from benchkit.runner import check_determinism as _check_determinism
from benchkit.runner import run_one as _run_one
from data import Instance, budget_for


def evaluate(instance: Instance, plan: Any):
    return validator.evaluate(instance, plan)


def run_one(
    instance: Instance,
    adapter_factory: Any,
    budget_s: Optional[float] = None,
    memory_mb: Optional[int] = DEFAULT_MEMORY_MB,
    capture_detail: bool = False,
) -> RunRecord:
    return _run_one(
        instance,
        adapter_factory,
        evaluate,
        budget_s=budget_s if budget_s is not None else budget_for(instance),
        memory_mb=memory_mb,
        describe=validator.describe if capture_detail else None,
    )


def check_determinism(
    instance: Instance, adapter_factory: Any, budget_s: Optional[float] = None
):
    return _check_determinism(
        instance,
        adapter_factory,
        evaluate,
        budget_s if budget_s is not None else budget_for(instance),
    )


def run_suite(
    instances: Sequence[Instance],
    adapter_factory: Any,
    budget_scale: float = 1.0,
    memory_mb: Optional[int] = DEFAULT_MEMORY_MB,
    progress: Optional[Any] = None,
) -> List[RunRecord]:
    records = []
    for instance in instances:
        if progress:
            progress(instance)
        records.append(
            run_one(
                instance,
                adapter_factory,
                budget_s=budget_for(instance) * budget_scale,
                memory_mb=memory_mb,
            )
        )
    return records
