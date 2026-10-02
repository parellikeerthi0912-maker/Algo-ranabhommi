"""Factory scoring configuration.

Aggregation lives in :mod:`benchkit.aggregate`. This file supplies only what
is specific to the factory: which profile values are the hard side of each
axis, the 65/25/10 split, and how to print a cost.
"""

from __future__ import annotations

from typing import Mapping

import _paths  # noqa: F401
from benchkit.aggregate import MissingAnchors, Scorer, load_anchors as _load_anchors
from benchkit.scoring import FACTORY_AXES
from validator import normalised_cost

REFERENCE_SCHEMA = "factory-reference-1.0"

SHIFT_MARKERS: Mapping[str, frozenset] = {
    "workload": frozenset({"bottleneck"}),
    "due": frozenset({"tight"}),
    "durations": frozenset({"heavy_tail"}),
    "routing": frozenset({"reentrant"}),
    "setups": frozenset({"groups"}),
}

SCORER = Scorer(SHIFT_MARKERS, FACTORY_AXES, normalised_cost, shift_threshold=3)

shift_degree = SCORER.shift_degree
is_shift = SCORER.is_shift
families_of = SCORER.families_of
score_instance = SCORER.score_instance
score_suite = SCORER.score_suite
format_summary = SCORER.format_summary

__all__ = [
    "MissingAnchors",
    "REFERENCE_SCHEMA",
    "SCORER",
    "SHIFT_MARKERS",
    "families_of",
    "format_summary",
    "is_shift",
    "load_anchors",
    "score_instance",
    "score_suite",
    "shift_degree",
]


def load_anchors(path: str):
    return _load_anchors(path, REFERENCE_SCHEMA)
