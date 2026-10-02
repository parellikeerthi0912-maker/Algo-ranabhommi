"""Delivery scoring configuration.

The aggregation itself -- checkpoints into an instance score, instances into
families, families into the three axes, penalties -- lives in
:mod:`benchkit.aggregate`.  This file says only what is specific to delivery:
which profile values are the hard side of each axis, the 70/20/10 split, and
how to print a cost.
"""

from __future__ import annotations

from typing import Mapping

import _paths  # noqa: F401
from benchkit.aggregate import MissingAnchors, Scorer, load_anchors as _load_anchors
from benchkit.scoring import DELIVERY_AXES
from validator import cost_minutes

REFERENCE_SCHEMA = "delivery-reference-1.0"

# Which profile values count as the "shifted" side of each axis. The private
# mixture leans on these; the public suite includes a few so the effect is
# visible before it is scored.
SHIFT_MARKERS: Mapping[str, frozenset] = {
    "layout": frozenset({"ring", "mixed"}),
    "demand": frozenset({"heavy_light"}),
    "windows": frozenset({"tight", "waves"}),
    "fleet": frozenset({"tight", "heterogeneous"}),
}

SCORER = Scorer(SHIFT_MARKERS, DELIVERY_AXES, cost_minutes)

# Module-level shims so the runners read naturally.
shift_degree = SCORER.shift_degree
is_shift = SCORER.is_shift
families_of = SCORER.families_of
score_instance = SCORER.score_instance
score_suite = SCORER.score_suite
format_summary = SCORER.format_summary

__all__ = [
    "MissingAnchors",
    "REFERENCE_SCHEMA",
    "SHIFT_MARKERS",
    "SCORER",
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
