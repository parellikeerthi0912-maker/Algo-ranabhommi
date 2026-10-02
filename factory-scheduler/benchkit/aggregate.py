"""Run records -> per-instance scores -> families -> the 100-point total.

Identical in all three benchmarks.  What each one supplies is its shift
markers (which profile values count as the hard side of an axis), its axis
weights, and a formatter that turns internal cost units into something a human
can read.
"""

from __future__ import annotations

import json
import os
from typing import Callable, Dict, List, Mapping, Optional, Sequence

from .budget import CHECKPOINTS, CHECKPOINT_WEIGHTS
from .runner import RunRecord
from .scoring import BASELINE_SCORE, AxisScore, checkpoint_score, engineering_score, mean, normalised


class MissingAnchors(RuntimeError):
    pass


def load_anchors(path: str, schema: str) -> Dict[str, Dict[str, float]]:
    if not os.path.exists(path):
        raise MissingAnchors(
            f"no reference file at {path}. Generate one with "
            f"'python private/make_reference.py' before scoring."
        )
    with open(path, "r", encoding="utf-8") as handle:
        document = json.load(handle)
    if document.get("schema") != schema:
        raise MissingAnchors(f"{path} is not a {schema} file")
    return document["instances"]


class Scorer:
    """Holds one benchmark's scoring configuration."""

    def __init__(
        self,
        shift_markers: Mapping[str, frozenset],
        axes: AxisScore,
        format_cost: Callable[[Optional[float]], Optional[float]],
        shift_threshold: int = 2,
    ) -> None:
        self.shift_markers = shift_markers
        self.axes = axes
        self.format_cost = format_cost
        self.shift_threshold = shift_threshold

    # -- classification -----------------------------------------------------

    def shift_degree(self, profile: Mapping[str, object]) -> int:
        return sum(
            1 for axis, shifted in self.shift_markers.items() if profile.get(axis) in shifted
        )

    def is_shift(self, profile: Mapping[str, object]) -> bool:
        """Several shifted axes at once is where tuned solvers come apart."""
        return self.shift_degree(profile) >= self.shift_threshold

    def families_of(self, profile: Mapping[str, object]) -> List[str]:
        return [f"{axis}:{profile.get(axis)}" for axis in self.shift_markers]

    # -- scoring ------------------------------------------------------------

    def score_instance(self, record: RunRecord, anchor: Mapping[str, float]) -> Dict[str, object]:
        anchor_digest = anchor.get("digest")
        if anchor_digest and anchor_digest != record.instance_digest:
            raise MissingAnchors(
                f"{record.instance}: anchor digest {anchor_digest} does not match "
                f"instance digest {record.instance_digest}; regenerate anchors"
            )
        baseline = float(anchor["baseline_cost_units"])
        reference = float(anchor["reference_cost_units"])
        baseline_curve = anchor.get("baseline_checkpoint_cost_units")
        if baseline_curve is None:
            # Legacy public anchors remain readable, but regenerated anchors
            # must carry a curve so the published baseline scores BASELINE_SCORE at each
            # scored checkpoint rather than only at the final checkpoint.
            baseline_curve = [baseline] * len(record.costs)
        if len(baseline_curve) != len(record.costs):
            raise MissingAnchors(f"{record.instance}: malformed baseline checkpoint curve")
        per_checkpoint = [
            normalised(cost, float(checkpoint_baseline), reference)
            for cost, checkpoint_baseline in zip(record.costs, baseline_curve)
        ]

        return {
            "instance": record.instance,
            "size": record.size,
            "profile": record.profile,
            "shifted": self.is_shift(record.profile),
            "budget_s": record.budget_s,
            "wall_s": record.wall_s,
            "valid": record.produced_valid,
            "error": record.error,
            "baseline_cost": self.format_cost(baseline),
            "reference_cost": self.format_cost(reference),
            "best_cost": self.format_cost(record.final_cost),
            "gap_to_reference_pct": (
                round(100.0 * (record.final_cost - reference) / reference, 2)
                if record.final_cost is not None and reference
                else None
            ),
            "checkpoint_fractions": list(CHECKPOINTS),
            "checkpoint_costs": [self.format_cost(c) for c in record.costs],
            "checkpoint_scores": [round(s, 4) for s in per_checkpoint],
            "score": round(
                sum(score * weight for score, weight in zip(per_checkpoint, CHECKPOINT_WEIGHTS))
                / sum(CHECKPOINT_WEIGHTS),
                4,
            ),
            "first_valid_at_s": record.sink.get("first_valid_at_s"),
            "candidates": {
                "accepted": record.sink.get("accepted"),
                "rejected": record.sink.get("rejected"),
                "reasons": record.sink.get("rejection_reasons"),
            },
        }

    def score_suite(
        self,
        records: Sequence[RunRecord],
        anchors: Mapping[str, Mapping[str, float]],
        deterministic: Optional[bool] = None,
    ) -> Dict[str, object]:
        missing = [r.instance for r in records if r.instance not in anchors]
        if missing:
            raise MissingAnchors(f"no anchors for: {', '.join(missing)}")

        scored = [self.score_instance(record, anchors[record.instance]) for record in records]
        quality = mean([entry["score"] for entry in scored])
        shifted = [entry["score"] for entry in scored if entry["shifted"]]
        robustness = mean(shifted) if shifted else quality

        by_family: Dict[str, List[float]] = {}
        valid_by_family: Dict[str, List[bool]] = {}
        for entry, record in zip(scored, records):
            for family in self.families_of(record.profile):
                by_family.setdefault(family, []).append(entry["score"])
                valid_by_family.setdefault(family, []).append(record.produced_valid)

        family_scores = {name: round(mean(values), 4) for name, values in by_family.items()}
        dead_families = [name for name, flags in valid_by_family.items() if not any(flags)]
        # Any family below the published baseline is penalised, whether or not
        # the submission is strong elsewhere.
        specialised = bool(family_scores and min(family_scores.values()) < BASELINE_SCORE)

        accepted = sum(r.sink.get("accepted", 0) for r in records)
        rejected = sum(r.sink.get("rejected", 0) for r in records)
        engineering, parts = engineering_score(
            deterministic=bool(deterministic) if deterministic is not None else True,
            rejected_candidates=rejected,
            accepted_candidates=accepted,
            instance_untouched=all(r.instance_untouched for r in records),
            within_budget=all(r.within_budget for r in records),
        )

        points = self.axes.total(
            quality,
            robustness,
            engineering,
            any_family_invalid=bool(dead_families),
            below_baseline_family=specialised,
        )

        return {
            "checkpoints": list(CHECKPOINTS),
            "checkpoint_weights": list(CHECKPOINT_WEIGHTS),
            "instances": scored,
            "families": family_scores,
            "aggregate": {
                "quality_mean": round(quality, 4),
                "robustness_mean": round(robustness, 4),
                "engineering": round(engineering, 4),
                "engineering_parts": {k: round(v, 3) for k, v in parts.items()},
                "shifted_instances": len(shifted),
                "dead_families": dead_families,
                "specialised": specialised,
                "determinism_checked": deterministic,
            },
            "points": points,
        }

    def format_summary(self, result: Mapping[str, object]) -> str:
        from .report import table

        rows = [
            [
                entry["instance"],
                entry["size"],
                "yes" if entry["shifted"] else "no",
                entry["best_cost"],
                entry["baseline_cost"],
                entry["reference_cost"],
                entry["gap_to_reference_pct"],
                entry["checkpoint_scores"][0],
                entry["score"],
            ]
            for entry in result["instances"]
        ]
        return table(
            ["instance", "size", "shift", "cost", "baseline", "reference",
             "gap%", "cp@1%", "score"],
            rows,
        )
