"""The scoring curve shared by all three benchmarks.

Two anchors define every score in this suite:

    published serious baseline  -> 0.25
    organizer reference solver  -> 1.00

Matching the published baseline is deliberately worth little: it ships with
the benchmark, so a copy of it should not carry a submission most of the way
to a good score.  Almost all of the scale is the distance from the baseline
to the reference.

Both are real programs that ship in the repository (the reference under
``private/``), not hand-chosen constants.  A submission is therefore always
being measured against code a contestant can read, reason about, and try to
beat, which is the only version of "difficulty" that stays honest over time.

The line extends below the baseline at the same slope, so a submission that is
as much worse than the baseline as the reference is better scores 0.  It does
not extend above 1: beating the organizer reference is worth exactly as much
as tying it, which removes any incentive to chase a single instance.
"""

from __future__ import annotations

from typing import Mapping, Optional, Sequence

from .budget import CHECKPOINT_WEIGHTS

# Normalised score of a cost equal to the published baseline.
BASELINE_SCORE = 0.25

# Engineering scales the total rather than adding to it: a clean submission
# keeps its full score, a sloppy one loses up to this fraction of it.
ENGINEERING_WEIGHT = 0.30


def normalised(cost: Optional[float], baseline: float, reference: float) -> float:
    """Map a cost onto [0, 1] against the two anchors. Lower cost is better."""
    if cost is None:
        return 0.0
    span = baseline - reference
    if span <= 0:
        # The reference failed to beat the baseline on this instance. Treat the
        # baseline as the only anchor and score pass/fail against it rather
        # than dividing by a degenerate span.
        return 1.0 if cost <= baseline else 0.0
    score = BASELINE_SCORE + (1.0 - BASELINE_SCORE) * (baseline - cost) / span
    return max(0.0, min(1.0, score))


def checkpoint_score(
    costs_by_checkpoint: Sequence[Optional[float]],
    baseline: float,
    reference: float,
    weights: Sequence[float] = CHECKPOINT_WEIGHTS,
) -> float:
    """Weighted mean of the normalised best-cost at each checkpoint."""
    if len(costs_by_checkpoint) != len(weights):
        raise ValueError("checkpoint count does not match weight count")
    total_weight = sum(weights)
    return sum(
        weight * normalised(cost, baseline, reference)
        for cost, weight in zip(costs_by_checkpoint, weights)
    ) / total_weight


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


class AxisScore:
    """The 100-point split, assembled from per-instance scores.

    Quality and robustness points sum to 100.  Engineering is a multiplier in
    ``[1 - ENGINEERING_WEIGHT, 1]`` on that sum, so it can only cost points,
    never hand them out for merely running cleanly.

    Quality and robustness read the same per-instance numbers; they differ only
    in which families they read.  Keeping that split here rather than in each
    benchmark is what makes "wins one family, collapses on another" a
    *structurally* losing strategy in all three benchmarks at once.
    """

    def __init__(self, quality_points: float, robustness_points: float) -> None:
        self.quality_points = quality_points
        self.robustness_points = robustness_points

    def total(
        self,
        quality: float,
        robustness: float,
        engineering: float,
        *,
        any_family_invalid: bool = False,
        below_baseline_family: bool = False,
    ) -> Mapping[str, float]:
        """Apply the axis weights and the two structural penalties."""
        q = quality * self.quality_points
        r = robustness * self.robustness_points
        multiplier = 1.0 - ENGINEERING_WEIGHT * (1.0 - engineering)

        penalties = []
        if any_family_invalid:
            # Producing nothing valid on a whole family is not a small miss --
            # it means the submission does not solve the problem as stated.
            q *= 0.5
            r *= 0.5
            penalties.append("family_with_no_valid_result")
        if below_baseline_family:
            # Falling below the published baseline on any family is the exact
            # failure mode these benchmarks exist to catch.
            r *= 0.5
            penalties.append("family_below_published_baseline")

        return {
            "quality": round(q, 3),
            "robustness": round(r, 3),
            "engineering": round(multiplier, 3),
            "total": round((q + r) * multiplier, 3),
            "penalties": penalties,
        }


# Quality/robustness split per benchmark; engineering is a multiplier on both.
DELIVERY_AXES = AxisScore(70, 30)
FACTORY_AXES = AxisScore(65, 35)
WAREHOUSE_AXES = AxisScore(70, 30)


def engineering_score(
    *,
    deterministic: bool,
    rejected_candidates: int,
    accepted_candidates: int,
    instance_untouched: bool,
    within_budget: bool,
) -> "tuple[float, dict]":
    """Engineering in [0, 1], measured rather than judged.

    Every component here is something the harness observed, so two runs of the
    same submission produce the same engineering score.
    """
    parts = {
        # Replay the solver and compare its scored outputs (checkpoint costs
        # and final cost) -- see benchkit.runner.scored_output_stability. A
        # submission whose score cannot reproduce cannot be verified by
        # anyone; the raw candidate stream is allowed to differ, since a
        # solver may legitimately bound its own work by wall-clock time.
        "determinism": 0.40 if deterministic else 0.0,
        # Invalid submissions are not free: they consume harness time and
        # signal a solver that does not understand its own constraints.
        "valid_output": 0.0,
        "instance_untouched": 0.20 if instance_untouched else 0.0,
        "within_budget": 0.10 if within_budget else 0.0,
    }
    submitted = rejected_candidates + accepted_candidates
    if submitted == 0:
        parts["valid_output"] = 0.0
    else:
        clean = 1.0 - (rejected_candidates / submitted)
        parts["valid_output"] = 0.30 * max(0.0, clean)

    return sum(parts.values()), parts
