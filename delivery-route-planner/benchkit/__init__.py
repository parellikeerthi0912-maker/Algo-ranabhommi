"""Shared machinery for the three optimisation benchmarks in this repository.

The delivery, factory, and warehouse benchmarks are different problems, but
they are the *same benchmark design*: deterministic integer-only instance
generation, an anytime candidate channel, evaluator-owned cost, four scored
checkpoints, and a two-anchor scoring curve.  Keeping that design in one place
means a fix to the scoring curve or the time accounting lands in all three at
once instead of drifting into three subtly different contests.

Each benchmark directory is otherwise self-contained.  To ship one on its own,
copy this package next to it -- it has no dependencies outside the standard
library.
"""

from .budget import CHECKPOINTS, CHECKPOINT_WEIGHTS, Budget, TimeBudgetExceeded
from .limits import hard_time_limit, memory_limit, peak_memory_mb
from .loader import AdapterError, load_adapter, reject_private_path
from .rng import Rng, derive_seed, digest_ints, fnv1a64
from .scoring import (
    AxisScore,
    checkpoint_score,
    engineering_score,
    mean,
    normalised,
)
from .sink import CandidateSink

__all__ = [
    "AdapterError",
    "AxisScore",
    "Budget",
    "CHECKPOINTS",
    "CHECKPOINT_WEIGHTS",
    "CandidateSink",
    "Rng",
    "TimeBudgetExceeded",
    "checkpoint_score",
    "derive_seed",
    "digest_ints",
    "engineering_score",
    "fnv1a64",
    "hard_time_limit",
    "load_adapter",
    "mean",
    "memory_limit",
    "normalised",
    "peak_memory_mb",
    "reject_private_path",
]

__version__ = "1.0.0"
