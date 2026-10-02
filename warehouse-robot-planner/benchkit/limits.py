"""Resource limits around an untrusted solver call.

The harness runs solvers in-process, which is fast and keeps tracebacks
readable.  The cost is that a runaway solver can take the harness with it, so
both escape hatches -- the wall clock and the address space -- are closed here
for the duration of a single ``solve`` call and reopened afterwards.

Both mechanisms are POSIX-only and degrade to no-ops elsewhere; the harness
still enforces the budget by rejecting late candidates, so a missing rlimit
changes performance reporting, never scoring.
"""

from __future__ import annotations

import contextlib
import signal
from typing import Iterator, Optional

from .budget import TimeBudgetExceeded

try:  # pragma: no cover - platform dependent
    import resource
except ImportError:  # pragma: no cover
    resource = None  # type: ignore[assignment]


@contextlib.contextmanager
def hard_time_limit(seconds: float) -> Iterator[None]:
    """Raise ``TimeBudgetExceeded`` inside the solver once ``seconds`` pass.

    This fires well after the scored deadline.  Its job is not scoring -- late
    candidates are already rejected -- but making sure one wedged submission
    cannot stall an entire evaluation run.
    """
    if not hasattr(signal, "setitimer") or seconds <= 0:
        yield
        return

    def _fire(signum, frame):  # noqa: ANN001 - signal handler signature
        raise TimeBudgetExceeded(f"hard limit of {seconds:.2f}s reached")

    previous = signal.signal(signal.SIGALRM, _fire)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


@contextlib.contextmanager
def memory_limit(megabytes: Optional[int]) -> Iterator[None]:
    """Cap address space for the solver, then restore the previous soft cap."""
    if resource is None or not megabytes:
        yield
        return
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    target = megabytes * 1024 * 1024
    if hard != resource.RLIM_INFINITY and target > hard:
        target = hard
    try:
        resource.setrlimit(resource.RLIMIT_AS, (target, hard))
    except (ValueError, OSError):  # pragma: no cover - hardened environments
        yield
        return
    try:
        yield
    finally:
        with contextlib.suppress(ValueError, OSError):
            resource.setrlimit(resource.RLIMIT_AS, (soft, hard))


def peak_memory_mb() -> Optional[float]:
    if resource is None:
        return None
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux reports kilobytes, macOS reports bytes.
    return round(usage / 1024, 1)
