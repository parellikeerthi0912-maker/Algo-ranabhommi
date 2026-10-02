"""Fast sanity run: two small public instances, well under 30 seconds.

Run this first and often. It answers the questions that matter early -- does
the submission import, does the evaluator accept what it returns, does it beat
the floor -- without waiting for the full suite.

    python self_check.py --adapter adapters.myteam:Router
"""

from __future__ import annotations

import _paths  # noqa: F401
from _cli import CLI
from benchkit.cli import self_check_main

if __name__ == "__main__":
    raise SystemExit(self_check_main(CLI))
