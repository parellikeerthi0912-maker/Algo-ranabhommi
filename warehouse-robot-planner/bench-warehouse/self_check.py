"""Fast sanity run: two small public instances, well under 30 seconds.

    python self_check.py --adapter adapters.myteam:Planner
"""

from __future__ import annotations

import _paths  # noqa: F401
from _cli import CLI
from benchkit.cli import self_check_main

if __name__ == "__main__":
    raise SystemExit(self_check_main(CLI))
