"""Full public evaluation and JSON report.

    python run.py --adapter adapters.myteam:Router --out report.json
    python run.py --adapter adapters.myteam:Router --seeds 11 29 47

The public suite is scored against published anchors. ``--seeds`` additionally
generates practice instances from the same generator; those have no published
anchors, so they are scored provisionally against a live baseline run.

This script never reads ``private/``, and refuses an output path that tries.
"""

from __future__ import annotations

import _paths  # noqa: F401
from _cli import CLI
from benchkit.cli import run_main

if __name__ == "__main__":
    raise SystemExit(run_main(CLI))
