"""Put the benchmark directory and its parent on ``sys.path``.

Every script here is run directly (``python run.py ...``) rather than as a
package, so the benchmark's own modules resolve via the script directory and
``benchkit`` resolves via its parent.  Importing this module first is the
whole mechanism.
"""

from __future__ import annotations

import pathlib
import sys

_HERE = pathlib.Path(__file__).resolve().parent
for _path in (_HERE, _HERE.parent):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

BENCH_DIR = _HERE
PRIVATE_DIR = _HERE / "private"
PUBLIC_REFERENCE = _HERE / "public_reference.json"
