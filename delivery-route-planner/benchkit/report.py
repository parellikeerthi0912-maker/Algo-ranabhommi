"""JSON reports and the console table both runners print."""

from __future__ import annotations

import datetime
import json
import platform
import sys
from typing import Any, Mapping, Sequence

SCHEMA_VERSION = "1.0"


def environment() -> Mapping[str, Any]:
    return {
        "python": sys.version.split()[0],
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
    }


def write_report(path: str, benchmark: str, payload: Mapping[str, Any]) -> None:
    document = {
        "schema_version": SCHEMA_VERSION,
        "benchmark": benchmark,
        "environment": environment(),
        **payload,
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(document, handle, indent=2, sort_keys=False)
        handle.write("\n")


def table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    """Fixed-width table. Numeric columns right-aligned."""
    cells = [[str(h) for h in headers]] + [[_fmt(c) for c in row] for row in rows]
    widths = [max(len(row[i]) for row in cells) for i in range(len(headers))]
    numeric = [
        all(_looks_numeric(row[i]) for row in cells[1:]) if len(cells) > 1 else False
        for i in range(len(headers))
    ]

    def render(row: Sequence[str]) -> str:
        return "  ".join(
            cell.rjust(widths[i]) if numeric[i] else cell.ljust(widths[i])
            for i, cell in enumerate(row)
        ).rstrip()

    lines = [render(cells[0]), "  ".join("-" * w for w in widths)]
    lines.extend(render(row) for row in cells[1:])
    return "\n".join(lines)


def _fmt(value: Any) -> str:
    if value is None:
        return "--"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _looks_numeric(text: str) -> bool:
    return text == "--" or text.replace(".", "", 1).replace("-", "", 1).isdigit()


def banner(title: str, width: int = 72) -> str:
    return f"\n{title}\n{'=' * min(width, max(len(title), 8))}"
