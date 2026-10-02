"""Resolve ``--adapter`` strings into a solver class."""

from __future__ import annotations

import importlib
import importlib.util
import inspect
import os
import sys
from typing import Any

PRIVATE_DIR_NAMES = {"private"}


class AdapterError(RuntimeError):
    pass


def load_adapter(spec: str, required_method: str = "solve") -> Any:
    """``package.module:ClassName`` or ``path/to/file.py:ClassName``."""
    if ":" not in spec:
        raise AdapterError(
            f"adapter must look like 'adapters.myteam:Router', got {spec!r}"
        )
    target, attribute = spec.rsplit(":", 1)

    if target.endswith(".py") or os.sep in target:
        module = _load_from_path(target)
    else:
        try:
            module = importlib.import_module(target)
        except ImportError as exc:
            raise AdapterError(f"cannot import {target!r}: {exc}") from exc

    try:
        obj = getattr(module, attribute)
    except AttributeError as exc:
        raise AdapterError(f"{target!r} has no attribute {attribute!r}") from exc

    if not callable(obj):
        raise AdapterError(f"{spec} is not callable")
    if inspect.isabstract(obj):
        raise AdapterError(f"{spec} is an abstract base, not a concrete submission")
    if not hasattr(obj, required_method):
        raise AdapterError(f"{spec} has no .{required_method}() method")
    return obj


def _load_from_path(path: str) -> Any:
    resolved = os.path.abspath(path)
    if not os.path.exists(resolved):
        raise AdapterError(f"no such file: {path}")
    name = "adapter_" + os.path.splitext(os.path.basename(resolved))[0]
    spec = importlib.util.spec_from_file_location(name, resolved)
    if spec is None or spec.loader is None:
        raise AdapterError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def reject_private_path(value: str) -> str:
    """Guard for every user-supplied path argument in a public runner.

    ``run.py`` accepts seeds and output paths from contestants.  Neither may be
    steered at organizer-only material, so the check lives here rather than
    being re-derived (and eventually forgotten) in each benchmark.
    """
    parts = {p.lower() for p in os.path.normpath(value).split(os.sep)}
    if parts & PRIVATE_DIR_NAMES:
        raise AdapterError(f"refusing to touch organizer-only path: {value}")
    return value
