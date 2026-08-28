"""
Utilities for turning TOML-loaded dicts into (possibly nested) Python object instances
with support for dynamically importing and instantiating objects from the config itself.

Convention
----------
Anywhere in the TOML you can write a table shaped like::

    [some.path]
    class = "package.module.ClassName"

    [some.path.kwargs]
    foo = 1
    bar = "baz"

and it will be turned into `ClassName(foo=1, bar="baz")` at load time.
"""

from __future__ import annotations

import importlib
from typing import Any


def instantiate(spec: dict, *args: Any) -> Any:
    """Instantiate a `{"class": ..., "kwargs": {...}}` spec as `cls(*args, **kwargs)`.

    `kwargs` values are resolved recursively, so specs may nest. `*args` lets callers
    supply positional arguments the target constructor needs *ahead of* `kwargs`.
    """
    if not _is_dynamic_spec(spec):
        raise ValueError(f"Not a class/kwargs spec: {spec!r}")
    cls = import_from_path(spec["class"])
    raw_kwargs = spec.get("kwargs", {})
    if not isinstance(raw_kwargs, dict):
        raise TypeError(f"'kwargs' must be a table, got {type(raw_kwargs).__name__}")
    kwargs = {k: resolve_value(v) for k, v in raw_kwargs.items()}
    return cls(*args, **kwargs)


def resolve_value(value: Any) -> Any:
    """Recursively resolve a raw value decoded from TOML.

    A dynamic `class`/`kwargs` dict is instantiated (with no extra positional args
    -- use `instantiate` directly if the target needs any). Any other dict or list
    is walked recursively so nested dynamic specs are still found; everything else
    is returned unchanged.
    """
    if _is_dynamic_spec(value):
        return instantiate(value)
    if isinstance(value, dict):
        return {k: resolve_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [resolve_value(v) for v in value]
    return value


def import_from_path(dotted_path: str) -> Any:
    """Import an attribute (class, function, ...) from a dotted path such as
    `"gymnasium.wrappers.vector.TransformAction"`.
    """
    module_path, sep, attr_name = dotted_path.rpartition(".")
    assert sep, (
        f"'{dotted_path}' is not a valid dotted import path "
        "(expected something like 'module.submodule.ClassName')."
    )
    module = importlib.import_module(module_path)
    assert hasattr(module, attr_name), f"Module '{module_path}' has no attribute '{attr_name}'"
    return getattr(module, attr_name)


# region Helpers


def _is_dynamic_spec(value: Any) -> bool:
    return isinstance(value, dict) and "class" in value
