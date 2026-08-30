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

import dataclasses
import importlib
import types
import typing
from typing import TYPE_CHECKING, Any, TypeVar, Union

import tomllib

if TYPE_CHECKING:
    from pathlib import Path

T = TypeVar("T")


def load_toml_as_dataclass(path: str | Path, cls: type[T]) -> T:
    """Read a TOML file and convert its full contents into an instance of dataclass `cls`
    via :func:`dict_to_dataclass`.
    """
    with open(path, "rb") as fh:
        data = tomllib.load(fh)
    return dict_to_dataclass(data, cls)


def dict_to_dataclass(data: dict, cls: type[T]) -> T:
    """Recursively convert a dict decoded from TOML into an instance of the
    dataclass `cls`.

    Per field, in priority order:

    1. If the raw value is a `class`/`kwargs` dynamic spec, it is instantiated via
       `resolve_value`, regardless of the field's declared type.
    2. If the field's type (after unwrapping `Optional`) is itself a dataclass and the
       raw value is a dict, it is recursively converted with `dict_to_dataclass`.
    3. If the field's type is `list[SomeDataclass]`, each element is recursively
       converted the same way.
    4. Otherwise the raw value is passed through `resolve_value`, which walks
       dicts/lists for nested dynamic specs.

    Fields absent from `data` are left to the dataclass's own default.
    """
    if not dataclasses.is_dataclass(cls):
        raise TypeError(f"{cls!r} is not a dataclass")
    if not isinstance(data, dict):
        raise TypeError(f"Expected a dict to build {cls.__name__}, got {type(data).__name__}")

    hints = typing.get_type_hints(cls)
    kwargs: dict[str, Any] = {}

    for f in dataclasses.fields(cls):
        if f.name not in data:
            continue
        raw = data[f.name]
        field_type, _ = _unwrap_optional(hints.get(f.name, f.type))
        origin = typing.get_origin(field_type)
        if raw is None:
            kwargs[f.name] = None
        elif _is_dynamic_spec(raw):
            kwargs[f.name] = resolve_value(raw)
        elif dataclasses.is_dataclass(field_type) and isinstance(raw, dict):
            kwargs[f.name] = dict_to_dataclass(raw, field_type)
        elif origin in (list, typing.List) and isinstance(raw, list):
            item_args = typing.get_args(field_type)
            item_type, _ = _unwrap_optional(item_args[0]) if item_args else (Any, False)
            if dataclasses.is_dataclass(item_type):
                kwargs[f.name] = [
                    dict_to_dataclass(item, item_type)
                    if isinstance(item, dict)
                    else resolve_value(item)
                    for item in raw
                ]
            else:
                kwargs[f.name] = [resolve_value(item) for item in raw]
        else:
            kwargs[f.name] = resolve_value(raw)
    return cls(**kwargs)


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
    try:
        module = importlib.import_module(module_path)
    except ModuleNotFoundError:
        # try to walk-back dotted path (nested classes)
        module = import_from_path(module_path)
    assert hasattr(module, attr_name), f"Module '{module_path}' has no attribute '{attr_name}'"
    return getattr(module, attr_name)


# region Helpers


def _is_dynamic_spec(value: Any) -> bool:
    return isinstance(value, dict) and "class" in value


def _unwrap_optional(tp: Any) -> tuple[Any, bool]:
    """Return `(inner_type, True)` for `Optional[X]` / `X | None`,
    otherwise `(tp, False)`."""
    origin = typing.get_origin(tp)
    if origin is Union or origin is getattr(types, "UnionType", None):
        non_none = [a for a in typing.get_args(tp) if a is not type(None)]
        if len(non_none) == 1:
            return non_none[0], True
    return tp, False
