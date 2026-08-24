from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, ClassVar

if TYPE_CHECKING:
    from collections.abc import Sequence

    import gymnasium as gym


class CurriculumTerm(ABC):
    """Base abstract class for curriculum learning terms.

    Args:
        paths: Sequence of dot-separated paths for target attributes
            (e.g., "env.reward.weights[0]").
    """

    TOKEN_RE: ClassVar = re.compile(r"([^.\[\]]+)|\[(\d+)\]")

    @dataclass(slots=True)
    class Address:
        """Dataclass for storing path-based attribute getter and setter functions."""

        get: Callable[[], Any]
        set: Callable[[Any], None]

    def __init__(self, paths: Sequence[str]):
        self.paths = paths
        self.addrs = None

    @abstractmethod
    def __call__(self, env: gym.Env, step: int) -> dict[str, Any] | None:
        """Update environment parameter values in place according to preset curriculum.

        Args:
            env: Gymnasium environment.
            step: Current environment step number.

        Returns:
            Optional dictionary of current parameters and their values.
        """
        ...

    def setup(self, env: gym.Env) -> None:
        """Cache the getters and setters for all managed path-based parameters."""
        self.addrs = [self._resolve_address(env, path) for path in self.paths]

    # region Helpers

    @staticmethod
    def _resolve_address(root: Any, path: str) -> CurriculumTerm.Address:
        """Factory for path-based :class:`Address` objects given the path's root object.

        Args:
            root: Root object of the given path, typically the environment.
            path: Dot-separated path for target attribute (e.g., "env.reward.weights[0]").

        Returns:
            Path address with registered getter and setter.
        """
        tokens = CurriculumTerm._tokenize(path)
        obj = root
        for tok in tokens[:-1]:
            obj = obj[tok] if isinstance(tok, int) or isinstance(obj, dict) else getattr(obj, tok)
        last = tokens[-1]
        if isinstance(last, int) or isinstance(obj, dict):
            return CurriculumTerm.Address(lambda: obj[last], lambda v: obj.__setitem__(last, v))
        return CurriculumTerm.Address(lambda: getattr(obj, last), lambda v: setattr(obj, last, v))

    @staticmethod
    def _tokenize(path: str) -> list[str | int]:
        """Tokenize path into names and indices.

        Examples:
            >>> CurriculumTerm._tokenize("env.reward.weights[0]")
            ["env", "reward", "weights", 0]
        """
        out: list[str | int] = []
        for name, idx in CurriculumTerm.TOKEN_RE.findall(path):
            out.append(name if name else int(idx))
        return out
