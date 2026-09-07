"""Local-only compatibility layer for legacy experiment tracking calls."""

from __future__ import annotations

from typing import Any


class _TensorBoard:
    @staticmethod
    def patch(*_: Any, **__: Any) -> None:
        return None


class Histogram:
    def __new__(cls, values: Any, *_: Any, **__: Any) -> Any:
        return values


class Image:
    def __new__(cls, value: Any, *_: Any, **__: Any) -> Any:
        return value


class Video:
    def __new__(cls, value: Any, *_: Any, **__: Any) -> Any:
        return value


tensorboard = _TensorBoard()
config: dict[str, Any] = {}


def init(*_: Any, **__: Any) -> None:
    return None


def login(*_: Any, **__: Any) -> None:
    return None


def log(*_: Any, **__: Any) -> None:
    return None


def finish(*_: Any, **__: Any) -> None:
    return None
