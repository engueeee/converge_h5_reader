"""Optional tqdm progress bars (the ``[progress]`` extra)."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any, TypeVar

from .exceptions import MissingBackendError

__all__ = ["wrap_progress"]

T = TypeVar("T")


def wrap_progress(items: Sequence[T] | Iterable[T], **kwargs: Any) -> Any:
    """Wrap an iterable in a tqdm bar, or explain how to get one."""
    try:
        from tqdm.auto import tqdm
    except ImportError as exc:
        raise MissingBackendError(
            "progress=True needs tqdm: pip install 'converge-h5-reader[progress]'"
        ) from exc
    return tqdm(items, **kwargs)
