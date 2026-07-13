"""Cell selection: composable predicates evaluated before any payload is read.

A predicate names the fields it needs, so filtering a 8M-cell stream on
``REGION_ID == 1`` costs one 32 MB read plus an 8 MB bool mask -- the requested
payload fields are then read for the selected rows only.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Protocol, Union, runtime_checkable

import numpy as np

if TYPE_CHECKING:
    import h5py

    from .stream import CellData

__all__ = [
    "And",
    "Field",
    "FieldPredicate",
    "Not",
    "Or",
    "Predicate",
    "as_mask",
    "region",
]

Op = Literal["==", "!=", "<", "<=", ">", ">="]

_OPS: dict[str, Callable[[Any, Any], Any]] = {
    "==": np.equal,
    "!=": np.not_equal,
    "<": np.less,
    "<=": np.less_equal,
    ">": np.greater,
    ">=": np.greater_equal,
}

#: Fraction of selected cells above which a contiguous read beats point selection.
#: h5py fancy indexing degrades badly for large index sets.
MASKED_READ_THRESHOLD = 0.3


@runtime_checkable
class Predicate(Protocol):
    """Anything that can turn a :class:`~converge_h5_reader.stream.CellData` into a bool mask."""

    def evaluate(self, cells: CellData) -> np.ndarray:
        """Return a bool mask of shape ``(n_cells,)``."""
        ...


Mask = Union[np.ndarray, Predicate]


class _Combinable:
    """Mixin giving predicates ``&``, ``|`` and ``~``."""

    def __and__(self, other: Predicate) -> And:
        return And(self, other)  # type: ignore[arg-type]

    def __or__(self, other: Predicate) -> Or:
        return Or(self, other)  # type: ignore[arg-type]

    def __invert__(self) -> Not:
        return Not(self)  # type: ignore[arg-type]


@dataclass(frozen=True)
class FieldPredicate(_Combinable):
    """Compare one cell field against a scalar."""

    field: str
    op: Op
    value: float

    def evaluate(self, cells: CellData) -> np.ndarray:
        """Read the field and compare it, returning a bool mask."""
        return np.asarray(_OPS[self.op](cells[self.field], self.value), dtype=bool)


@dataclass(frozen=True)
class And(_Combinable):
    """Logical conjunction of two predicates."""

    left: Predicate
    right: Predicate

    def evaluate(self, cells: CellData) -> np.ndarray:
        """Return the intersection of the two operands' masks."""
        return self.left.evaluate(cells) & self.right.evaluate(cells)


@dataclass(frozen=True)
class Or(_Combinable):
    """Logical disjunction of two predicates."""

    left: Predicate
    right: Predicate

    def evaluate(self, cells: CellData) -> np.ndarray:
        """Return the union of the two operands' masks."""
        return self.left.evaluate(cells) | self.right.evaluate(cells)


@dataclass(frozen=True)
class Not(_Combinable):
    """Logical negation of a predicate."""

    inner: Predicate

    def evaluate(self, cells: CellData) -> np.ndarray:
        """Return the complement of the operand's mask."""
        return ~self.inner.evaluate(cells)


class Field:
    """Sugar for building predicates: ``Field("REGION_ID") == 1``."""

    __slots__ = ("name",)

    def __init__(self, name: str) -> None:
        self.name = name

    def __eq__(self, other: object) -> FieldPredicate:  # type: ignore[override]
        return FieldPredicate(self.name, "==", float(other))  # type: ignore[arg-type]

    def __ne__(self, other: object) -> FieldPredicate:  # type: ignore[override]
        return FieldPredicate(self.name, "!=", float(other))  # type: ignore[arg-type]

    def __lt__(self, other: float) -> FieldPredicate:
        return FieldPredicate(self.name, "<", float(other))

    def __le__(self, other: float) -> FieldPredicate:
        return FieldPredicate(self.name, "<=", float(other))

    def __gt__(self, other: float) -> FieldPredicate:
        return FieldPredicate(self.name, ">", float(other))

    def __ge__(self, other: float) -> FieldPredicate:
        return FieldPredicate(self.name, ">=", float(other))

    __hash__ = None  # type: ignore[assignment]


@dataclass(frozen=True)
class _RegionPredicate(_Combinable):
    """``REGION_ID == region_id``, falling back to the legacy ``IDREG`` dataset."""

    region_id: int

    def evaluate(self, cells: CellData) -> np.ndarray:
        name = "REGION_ID"
        if name not in cells and "IDREG" in cells:
            warnings.warn(
                "REGION_ID is absent; falling back to the legacy IDREG dataset",
                DeprecationWarning,
                stacklevel=2,
            )
            name = "IDREG"
        # REGION_ID is stored as float32; compare after rounding to be safe.
        values = np.rint(cells[name]).astype(np.int32, copy=False)
        return values == self.region_id


def region(region_id: int = 1) -> Predicate:
    """Select the cells of one CONVERGE region (region 1 is the cylinder)."""
    return _RegionPredicate(region_id)


def as_mask(where: Mask | None, cells: CellData, n_cells: int) -> np.ndarray | None:
    """Normalise ``where`` into a bool mask, or ``None`` when nothing is selected out."""
    if where is None:
        return None
    if isinstance(where, np.ndarray):
        if where.dtype != bool:
            raise TypeError(f"a mask array must be boolean, got dtype {where.dtype}")
        if where.shape != (n_cells,):
            raise ValueError(f"mask has shape {where.shape}, expected ({n_cells},)")
        return where
    mask = where.evaluate(cells)
    if mask.shape != (n_cells,):
        raise ValueError(f"predicate produced shape {mask.shape}, expected ({n_cells},)")
    return mask


def read_masked(
    dataset: h5py.Dataset,
    mask: np.ndarray | None,
    *,
    threshold: float = MASKED_READ_THRESHOLD,
) -> np.ndarray:
    """Read a dataset, applying ``mask`` with whichever strategy suits its selectivity."""
    if mask is None:
        return np.asarray(dataset[...])
    if mask.mean() > threshold:
        return np.asarray(dataset[...])[mask]
    indices = np.flatnonzero(mask)
    if indices.size == 0:
        return np.empty(0, dtype=dataset.dtype)
    return np.asarray(dataset[indices])
