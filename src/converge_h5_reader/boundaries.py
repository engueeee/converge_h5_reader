"""The ``BOUNDARIES`` table of a CONVERGE ``post*.h5`` file."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, overload

import numpy as np

from .naming import normalize_stream

if TYPE_CHECKING:
    import h5py
    import pandas as pd

__all__ = ["Boundary", "BoundaryTable"]


def _decode(raw: object) -> str:
    """Decode a fixed-width HDF5 byte string, stripping NULs and padding."""
    if isinstance(raw, bytes):
        return raw.decode("utf-8", errors="replace").rstrip("\x00").strip()
    return str(raw).rstrip("\x00").strip()


@dataclass(frozen=True)
class Boundary:
    """One row of the ``BOUNDARIES`` table."""

    id: int
    name: str
    stream: str
    n_elements: int
    n_points: int
    center: tuple[float, float, float]

    @property
    def is_empty(self) -> bool:
        """True when the boundary carries no surface elements."""
        return self.n_elements == 0


class BoundaryTable(Sequence[Boundary]):
    """Sequence of :class:`Boundary`, indexable by position, id or name."""

    def __init__(self, boundaries: Sequence[Boundary]) -> None:
        self._items: tuple[Boundary, ...] = tuple(boundaries)

    @classmethod
    def from_h5(cls, group: h5py.Group) -> BoundaryTable:
        """Build the table from the file's ``BOUNDARIES`` group."""
        ids = np.asarray(group["BOUNDARY_IDS"][:])
        names = group["BOUNDARY_NAMES"][:]
        streams = group["STREAMS"][:]
        n_elements = np.asarray(group["NUM_ELEMENTS"][:])
        n_points = np.asarray(group["NUM_POINTS"][:])
        centers = [
            np.asarray(group[f"GEOMETRIC_CENTER_COORDINATE_{axis}"][:]) for axis in ("X", "Y", "Z")
        ]

        items = [
            Boundary(
                id=int(ids[i]),
                name=_decode(names[i]),
                stream=normalize_stream(_decode(streams[i])),
                n_elements=int(n_elements[i]),
                n_points=int(n_points[i]),
                center=(float(centers[0][i]), float(centers[1][i]), float(centers[2][i])),
            )
            for i in range(len(ids))
        ]
        return cls(items)

    @overload
    def __getitem__(self, key: int | str) -> Boundary: ...

    @overload
    def __getitem__(self, key: slice) -> BoundaryTable: ...

    def __getitem__(self, key: int | str | slice) -> Boundary | BoundaryTable:
        """Look up by boundary id (int), name (str) or slice.

        Note the int path is a *boundary id* lookup, not a positional one --
        CONVERGE ids are 1-based and need not be contiguous.
        """
        if isinstance(key, slice):
            return BoundaryTable(self._items[key])
        if isinstance(key, str):
            return self.by_name(key)
        return self.by_id(key)

    def by_id(self, boundary_id: int) -> Boundary:
        """Return the boundary with the given ``BOUNDARY_ID``."""
        for item in self._items:
            if item.id == boundary_id:
                return item
        raise KeyError(f"no boundary with id {boundary_id}; known ids: {list(self.ids())}")

    def by_name(self, name: str, *, case_sensitive: bool = False) -> Boundary:
        """Return the boundary with the given name."""
        target = name if case_sensitive else name.casefold()
        for item in self._items:
            candidate = item.name if case_sensitive else item.name.casefold()
            if candidate == target:
                return item
        raise KeyError(f"no boundary named {name!r}; known names: {list(self.names())}")

    def for_stream(self, stream: int | str) -> BoundaryTable:
        """Return only the boundaries belonging to ``stream``."""
        wanted = normalize_stream(stream)
        return BoundaryTable([b for b in self._items if b.stream == wanted])

    def nonempty(self) -> BoundaryTable:
        """Return only the boundaries that carry surface elements."""
        return BoundaryTable([b for b in self._items if not b.is_empty])

    def names(self) -> tuple[str, ...]:
        """Boundary names, in file order."""
        return tuple(b.name for b in self._items)

    def ids(self) -> tuple[int, ...]:
        """Boundary ids, in file order."""
        return tuple(b.id for b in self._items)

    def to_dataframe(self) -> pd.DataFrame:
        """Return the table as a pandas DataFrame."""
        import pandas as pd

        return pd.DataFrame(
            {
                "id": [b.id for b in self._items],
                "name": [b.name for b in self._items],
                "stream": [b.stream for b in self._items],
                "n_elements": [b.n_elements for b in self._items],
                "n_points": [b.n_points for b in self._items],
                "center_x": [b.center[0] for b in self._items],
                "center_y": [b.center[1] for b in self._items],
                "center_z": [b.center[2] for b in self._items],
            }
        )

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self) -> Iterator[Boundary]:
        return iter(self._items)

    def __repr__(self) -> str:
        return f"BoundaryTable({len(self._items)} boundaries: {', '.join(self.names())})"
