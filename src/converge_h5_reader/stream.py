"""Per-stream access: cell-centre variables, vertices and connectivity."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from functools import cached_property
from typing import TYPE_CHECKING, Any

import numpy as np

from .exceptions import MissingFieldError
from .selection import Mask, as_mask, read_masked

if TYPE_CHECKING:
    import h5py
    import pandas as pd

    from .file import ConvergeFile

__all__ = ["CellData", "Connectivity", "Stream"]


def _scalarize(attrs: Mapping[str, Any]) -> dict[str, Any]:
    """Turn CONVERGE's 1-element attribute arrays into plain Python scalars."""
    out: dict[str, Any] = {}
    for key, value in attrs.items():
        arr = np.asarray(value)
        if arr.ndim == 1 and arr.size == 1:
            out[key] = arr[0].item()
        elif arr.ndim == 0:
            out[key] = arr.item()
        else:
            out[key] = arr
    return out


class Connectivity:
    """Lazy view on ``CONNECTIVITY``; exposes handles, never materialises arrays.

    The three datasets describe a polyhedral cut-cell mesh:

    ``POLYGON_OFFSET``
        ``n_polygons + 1`` offsets into ``POLYGON_TO_VERTEX``.
    ``POLYGON_TO_VERTEX``
        Concatenated vertex ids of every polygon.
    ``CONNECTED_CELLS``
        Two cell ids per polygon (the cells it separates).

    On a production file these hold 27.8M / 112M / 55.6M entries, so nothing is
    read until you ask for it.
    """

    def __init__(self, group: h5py.Group) -> None:
        self._group = group

    @property
    def polygon_offset(self) -> h5py.Dataset:
        """Handle on ``POLYGON_OFFSET`` (no read)."""
        return self._group["POLYGON_OFFSET"]

    @property
    def polygon_to_vertex(self) -> h5py.Dataset:
        """Handle on ``POLYGON_TO_VERTEX`` (no read)."""
        return self._group["POLYGON_TO_VERTEX"]

    @property
    def connected_cells(self) -> h5py.Dataset:
        """Handle on ``CONNECTED_CELLS`` (no read)."""
        return self._group["CONNECTED_CELLS"]

    @property
    def n_polygons(self) -> int:
        """Number of polygons (faces) in the mesh."""
        return int(self.polygon_offset.shape[0]) - 1

    def polygon(self, index: int) -> np.ndarray:
        """Vertex ids of a single polygon."""
        start, stop = self.polygon_offset[index : index + 2]
        return np.asarray(self.polygon_to_vertex[int(start) : int(stop)])

    def cell_map(self) -> tuple[np.ndarray, np.ndarray]:
        """Build a CSR cell -> polygons map.

        Reads all of ``CONNECTED_CELLS`` and allocates two arrays of comparable
        size (hundreds of MB on a production file), so this is opt-in.
        Returns ``(offsets, polygons)`` where cell ``i`` owns
        ``polygons[offsets[i]:offsets[i + 1]]``.
        """
        pairs = np.asarray(self.connected_cells[...]).reshape(-1, 2)
        n_polygons = pairs.shape[0]
        poly_ids = np.repeat(np.arange(n_polygons, dtype=np.int64), 2)
        cell_ids = pairs.ravel()

        interior = cell_ids >= 0
        poly_ids = poly_ids[interior]
        cell_ids = cell_ids[interior]

        order = np.argsort(cell_ids, kind="stable")
        cell_ids = cell_ids[order]
        polygons = poly_ids[order]

        n_cells = int(cell_ids[-1]) + 1 if cell_ids.size else 0
        counts = np.bincount(cell_ids, minlength=n_cells)
        offsets = np.zeros(n_cells + 1, dtype=np.int64)
        np.cumsum(counts, out=offsets[1:])
        return offsets, polygons


class CellData(Mapping[str, np.ndarray]):
    """Lazy mapping over ``CELL_CENTER_DATA``.

    ``__getitem__`` performs exactly one HDF5 read and does **not** cache: a
    single field of a production stream is 32 MB and the full set is 1.16 GB.
    """

    def __init__(self, stream: Stream, group: h5py.Group) -> None:
        self._stream = stream
        self._group = group

    @property
    def _aliases(self) -> Any:
        return self._stream.file.aliases

    def dataset(self, name: str) -> h5py.Dataset:
        """Return the raw HDF5 handle for a field, resolving aliases. Reads nothing."""
        resolved = self._aliases.resolve(name)
        try:
            return self._group[resolved]
        except KeyError:
            raise MissingFieldError(
                f"{name!r} (resolved to {resolved!r}) is not in "
                f"{self._stream.name}/CELL_CENTER_DATA; available: {sorted(self._group)}"
            ) from None

    def __getitem__(self, name: str) -> np.ndarray:
        """Read one field in full, preserving its on-disk dtype."""
        return np.asarray(self.dataset(name)[...])

    def read(
        self,
        name: str,
        *,
        where: Mask | None = None,
        dtype: np.dtype[Any] | str | None = None,
    ) -> np.ndarray:
        """Read one field, optionally restricted to the cells selected by ``where``."""
        mask = as_mask(where, self, self._stream.n_cells)
        values = read_masked(self.dataset(name), mask)
        if dtype is not None:
            values = values.astype(dtype, copy=False)
        return values

    def to_arrays(
        self,
        fields: Sequence[str] | None = None,
        *,
        where: Mask | None = None,
        dtype: np.dtype[Any] | str | None = None,
    ) -> dict[str, np.ndarray]:
        """Read several fields at once, sharing a single evaluation of ``where``."""
        names = list(fields) if fields is not None else list(self.keys())
        mask = as_mask(where, self, self._stream.n_cells)
        out: dict[str, np.ndarray] = {}
        for name in names:
            values = read_masked(self.dataset(name), mask)
            out[name] = values.astype(dtype, copy=False) if dtype is not None else values
        return out

    def to_dataframe(
        self,
        fields: Sequence[str] | None = None,
        *,
        where: Mask | None = None,
        dtype: np.dtype[Any] | str | None = None,
        rename: bool = True,
        add_time: bool = True,
        add_cad: bool = True,
    ) -> pd.DataFrame:
        """Read fields into a DataFrame.

        ``fields=None`` reads *every* variable, which on a production stream is
        36 x 8.06M float32 = 1.16 GB -- pass an explicit list unless you mean it.
        Columns keep their on-disk dtype (float32) unless ``dtype`` says otherwise.
        With ``rename=True`` columns use short aliases (``TEMPERATURE`` -> ``T``).
        """
        import pandas as pd

        names = list(fields) if fields is not None else list(self.keys())
        arrays = self.to_arrays(names, where=where, dtype=dtype)

        columns: dict[str, np.ndarray] = {}
        for name in names:
            label = self._aliases.alias_for(self._aliases.resolve(name)) if rename else name
            columns[label] = arrays[name]

        frame = pd.DataFrame(columns, copy=False)
        if add_time:
            frame["time"] = self._stream.file.time
        if add_cad:
            frame["cad"] = self._stream.file.crank_angle
        return frame

    def iter_chunks(
        self,
        fields: Sequence[str],
        *,
        chunk: int = 1_000_000,
        rename: bool = True,
    ) -> Iterator[pd.DataFrame]:
        """Stream fields in contiguous row blocks, bounding memory to ``chunk * n_fields * 4`` B."""
        import pandas as pd

        if chunk <= 0:
            raise ValueError(f"chunk must be positive, got {chunk}")
        datasets = {name: self.dataset(name) for name in fields}
        n_cells = self._stream.n_cells
        for start in range(0, n_cells, chunk):
            stop = min(start + chunk, n_cells)
            columns = {}
            for name, ds in datasets.items():
                label = self._aliases.alias_for(ds.name.rsplit("/", 1)[-1]) if rename else name
                columns[label] = np.asarray(ds[start:stop])
            yield pd.DataFrame(columns, copy=False)

    def keys(self) -> Any:
        """Dataset names present in the stream, in the file's declared order."""
        return self._stream.variables

    def aliases(self) -> tuple[str, ...]:
        """Short names available for the datasets in this stream."""
        return tuple(self._aliases.alias_for(name) for name in self.keys())

    def dtype(self, name: str) -> np.dtype[Any]:
        """Dtype of a field, without reading it."""
        return self.dataset(name).dtype

    def shape(self, name: str) -> tuple[int, ...]:
        """Shape of a field, without reading it."""
        return tuple(self.dataset(name).shape)

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and self._aliases.resolve(name) in self._group

    def __iter__(self) -> Iterator[str]:
        return iter(self.keys())

    def __len__(self) -> int:
        return len(self._stream.variables)

    def __repr__(self) -> str:
        return f"CellData({self._stream.name}: {len(self)} variables)"


class Stream:
    """One ``STREAM_NN`` group."""

    def __init__(self, file: ConvergeFile, name: str) -> None:
        self.file = file
        self.name = name

    @property
    def _group(self) -> h5py.Group:
        return self.file.h5[self.name]

    @cached_property
    def attrs(self) -> Mapping[str, Any]:
        """Stream attributes, scalarised (includes ``CELL_COUNT``)."""
        return _scalarize(self._group.attrs)

    @cached_property
    def n_cells(self) -> int:
        """Number of cells, from the ``CELL_COUNT`` attribute."""
        count = self.attrs.get("CELL_COUNT")
        if count is None:
            return int(self._group["CELL_CENTER_DATA"][self.variables[0]].shape[0])
        return int(count)

    @property
    def n_vertices(self) -> int:
        """Number of mesh vertices."""
        return int(self._group["VERTEX_COORDINATES/X"].shape[0])

    @cached_property
    def variables(self) -> tuple[str, ...]:
        """Cell variable names, from the authoritative ``VARIABLE_NAMES/CELL_VARIABLES``.

        Falls back to the ``CELL_CENTER_DATA`` keys when that list is absent.
        """
        group = self._group
        path = "VARIABLE_NAMES/CELL_VARIABLES"
        if path not in group:
            return tuple(sorted(group["CELL_CENTER_DATA"]))
        raw = group[path][:]
        available = set(group["CELL_CENTER_DATA"])
        names = [
            item.decode("utf-8", errors="replace").rstrip("\x00").strip()
            if isinstance(item, bytes)
            else str(item).strip()
            for item in raw
        ]
        return tuple(name for name in names if name in available)

    @cached_property
    def cells(self) -> CellData:
        """Lazy accessor for the cell-centre variables."""
        return CellData(self, self._group["CELL_CENTER_DATA"])

    @cached_property
    def connectivity(self) -> Connectivity:
        """Lazy accessor for the polyhedral connectivity."""
        return Connectivity(self._group["CONNECTIVITY"])

    def vertices(self, *, dtype: Any = np.float32) -> np.ndarray:
        """Read the mesh vertices as an ``(n_vertices, 3)`` array."""
        group = self._group["VERTEX_COORDINATES"]
        return np.column_stack(
            [np.asarray(group[axis][...], dtype=dtype) for axis in ("X", "Y", "Z")]
        )

    def __repr__(self) -> str:
        return f"Stream({self.name!r}, n_cells={self.n_cells})"
