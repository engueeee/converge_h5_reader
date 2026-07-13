"""The top-level handle on a CONVERGE ``post*.h5`` file."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from functools import cached_property
from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING, Any

import h5py

from .aliases import DEFAULT_ALIASES, AliasRegistry
from .boundaries import BoundaryTable
from .exceptions import ClosedFileError, StreamNotFoundError
from .naming import cad_from_filename, normalize_stream
from .stream import Stream, _scalarize

if TYPE_CHECKING:
    import pyvista as pv

__all__ = ["ConvergeFile", "open_file"]


class ConvergeFile:
    """Lazy handle on one ``post*.h5``.

    Opening reads nothing: the HDF5 file is opened on first access and only the
    (tiny) attribute and boundary metadata is cached.  Bulk cell data is read on
    demand through :attr:`Stream.cells`.

    Use as a context manager, or call :meth:`close` explicitly.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        aliases: AliasRegistry = DEFAULT_ALIASES,
        mode: str = "r",
    ) -> None:
        self.path = Path(path)
        self.aliases = aliases
        self.mode = mode
        self._h5: h5py.File | None = None
        self._closed = False
        self._streams: dict[str, Stream] = {}

    @property
    def h5(self) -> h5py.File:
        """The underlying :class:`h5py.File`, opened on first access."""
        if self._closed:
            raise ClosedFileError(f"{self.path.name} has been closed")
        if self._h5 is None:
            self._h5 = h5py.File(self.path, self.mode)
        return self._h5

    def close(self) -> None:
        """Close the file. Further access raises :class:`ClosedFileError`."""
        if self._h5 is not None:
            self._h5.close()
            self._h5 = None
        self._closed = True

    def __enter__(self) -> ConvergeFile:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    @cached_property
    def attrs(self) -> Mapping[str, Any]:
        """Root attributes, with CONVERGE's 1-element arrays reduced to scalars."""
        return _scalarize(self.h5.attrs)

    @property
    def crank_angle(self) -> float:
        """Crank angle of this output, in degrees (``CRANK_ANGLE``)."""
        return float(self.attrs["CRANK_ANGLE"])

    @property
    def time(self) -> float:
        """Physical time of this output, in seconds (``OUTPUT_TIME_SEC``)."""
        return float(self.attrs["OUTPUT_TIME_SEC"])

    @property
    def rpm(self) -> float:
        """Engine speed (``RPM``)."""
        return float(self.attrs["RPM"])

    @property
    def version(self) -> tuple[int, int, int]:
        """CONVERGE version that wrote the file."""
        return (
            int(self.attrs["VERSION_NUM1"]),
            int(self.attrs["VERSION_NUM2"]),
            int(self.attrs["VERSION_NUM3"]),
        )

    @property
    def cad_from_name(self) -> float | None:
        """Crank angle encoded in the filename, or ``None`` if it does not match."""
        return cad_from_filename(self.path, strict=False)

    @cached_property
    def stream_names(self) -> tuple[str, ...]:
        """Names of the ``STREAM_NN`` groups, sorted."""
        return tuple(sorted(key for key in self.h5 if key.upper().startswith("STREAM_")))

    def stream(self, stream: int | str = 0) -> Stream:
        """Return one stream by index (``0``) or name (``"STREAM_00"``)."""
        name = normalize_stream(stream)
        if name not in self.stream_names:
            raise StreamNotFoundError(
                f"{name} is not in {self.path.name}; available: {list(self.stream_names)}"
            )
        if name not in self._streams:
            self._streams[name] = Stream(self, name)
        return self._streams[name]

    def __getitem__(self, stream: int | str) -> Stream:
        """Sugar for :meth:`stream`."""
        return self.stream(stream)

    def __iter__(self) -> Iterator[Stream]:
        return (self.stream(name) for name in self.stream_names)

    def __len__(self) -> int:
        return len(self.stream_names)

    @cached_property
    def boundaries(self) -> BoundaryTable:
        """The ``BOUNDARIES`` table (30 rows on a typical engine case)."""
        if "BOUNDARIES" not in self.h5:
            return BoundaryTable([])
        return BoundaryTable.from_h5(self.h5["BOUNDARIES"])

    def mesh(self, *, backend: str = "vtk", **kwargs: Any) -> pv.DataSet | pv.MultiBlock:
        """Build a PyVista mesh. Requires the ``[mesh]`` extra (vtk/pyvista).

        Raises :class:`~converge_h5_reader.exceptions.MissingBackendError` if the
        backend's dependencies are not installed.
        """
        from .mesh import read_mesh

        return read_mesh(self.path, backend=backend, **kwargs)

    def __repr__(self) -> str:
        state = "closed" if self._closed else "open"
        return f"ConvergeFile({self.path.name!r}, {state})"


def open_file(path: str | Path, **kwargs: Any) -> ConvergeFile:
    """Open a CONVERGE ``post*.h5`` file. Alias for :class:`ConvergeFile`."""
    return ConvergeFile(path, **kwargs)
