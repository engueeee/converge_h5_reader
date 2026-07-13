"""Modular, lazy reader for CONVERGE CFD HDF5 ``post*.h5`` output.

The core depends only on h5py, numpy and pandas.  VTK and PyVista are optional
and live behind the ``[mesh]`` extra -- importing this package never imports them.

    >>> from converge_h5_reader import ConvergeFile, Field, region
    >>> with ConvergeFile("post000061_-7.40757e+01.h5") as f:
    ...     stream = f[0]
    ...     df = stream.cells.to_dataframe(["T", "P", "u", "v", "w"], where=region(1))
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from typing import Any

from .aliases import DEFAULT_ALIASES, DEFAULT_PREFIXES, AliasRegistry
from .boundaries import Boundary, BoundaryTable
from .exceptions import (
    ClosedFileError,
    ConvergeH5Error,
    MissingBackendError,
    MissingFieldError,
    StreamNotFoundError,
)
from .file import ConvergeFile, open_file
from .index import INDEX_COLUMNS, RunConfig, build_index, find_converge_h5_files, select_at_cad
from .naming import H5_PATTERN, cad_from_filename, normalize_stream
from .selection import And, Field, FieldPredicate, Not, Or, Predicate, region
from .stream import CellData, Connectivity, Stream

try:
    __version__ = version("converge-h5-reader")
except PackageNotFoundError:  # pragma: no cover - running from a source tree
    __version__ = "0.0.0.dev0"

__all__ = [
    "DEFAULT_ALIASES",
    "DEFAULT_PREFIXES",
    "H5_PATTERN",
    "INDEX_COLUMNS",
    "AliasRegistry",
    "And",
    "Boundary",
    "BoundaryTable",
    "CellData",
    "ClosedFileError",
    "Connectivity",
    "ConvergeFile",
    "ConvergeH5Error",
    "Field",
    "FieldPredicate",
    "MissingBackendError",
    "MissingFieldError",
    "Not",
    "Or",
    "Predicate",
    "RunConfig",
    "Stream",
    "StreamNotFoundError",
    "__version__",
    "build_index",
    "cad_from_filename",
    "find_converge_h5_files",
    "normalize_stream",
    "open_file",
    "region",
    "select_at_cad",
]


def __getattr__(name: str) -> Any:
    """Expose ``converge_h5_reader.mesh`` without importing vtk/pyvista at import time."""
    if name == "mesh":
        import importlib

        return importlib.import_module(".mesh", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
