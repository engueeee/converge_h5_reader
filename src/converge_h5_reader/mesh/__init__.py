"""Optional mesh backends (the ``[mesh]`` extra).

Importing this package does **not** import vtk or pyvista: those live at module
scope of the backend submodules, which :func:`get_backend` imports lazily and
guards with a :class:`~converge_h5_reader.exceptions.MissingBackendError`.

Two backends ship with the package:

``vtk`` (default)
    ``vtkCONVERGECFDReader`` -> ``pyvista.MultiBlock``.  The C++ reader already
    assembles the polyhedral cut cells, the named boundary surfaces and the
    multiple internal timesteps a single ``post*.h5`` can hold.
``h5``
    Builds a ``VTK_POLYHEDRON`` grid straight from ``CONNECTIVITY``.  The escape
    hatch for VTK builds without the CONVERGE reader, and for subset meshes.

Third parties can register their own through the
``converge_h5_reader.mesh_backends`` entry-point group.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import importlib.util
from pathlib import Path
from typing import Any

from ..exceptions import MissingBackendError
from .base import MeshBackend

__all__ = ["MeshBackend", "available_backends", "get_backend", "read_mesh", "register_backend"]

ENTRY_POINT_GROUP = "converge_h5_reader.mesh_backends"

#: Backends bundled with the package, so we work from a source tree too.
_BUILTIN = {
    "vtk": "converge_h5_reader.mesh.vtk_reader:VTKConvergeBackend",
    "h5": "converge_h5_reader.mesh.polyhedral:PolyhedralBackend",
}

_REGISTERED: dict[str, type[MeshBackend]] = {}


def register_backend(cls: type[MeshBackend]) -> type[MeshBackend]:
    """Register a backend class under its ``name``. Usable as a decorator."""
    _REGISTERED[cls.name] = cls
    return cls


def _entry_points() -> dict[str, str]:
    """Map backend name -> ``module:attr`` from entry points, merged over the builtins."""
    targets = dict(_BUILTIN)

    entry_points = importlib.metadata.entry_points()
    if hasattr(entry_points, "select"):  # Python >= 3.10
        found = entry_points.select(group=ENTRY_POINT_GROUP)
    else:  # pragma: no cover - Python 3.9 returns a plain dict
        found = entry_points.get(ENTRY_POINT_GROUP, [])

    for entry in found:
        targets[entry.name] = entry.value
    return targets


def _load(name: str) -> type[MeshBackend]:
    """Import and return a backend class, translating ImportError into a helpful message."""
    if name in _REGISTERED:
        return _REGISTERED[name]

    targets = _entry_points()
    if name not in targets:
        raise MissingBackendError(
            f"unknown mesh backend {name!r}; known backends: {sorted(targets)}"
        )

    module_name, _, attr = targets[name].partition(":")
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise MissingBackendError(
            f"the {name!r} mesh backend needs vtk/pyvista, which are not installed.\n"
            f"  pip install 'converge-h5-reader[mesh]'\n"
            f"original import error: {exc}"
        ) from exc

    cls: type[MeshBackend] = getattr(module, attr)
    return register_backend(cls)


def get_backend(name: str = "vtk") -> MeshBackend:
    """Return an instance of the named backend, importing it on demand."""
    return _load(name)()


def available_backends() -> tuple[str, ...]:
    """Names of the backends whose dependencies are actually importable."""
    names = []
    for name, target in sorted(_entry_points().items()):
        module_name = target.partition(":")[0]
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue
        attr = target.partition(":")[2]
        cls = getattr(module, attr, None)
        if cls is not None:
            names.append(name)
    return tuple(names)


def read_mesh(path: str | Path, *, backend: str = "vtk", **kwargs: Any) -> Any:
    """Read a ``post*.h5`` into a PyVista mesh using the named backend."""
    return get_backend(backend).read(Path(path), **kwargs)
