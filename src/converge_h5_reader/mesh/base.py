"""The mesh backend protocol.

This module must stay free of vtk/pyvista imports -- it is imported by the
registry, which is reachable without the ``[mesh]`` extra.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar, Protocol, runtime_checkable

__all__ = ["MeshBackend"]


@runtime_checkable
class MeshBackend(Protocol):
    """Turns a ``post*.h5`` path into a PyVista dataset."""

    #: Registry key, e.g. ``"vtk"``.
    name: ClassVar[str]
    #: Importable module names this backend needs, e.g. ``("vtk", "pyvista")``.
    requires: ClassVar[tuple[str, ...]]

    def read(self, path: Path, **kwargs: Any) -> Any:
        """Build and return the mesh."""
        ...
