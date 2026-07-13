"""The ``h5`` mesh backend: build a polyhedral grid straight from ``CONNECTIVITY``.

This bypasses ``vtkCONVERGECFDReader`` entirely (it still needs VTK itself, to
hold the grid).  Use it when the CONVERGE reader is unavailable, or -- its real
strength -- to build a mesh for a *subset* of cells selected with a
:class:`~converge_h5_reader.selection.Predicate`, without materialising the whole
8M-cell grid.

Cell assembly runs in Python, so it is markedly slower than the C++ reader.
``backend="vtk"`` remains the recommended default for whole-domain reads.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
import pyvista as pv
import vtk
from vtk.util.numpy_support import numpy_to_vtk

from ..file import ConvergeFile
from ..selection import Mask, as_mask

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = ["PolyhedralBackend", "build_polyhedral_grid"]


def build_polyhedral_grid(
    path: str | Path,
    *,
    stream: int | str = 0,
    fields: Sequence[str] | None = None,
    where: Mask | None = None,
    point_dtype: Any = np.float32,
) -> pv.UnstructuredGrid:
    """Assemble a ``VTK_POLYHEDRON`` grid for a stream, optionally restricted to a cell subset.

    ``fields`` are attached as cell data (aliases resolved); ``None`` attaches none.
    ``where`` selects the cells to include.
    """
    with ConvergeFile(path) as f:
        st = f.stream(stream)
        conn = st.connectivity

        mask = as_mask(where, st.cells, st.n_cells)
        cell_ids = np.arange(st.n_cells) if mask is None else np.flatnonzero(mask)
        if cell_ids.size == 0:
            raise ValueError("the selection is empty; no cells to build a mesh from")

        offsets, cell_polygons = conn.cell_map()
        polygon_offset = np.asarray(conn.polygon_offset[...])
        polygon_to_vertex = np.asarray(conn.polygon_to_vertex[...])

        grid = vtk.vtkUnstructuredGrid()

        points = vtk.vtkPoints()
        points.SetData(numpy_to_vtk(st.vertices(dtype=point_dtype), deep=True))
        grid.SetPoints(points)

        grid.Allocate(cell_ids.size)
        id_list = vtk.vtkIdList()
        for cell in cell_ids:
            faces = cell_polygons[offsets[cell] : offsets[cell + 1]]
            id_list.Reset()
            id_list.InsertNextId(len(faces))
            for face in faces:
                vertices = polygon_to_vertex[polygon_offset[face] : polygon_offset[face + 1]]
                id_list.InsertNextId(len(vertices))
                for vertex in vertices:
                    id_list.InsertNextId(int(vertex))
            grid.InsertNextCell(vtk.VTK_POLYHEDRON, id_list)

        wrapped = pv.wrap(grid)
        if fields:
            arrays = st.cells.to_arrays(list(fields), where=mask)
            for name, values in arrays.items():
                wrapped.cell_data[name] = values
        return wrapped


class PolyhedralBackend:
    """Mesh backend that assembles the grid from the raw HDF5 connectivity."""

    name: ClassVar[str] = "h5"
    requires: ClassVar[tuple[str, ...]] = ("vtk", "pyvista")

    def read(
        self,
        path: Path,
        *,
        stream: int | str = 0,
        fields: Sequence[str] | None = None,
        where: Mask | None = None,
    ) -> pv.UnstructuredGrid:
        """Build the grid for one stream."""
        return build_polyhedral_grid(path, stream=stream, fields=fields, where=where)
