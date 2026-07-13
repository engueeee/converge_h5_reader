"""The default mesh backend: ``vtkCONVERGECFDReader`` -> ``pyvista.MultiBlock``.

Ported from ``converge_post/helpers.py``.  vtk/pyvista are imported at module
scope here on purpose -- this module is only ever imported through
:func:`converge_h5_reader.mesh.get_backend`, which turns the ImportError into a
:class:`~converge_h5_reader.exceptions.MissingBackendError`.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import ClassVar

import numpy as np
import pyvista as pv
import vtk

from ..naming import H5_PATTERN

__all__ = [
    "VTKConvergeBackend",
    "calculate_enstrophy",
    "filter_by_region",
    "get_z_piston",
    "prepare_cylinder_mesh",
    "read_converge_h5",
    "read_timesteps",
]

_VOLUMETRIC_CELL_TYPES = frozenset(
    {
        vtk.VTK_TETRA,
        vtk.VTK_HEXAHEDRON,
        vtk.VTK_WEDGE,
        vtk.VTK_PYRAMID,
        vtk.VTK_VOXEL,
        vtk.VTK_PENTAGONAL_PRISM,
        vtk.VTK_HEXAGONAL_PRISM,
        vtk.VTK_POLYHEDRON,
        vtk.VTK_QUADRATIC_TETRA,
        vtk.VTK_QUADRATIC_HEXAHEDRON,
        vtk.VTK_QUADRATIC_WEDGE,
        vtk.VTK_QUADRATIC_PYRAMID,
    }
)


def _make_reader(path: Path) -> vtk.vtkCONVERGECFDReader:
    reader = vtk.vtkCONVERGECFDReader()
    reader.SetFileName(str(path))
    reader.UpdateInformation()
    return reader


def read_timesteps(path: str | Path) -> list[float]:
    """Return the internal VTK timesteps of a file, reading metadata only.

    These are *run* crank angles, on the same scale as the number in the filename.
    """
    reader = _make_reader(Path(path).expanduser().resolve())
    info = reader.GetOutputInformation(0)
    key = vtk.vtkStreamingDemandDrivenPipeline.TIME_STEPS()
    if info is None or not info.Has(key):
        cad = H5_PATTERN.search(Path(path).name)
        return [float(cad.group(1))] if cad else []
    return [float(t) for t in info.Get(key)]


def read_converge_h5(
    path: str | Path,
    *,
    cad_run: float | None = None,
    echo_path: bool = False,
) -> pv.MultiBlock:
    """Read a CONVERGE ``post*.h5`` into a MultiBlock of named blocks.

    One file can hold **many** VTK timesteps, whose times are *run* CAD (the same
    scale as the number in the filename), not cycle-local CAD.  A plain
    ``Update()`` would silently load only the first step, so:

    * ``cad_run`` given -> snap to the nearest internal step.  If it falls outside
      the file's time range (usually because cycle-local CAD was passed by
      mistake) fall back to the CAD parsed from the filename, and raise if that
      does not fit either.
    * ``cad_run`` omitted -> use the **last** internal step.

    Blocks that share a VTK ``NAME`` (a volume region and its boundary surface, or
    a surface chunked across MPI ranks) would overwrite each other in a MultiBlock,
    so colliding names are suffixed ``name#0``, ``name#1``, ...
    """
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"expected a single existing .h5 file, got: {path}")

    if cad_run is not None and not np.isfinite(float(cad_run)):
        cad_run = None

    reader = _make_reader(path)
    times = np.asarray(read_timesteps(path), dtype=float)

    if times.size == 0:
        reader.Update()
    else:
        target = _choose_time(times, cad_run, path)
        if echo_path:
            print(f"[read_converge_h5] {path.name}: UpdateTimeStep({target:.12g})", flush=True)
        reader.UpdateTimeStep(target)

    return _wrap_partitions(reader.GetOutput())


def _choose_time(times: np.ndarray, cad_run: float | None, path: Path) -> float:
    """Pick the timestep to load, defaulting to the one this file actually holds."""
    lo, hi = float(times[0]), float(times[-1])

    if cad_run is None:
        # vtkCONVERGECFDReader globs the *directory* as a time series, so the steps
        # it offers include the neighbouring post*.h5 files. Reading "the last step"
        # would silently return another file's data; the CAD in this file's name is
        # the one the caller asked for.
        match = H5_PATTERN.search(path.name)
        if match is not None:
            return float(times[int(np.argmin(np.abs(times - float(match.group(1)))))])
        return float(times[-1])

    requested = float(cad_run)
    if not (lo - 1e-6 <= requested <= hi + 1e-6):
        match = H5_PATTERN.search(path.name)
        fallback = float(match.group(1)) if match else None
        if fallback is None or not (lo - 1e-6 <= fallback <= hi + 1e-6):
            raise ValueError(
                f"cad_run={requested} is outside the VTK time range [{lo}, {hi}] for "
                f"{path.name}; pass run CAD (index column 'cad_run'), not cad_local"
            )
        requested = fallback

    return float(times[int(np.argmin(np.abs(times - requested)))])


def _wrap_partitions(pdc: vtk.vtkPartitionedDataSetCollection) -> pv.MultiBlock:
    """Turn the reader's partitioned dataset collection into a named MultiBlock."""
    name_key = vtk.vtkCompositeDataSet.NAME()

    raw: list[tuple[str, pv.DataSet]] = []
    for i in range(pdc.GetNumberOfPartitionedDataSets()):
        meta = pdc.GetMetaData(i)
        base = meta.Get(name_key) if meta is not None and meta.Has(name_key) else f"block_{i}"
        for j in range(pdc.GetNumberOfPartitions(i)):
            part = pdc.GetPartition(i, j)
            if part is None or part.GetNumberOfPoints() == 0:
                continue
            raw.append((base, pv.wrap(part)))

    if not raw:
        raise RuntimeError("no mesh partitions found in the CONVERGE file")

    counts: dict[str, int] = {}
    for name, _ in raw:
        counts[name] = counts.get(name, 0) + 1

    seen: dict[str, int] = {}
    blocks: dict[str, pv.DataSet] = {}
    for name, dataset in raw:
        if counts[name] == 1:
            blocks[name] = dataset
        else:
            k = seen.get(name, 0)
            seen[name] = k + 1
            blocks[f"{name}#{k}"] = dataset

    return pv.MultiBlock(blocks)


class VTKConvergeBackend:
    """Mesh backend built on VTK's native CONVERGE reader."""

    name: ClassVar[str] = "vtk"
    requires: ClassVar[tuple[str, ...]] = ("vtk", "pyvista")

    def read(
        self,
        path: Path,
        *,
        cad_run: float | None = None,
        echo_path: bool = False,
    ) -> pv.MultiBlock:
        """Read the file into a named MultiBlock."""
        return read_converge_h5(path, cad_run=cad_run, echo_path=echo_path)


def filter_by_region(mesh: pv.DataSet, region_id: int = 1, *, tol: float = 0.1) -> pv.DataSet:
    """Keep only the cells of one CONVERGE region (region 1 is the cylinder)."""
    if "REGION_ID" in mesh.cell_data:
        scalars = "REGION_ID"
    elif "IDREG" in mesh.cell_data:
        warnings.warn(
            "REGION_ID is absent; falling back to the legacy IDREG array",
            DeprecationWarning,
            stacklevel=2,
        )
        scalars = "IDREG"
    else:
        raise KeyError("neither REGION_ID nor IDREG is present in the cell data")

    return mesh.threshold(
        value=(region_id - tol, region_id + tol),
        scalars=scalars,
        preference="cell",
    )


def calculate_enstrophy(mesh: pv.DataSet) -> None:
    """Attach the cell field ``enstrophy`` = 1/2 |omega|^2, computed from ``VORTICITY``."""
    vorticity = np.asarray(mesh.cell_data["VORTICITY"])
    if vorticity.ndim == 2:
        mesh.cell_data["enstrophy"] = 0.5 * np.sum(vorticity * vorticity, axis=1)
    else:
        mesh.cell_data["enstrophy"] = 0.5 * np.squeeze(vorticity) ** 2


def get_z_piston(mesh: pv.MultiBlock, piston_id: int = 24) -> float:
    """Return the z of the piston crown (the lowest z of the piston block)."""
    block = mesh.get_block(piston_id)
    if block is None:
        raise KeyError(f"block {piston_id} is empty in this MultiBlock")
    return float(block.bounds.z_min)


def prepare_cylinder_mesh(
    multiblock: pv.MultiBlock,
    *,
    region_id: int | None = 1,
    region_tol: float = 0.1,
) -> pv.DataSet:
    """Combine the volumetric blocks into one grid and keep the requested region.

    The named CONVERGE boundary surfaces come back as ``PolyData`` and the volume
    mesh as ``UnstructuredGrid``, so surfaces are dropped by type and then by
    cell dimensionality.
    """
    volumetric: list[pv.UnstructuredGrid] = []
    for block in multiblock:
        if block is None or not isinstance(block, pv.UnstructuredGrid) or block.n_cells == 0:
            continue
        types = set(np.unique(block.celltypes).tolist())
        if types & _VOLUMETRIC_CELL_TYPES:
            volumetric.append(block)

    if not volumetric:
        raise RuntimeError("no volumetric (3D) blocks found in the MultiBlock")

    combined = volumetric[0] if len(volumetric) == 1 else pv.MultiBlock(volumetric).combine()
    if region_id is None:
        return combined
    return filter_by_region(combined, region_id=region_id, tol=region_tol)
