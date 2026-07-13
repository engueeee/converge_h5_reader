"""Slice, clip and resample a CONVERGE mesh, then write it out as ``.vti``.

Everything here operates on PyVista datasets, so it lives behind the ``[mesh]``
extra alongside the backends.  The usual pipeline is:

    mesh  = read_mesh(path)                     # -> MultiBlock
    vol   = to_volume(mesh, region_id=1)        # the cylinder region only
    plane = tumble_slice(vol)                   # cut normal to Y
    image = sample_to_image(plane, spacing=50e-6)
    save_vti(image, "tumble.vti")

Sampling is what turns CONVERGE's polyhedral cut cells into the uniform grid a
``.vti`` requires: the cells themselves are not axis-aligned, so an ImageData is
built over the bounds and interpolated from the source mesh.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pyvista as pv

from .vtk_reader import filter_by_region, prepare_cylinder_mesh

__all__ = [
    "axis_slice",
    "clip_box",
    "clip_cylinder",
    "clip_sphere",
    "clip_surface",
    "drop_below_z",
    "image_grid",
    "injector_slice",
    "piston_crown_z",
    "sample_to_image",
    "save_vti",
    "slice_plane",
    "tetrahedralize",
    "to_volume",
    "tumble_slice",
]

Vector = Sequence[float]
Bounds = Sequence[float]
#: Default gap below which a sample point is still considered inside a cell.
SAMPLE_TOLERANCE = 100e-6
#: The boundary whose surface is the piston crown; everything below it is crevice.
PISTON_CROWN_BOUNDARY = "PISTONHEAD"


# --------------------------------------------------------------------------- #
# Volume extraction
# --------------------------------------------------------------------------- #


def piston_crown_z(mesh: pv.MultiBlock, *, boundary: str = PISTON_CROWN_BOUNDARY) -> float:
    """Return the z of the piston crown, from the ``PISTONHEAD`` boundary surface.

    The *lowest* point of the crown surface is used, so a bowl or a dome is kept
    whole; only what lies below the piston is crevice.
    """
    # MultiBlock.keys() is the block-name list, not a mapping's keys.
    block_names = list(mesh.keys())

    crowns = []
    for name in block_names:
        # A boundary split across MPI ranks comes back as "NAME#0", "NAME#1", ...
        if name.split("#")[0] != boundary:
            continue
        block = mesh[name]
        if block is not None:
            crowns.append(block)

    if not crowns:
        raise KeyError(
            f"no {boundary!r} block in this mesh, so the piston crown cannot be located; "
            f"pass z_piston explicitly. Blocks: {sorted({n.split('#')[0] for n in block_names})}"
        )
    return min(float(block.bounds[4]) for block in crowns)


def drop_below_z(mesh: pv.DataSet, z: float) -> pv.DataSet:
    """Drop the cells whose **centre** lies below ``z``.

    The centre is where CONVERGE stores the cell value, so this is what "filter out
    the data below z" means.  Cells are kept or dropped whole -- nothing is cut --
    which needs no tetrahedralization and so stays cheap on a multi-million-cell
    mesh.  The consequence is that a tall cell straddling ``z`` is kept in full, and
    the resulting mesh's *geometric* floor can dip somewhat below ``z`` even though
    none of its data does.
    """
    centers = np.asarray(mesh.cell_centers().points)
    keep = np.flatnonzero(centers[:, 2] >= z).astype(np.int64, copy=False)
    if keep.size == 0:
        raise RuntimeError(f"no cells lie above z={z}; check z_piston and the mesh units")
    return mesh.extract_cells(keep)  # type: ignore[arg-type]


def to_volume(
    mesh: pv.MultiBlock | pv.DataSet,
    *,
    region_id: int | None = None,
    region_tol: float = 0.1,
    exclude_crevice: bool = True,
    z_piston: float | None = None,
) -> pv.DataSet:
    """Return the 3D fluid volume, optionally restricted to one CONVERGE region.

    ``region_id=None`` keeps the full domain (cylinder + ports + pipes);
    ``region_id=1`` is the cylinder.  Boundary surfaces (``PolyData``) are dropped;
    only volumetric blocks survive.

    The cylinder region also contains the **crevice**: the thin annular gap between
    the piston and the liner, which runs far below the piston crown (~10 cm on a
    typical case) and is full of cold, stagnant gas that skews any average or plot.
    ``exclude_crevice=True`` (the default) drops every cell whose centre lies below
    the crown.  ``z_piston`` overrides the crown position; omitted, it is read from
    the ``PISTONHEAD`` boundary, which requires ``mesh`` to be the full MultiBlock.
    """
    if isinstance(mesh, pv.MultiBlock):
        volume = prepare_cylinder_mesh(mesh, region_id=region_id, region_tol=region_tol)
    elif region_id is None:
        volume = mesh
    else:
        volume = filter_by_region(mesh, region_id=region_id, tol=region_tol)

    if not exclude_crevice:
        return volume

    if z_piston is None:
        if not isinstance(mesh, pv.MultiBlock):
            raise ValueError(
                "z_piston is needed to exclude the crevice from a plain dataset; pass it, "
                "or pass the full MultiBlock so the PISTONHEAD boundary can be used, "
                "or set exclude_crevice=False"
            )
        z_piston = piston_crown_z(mesh)

    return drop_below_z(volume, z_piston)


def tetrahedralize(mesh: pv.DataSet) -> pv.DataSet:
    """Split polyhedral cells into tetrahedra, which VTK's clip filters require.

    CONVERGE writes cut cells as ``VTK_POLYHEDRON``.  VTK's clippers do not handle
    that type -- they return an *empty* mesh rather than failing -- so any clip has
    to go through a tetrahedral mesh.  Cell data is inherited by each tetrahedron,
    and the total volume is preserved.  Slicing needs none of this.
    """
    if pv.CellType.POLYHEDRON in set(np.unique(mesh.celltypes).tolist()):
        return mesh.triangulate()
    return mesh


def clip_box(mesh: pv.DataSet, bounds: Bounds, *, invert: bool = False) -> pv.DataSet:
    """Keep the cells inside an axis-aligned box ``(xmin, xmax, ymin, ymax, zmin, zmax)``.

    ``invert=True`` keeps the outside instead.
    """
    if len(bounds) != 6:
        raise ValueError(f"bounds must have 6 values, got {len(bounds)}")
    clipped = tetrahedralize(mesh).clip_box(list(bounds), invert=invert)
    return _check_not_empty(clipped)


def clip_sphere(
    mesh: pv.DataSet,
    center: Vector,
    radius: float,
    *,
    invert: bool = False,
    resolution: int = 60,
) -> pv.DataSet:
    """Keep the cells inside a sphere (``invert=True`` keeps the outside)."""
    sphere = pv.Sphere(
        radius=radius,
        center=tuple(center),
        theta_resolution=resolution,
        phi_resolution=resolution,
    )
    return clip_surface(mesh, sphere, invert=invert)


def clip_cylinder(
    mesh: pv.DataSet,
    center: Vector,
    direction: Vector,
    radius: float,
    height: float,
    *,
    invert: bool = False,
    resolution: int = 120,
) -> pv.DataSet:
    """Keep the cells inside a finite cylinder -- e.g. the bore, or an injector plume."""
    cylinder = pv.Cylinder(
        center=tuple(center),
        direction=tuple(direction),
        radius=radius,
        height=height,
        resolution=resolution,
        capping=True,
    ).triangulate()
    return clip_surface(mesh, cylinder, invert=invert)


def clip_surface(mesh: pv.DataSet, surface: pv.PolyData, *, invert: bool = False) -> pv.DataSet:
    """Keep the cells inside an arbitrary closed surface (any STL you can load).

    The surface must be closed and its units must match the mesh (CONVERGE writes
    metres).  A surface finer than the mesh can slip between vertices and clip away
    everything -- that is what an empty result usually means.
    """
    # PyVista keeps the *negative* side (the inside) when invert=True.
    clipped = tetrahedralize(mesh).clip_surface(surface, invert=not invert)
    return _check_not_empty(clipped)


def _check_not_empty(mesh: pv.DataSet) -> pv.DataSet:
    if mesh.n_cells == 0:
        raise RuntimeError(
            "the clip produced an empty mesh; check the geometry, its units, and that it "
            "is not finer than the mesh it clips"
        )
    return mesh


# --------------------------------------------------------------------------- #
# Slices
# --------------------------------------------------------------------------- #


def slice_plane(
    mesh: pv.DataSet,
    *,
    normal: Vector,
    origin: Vector = (0.0, 0.0, 0.0),
    keep_largest: bool = False,
    generate_triangles: bool = True,
) -> pv.PolyData:
    """Cut the mesh with a plane, preserving cell connectivity.

    ``keep_largest=True`` keeps only the largest connected piece, which is how you
    drop the ports and crevices that a cylinder cut also intersects.
    """
    cut = mesh.slice(
        normal=tuple(normal),
        origin=tuple(origin),
        generate_triangles=generate_triangles,
    )
    if cut.n_cells == 0:
        raise RuntimeError(f"empty slice: plane normal={tuple(normal)} origin={tuple(origin)}")
    if keep_largest:
        cut = cut.connectivity("largest")
    return cut


def tumble_slice(mesh: pv.DataSet, y: float = 0.0, **kwargs: object) -> pv.PolyData:
    """Cut the tumble plane: the plane normal to **Y**, at ``y``."""
    return slice_plane(mesh, normal=(0.0, 1.0, 0.0), origin=(0.0, y, 0.0), **kwargs)  # type: ignore[arg-type]


def axis_slice(
    mesh: pv.DataSet, axis: str = "z", position: float = 0.0, **kwargs: object
) -> pv.PolyData:
    """Cut the plane normal to a coordinate axis (``"x"``, ``"y"`` or ``"z"``)."""
    normals = {"x": (1.0, 0.0, 0.0), "y": (0.0, 1.0, 0.0), "z": (0.0, 0.0, 1.0)}
    if axis not in normals:
        raise ValueError(f"axis must be one of {sorted(normals)}, got {axis!r}")
    normal = normals[axis]
    origin = tuple(position * c for c in normal)
    return slice_plane(mesh, normal=normal, origin=origin, **kwargs)  # type: ignore[arg-type]


def injector_slice(
    mesh: pv.DataSet,
    *,
    origin: Vector,
    axis: Vector = (0.0, 0.0, 1.0),
    normal: Vector | None = None,
    **kwargs: object,
) -> pv.PolyData:
    """Cut a plane that **contains** the injector axis.

    ``axis`` is the injector direction and ``origin`` a point on it (the nozzle).
    The cutting plane is perpendicular to ``normal``, which must itself be
    perpendicular to ``axis``; omitted, an arbitrary perpendicular is chosen, which
    is enough when you only want *a* plane through the spray.
    """
    direction = np.asarray(axis, dtype=float)
    if not np.any(direction):
        raise ValueError("axis must be a non-zero vector")
    direction /= np.linalg.norm(direction)

    if normal is None:
        # Any vector not parallel to the axis gives a valid perpendicular.
        seed = np.array([1.0, 0.0, 0.0])
        if abs(float(direction @ seed)) > 0.9:
            seed = np.array([0.0, 1.0, 0.0])
        plane_normal = np.cross(direction, seed)
    else:
        plane_normal = np.asarray(normal, dtype=float)
        if abs(float(plane_normal @ direction)) > 1e-6:
            raise ValueError("normal must be perpendicular to axis for the plane to contain it")

    plane_normal /= np.linalg.norm(plane_normal)
    return slice_plane(mesh, normal=plane_normal, origin=origin, **kwargs)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Resampling onto a uniform grid, and .vti output
# --------------------------------------------------------------------------- #


def image_grid(
    bounds: Bounds,
    spacing: float | Vector = 50e-6,
) -> pv.ImageData:
    """Build a uniform grid spanning ``bounds`` at ``spacing``.

    An axis whose extent is zero (a plane) collapses to a single point, which is how
    a tumble-plane cut becomes a 2D ``.vti``.
    """
    if len(bounds) != 6:
        raise ValueError(f"bounds must have 6 values, got {len(bounds)}")

    steps: tuple[float, ...]
    if isinstance(spacing, (int, float)):
        steps = (float(spacing),) * 3
    else:
        steps = tuple(float(s) for s in spacing)
    if len(steps) != 3 or any(s <= 0 for s in steps):
        raise ValueError(f"spacing must be 3 positive values, got {steps}")

    origin = (float(bounds[0]), float(bounds[2]), float(bounds[4]))
    lengths = (
        float(bounds[1]) - origin[0],
        float(bounds[3]) - origin[1],
        float(bounds[5]) - origin[2],
    )
    if any(length < 0 for length in lengths):
        raise ValueError(f"bounds are inverted: {tuple(bounds)}")

    dimensions = tuple(
        max(1, round(length / step) + 1) for length, step in zip(lengths, steps)
    )
    return pv.ImageData(origin=origin, spacing=steps, dimensions=dimensions)


def sample_to_image(
    source: pv.DataSet,
    *,
    bounds: Bounds | None = None,
    spacing: float | Vector = 50e-6,
    fields: Sequence[str] | None = None,
    tolerance: float = SAMPLE_TOLERANCE,
    z_piston: float | None = None,
) -> pv.ImageData:
    """Resample a mesh or slice onto a uniform grid, ready to be written as ``.vti``.

    ``bounds`` defaults to the source's own bounds.  Points that fall outside the
    mesh are flagged by the ``vtkValidPointMask`` array that VTK adds.
    ``fields`` keeps only the named arrays (plus the mask).

    ``z_piston`` raises the floor of the grid to the piston crown.  You only need it
    when sampling a source whose crevice was *not* already removed
    (:func:`to_volume` does that by default) -- otherwise the grid would waste most
    of its rows on the empty column below the piston.
    """
    grid_bounds = list(source.bounds if bounds is None else bounds)
    if z_piston is not None:
        grid_bounds[4] = max(float(grid_bounds[4]), float(z_piston))
        if grid_bounds[4] > grid_bounds[5]:
            raise ValueError(
                f"z_piston={z_piston} is above the source's top (z_max={grid_bounds[5]})"
            )

    grid = image_grid(grid_bounds, spacing)
    sampled = grid.sample(
        source,
        tolerance=tolerance,
        pass_cell_data=True,
        pass_point_data=True,
    )

    if fields is not None:
        keep = set(fields) | {"vtkValidPointMask"}
        for name in list(sampled.point_data.keys()):
            if name not in keep:
                sampled.point_data.remove(name)

    return sampled


def save_vti(image: pv.ImageData, path: str | Path) -> Path:
    """Write a uniform grid to a ``.vti`` file."""
    target = Path(path)
    if target.suffix != ".vti":
        target = target.with_suffix(".vti")
    target.parent.mkdir(parents=True, exist_ok=True)
    image.save(target)
    return target
