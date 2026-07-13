"""Synthetic CONVERGE H5 files mirroring the real schema, so tests need no 2 GB dataset."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import h5py
import numpy as np
import pytest

# Deliberately NOT in sorted order: the reader must honour VARIABLE_NAMES, not
# sorted(CELL_CENTER_DATA).
CELL_VARS = [
    "TEMPERATURE",
    "PRESSURE",
    "DENSITY",
    "VELOCITY_X",
    "MASSFRAC_H2",
    "REGION_ID",
    "XCEN_X",
]

#: STREAM_00 has 8 cells; 4 of them are in region 1.
REGION_IDS = np.array([1, 1, 1, 1, 2, 2, 0, 0], dtype=np.float32)

ROOT_ATTRS: dict[str, np.ndarray] = {
    "BLM_FLAG": np.array([0], dtype=np.int32),
    "CRANK_ANGLE": np.array([-74.07573604]),
    "CRANK_FLAG": np.array([1], dtype=np.int32),
    "OUTPUT_TIME": np.array([-74.07573604]),
    "OUTPUT_TIME_SEC": np.array([-0.01543245]),
    "RPM": np.array([800.0]),
    "VERSION_FLAG": np.array([4], dtype=np.int32),
    "VERSION_NUM1": np.array([5], dtype=np.int32),
    "VERSION_NUM2": np.array([1], dtype=np.int32),
    "VERSION_NUM3": np.array([1], dtype=np.int32),
}


def _cell_values(name: str, n_cells: int) -> np.ndarray:
    """Deterministic, distinguishable values for each variable."""
    base = np.arange(n_cells, dtype=np.float32)
    if name in ("REGION_ID", "IDREG"):
        return REGION_IDS[:n_cells].copy()
    if name == "TEMPERATURE":
        return 300.0 + 100.0 * base
    if name == "PRESSURE":
        return 1.0e5 + 1.0e3 * base
    if name == "DENSITY":
        return 1.2 - 0.01 * base
    return base.copy()


#: STREAM_00's mesh: a 1x1xN column of unit hexahedra, one per cell.
N_MESH_CELLS = 8
N_MESH_VERTICES = 4 * (N_MESH_CELLS + 1)
#: N + 1 horizontal quads (the caps and the separators) plus 4 sides per cell.
N_MESH_POLYGONS = (N_MESH_CELLS + 1) + 4 * N_MESH_CELLS


def _column_vertices() -> np.ndarray:
    """4 vertices per level, ``N_MESH_CELLS + 1`` levels stacked along z."""
    corners = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
    return np.array(
        [(x, y, float(level)) for level in range(N_MESH_CELLS + 1) for (x, y) in corners],
        dtype=np.float32,
    )


def _column_connectivity() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Faces of the hex column, with ``CONNECTED_CELLS`` pairs (-1 marks the exterior)."""
    faces: list[list[int]] = []
    connected: list[tuple[int, int]] = []

    # Horizontal quads: the one at level L separates cell L-1 (below) from cell L (above).
    for level in range(N_MESH_CELLS + 1):
        base = 4 * level
        faces.append([base, base + 1, base + 2, base + 3])
        below = level - 1 if level >= 1 else -1
        above = level if level < N_MESH_CELLS else -1
        connected.append((below, above))

    # Four side quads per cell, all on the exterior.
    for cell in range(N_MESH_CELLS):
        lo, hi = 4 * cell, 4 * (cell + 1)
        for i in range(4):
            j = (i + 1) % 4
            faces.append([lo + i, lo + j, hi + j, hi + i])
            connected.append((cell, -1))

    polygon_offset = np.arange(0, 4 * len(faces) + 1, 4, dtype=np.int32)
    polygon_to_vertex = np.asarray(faces, dtype=np.int32).ravel()
    connected_cells = np.asarray(connected, dtype=np.int32).ravel()
    return polygon_offset, polygon_to_vertex, connected_cells


def _make_stream(
    handle: h5py.File,
    name: str,
    *,
    n_cells: int,
    variables: list[str],
    with_mesh: bool,
) -> None:
    group = handle.create_group(name)
    for key, value in ROOT_ATTRS.items():
        group.attrs[key] = value
    group.attrs["CELL_COUNT"] = np.array([n_cells], dtype=np.int32)

    cells = group.create_group("CELL_CENTER_DATA")
    for var in variables:
        cells.create_dataset(var, data=_cell_values(var, n_cells), dtype=np.float32)

    group.create_dataset(
        "VARIABLE_NAMES/CELL_VARIABLES",
        data=np.array([v.encode() for v in variables], dtype="S22"),
    )

    if not with_mesh:
        return

    vertices = _column_vertices()
    for i, axis in enumerate("XYZ"):
        group.create_dataset(f"VERTEX_COORDINATES/{axis}", data=vertices[:, i], dtype=np.float32)

    offset, to_vertex, connected = _column_connectivity()
    group.create_dataset("CONNECTIVITY/POLYGON_OFFSET", data=offset)
    group.create_dataset("CONNECTIVITY/POLYGON_TO_VERTEX", data=to_vertex)
    group.create_dataset("CONNECTIVITY/CONNECTED_CELLS", data=connected)


def _write_boundaries(handle: h5py.File) -> None:
    group = handle.create_group("BOUNDARIES")
    group.create_dataset("BOUNDARY_IDS", data=np.array([4, 14, 9, 28], dtype=np.int32))
    group.create_dataset(
        "BOUNDARY_NAMES",
        data=np.array([b"PISTON", b"LINER", b"CYLINDERHEAD", b"EMPTY_BND"], dtype="S29"),
    )
    group.create_dataset(
        "STREAMS",
        data=np.array([b"Stream_00", b"Stream_00", b"Stream_00", b"Stream_01"], dtype="S10"),
    )
    group.create_dataset("NUM_ELEMENTS", data=np.array([4, 4, 2, 0], dtype=np.int32))
    group.create_dataset("NUM_POINTS", data=np.array([6, 6, 4, 0], dtype=np.int32))
    for axis, values in zip(
        "XYZ",
        ([0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [-1.0, 0.0, 1.0, 0.0]),
    ):
        group.create_dataset(
            f"GEOMETRIC_CENTER_COORDINATE_{axis}", data=np.array(values, dtype=np.float64)
        )


def _write_file(path: Path, *, variables: list[str] | None = None) -> Path:
    variables = variables if variables is not None else CELL_VARS
    with h5py.File(path, "w") as handle:
        for key, value in ROOT_ATTRS.items():
            handle.attrs[key] = value
        _write_boundaries(handle)
        # STREAM_00 carries the 2-hex mesh; STREAM_01 proves multi-stream handling.
        _make_stream(handle, "STREAM_00", n_cells=8, variables=variables, with_mesh=True)
        _make_stream(handle, "STREAM_01", n_cells=3, variables=variables, with_mesh=False)
    return path


@pytest.fixture(scope="session")
def synthetic_h5(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A tiny valid CONVERGE file: 2 streams, 7 cell variables, 4 boundaries."""
    path = tmp_path_factory.mktemp("data") / "post000061_-7.40757e+01.h5"
    return _write_file(path)


@pytest.fixture(scope="session")
def legacy_region_h5(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A file using the legacy IDREG name instead of REGION_ID."""
    path = tmp_path_factory.mktemp("legacy") / "post000001_+0.00000e+00.h5"
    variables = [v if v != "REGION_ID" else "IDREG" for v in CELL_VARS]
    return _write_file(path, variables=variables)


@pytest.fixture
def run_tree(tmp_path: Path, synthetic_h5: Path) -> Path:
    """A directory of three snapshots, for the index tests."""
    root = tmp_path / "runA"
    root.mkdir()
    for name in (
        "post000001_+0.00000e+00.h5",
        "post000061_-7.40757e+01.h5",
        "post000224_+7.00004e+02.h5",
    ):
        shutil.copyfile(synthetic_h5, root / name)
    return root


@pytest.fixture(scope="session")
def real_h5() -> Path:
    """The real CONVERGE file, for tests marked ``realdata``."""
    raw = os.environ.get("CONVERGE_H5_TEST_FILE")
    if not raw:
        pytest.skip("set CONVERGE_H5_TEST_FILE to run the real-data tests")
    path = Path(raw)
    if not path.is_file():
        pytest.skip(f"CONVERGE_H5_TEST_FILE does not exist: {path}")
    return path
