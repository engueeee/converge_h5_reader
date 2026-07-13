# converge_h5_reader

A modular, lazy reader for CONVERGE CFD HDF5 (`post*.h5`) output.

The core depends only on **h5py, numpy and pandas**. VTK and PyVista are optional and live
behind the `[mesh]` extra — importing the package never imports them, so you can post-process
cell data on a machine with no VTK at all.

## Install

```bash
pip install -e .            # core: h5py + numpy + pandas
pip install -e '.[mesh]'    # adds the vtk / pyvista mesh backends
pip install -e '.[dev]'     # everything, plus pytest / ruff / mypy
```

## Quickstart

```python
from converge_h5_reader import ConvergeFile, Field, region

with ConvergeFile("post000061_-7.40757e+01.h5") as f:
    f.crank_angle          # -74.0757  (from the CRANK_ANGLE root attribute)
    f.rpm, f.version       # 800.0, (5, 1, 1)
    f.stream_names         # ('STREAM_00',)
    f.boundaries["PISTON"] # Boundary(id=4, name='PISTON', n_elements=588880, ...)

    stream = f[0]                    # or f["STREAM_00"]
    stream.n_cells                   # 8060563
    stream.variables                 # ('DENSITY', 'ENTHALPY', ..., 'YPLUS')

    T = stream.cells["T"]            # alias for TEMPERATURE; float32, one HDF5 read

    # Filter before reading the payload: only the selected cells are loaded.
    df = stream.cells.to_dataframe(["T", "P", "u", "v", "w"], where=region(1))

    hot = (Field("REGION_ID") == 1) & (Field("TEMPERATURE") > 1500)
    df = stream.cells.to_dataframe(["T"], where=hot)

    # Bounded memory regardless of how many cells the file holds.
    for chunk in stream.cells.iter_chunks(["T", "P"], chunk=1_000_000):
        ...
```

### Laziness

Opening a file reads nothing. Only the (tiny) root attributes and boundary table are cached;
cell arrays are read on demand and **not** cached — one field of a production stream is 32 MB
and the full set of 36 is 1.16 GB. `where=` is evaluated before any payload field is read, so a
`region(1)` filter costs one field read plus a bool mask.

`to_dataframe()` with no `fields` reads *every* variable. Pass an explicit list unless you mean it.

### Aliases

Short names resolve to CONVERGE dataset names through a registry, so nothing is hardcoded:

| alias | dataset | alias | dataset |
|---|---|---|---|
| `T` | `TEMPERATURE` | `u`, `v`, `w` | `VELOCITY_X/Y/Z` |
| `P` | `PRESSURE` | `x`, `y`, `z` | `XCEN_X/Y/Z` |
| `rho` | `DENSITY` | `Y_H2` | `MASSFRAC_H2` |
| `phi` | `EQUIV_RATIO` | `X_O2` | `MOLEFRAC_O2` |

Unknown names pass through untouched, and a genuinely missing dataset raises `MissingFieldError`
listing what *is* available. Extend or replace the registry per file:

```python
from converge_h5_reader import DEFAULT_ALIASES, AliasRegistry, ConvergeFile

aliases = DEFAULT_ALIASES.with_(HRR="MY_HEAT_RELEASE")
with ConvergeFile(path, aliases=aliases) as f:
    ...

AliasRegistry.from_toml("my_aliases.toml")   # [aliases] and [prefixes] tables
```

## Indexing many files across runs

```python
from converge_h5_reader import RunConfig, build_index, select_at_cad

runs = [RunConfig(name="caseA", root="outputs/", n_cycles=3, cad_per_cycle=720.0)]
index = build_index(runs)        # DataFrame: run, root, path, cad_run, cycle, cad_local
snaps = select_at_cad(index, target_cad=-90.0, tol=0.5)
```

The index is built from filenames (`post000061_-7.40757e+01.h5`) and needs no VTK. Pass
`scan_timesteps=True` to open each file with the VTK reader and emit one row per internal
timestep (a single `.h5` can hold several) — that requires the `[mesh]` extra.

## Meshes (optional)

```python
from converge_h5_reader.mesh import read_mesh

mesh = read_mesh(path, backend="vtk", cad_run=-74.0757)   # -> pv.MultiBlock, named blocks
```

Two backends:

- **`vtk`** (default) — `vtkCONVERGECFDReader`. The C++ reader assembles the polyhedral cut
  cells, the named boundary surfaces (`PISTON`, `LINER`, …) and handles the multiple internal
  timesteps a file can hold. A plain `Update()` would silently load only the *first* step, so
  pass `cad_run` (the `cad_run` column of the index); omitted, the **last** step is used.
- **`h5`** — builds a `VTK_POLYHEDRON` grid straight from `CONNECTIVITY`, bypassing the CONVERGE
  reader. Slower, but it can build a mesh for a *subset* of cells:

  ```python
  grid = read_mesh(path, backend="h5", where=region(1), fields=["TEMPERATURE"])
  ```

Without the extra installed, any of this raises `MissingBackendError` telling you to
`pip install 'converge-h5-reader[mesh]'`.

> **`cad_run` and the directory time series.** `vtkCONVERGECFDReader` globs the *whole directory*
> as a time series, so the timesteps it offers include the neighbouring `post*.h5` files. Reading
> "the last step" would silently hand you another file's mesh. `read_mesh` therefore defaults to
> the CAD in the file's own name; pass `cad_run` (the index column) to override.

## Slicing, clipping and `.vti` export

```python
from converge_h5_reader.mesh import extract, read_mesh

mesh     = read_mesh("post000061_-7.40757e+01.h5")
cylinder = extract.to_volume(mesh, region_id=1)      # region 1; None = full domain

# Slices
plane = extract.tumble_slice(cylinder, y=0.0, keep_largest=True)   # normal to Y
plane = extract.axis_slice(cylinder, axis="z", position=-0.01)
plane = extract.injector_slice(cylinder, origin=(0.0, 0.0, 0.005), axis=(0, 0, 1))

# Volumes: by box, sphere, cylinder, or any closed surface you can load
core = extract.clip_cylinder(cylinder, center=(0, 0, -0.02), direction=(0, 0, 1),
                             radius=0.03, height=0.04)
core = extract.clip_box(cylinder, (-0.04, 0.04, -0.04, 0.04, -0.05, 0.0))
core = extract.clip_surface(cylinder, pv.read("piston_bowl.stl"))

# Resample onto a uniform grid and write it out
image = extract.sample_to_image(plane, spacing=50e-6, fields=["TEMPERATURE", "VELOCITY"])
extract.save_vti(image, "tumble.vti")
```

`sample_to_image` is what makes a `.vti` possible at all: CONVERGE cells are polyhedral cut cells,
so an `ImageData` is built over the bounds and interpolated from the mesh. Points falling outside
the mesh are flagged in `vtkValidPointMask`. An axis with zero extent (a slice) collapses to one
point, giving a 2D `.vti`.

Clipping tetrahedralizes first — VTK's clip filters do not handle `VTK_POLYHEDRON` and return an
empty mesh rather than an error. `extract.tetrahedralize` is exposed if you need it directly.
Slicing needs no such conversion.

## Tests

```bash
pytest                                                      # synthetic fixture; no big file needed
CONVERGE_H5_TEST_FILE=/path/to/post*.h5 pytest -m realdata  # against a real file
```

The suite builds a small valid CONVERGE file in a fixture, so it never needs the multi-GB dataset.

## The file layout this reader expects

```
/                          attrs: CRANK_ANGLE, RPM, OUTPUT_TIME_SEC, VERSION_NUM1/2/3, ...
├── BOUNDARIES/            BOUNDARY_IDS, BOUNDARY_NAMES, STREAMS, NUM_ELEMENTS, NUM_POINTS,
│                          GEOMETRIC_CENTER_COORDINATE_X/Y/Z
└── STREAM_00/             attrs: CELL_COUNT
    ├── CELL_CENTER_DATA/  TEMPERATURE, PRESSURE, VELOCITY_X/Y/Z, MASSFRAC_*, REGION_ID, ...
    ├── VARIABLE_NAMES/    CELL_VARIABLES  (the authoritative variable list)
    ├── VERTEX_COORDINATES/X, Y, Z
    └── CONNECTIVITY/      POLYGON_OFFSET, POLYGON_TO_VERTEX, CONNECTED_CELLS
```
