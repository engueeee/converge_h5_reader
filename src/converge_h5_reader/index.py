"""Multi-file discovery and crank-angle indexing across runs.

Ported from ``converge_post/configBuilder.py``, with one deliberate change: the
index is built from filenames alone and needs no VTK.  Scanning the internal VTK
timesteps of each file is opt-in (``scan_timesteps=True``) and then genuinely
requires the ``[mesh]`` extra instead of silently yielding nothing.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from .naming import cad_from_filename

if TYPE_CHECKING:
    import pandas as pd

__all__ = [
    "INDEX_COLUMNS",
    "RunConfig",
    "build_index",
    "find_converge_h5_files",
    "select_at_cad",
]

INDEX_COLUMNS = ["run", "root", "path", "cad_run", "cycle", "cad_local"]


@dataclass(frozen=True)
class RunConfig:
    """One CONVERGE run: where its outputs live and how run-CAD maps onto cycles."""

    name: str
    root: Path | str | Sequence[Path | str]
    start_cycle: int = 1
    n_cycles: int = 1
    cad_per_cycle: float = 720.0
    cad_cycle_start: float = 0.0
    cad_run_offset: float = 0.0
    _roots: tuple[Path, ...] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        raw = self.root
        roots = (raw,) if isinstance(raw, (str, Path)) else tuple(raw)
        object.__setattr__(self, "_roots", tuple(Path(r) for r in roots))

    @property
    def roots(self) -> tuple[Path, ...]:
        """The output directories of this run."""
        return self._roots

    @property
    def cad_run_min(self) -> float:
        """First run-CAD covered by this run."""
        return self.cad_run_offset + self.cad_cycle_start

    @property
    def cad_run_max(self) -> float:
        """Last run-CAD covered by this run."""
        return self.cad_run_min + self.n_cycles * self.cad_per_cycle

    def cycle_bounds(self, absolute_cycle: int) -> tuple[float, float]:
        """Run-CAD span ``[start, end)`` of one absolute cycle number."""
        offset = absolute_cycle - self.start_cycle
        start = self.cad_run_min + offset * self.cad_per_cycle
        return start, start + self.cad_per_cycle

    def cad_run_to_local(self, cad_run: float) -> tuple[int, float]:
        """Map a run-CAD onto ``(absolute_cycle, cad_local)``.

        ``cad_local`` lies in ``[cad_cycle_start, cad_cycle_start + cad_per_cycle)``.
        """
        elapsed = cad_run - self.cad_run_min
        index = int(elapsed // self.cad_per_cycle)
        cad_local = self.cad_cycle_start + (elapsed - index * self.cad_per_cycle)
        return self.start_cycle + index, cad_local


def find_converge_h5_files(
    root: Path | str,
    target_cad: float | None = None,
    tol: float = 1e-3,
    *,
    recursive: bool = False,
) -> list[tuple[float, Path]]:
    """Find ``post*.h5`` files under ``root``, sorted by the CAD in their filename.

    With ``target_cad`` set, keep only files within ``tol`` of it.
    """
    directory = Path(root)
    paths = directory.rglob("*.h5") if recursive else directory.glob("*.h5")

    found: list[tuple[float, Path]] = []
    for path in paths:
        cad = cad_from_filename(path, strict=False)
        if cad is None:
            continue
        if target_cad is not None and abs(cad - target_cad) > tol:
            continue
        found.append((cad, path))
    return sorted(found, key=lambda item: (item[0], str(item[1])))


def build_index(
    runs: Sequence[RunConfig],
    *,
    recursive: bool = False,
    scan_timesteps: bool = False,
    progress: bool = False,
) -> pd.DataFrame:
    """Index every ``post*.h5`` of every run into a DataFrame.

    Columns: ``run, root, path, cad_run, cycle, cad_local``.

    ``scan_timesteps=True`` opens each file with the VTK CONVERGE reader to emit
    one row per internal timestep (a single ``.h5`` can hold several).  That needs
    the ``[mesh]`` extra.
    """
    import pandas as pd

    iterator: Sequence[RunConfig] = runs
    if progress:
        from .progress import wrap_progress

        iterator = wrap_progress(runs, desc="indexing runs")

    rows: list[dict[str, object]] = []
    for run in iterator:
        for root in run.roots:
            for cad_file, path in find_converge_h5_files(root, recursive=recursive):
                cads = _timesteps(path) if scan_timesteps else [cad_file]
                for cad_run in cads:
                    cycle, cad_local = run.cad_run_to_local(cad_run)
                    rows.append(
                        {
                            "run": run.name,
                            "root": str(root),
                            "path": str(path),
                            "cad_run": float(cad_run),
                            "cycle": int(cycle),
                            "cad_local": float(cad_local),
                        }
                    )

    frame = pd.DataFrame(rows, columns=INDEX_COLUMNS)
    return frame.sort_values(["run", "cad_run"]).reset_index(drop=True)


def _timesteps(path: Path) -> list[float]:
    """Read the internal VTK timesteps of a file (metadata only). Needs the mesh extra."""
    from .mesh.vtk_reader import read_timesteps

    return read_timesteps(path)


def select_at_cad(
    index: pd.DataFrame,
    target_cad: float,
    *,
    tol: float = 1e-2,
    cycles: Sequence[int] | None = None,
    runs: Sequence[str] | None = None,
    nearest: bool = False,
) -> pd.DataFrame:
    """Select the indexed snapshots at one local crank angle.

    With ``nearest=True`` the closest snapshot of each ``(run, cycle)`` is kept
    regardless of ``tol``; otherwise every snapshot within ``tol`` is returned.
    """
    frame = index
    if runs is not None:
        frame = frame[frame["run"].isin(list(runs))]
    if cycles is not None:
        frame = frame[frame["cycle"].isin(list(cycles))]
    if frame.empty:
        return frame.copy()

    delta = (frame["cad_local"] - target_cad).abs()
    if not nearest:
        return frame[delta <= tol].copy()

    frame = frame.assign(_delta=delta)
    picked = frame.loc[frame.groupby(["run", "cycle"])["_delta"].idxmin()]
    return picked.drop(columns="_delta").reset_index(drop=True)
