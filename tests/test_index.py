from __future__ import annotations

import pytest

from converge_h5_reader import (
    INDEX_COLUMNS,
    RunConfig,
    build_index,
    find_converge_h5_files,
    select_at_cad,
)


def test_find_files_sorted_by_cad(run_tree):
    found = find_converge_h5_files(run_tree)
    assert [round(cad, 3) for cad, _ in found] == [-74.076, 0.0, 700.004]


def test_find_files_at_a_target_cad(run_tree):
    found = find_converge_h5_files(run_tree, target_cad=700.004, tol=1e-2)
    assert len(found) == 1
    assert found[0][1].name == "post000224_+7.00004e+02.h5"


def test_cad_run_to_local():
    run = RunConfig(name="A", root=".", start_cycle=1, n_cycles=3, cad_per_cycle=720.0)
    assert run.cad_run_to_local(0.0) == (1, 0.0)
    assert run.cad_run_to_local(360.0) == (1, 360.0)
    # Exactly on a cycle boundary: the next cycle starts here.
    assert run.cad_run_to_local(720.0) == (2, 0.0)
    assert run.cad_run_to_local(1445.0) == (3, 5.0)
    assert run.cad_run_min == 0.0
    assert run.cad_run_max == 2160.0


def test_cad_run_to_local_with_offsets():
    run = RunConfig(
        name="A", root=".", start_cycle=5, cad_per_cycle=720.0, cad_cycle_start=-360.0
    )
    assert run.cycle_bounds(5) == (-360.0, 360.0)
    cycle, cad_local = run.cad_run_to_local(-360.0)
    assert (cycle, cad_local) == (5, -360.0)


def test_build_index(run_tree):
    run = RunConfig(name="A", root=run_tree, cad_per_cycle=720.0, cad_cycle_start=-360.0)
    index = build_index([run])

    assert list(index.columns) == INDEX_COLUMNS
    assert len(index) == 3
    assert index["run"].unique().tolist() == ["A"]
    assert index["cad_run"].tolist() == pytest.approx([-74.0757, 0.0, 700.004], abs=1e-3)
    # cad_local == cad_run for the first cycle of a run with no offset.
    assert index["cycle"].tolist() == [1, 1, 2]


def test_select_at_cad(run_tree):
    run = RunConfig(name="A", root=run_tree)
    index = build_index([run])

    exact = select_at_cad(index, 0.0, tol=1e-6)
    assert len(exact) == 1

    assert select_at_cad(index, 123.0, tol=1e-2).empty

    nearest = select_at_cad(index, 123.0, nearest=True)
    # One row per (run, cycle); cycle 1 holds -74.08 and 0.0, cycle 2 holds 700.004.
    assert len(nearest) == 2
    assert nearest.loc[nearest["cycle"] == 1, "cad_local"].item() == pytest.approx(0.0)


def test_select_filters_by_run(run_tree):
    index = build_index([RunConfig(name="A", root=run_tree)])
    assert select_at_cad(index, 0.0, tol=1e-6, runs=["B"]).empty
