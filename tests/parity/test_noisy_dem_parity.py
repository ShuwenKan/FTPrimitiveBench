"""Noisy-DEM parity: chunks vs legacy produce equivalent NOISY circuits.

The noiseless DEM checks only structural equivalence (detector and observable
counts, coordinates). It misses TICK/SHIFT_COORDS placement bugs that don't
affect the noiseless DEM but DO change noise channel insertion. These tests
add the project's noise model to both backends and assert the noisy DEMs match
under time-coordinate canonicalization.
"""

from __future__ import annotations

import pytest

from ft_primitive_bench.noise_models import uniform_depolarizing
from ft_primitive_bench.surface_code.circuits import lattice_surgery, memory, transversal_h

from ._harness import _coord_canonical_dem

_NM = uniform_depolarizing(p=1e-3)


def _assert_noisy_dem_equivalent(legacy, chunks) -> None:
    legacy_noisy = _NM.noisy_circuit(legacy)
    chunks_noisy = _NM.noisy_circuit(chunks)
    a, _, _ = _coord_canonical_dem(legacy_noisy)
    b, _, _ = _coord_canonical_dem(chunks_noisy)
    if a != b:
        raise AssertionError(
            f"Noisy DEM error multisets differ.\n"
            f"  only in legacy: {sum((a - b).values())} errors\n"
            f"  only in chunks: {sum((b - a).values())} errors"
        )


@pytest.mark.parametrize("d", [3, 5])
@pytest.mark.parametrize("rounds", [1, 3])
@pytest.mark.parametrize("meas_basis", ["Z", "X"])
def test_memory_noisy_dem(d, rounds, meas_basis):
    legacy = memory(d, d, rounds, meas_basis=meas_basis, backend="legacy")
    chunks = memory(d, d, rounds, meas_basis=meas_basis, backend="stimflow")
    _assert_noisy_dem_equivalent(legacy, chunks)


@pytest.mark.parametrize("d", [3, 5])
@pytest.mark.parametrize("pre,post", [(0, 0), (1, 1), (2, 3)])
@pytest.mark.parametrize("meas_basis", ["Z", "X"])
def test_transversal_h_noisy_dem(d, pre, post, meas_basis):
    legacy = transversal_h(d, d, pre, post, meas_basis=meas_basis, backend="legacy")
    chunks = transversal_h(d, d, pre, post, meas_basis=meas_basis, backend="stimflow")
    _assert_noisy_dem_equivalent(legacy, chunks)


@pytest.mark.parametrize("d", [3, 5])
@pytest.mark.parametrize("pre,merge,post", [(0, 1, 0), (1, 2, 1), (2, 3, 2)])
@pytest.mark.parametrize("meas_basis", ["Z", "X"])
def test_lattice_surgery_noisy_dem(d, pre, merge, post, meas_basis):
    legacy = lattice_surgery(d, d, 1, pre, merge, post, meas_basis=meas_basis, backend="legacy")
    chunks = lattice_surgery(d, d, 1, pre, merge, post, meas_basis=meas_basis, backend="stimflow")
    _assert_noisy_dem_equivalent(legacy, chunks)
