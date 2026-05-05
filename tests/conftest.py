"""Shared pytest fixtures and helpers."""

from __future__ import annotations

import pytest
import stim


def assert_decodes_clean(circuit: stim.Circuit) -> None:
    """Build the DEM with the standard flags. Raises if Stim cannot decompose."""
    circuit.detector_error_model(decompose_errors=True, approximate_disjoint_errors=True)


@pytest.fixture
def small_distances():
    return {"x_distance": 3, "z_distance": 3}
