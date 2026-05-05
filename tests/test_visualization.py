"""Tests for surface_code.visualization (matplotlib + plotly).

These tests are skipped if matplotlib / plotly aren't installed (the optional
``[viz]`` extra). On a system with the extra installed, they verify that each
plotting helper runs without raising.
"""

from __future__ import annotations

import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")  # headless backend for CI
plotly = pytest.importorskip("plotly")

from ft_primitive_bench.surface_code.circuits import rectangular_surface_code_patch  # noqa: E402
from ft_primitive_bench.surface_code.visualization import (  # noqa: E402
    plot_lattice_surgery_timeline,
    plot_memory_timeline,
    plot_patch,
    plot_s_gate_timeline,
    plot_transversal_h_timeline,
)


def test_plot_patch_runs():
    patch = rectangular_surface_code_patch(x_distance=3, z_distance=3)
    fig, ax = plot_patch(patch, show=False)
    assert fig is not None
    assert ax is not None


def test_plot_memory_timeline_runs():
    fig = plot_memory_timeline(x_distance=3, z_distance=3, rounds=2, meas_basis="Z")
    assert fig is not None


def test_plot_transversal_h_timeline_runs():
    fig = plot_transversal_h_timeline(
        x_distance=3, z_distance=3, pre_rounds=1, post_rounds=1, meas_basis="Z"
    )
    assert fig is not None


def test_plot_lattice_surgery_timeline_runs():
    fig = plot_lattice_surgery_timeline(
        x_distance=3,
        z_distance=3,
        bridge_length=1,
        pre_rounds=1,
        merge_rounds=1,
        post_rounds=1,
    )
    assert fig is not None


def test_plot_s_gate_timeline_runs():
    fig = plot_s_gate_timeline(
        distance=3,
        bridge_length=1,
        pre_rounds=1,
        merge_rounds=1,
        boundary_rounds=1,
    )
    assert fig is not None
