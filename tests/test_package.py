"""Smoke tests for top-level package imports."""
from __future__ import annotations


def test_top_level_import():
    import ft_primitive_bench

    assert hasattr(ft_primitive_bench, "surface_code")
    assert hasattr(ft_primitive_bench, "noise_models")
    assert hasattr(ft_primitive_bench.surface_code, "circuits")
    assert hasattr(ft_primitive_bench.surface_code, "visualization")


def test_circuits_public_api():
    from ft_primitive_bench.surface_code.circuits import (
        boundary_tiles,
        build_merged_patch,
        checkerboard_weight4_patch,
        create_individual_patch,
        lattice_surgery,
        memory,
        patch_midlines,
        rectangular_surface_code_patch,
        s_gate,
        transversal_h,
    )

    for fn in (
        boundary_tiles,
        build_merged_patch,
        checkerboard_weight4_patch,
        create_individual_patch,
        lattice_surgery,
        memory,
        patch_midlines,
        rectangular_surface_code_patch,
        s_gate,
        transversal_h,
    ):
        assert callable(fn)


def test_noise_models_public_api():
    from ft_primitive_bench.noise_models import (
        Coherence,
        CompiledCircuit,
        ConfiguredNoiseModel,
        NoiseModel,
        NoiseModelConfig,
        NoiseProfile,
        SampledFactorSnapshot,
        idle_pauli_channel_from_T1T2,
        infer_moment_rounds,
        measurement_biased,
        noise_model,
        nonuniform,
        pauli_biased,
        strip_noise_channels,
        uniform_depolarizing,
    )

    assert NoiseProfile is not None
    assert callable(noise_model)
    assert callable(uniform_depolarizing)
    assert callable(pauli_biased)
    assert callable(measurement_biased)
    assert callable(nonuniform)


def test_visualization_public_api():
    from ft_primitive_bench.surface_code.visualization import (
        plot_lattice_surgery_timeline,
        plot_memory_timeline,
        plot_patch,
        plot_patch_timeline,
        plot_s_gate_timeline,
        plot_transversal_h_timeline,
    )

    for fn in (
        plot_patch,
        plot_patch_timeline,
        plot_memory_timeline,
        plot_transversal_h_timeline,
        plot_lattice_surgery_timeline,
        plot_s_gate_timeline,
    ):
        assert callable(fn)
