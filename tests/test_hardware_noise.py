"""Tests for surface_code.noise_models.hardware_noise public surface."""
from __future__ import annotations

import stim

from ft_primitive_bench.surface_code.circuits import memory
from ft_primitive_bench.noise_models import (
    NoiseModel,
    infer_moment_rounds,
    strip_noise_channels,
    uniform_depolarizing,
)


def test_noise_model_is_callable_via_factory():
    m = uniform_depolarizing(p=1e-3)
    assert isinstance(m, NoiseModel)


def test_strip_noise_round_trips_to_clean():
    clean = memory(x_distance=3, z_distance=3, rounds=2, meas_basis="Z")
    noisy = uniform_depolarizing(p=1e-3).noisy_circuit(clean)
    stripped = strip_noise_channels(noisy)
    assert str(stripped) == str(clean)


def test_strip_noise_idempotent():
    clean = memory(x_distance=3, z_distance=3, rounds=2, meas_basis="Z")
    noisy = uniform_depolarizing(p=1e-3).noisy_circuit(clean)
    stripped_once = strip_noise_channels(noisy)
    stripped_twice = strip_noise_channels(stripped_once)
    assert str(stripped_once) == str(stripped_twice)


def test_strip_noise_no_op_on_clean_circuit():
    clean = memory(x_distance=3, z_distance=3, rounds=2, meas_basis="Z")
    stripped = strip_noise_channels(clean)
    assert str(stripped) == str(clean)


def test_infer_moment_rounds_memory_advances():
    clean = memory(x_distance=3, z_distance=3, rounds=3, meas_basis="Z")
    rounds = infer_moment_rounds(clean.flattened())
    assert max(rounds) >= 3
    assert min(rounds) >= 0


def test_infer_moment_rounds_zero_reset_circuit():
    circuit = stim.Circuit("H 0\nCX 0 1\n")
    rounds = infer_moment_rounds(circuit)
    assert all(r == 0 for r in rounds)
