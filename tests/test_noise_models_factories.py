"""Tests for surface_code.noise_models.profiles factories."""
from __future__ import annotations

import pytest
import stim

from ft_primitive_bench.surface_code.circuits import memory
from ft_primitive_bench.noise_models import (
    Coherence,
    NoiseProfile,
    measurement_biased,
    noise_model,
    nonuniform,
    pauli_biased,
    strip_noise_channels,
    uniform_depolarizing,
)
from tests.conftest import assert_decodes_clean


@pytest.fixture
def clean_memory():
    return memory(x_distance=3, z_distance=3, rounds=2, meas_basis="Z")


# ── noise_model() baseline + custom modes ──────────────────────────────────

def test_noise_model_baseline(clean_memory):
    m = noise_model(p=1e-3)
    noisy = m.noisy_circuit(clean_memory)
    assert isinstance(noisy, stim.Circuit)
    assert_decodes_clean(noisy)


def test_noise_model_per_class_rates(clean_memory):
    m = noise_model(p=1e-3, p_1q=1e-4, p_2q=2e-3, p_meas=5e-3, p_reset=1e-3,
                    p_idle=1e-4, p_idle_meas=1e-3)
    noisy = m.noisy_circuit(clean_memory)
    assert_decodes_clean(noisy)


def test_noise_model_mode_b_with_coherence(clean_memory):
    m = noise_model(
        p=1e-3,
        p_1q=(1e-4, 40e-9), p_2q=(2e-3, 120e-9),
        p_meas=(5e-3, 200e-9), p_reset=(1e-3, 100e-9),
        coherence=(30e-6, 20e-6),
    )
    noisy = m.noisy_circuit(clean_memory)
    assert_decodes_clean(noisy)


def test_noise_model_mode_b_with_coherence_object(clean_memory):
    m = noise_model(
        p=1e-3,
        p_1q=(1e-4, 40e-9), p_2q=(2e-3, 120e-9),
        p_meas=(5e-3, 200e-9), p_reset=(1e-3, 100e-9),
        coherence=Coherence(T1=30e-6, T2=20e-6),
    )
    noisy = m.noisy_circuit(clean_memory)
    assert_decodes_clean(noisy)


def test_noise_model_baseline_p_out_of_range_raises():
    with pytest.raises(ValueError):
        noise_model(p=1.5)


def test_noise_model_mode_a_rejects_tuple():
    with pytest.raises(ValueError):
        noise_model(p=1e-3, p_1q=(1e-4, 40e-9))


def test_noise_model_mode_b_requires_tuple():
    with pytest.raises(ValueError):
        noise_model(p=1e-3, p_1q=1e-4, coherence=(30e-6, 20e-6))


def test_noise_model_mode_b_forbids_p_idle():
    with pytest.raises(ValueError):
        noise_model(p=1e-3, p_idle=1e-4, coherence=(30e-6, 20e-6))


def test_noise_model_p_idle_meas_requires_p_idle():
    with pytest.raises(ValueError):
        noise_model(p=1e-3, p_idle_meas=1e-3)


def test_noise_model_custom_mode_no_baseline(clean_memory):
    profile = NoiseProfile({(None, None): {"p_1q": 1e-3, "p_meas": 5e-3,
                                           "p_2q": 2e-3, "p_reset": 1e-3,
                                           "p_idle": 1e-4, "p_idle_meas": 1e-3}})
    m = noise_model(profile=profile)
    noisy = m.noisy_circuit(clean_memory)
    assert_decodes_clean(noisy)


def test_noise_model_custom_mode_rejects_baseline_kwargs():
    profile = NoiseProfile({(None, None): {"p_1q": 1e-3}})
    with pytest.raises(ValueError):
        noise_model(profile=profile, p_meas=1e-3)


def test_noise_model_with_no_p_or_profile_raises():
    with pytest.raises(ValueError):
        noise_model()


def test_noise_model_with_overrides(clean_memory):
    profile = NoiseProfile({(5, None): {"p_meas": 1e-2}})
    m = noise_model(p=1e-3, profile=profile)
    noisy = m.noisy_circuit(clean_memory)
    assert_decodes_clean(noisy)


# ── uniform_depolarizing ───────────────────────────────────────────────────

def test_uniform_depolarizing(clean_memory):
    m = uniform_depolarizing(p=1e-3)
    noisy = m.noisy_circuit(clean_memory)
    assert_decodes_clean(noisy)


def test_uniform_depolarizing_zero_p(clean_memory):
    m = uniform_depolarizing(p=0.0)
    noisy = m.noisy_circuit(clean_memory)
    assert isinstance(noisy, stim.Circuit)


# ── pauli_biased ───────────────────────────────────────────────────────────

@pytest.mark.parametrize("axis", ["Z", "X"])
def test_pauli_biased_axes(clean_memory, axis):
    m = pauli_biased(p=1e-3, axis=axis, bias_factor=5.0)
    noisy = m.noisy_circuit(clean_memory)
    assert_decodes_clean(noisy)


def test_pauli_biased_bias_factor_one_matches_uniform(clean_memory):
    a = pauli_biased(p=1e-3, axis="Z", bias_factor=1.0).noisy_circuit(clean_memory)
    b = uniform_depolarizing(p=1e-3).noisy_circuit(clean_memory)
    assert str(a) == str(b)


def test_pauli_biased_bias_factor_below_one_raises():
    with pytest.raises(ValueError):
        pauli_biased(p=1e-3, axis="Z", bias_factor=0.5)


def test_pauli_biased_invalid_axis():
    with pytest.raises(ValueError):
        pauli_biased(p=1e-3, axis="Y", bias_factor=2.0)


# ── measurement_biased ─────────────────────────────────────────────────────

def test_measurement_biased(clean_memory):
    m = measurement_biased(p=1e-3, bias_factor=5.0)
    noisy = m.noisy_circuit(clean_memory)
    assert_decodes_clean(noisy)


def test_measurement_biased_bias_below_one_raises():
    with pytest.raises(ValueError):
        measurement_biased(p=1e-3, bias_factor=0.5)


# ── nonuniform ─────────────────────────────────────────────────────────────

def test_nonuniform_space_time(clean_memory):
    m = nonuniform(p=1e-3, sigma=0.3, variant="space_time", seed=42)
    noisy = m.noisy_circuit(clean_memory)
    assert_decodes_clean(noisy)


def test_nonuniform_space_only(clean_memory):
    m = nonuniform(p=1e-3, sigma=0.3, variant="space_only", seed=42)
    noisy = m.noisy_circuit(clean_memory)
    assert_decodes_clean(noisy)


def test_nonuniform_seed_reproducibility(clean_memory):
    a = nonuniform(p=1e-3, sigma=0.3, seed=42).noisy_circuit(clean_memory)
    b = nonuniform(p=1e-3, sigma=0.3, seed=42).noisy_circuit(clean_memory)
    assert str(a) == str(b)


def test_nonuniform_different_seeds_differ(clean_memory):
    a = nonuniform(p=1e-3, sigma=0.3, seed=42).noisy_circuit(clean_memory)
    b = nonuniform(p=1e-3, sigma=0.3, seed=43).noisy_circuit(clean_memory)
    assert str(a) != str(b)


def test_nonuniform_sigma_zero_recovers_uniform(clean_memory):
    a = nonuniform(p=1e-3, sigma=0.0, seed=42).noisy_circuit(clean_memory)
    b = uniform_depolarizing(p=1e-3).noisy_circuit(clean_memory)
    assert str(a) == str(b)


def test_nonuniform_negative_sigma_raises():
    with pytest.raises(ValueError):
        nonuniform(p=1e-3, sigma=-0.1)


def test_nonuniform_invalid_variant_raises():
    with pytest.raises(ValueError):
        nonuniform(p=1e-3, sigma=0.1, variant="rainbow")


def test_nonuniform_invalid_distribution_raises():
    with pytest.raises(ValueError):
        nonuniform(p=1e-3, sigma=0.1, distribution="cauchy")


def test_nonuniform_min_max_factor_invalid():
    with pytest.raises(ValueError):
        nonuniform(p=1e-3, sigma=0.1, min_factor=2.0, max_factor=1.0)


# ── strip_noise_channels round-trip ────────────────────────────────────────

def test_strip_noise_channels_round_trip(clean_memory):
    m = uniform_depolarizing(p=1e-3)
    noisy = m.noisy_circuit(clean_memory)
    stripped = strip_noise_channels(noisy)
    assert str(stripped) == str(clean_memory)
