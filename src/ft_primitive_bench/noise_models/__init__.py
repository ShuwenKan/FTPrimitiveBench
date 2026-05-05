"""Noise models for surface-code primitive benchmarks.

Public API
----------
- ``noise_model(p=..., *, p_*=..., profile=...)`` — the one factory. Two modes:
  baseline (``p`` set) or fully custom (``p`` None, ``profile`` carries everything).
- ``uniform_depolarizing``, ``pauli_biased``, ``measurement_biased``, ``nonuniform`` — pre-packaged.
- ``NoiseProfile`` — dict subclass mapping ``(component, round)`` → rate dict.
- ``Coherence`` — typed (T1, T2) wrapper.
- ``NoiseModel`` — engine class; call ``model.noisy_circuit(clean)``.
- ``strip_noise_channels``, ``infer_moment_rounds``, ``idle_pauli_channel_from_T1T2``.
"""

from .hardware_noise import (
    CompiledCircuit,
    NoiseModel,
    idle_pauli_channel_from_T1T2,
    infer_moment_rounds,
    strip_noise_channels,
)
from .noise_profile import Coherence, NoiseProfile
from .profiles import (
    ConfiguredNoiseModel,
    NoiseModelConfig,
    SampledFactorSnapshot,
    measurement_biased,
    noise_model,
    nonuniform,
    pauli_biased,
    uniform_depolarizing,
)

__all__ = [
    # Types
    "Coherence",
    "NoiseProfile",
    "NoiseModel",
    "NoiseModelConfig",
    "ConfiguredNoiseModel",
    "SampledFactorSnapshot",
    "CompiledCircuit",
    # Factories
    "noise_model",
    "uniform_depolarizing",
    "pauli_biased",
    "measurement_biased",
    "nonuniform",
    # Auxiliary helpers
    "infer_moment_rounds",
    "idle_pauli_channel_from_T1T2",
    "strip_noise_channels",
]
