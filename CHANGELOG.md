# Changelog

All notable changes to FTPrimitiveBench are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - Initial release

Companion to *FTPrimitiveBench: A Benchmark Suite For Logical Computation
Under Hardware-Motivated and Biased Noise Models* (arXiv submission pending).

### Added

- Surface-code primitive builders:
  - `memory(x_distance, z_distance, rounds, meas_basis)` — single-patch
    stabilizer-measurement memory.
  - `transversal_h(x_distance, z_distance, pre_rounds, post_rounds, meas_basis)` —
    transversal logical Hadamard with automatic X↔Z schedule swap.
  - `lattice_surgery(x_distance, z_distance, bridge_length, pre_rounds,
    merge_rounds, post_rounds, meas_basis)` — MZZ / MXX merge-and-split.
  - `s_gate(distance, bridge_length, pre_rounds, merge_rounds,
    boundary_rounds, post_rounds)` — ZZ surgery + teleported logical-Y
    magic-state measurement.
  - All primitives emit `DETECTOR` and `OBSERVABLE_INCLUDE` annotations and
    decode cleanly under `detector_error_model(decompose_errors=True,
    approximate_disjoint_errors=True)`.
- Lattice-surgery parity validation: `(stack_dim + bridge_length) % 2 == 0`.
- Noise model API:
  - Single canonical factory `noise_model(p, *, p_*=..., profile=...,
    coherence=...)` with two modes — baseline + overrides, or fully custom
    profile.
  - Pre-packaged factories: `uniform_depolarizing`, `pauli_biased(axis,
    bias_factor)`, `measurement_biased(bias_factor)`,
    `nonuniform(sigma, variant, seed)`.
  - `NoiseProfile` dict subclass with validated `(component, round)` keys.
  - `Coherence(T1, T2)` for Mode B (PTA-derived idle).
  - 5-tier override resolution: `(None, None) → (component, None) →
    (None, round) → (component, round)`.
  - Auxiliary helpers: `strip_noise_channels`, `infer_moment_rounds`,
    `idle_pauli_channel_from_T1T2`.
- Visualization:
  - `plot_patch` — matplotlib 2D patch plot.
  - `plot_*_timeline` — plotly 3D spacetime timelines for every primitive.
- Tutorials: `tutorials/noise_model_tutorial.ipynb` (primitives × noise),
  `tutorials/visualization_tutorial.ipynb` (code-only render-and-save notebook).
- Tests: 147 pytest tests across 12 modules.
- CI: GitHub Actions matrix across Python 3.10/3.11/3.12 and
  Ubuntu/macOS/Windows.

### Attribution

- The vendored Y-magic-measurement subcircuit at
  `src/ft_primitive_bench/surface_code/circuits/_y_measurement/` is adapted from Craig Gidney's
  "Inplace Access to the Surface Code Y Basis" (CC-BY 4.0).
- The noise-engine helpers in
  `src/ft_primitive_bench/noise_models/_noise_core.py` are adapted from the same
  author's `inplace-y-basis-2023/midout/gen/_noise.py` (CC-BY 4.0).

[Unreleased]: https://github.com/ShuwenKan/FTPrimitiveBench/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/ShuwenKan/FTPrimitiveBench/releases/tag/v0.1.0
