# FTPrimitiveBench

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

> **Companion paper:** *FTPrimitiveBench: A Benchmark Suite For Logical
> Computation Under Hardware-Motivated and Biased Noise Models* — citation
> entry pending arXiv submission.

FTPrimitiveBench is a Python library for constructing, instrumenting, and
benchmarking the **primitives** of fault-tolerant quantum computation on the
rotated surface code: stabilizer-measurement memory, transversal Hadamard,
lattice surgery, and a logical S gate. It builds noiseless stabilizer circuit in
[Stim](https://github.com/quantumlib/Stim) format for each primitive,
wraps them in configurable noise models - uniform depolarizing, Pauli-biased,
measurement-biased, Gaussian per-component scatter, and T1/T2-derived
hardware noise with per-qubit / per-pair / per-round overrides for porting
real calibration data - and exports noisy circuits ready for
[`sinter`](https://github.com/quantumlib/Stim/tree/main/glue/sinter) /
[`pymatching`](https://github.com/oscarhiggott/PyMatching) sampling and
detector-error-model decoding. A pair of plotting helpers
(matplotlib 2D + plotly 3D) renders patches and spacetime timelines for
inspection or paper figures.

## Primitives

- **Memory** — `N` rounds of stabilizer measurement on a single patch followed by a basis-aligned final measurement.
- **Transversal H** — `pre_rounds` rounds of stabilizer measurement, **one** transversal Hadamard layer (with automatic X↔Z schedule swap), then `post_rounds` rounds. The round structure is `pre_rounds + 1 + post_rounds`, with the H layer in the middle.
- **Lattice surgery** — two patches separated by a configurable bridge, with MZZ / MXX merge and split phases.
- **Logical S gate** — ZZ lattice surgery spliced with a teleported logical-Y magic-state measurement.

Every primitive returns a `stim.Circuit` with `DETECTOR` and `OBSERVABLE_INCLUDE` annotations already in place.

## Noise models

A single canonical factory `noise_model(...)` plus four hardware-motivated pre-packaged builders. All accept the same per-class rate kwargs and per-component override hierarchy.

```python
from ft_primitive_bench.noise_models import noise_model, pauli_biased, NoiseProfile

# Baseline mode — uniform rates everywhere
m = noise_model(p=1e-3)

# SI1000-style asymmetric rates
m = noise_model(p=1e-3, p_1q=1e-4, p_2q=1e-3, p_meas=5e-3, p_reset=2e-3,
                p_idle=1e-4, p_idle_meas=2e-3)

# Mode B — physical T1/T2 with bundled (rate, duration) tuples
m = noise_model(p=1e-3,
                p_1q=(1e-4, 40e-9), p_2q=(1e-3, 120e-9),
                p_meas=(5e-3, 200e-9), p_reset=(2e-3, 100e-9),
                coherence=(30e-6, 20e-6))

# Per-component overrides via NoiseProfile (or a plain dict)
m = noise_model(p=1e-3, profile={
    (5, None):    {"p_meas": 1e-2},      # qubit 5, all rounds
    ((3, 4), None): {"p_2q": 5e-3},      # pair (3, 4), all rounds
    (None, 7):    {"p_1q": 2e-3},        # round 7, all components
    (5, 7):       {"p_meas": 5e-2},      # qubit 5 in round 7
})

# Fully custom — `profile` is the entire spec, no synthetic defaults
m = noise_model(profile=NoiseProfile({(None, None): {"p_1q": 1e-3, "p_meas": 5e-3}}))
```

The four pre-packaged builders sit on top of `noise_model`:

- `uniform_depolarizing(p, ...)` — direct alias.
- `pauli_biased(p, axis, bias_factor, ...)` — Pauli-axis bias (`axis ∈ {"Z", "X"}`); `bias_factor=1` recovers depolarizing.
- `measurement_biased(p, bias_factor, ...)` — uniform SPAM amplification.
- `nonuniform(p, sigma, *, variant="space_time", seed=...)` — Gaussian per-component scatter.

A utility helper `strip_noise_channels(circuit)` round-trips a noisy circuit back to its source.

## Requirements

| Group | Packages | Tested with |
|---|---|---|
| Python | — | 3.10, 3.11, 3.12 |
| Core (always installed) | `stim>=1.13`, `numpy>=1.21` | stim 1.15.x, numpy 1.21–2.4 |
| `[viz]` extra | `matplotlib>=3.5`, `plotly>=5.0` | matplotlib 3.9, plotly 5.24 |
| `[sampling]` extra | `sinter>=1.13`, `pymatching>=2.1` | sinter 1.15, pymatching 2.x |
| `[dev]` extra | `pytest>=7`, `pytest-cov`, `ruff`, `black`, `isort`, `build`, `twine`, `nbformat`, `jupyter` | latest each |

The core install is small and pure-Python on top of `stim` + `numpy` — works on
Linux, macOS, and Windows out of the box, with no compiler toolchain required.

## Installation

PyPI install (`pip install ft-primitive-bench` or
`uv pip install ft-primitive-bench`) coming soon. From source:

```bash
git clone https://github.com/ShuwenKan/FTPrimitiveBench.git
cd FTPrimitiveBench
```

Using [`uv`](https://docs.astral.sh/uv/) (recommended):

```bash
uv venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
uv pip install -e .                      # or: -e ".[viz,sampling,dev]"  /  -e ".[all]"
```

Using stock `pip` + `venv`:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .                         # or: -e ".[viz,sampling,dev]"  /  -e ".[all]"
```

Extras: `[viz]` (matplotlib + plotly), `[sampling]` (sinter + pymatching),
`[dev]` (pytest, ruff, black, isort, build, jupyter), `[all]` (everything).

## Quick start

```python
from ft_primitive_bench.surface_code.circuits import memory
from ft_primitive_bench.noise_models import pauli_biased

# 1. Noiseless circuit.
clean = memory(x_distance=3, z_distance=3, rounds=3, meas_basis="Z")

# 2. Noise model.
model = pauli_biased(p=1e-3, axis="Z", bias_factor=5.0)

# 3. Noisy circuit, ready for sinter / pymatching.
noisy = model.noisy_circuit(clean)

# 4. DEM extraction (biased presets emit PAULI_CHANNEL_*; pass approximate_disjoint_errors=True).
dem = noisy.detector_error_model(decompose_errors=True, approximate_disjoint_errors=True)
```

The same shape applies to every primitive: build a clean circuit, build a noise
model, call `model.noisy_circuit(clean)`.

## Repository layout

```
src/ft_primitive_bench/
    surface_code/
        circuits/            primitive builders + vendored Y-magic-measurement
                             subpackage (circuits/_y_measurement/)
        visualization/       matplotlib 2D patch plots + plotly 3D timelines
    noise_models/            NoiseModel, NoiseProfile, the packaged factories,
                             strip_noise_channels, infer_moment_rounds
tests/                   pytest suite (one file per source module)
tutorials/               end-to-end walkthrough notebooks
figures/                 generated plot outputs (PNG / SVG / HTML)
```

## Tutorials

- [`tutorials/noise_model_tutorial.ipynb`](tutorials/noise_model_tutorial.ipynb) —
  primitives, noise models, the `NoiseProfile` override hierarchy, and a
  synthetic calibration-data porting example.
- [`tutorials/visualization_tutorial.ipynb`](tutorials/visualization_tutorial.ipynb) —
  code-only notebook that builds + saves every visualization (`plot_patch`
  and the `plot_*_timeline` helpers) into `figures/`.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, testing, and PR
conventions. Notable changes per release are documented in
[CHANGELOG.md](CHANGELOG.md).

## Attribution

The vendored logical-Y magic-measurement subcircuit and the noise-engine
helpers in `src/ft_primitive_bench/noise_models/_noise_core.py` are adapted from
prior art under CC-BY 4.0:

> Gidney, C. (2022). *Data for "Inplace Access to the Surface Code Y Basis."*
> Zenodo. https://doi.org/10.5281/zenodo.7487893

See file headers and `LICENSE` for attribution details.

## License

MIT — see [`LICENSE`](LICENSE).
