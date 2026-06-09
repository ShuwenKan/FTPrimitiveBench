# FTPrimitiveBench

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-2605.04049-b31b1b.svg)](https://arxiv.org/abs/2605.04049)

**FTPrimitiveBench** is a Python library for constructing and benchmarking the **primitives** of fault-tolerant computation under hardware-motivated noise models. 

 

## Primitives
Rotated Surface Code:
- **Memory** — `N` rounds of stabilizer measurement on a single patch followed by a basis-aligned final measurement.
- **Transversal H** — `pre_rounds` rounds of stabilizer measurement, **one** transversal Hadamard layer (with automatic X↔Z schedule swap), then `post_rounds` rounds. The round structure is `pre_rounds + 1 + post_rounds`, with the H layer in the middle.
- **Lattice surgery** — two patches separated by a configurable bridge, with MZZ / MXX merge and split phases.
- **Logical S gate** — ZZ lattice surgery spliced with a teleported logical-Y magic-state measurement.

Every primitive returns a `stim.Circuit` with `DETECTOR` and `OBSERVABLE_INCLUDE` annotations already in place.

## Noise models

A single interface exposes parameters at four levels of granularity:
**global**, **per-component**, **per-round**, and **per-(component, round)**.
Three usage modes sit on top:

- **Pre-packaged structured-noise families** — Pauli bias, measurement bias,
  spatial / spatio-temporal non-uniformity — for controlled comparative
  studies.
- **Selective per-qubit and per-round overrides** for device heterogeneity and
  drift.
- **Fully customized hardware-aligned profiles** — gate and SPAM rates sourced
  from device calibration data, while idle channels are accumulated from the
  compiled syndrome-extraction schedule via per-qubit T1/T2.
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
| `[stimflow]` extra | `stimflow` (pinned git commit; required for `backend="stimflow"`) | stimflow 0.1.0 |
| `[dev]` extra | `pytest>=7`, `pytest-cov`, `ruff`, `black`, `isort`, `build`, `twine`, `nbformat`, `jupyter` | latest each |

The core install is small and pure-Python on top of `stim` + `numpy` — works on
Linux, macOS, and Windows out of the box, with no compiler toolchain required.

## Installation

PyPI install (`pip install ft-primitive-bench` or `uv pip install ft-primitive-bench`)
coming soon. From source:

```bash
git clone https://github.com/ShuwenKan/FTPrimitiveBench.git
cd FTPrimitiveBench
```

### Using [`uv`](https://docs.astral.sh/uv/) (recommended)

**macOS / Linux (bash/zsh):**

```bash
uv venv
source .venv/bin/activate
uv pip install -e .                      # or: -e ".[viz,sampling,dev]"  /  -e ".[all]"
```

**Windows (PowerShell):**

```powershell
uv venv
.\.venv\Scripts\Activate.ps1
uv pip install -e .                      # or: -e ".[viz,sampling,dev]"  /  -e ".[all]"
```

**Windows (Command Prompt):**

```cmd
uv venv
.venv\Scripts\activate.bat
uv pip install -e .
```

### Using stock `pip` + `venv`

**macOS / Linux (bash/zsh):**

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .                         # or: -e ".[viz,sampling,dev]"  /  -e ".[all]"
```

**Windows (PowerShell):**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .                         # or: -e ".[viz,sampling,dev]"  /  -e ".[all]"
```

**Windows (Command Prompt):**

```cmd
python -m venv .venv
.venv\Scripts\activate.bat
pip install -e .
```

### Optional extras

- `[viz]` — matplotlib + plotly
- `[sampling]` — sinter + pymatching
- `[dev]` — pytest, ruff, black, isort, build, jupyter
- `[all]` — everything

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

## Circuit construction backends

Every primitive accepts a keyword-only `backend` argument:

```python
clean = memory(x_distance=3, z_distance=3, rounds=3, backend="legacy")  # default
clean = memory(x_distance=3, z_distance=3, rounds=3, backend="stimflow")  # stimflow
```

`backend="legacy"` (the default) uses hand-rolled `stim.target_rec` offset
arithmetic — a self-contained construction that depends only on `stim` and
`numpy`. `backend="stimflow"` uses [stimflow](https://github.com/quantumlib/Stim/tree/main/glue/stimflow)'s
`Chunk` / `ChunkCompiler` framework, requiring the optional `[stimflow]` extra.
Both backends produce DEM-equivalent circuits — verified by parity tests over a
grid of distances, round counts, and measurement bases.

The two backends exist side-by-side intentionally: legacy is the source of
truth for correctness and the no-dependency install path; chunks is the
declarative form that makes new primitives easier to extend and provides
`chunk.verify()` as a free stabilizer-flow sanity check on most chunks. The
chunks pipeline is also where future work like `Chunk.to_html_viewer()`
integration lands.

### The chunks design: open flows as measurement-record accounts

Each primitive is decomposed into a sequence of `stimflow.Chunk` objects. The
`ChunkCompiler` threads an "open flows" dictionary across chunks, keyed by
`PauliMap`, accumulating measurement records as flows pass through. Each
chunk's per-tile work boils down to one of five `add_flow` calls:

| Operation | API | Effect on the open-flows dict |
|---|---|---|
| emit | `add_flow(end=P, measurements=[m])` | open a new account labelled `P` containing `m` |
| absorb | `add_flow(start=P, measurements=[m])` | close `P`, emit `DETECTOR(rec[account] XOR rec[m])` |
| passthrough | `add_flow(start=P, end=P, measurements=[m])` | deposit `m` into account `P` |
| transition | `add_flow(start=P, end=Q)` | rename account `P` → `Q` |
| self-detector | `add_flow(measurements=[m])` | emit `DETECTOR(rec[m])` with no chain |

The compiler trusts these declarations and emits `DETECTOR` /
`OBSERVABLE_INCLUDE` instructions automatically — the rec-offset arithmetic
that dominates the legacy backend disappears entirely. The eight surface-code
phases (data init, pre-merge syndrome, transition, merged syndrome, ancilla
destruct, post-merge syndrome, final data measurement) each become a chunk
whose per-tile logic is one or two flow declarations.

### How lattice surgery handles a fundamental stim-flow tension

The merge-readout observable `L1 * L2` is the product of the two patches'
logical operators. Across the whole circuit it has a deterministic
relationship to the bridge stabilizer measurements; **but in any single merge
round it anticommutes with the X-stab `MX` measurements**. Stim's
`circuit.has_flow` therefore refuses to recognize `L1*L2 → bridge_anc_recs` as
a single-chunk flow — both the `measurements="auto"` solver and explicit
PauliMap composition with ancilla-init stabilizers reject it.

The fix exploits the fact that `ChunkCompiler.append` doesn't call
`chunk.verify()`. The compiler chains open flows by PauliMap identity, not by
algebraic validation. So the merge-last chunk just declares the observable
absorb manually:

```python
merge_last.add_flow(start=L_merged, measurements=bridge_anc_keys)
```

The compiler emits `OBSERVABLE_INCLUDE(rec[bridge_anc_keys])` — bit-identical
to legacy. End-to-end DEM parity against the legacy backend is the
authoritative correctness check; per-chunk `verify()` is preserved only for
chunks whose flows are deterministic in isolation (init, pre/post rounds).

The same trick — declaring flows manually where stim can't auto-verify them —
handles the TYPE B `ancilla_basis` 2-rec detector. When the bridge data is
initialized in `RX`, the X stabilizer on each bridge data qubit is `+1`
trivially, so the merged-X stab measurement equals the prior individual-X
stab measurement:

```python
transition.add_flow(start=individual_X_pm, measurements=[anc_curr])
```

No `ChunkReflow` needed; the bridge `+1` X stabilizers are implicit. The
declared absorb closes the prior round's open `individual_X` account and the
compiler emits `DETECTOR(rec[anc_prev], rec[anc_curr])`.

This is the general pattern: where stim's local flow analysis would reject a
declaration that is correct end-to-end, the chunks compiler still threads
the measurement records through and produces the right circuit. DEM-parity
tests against the legacy backend gate every primitive.

## Repository layout

```
src/ft_primitive_bench/
    surface_code/
        circuits/            primitive builders (legacy backend) + vendored
                             Y-magic-measurement subpackage (_y_measurement/)
        circuits/_stimflow/  stimflow backend: stimflow-based reimplementations
                             of memory, transversal_h, lattice_surgery
        visualization/       matplotlib 2D patch plots + plotly 3D timelines
    noise_models/            NoiseModel, NoiseProfile, the packaged factories,
                             strip_noise_channels, infer_moment_rounds
tests/                   pytest suite (one file per source module)
tests/parity/            stimflow-vs-legacy DEM equivalence tests
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
