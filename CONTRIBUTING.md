# Contributing to FTPrimitiveBench

Thanks for your interest in contributing! This document covers the local
development setup, the test/lint conventions, and the pull-request process.

## Development setup

Requires Python ≥ 3.10. Both [`uv`](https://docs.astral.sh/uv/) and stock
`pip` + `venv` are supported.

```bash
git clone https://github.com/ShuwenKan/FTPrimitiveBench.git
cd FTPrimitiveBench

# uv (recommended)
uv venv && source .venv/bin/activate         # Windows: .venv\Scripts\activate
uv pip install -e ".[dev]"

# or pip
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

`[dev]` pulls `pytest`, `pytest-cov`, `ruff`, `black`, `isort`, `build`,
`twine`, `nbformat`, `jupyter`. Use `[viz]`, `[sampling]`, or `[all]` for
the visualization / sampling stacks.

## Running tests

The full suite is `pytest`-based and lives under `tests/`:

```bash
pytest                              # full suite
pytest tests/test_memory.py         # one file
pytest -k "lattice_surgery"         # match by keyword
pytest --cov=surface_code           # with coverage
```

Tests stay fast (small distances, few rounds) so they can run on every commit.
If you add a new public function, add a test for it in the matching
`tests/test_<module>.py` file.

## Code quality

Three tools run before merge:

```bash
ruff check src tests                # lint
black src tests                     # auto-format
isort src tests                     # import order
```

`black` and `isort` are configured to agree (`profile = "black"` in
`pyproject.toml`). `ruff` enforces `E`, `F`, `W`, and `I` checks at
line length 100. Please run all three before opening a PR.

## Pull-request process

1. **Open an issue first** for non-trivial changes so we can discuss the
   design before you sink time into an implementation.
2. **Fork** the repository and create a topic branch off `main`:
   ```bash
   git checkout -b feature/my-change
   ```
3. **Write the change** alongside tests. New primitives need DEM-decoding
   tests (verify `noisy.detector_error_model(decompose_errors=True,
   approximate_disjoint_errors=True)` runs cleanly). New noise-model
   features need round-trip tests against `strip_noise_channels`.
4. **Run the full suite locally** (`pytest`, `ruff`, `black --check`,
   `isort --check-only`).
5. **Open a PR** against `main` with a clear description of the change,
   the motivation, and any benchmark / DEM evidence if relevant.
6. **Address review feedback** with new commits; we squash on merge.

## Reporting issues

Bugs, regressions, and feature requests go in the
[issue tracker](https://github.com/ShuwenKan/FTPrimitiveBench/issues).

A good bug report includes:

- The minimal `surface_code` call that triggers the bug.
- The Python version, `stim` version, and OS.
- The stim circuit (or DEM) that reproduces the issue, if applicable.
- The expected vs. observed behavior.

## Project layout

```
src/ft_primitive_bench/
    surface_code/
        circuits/            primitive builders
        visualization/       matplotlib + plotly helpers
    noise_models/            NoiseModel, NoiseProfile, packaged factories
tests/                   pytest suite (one file per source module)
tutorials/               end-to-end walkthrough notebooks
figures/                 generated plot outputs
```

## Code style

- **Type hints on public APIs.** Internal helpers may be lighter.
- **Docstrings on public functions** explaining what the function returns
  and what each non-obvious argument does.
- **No silent `try/except`** — let exceptions propagate; raise
  `ValueError` / `TypeError` on bad user input with a message that names
  the bad argument.
- **Don't add backwards-compatibility shims** while we're pre-1.0; we
  prefer clean breaks with a clear changelog entry.

## License

By contributing, you agree that your contribution is licensed under the
MIT License (see [`LICENSE`](LICENSE)). The vendored Y-magic-measurement
subcircuit and the noise-engine helpers retain their original CC-BY 4.0
attribution; do not relicense those files.
