"""FTPrimitiveBench: surface-code primitives + configurable noise models.

The package exposes two sibling sub-packages:

- :mod:`ft_primitive_bench.surface_code` — primitive builders and visualization.
- :mod:`ft_primitive_bench.noise_models` — the noise-model engine and factories.

Either can be imported directly:

>>> from ft_primitive_bench.surface_code import memory
>>> from ft_primitive_bench.noise_models import pauli_biased
"""
from importlib import metadata as _metadata

from . import noise_models, surface_code

try:
    __version__ = _metadata.version("ft-primitive-bench")
except _metadata.PackageNotFoundError:  # pragma: no cover - source checkout
    __version__ = "0.0.0+local"

__all__ = ["__version__", "noise_models", "surface_code"]
