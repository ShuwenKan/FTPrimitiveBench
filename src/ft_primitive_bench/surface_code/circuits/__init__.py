"""Public circuit builders for the :mod:`surface_code.circuits` package."""

from . import _build_patch
from . import _make_circuit as make_circuit_module

from ._build_patch import (
    boundary_tiles,
    checkerboard_weight4_patch,
    create_individual_patch,
    rectangular_surface_code_patch,
)
from ._make_circuit import (
    build_merged_patch,
    lattice_surgery,
    memory,
    patch_midlines,
    transversal_h,
)
from .S_gate import s_gate

__all__ = [
    "_build_patch",
    "boundary_tiles",
    "build_merged_patch",
    "checkerboard_weight4_patch",
    "create_individual_patch",
    "lattice_surgery",
    "make_circuit_module",
    "memory",
    "patch_midlines",
    "rectangular_surface_code_patch",
    "s_gate",
    "transversal_h",
]
