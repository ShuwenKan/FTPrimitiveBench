"""Public entry point for the ported Y-magic-measure circuit generator.

Inlines the ``basis='Y_magic_measure'`` branch of the upstream
``midout._make_circuit.make_circuit`` dispatcher (Craig Gidney, 2023), with all
code paths for other bases / noise injection / CZ conversion / debug output
removed.
"""

import stim

from . import _gen as gen
from ._circuits._y_memory_circuit import make_y_measure_chunks_with_magic_before


def make_y_measurement_circuit(
    *,
    boundary_rounds: int,
    memory_rounds: int,
    distance: int,
) -> stim.Circuit:
    """Build the logical-Y magic-measure circuit used by the S-gate splice.

    Mirrors ``midout.make_circuit(basis='Y_magic_measure', noise=None,
    convert_to_cz=False, boundary_rounds=..., memory_rounds=...,
    distance=...)``. The returned circuit still contains the leading MPP
    "magic initialization" block; the S-gate splicer skips it by iterating
    the circuit starting from the first post-MPP TICK segment, so the merged
    output has no MPP instructions.
    """
    chunks = make_y_measure_chunks_with_magic_before(
        distance=distance,
        boundary_rounds=boundary_rounds,
        memory_rounds=memory_rounds,
    )
    return gen.compile_chunks_into_circuit(chunks)
