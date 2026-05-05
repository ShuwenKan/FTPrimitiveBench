"""Y-magic-measure patch helpers.

Slimmed port of ``midout.circuits.steps._patches`` (Craig Gidney, 2023). Only
the two patches reached by the ``Y_magic_measure`` entry point are retained:

* :func:`make_ztop_yboundary_patch` — the Z/top, Y-boundary patch used by the
  final magic-state transition round.
* :func:`make_xtop_qubit_patch` — the standard X/top qubit patch used by the
  memory-round and initialization chunks.

The upstream ``make_stability_patch`` helper and the unused ``DIRS`` slice
export have been dropped because they are not reachable from this entry point.
"""

import functools
from typing import Callable, Iterable, List, Optional

from .. import _gen as gen

DIRS = [(0.5 + 0.5j) * 1j**d for d in range(4)]
DR, DL, UL, UR = DIRS


def surface_code_patch(
    *,
    possible_data_qubits: Iterable[complex],
    basis: Callable[[complex], str],
    is_boundary_x: Callable[[complex], bool],
    is_boundary_z: Callable[[complex], bool],
    order_func: Callable[[complex], Iterable[Optional[complex]]],
) -> gen.Patch:
    possible_data_qubits = set(possible_data_qubits)
    possible_measure_qubits = {q + d for q in possible_data_qubits for d in DIRS}
    measure_qubits = {
        m
        for m in possible_measure_qubits
        if sum(m + d in possible_data_qubits for d in DIRS) > 1
        if is_boundary_x(m) <= (basis(m) == "X")
        if is_boundary_z(m) <= (basis(m) == "Z")
    }
    data_qubits = {
        q for q in possible_data_qubits if sum(q + d in measure_qubits for d in DIRS) > 1
    }

    tiles = []
    for m in measure_qubits:
        tiles.append(
            gen.Tile(
                bases=basis(m),
                measurement_qubit=m,
                ordered_data_qubits=[
                    m + d if d is not None and m + d in data_qubits else None for d in order_func(m)
                ],
            )
        )
    return gen.Patch(tiles)


def rectangular_surface_code_patch(
    *,
    width: int,
    height: int,
    top_basis: str,
    bot_basis: str,
    left_basis: str,
    right_basis: str,
    order_func: Callable[[complex], Iterable[Optional[complex]]],
) -> gen.Patch:
    def is_boundary(m: complex, *, b: str) -> bool:
        if top_basis == b and m.imag == -0.5:
            return True
        if left_basis == b and m.real == -0.5:
            return True
        if bot_basis == b and m.imag == height - 0.5:
            return True
        if right_basis == b and m.real == width - 0.5:
            return True
        return False

    return surface_code_patch(
        possible_data_qubits=[x + 1j * y for x in range(width) for y in range(height)],
        basis=gen.checkerboard_basis,
        is_boundary_x=functools.partial(is_boundary, b="X"),
        is_boundary_z=functools.partial(is_boundary, b="Z"),
        order_func=order_func,
    )


def make_ztop_yboundary_patch(*, distance: int) -> gen.Patch:
    def order_func(m: complex) -> List[complex]:
        order_S = [UR, UL, DR, DL]
        order_N = [UR, DR, UL, DL]
        if m.real > m.imag and False:
            if gen.checkerboard_basis(m) == "X":
                return order_N
            else:
                return order_S
        else:
            if gen.checkerboard_basis(m) == "X":
                return order_S
            else:
                return order_N

    return rectangular_surface_code_patch(
        width=distance,
        height=distance,
        top_basis="Z",
        right_basis="X",
        bot_basis="X",
        left_basis="Z",
        order_func=order_func,
    )


def make_xtop_qubit_patch(*, distance: int) -> gen.Patch:
    def order_func(m: complex) -> List[complex]:
        order_S = [UR, UL, DR, DL]
        order_N = [UR, DR, UL, DL]
        if gen.checkerboard_basis(m) == "X":
            return order_S
        else:
            return order_N

    return rectangular_surface_code_patch(
        width=distance,
        height=distance,
        top_basis="X",
        right_basis="Z",
        bot_basis="X",
        left_basis="Z",
        order_func=order_func,
    )
