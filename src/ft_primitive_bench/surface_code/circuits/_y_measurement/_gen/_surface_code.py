"""Surface-code basis helpers.

Ported from ``midout.gen._surface_code``, trimmed to only the symbols actually
used by the ``Y_magic_measure`` code path (``checkerboard_basis``). The original
``surface_code_patch`` helper is left out because it is not reached by the
Y-measurement circuit.
"""


def checkerboard_basis(q: complex) -> str:
    """Classifies a coordinate as X type or Z type according to a checkerboard."""
    is_x = int(q.real + q.imag) & 1 == 0
    return "X" if is_x else "Z"
