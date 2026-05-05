"""Vendored logical-Y magic-measurement circuit generator.

This subpackage inlines the minimum subset of Craig Gidney's "Inplace Access to
the Surface Code Y Basis" codebase (midout, 2023 arXiv preprint) required to
produce the logical-Y magic-measure circuit that the :mod:`surface_code` S-gate
builder splices into a ZZ lattice-surgery base circuit.

Only the ``basis='Y_magic_measure'``, ``noise=None``, ``convert_to_cz=False``
code path is retained; all other bases, the noise compiler, the CZ-interaction
translator, HTML / SVG viewers, and the interaction planner have been dropped.

The upstream ``midout.gen`` helpers live under :mod:`._gen` and the circuit
generators live under :mod:`._circuits`. Downstream code should only import
:func:`make_y_measurement_circuit` from this package; the internal modules are
considered private.

Original authorship: Craig Gidney, "Inplace Access to the Surface Code Y
Basis", arXiv:2302.07395 (2023). See ``Y_state/src/midout/`` in earlier
revisions of this repository for the full upstream tree, including basis
variants not ported here.
"""

from ._entry import make_y_measurement_circuit

__all__ = ["make_y_measurement_circuit"]
