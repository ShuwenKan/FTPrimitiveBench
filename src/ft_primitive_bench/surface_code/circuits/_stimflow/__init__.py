"""Chunks-backend primitive builders (stimflow-based).

This package mirrors the public surface of the legacy rec-offset builders
in ``_make_circuit.py`` and ``S_gate.py``, but constructs each primitive
as a sequence of ``stimflow.Chunk`` objects with declared stabilizer
flows. The compiler emits all DETECTOR / OBSERVABLE_INCLUDE annotations
from the flow declarations, eliminating the hand-rolled
``stim.target_rec`` arithmetic.

Status: scaffold only. Primitives are ported one at a time; each new
implementation is gated through the ``backend`` kwarg on the public
builders in the parent ``circuits`` package and validated by
``tests/parity/`` before becoming the default.

Importing this package raises ``ImportError`` if stimflow is not
installed. Install with ``pip install -e ".[stimflow]"``.
"""

from __future__ import annotations

try:
    import stimflow as _stimflow  # noqa: F401
except ImportError as exc:  # pragma: no cover - exercised in environments without the extra
    raise ImportError(
        "The stimflow backend requires the optional 'stimflow' extra. "
        "Install it with: pip install -e '.[stimflow]'"
    ) from exc

__all__: list[str] = []
