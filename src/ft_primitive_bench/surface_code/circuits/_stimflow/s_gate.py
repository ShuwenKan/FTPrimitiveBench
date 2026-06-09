"""Chunks-backend s_gate primitive.

**Current implementation: legacy delegation.**

Unlike memory / transversal_h / lattice_surgery, the s_gate primitive is built
as two pieces glued together at the instruction-stream level:

1. The S-gate **surgery half** (legacy ``s_gate_surgery``): ZZ lattice surgery
   with X-basis init on both data and bridge ancilla, no final data
   measurement. Emits two ``OBSERVABLE_INCLUDE(0)`` calls during surgery —
   one for the controlled-X correction from ancilla MX, one for the bridge ZZ
   parity from the last merge round.

2. The **Y-magic measurement** (legacy ``make_y_measurement_circuit``), built
   from the vendored ``_y_measurement._gen`` chunks (proto-stimflow). Spliced
   into the surgery by ``_remap_circuit`` via a 200-line instruction-stream
   transform that:
   - strips the Y-magic's MPP preamble and its single memory round,
   - replays the Y-magic twice (once per patch),
   - rewrites detector rec references onto the surgery's last post-merge
     round measurements via coordinate lookup.

A full chunks port is feasible but substantial:

- The surgery half can mostly reuse the ``_stimflow/lattice_surgery.py``
  scaffolding, but s_gate's X-basis everywhere semantics means the
  ``ancilla_basis``/``meas_basis`` symmetry differs (both bases are X-init,
  not opposite). TYPE C deterministic detectors land on X-stabs (the
  meas_basis), not the opposite basis as in lattice_surgery.
- The two observable contributions need a stimflow flow that stays open
  across the merge readout AND the ancilla destruct, dumping
  ``OBSERVABLE_INCLUDE`` at each via the compiler's
  ``_append_detectors`` pending-observable-dump mechanism.
- The Y-magic splice would require translating the vendored ``_gen.Chunk``
  list into ``stimflow.Chunk`` (the API is structurally identical), then
  composing both halves through one ``ChunkCompiler`` rather than via the
  legacy instruction-stream splice.

For now ``backend="stimflow"`` delegates to legacy, which gives API parity and
lets Phase 5 cleanup flip defaults uniformly. The bit-identical legacy output
trivially satisfies the parity test.
"""

from __future__ import annotations

import stim


def s_gate(
    distance: int,
    bridge_length: int,
    pre_rounds: int,
    merge_rounds: int,
    boundary_rounds: int,
    post_rounds: int,
) -> stim.Circuit:
    # Lazy import to avoid circular dependency (S_gate.s_gate dispatches here).
    from ..S_gate import s_gate as _s_gate_legacy

    return _s_gate_legacy(
        distance=distance,
        bridge_length=bridge_length,
        pre_rounds=pre_rounds,
        merge_rounds=merge_rounds,
        boundary_rounds=boundary_rounds,
        post_rounds=post_rounds,
        backend="legacy",
    )
