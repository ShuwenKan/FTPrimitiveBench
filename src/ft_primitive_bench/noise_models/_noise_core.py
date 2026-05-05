"""Shared constants, dataclasses and helpers for the noise engine.

Attribution
-----------
Modified from Craig Gidney's noise-model code in the *Inplace Access to the
Surface Code Y Basis* repository:
https://github.com/Strilanc/inplace-y-basis-2023/blob/main/src/midout/gen/_noise.py
(same source as the vendored logical-Y magic-state circuit under
``surface_code/circuits/_y_measurement/``).

What is taken structurally from the upstream
--------------------------------------------
- The op-type classification table (``OP_TYPES``) — separating Cliffords by
  arity, measurement / reset / measure-reset variants, MPP family, annotations,
  and noise channels.
- The moment iterator (``_iter_split_op_moments``): walking a Stim circuit
  TICK-by-TICK with REPEAT-block handling and per-target splitting for ops
  that touch immune qubits or classical-control targets.
- Per-target splitting helpers (``_split_targets_if_needed*``) for 1Q
  Cliffords, 2Q Cliffords, MPP, and 1Q measurement / reset.
- The classical-control detection (``occurs_in_classical_control_system``).
- The basis-extraction helper (``_measure_basis``) for MZ / MX / MY / MPP /
  MR / MRX / MRY / MRZ.

What we added or changed
------------------------
- Added ``XCY`` and several other 2Q Clifford variants (``CXSWAP``, ``SWAPCX``,
  ``CZSWAP``, ``SWAPCZ``, ``ZCX``, ``ZCY``, ``ZCZ``, ``II``) to the
  classification table to cover the s-gate primitive's gate set.
- Added the ``ANNOTATION`` mapping for ``MPAD`` and ``E`` so they pass through
  the engine unchanged.
- Added new pure-noise channel names (``I_ERROR``, ``II_ERROR``,
  ``HERALDED_ERASE``, ``HERALDED_PAULI_CHANNEL_1``).
- Added ``idle_pauli_channel_from_T1T2`` — the standard Pauli-twirled
  approximation to amplitude damping + dephasing, exposed as a public helper
  for users who want the formula directly.
- Added ``_normalize_pair_value`` for normalizing user-side 2-qubit noise
  specs into a uniform list form.
- The downstream ``hardware_noise.py`` engine and the ``NoiseProfile`` /
  ``noise_model`` API on top are independent of the upstream.

License
-------
Upstream code is distributed under CC-BY 4.0. Modifications copyright the
FTPrimitiveBench authors, distributed under the same license.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Iterator, List, Mapping, Optional, Tuple, Union

import stim

# ---------------------------------------------------------------------------
# Op-type tags
# ---------------------------------------------------------------------------
CLIFFORD_1Q = "C1"
CLIFFORD_2Q = "C2"
ANNOTATION = "info"
MPP = "MPP"
MEASURE_RESET_1Q = "MR1"
JUST_MEASURE_1Q = "M1"
JUST_RESET_1Q = "R1"
NOISE = "!?"

OP_TYPES = {
    "I": CLIFFORD_1Q,
    "X": CLIFFORD_1Q,
    "Y": CLIFFORD_1Q,
    "Z": CLIFFORD_1Q,
    "H": CLIFFORD_1Q,
    "H_XY": CLIFFORD_1Q,
    "H_XZ": CLIFFORD_1Q,
    "H_YZ": CLIFFORD_1Q,
    "H_NXY": CLIFFORD_1Q,
    "H_NXZ": CLIFFORD_1Q,
    "H_NYZ": CLIFFORD_1Q,
    "S": CLIFFORD_1Q,
    "S_DAG": CLIFFORD_1Q,
    "SQRT_X": CLIFFORD_1Q,
    "SQRT_X_DAG": CLIFFORD_1Q,
    "SQRT_Y": CLIFFORD_1Q,
    "SQRT_Y_DAG": CLIFFORD_1Q,
    "SQRT_Z": CLIFFORD_1Q,
    "SQRT_Z_DAG": CLIFFORD_1Q,
    "C_XYZ": CLIFFORD_1Q,
    "C_ZYX": CLIFFORD_1Q,
    "C_NXYZ": CLIFFORD_1Q,
    "C_NZYX": CLIFFORD_1Q,
    "C_XNYZ": CLIFFORD_1Q,
    "C_XYNZ": CLIFFORD_1Q,
    "C_ZNYX": CLIFFORD_1Q,
    "C_ZYNX": CLIFFORD_1Q,
    "CNOT": CLIFFORD_2Q,
    "CX": CLIFFORD_2Q,
    "CY": CLIFFORD_2Q,
    "CZ": CLIFFORD_2Q,
    "ISWAP": CLIFFORD_2Q,
    "ISWAP_DAG": CLIFFORD_2Q,
    "SQRT_XX": CLIFFORD_2Q,
    "SQRT_XX_DAG": CLIFFORD_2Q,
    "SQRT_YY": CLIFFORD_2Q,
    "XCY": CLIFFORD_2Q,
    "SQRT_YY_DAG": CLIFFORD_2Q,
    "SQRT_ZZ": CLIFFORD_2Q,
    "SQRT_ZZ_DAG": CLIFFORD_2Q,
    "SWAP": CLIFFORD_2Q,
    "XCX": CLIFFORD_2Q,
    "XCZ": CLIFFORD_2Q,
    "YCX": CLIFFORD_2Q,
    "YCY": CLIFFORD_2Q,
    "YCZ": CLIFFORD_2Q,
    "ZCX": CLIFFORD_2Q,
    "ZCY": CLIFFORD_2Q,
    "ZCZ": CLIFFORD_2Q,
    "CXSWAP": CLIFFORD_2Q,
    "SWAPCX": CLIFFORD_2Q,
    "CZSWAP": CLIFFORD_2Q,
    "SWAPCZ": CLIFFORD_2Q,
    "II": CLIFFORD_2Q,
    "MPP": MPP,
    "MXX": MPP,
    "MYY": MPP,
    "MZZ": MPP,
    "MR": MEASURE_RESET_1Q,
    "MRX": MEASURE_RESET_1Q,
    "MRY": MEASURE_RESET_1Q,
    "MRZ": MEASURE_RESET_1Q,
    "M": JUST_MEASURE_1Q,
    "MX": JUST_MEASURE_1Q,
    "MY": JUST_MEASURE_1Q,
    "MZ": JUST_MEASURE_1Q,
    "MPAD": ANNOTATION,
    "R": JUST_RESET_1Q,
    "RX": JUST_RESET_1Q,
    "RY": JUST_RESET_1Q,
    "RZ": JUST_RESET_1Q,
    "DETECTOR": ANNOTATION,
    "OBSERVABLE_INCLUDE": ANNOTATION,
    "QUBIT_COORDS": ANNOTATION,
    "SHIFT_COORDS": ANNOTATION,
    "TICK": ANNOTATION,
    "E": ANNOTATION,
    "DEPOLARIZE1": NOISE,
    "DEPOLARIZE2": NOISE,
    "PAULI_CHANNEL_1": NOISE,
    "PAULI_CHANNEL_2": NOISE,
    "X_ERROR": NOISE,
    "Y_ERROR": NOISE,
    "Z_ERROR": NOISE,
    "I_ERROR": NOISE,
    "II_ERROR": NOISE,
    "ELSE_CORRELATED_ERROR": NOISE,
    "HERALDED_ERASE": NOISE,
    "HERALDED_PAULI_CHANNEL_1": NOISE,
}

OP_MEASURE_BASES = {
    "M": "Z",
    "MX": "X",
    "MY": "Y",
    "MZ": "Z",
    "MPP": "",
    "MXX": "XX",
    "MYY": "YY",
    "MZZ": "ZZ",
    "MR": "Z",
    "MRX": "X",
    "MRY": "Y",
    "MRZ": "Z",
}

COLLAPSING_OPS = {
    op
    for op, typ in OP_TYPES.items()
    if typ in {JUST_RESET_1Q, JUST_MEASURE_1Q, MPP, MEASURE_RESET_1Q}
}

MEASUREMENT_OPS = {"M", "MX", "MY", "MZ", "MPP", "MR", "MRX", "MRY", "MRZ"}
RESET_OPS = {"R", "RX", "RY", "RZ"}


# ---------------------------------------------------------------------------
# Shared dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GateDurations:
    """Wall-clock durations (in consistent time units) for core operations."""

    one_qubit: float
    two_qubit: float
    measurement: float

    def __post_init__(self) -> None:
        """Validate that all durations are non-negative."""
        for name, value in (
            ("one_qubit", self.one_qubit),
            ("two_qubit", self.two_qubit),
            ("measurement", self.measurement),
        ):
            if value < 0:
                raise ValueError(f"{name} duration must be non-negative, got {value}")


@dataclass(frozen=True)
class TwoQubitNoiseSpec:
    """Collection of noise instructions to apply after a specific 2-qubit gate."""

    instructions: List[Tuple[str, Union[float, List[float]]]]


# ---------------------------------------------------------------------------
# Shared helper functions
# ---------------------------------------------------------------------------


def idle_pauli_channel_from_T1T2(t: float, T1: float, T2: float) -> Tuple[float, float, float]:
    """Return (px, py, pz) for an idle period of duration ``t``."""
    px = 0.25 * (1 - math.exp(-t / T1))
    py = px
    pz = 0.5 * (1 - math.exp(-t / T2)) - 0.25 * (1 - math.exp(-t / T1))
    return max(px, 0.0), max(py, 0.0), max(pz, 0.0)


def occurs_in_classical_control_system(op: stim.CircuitInstruction) -> bool:
    """Determine whether an operation only touches classical data."""
    typ = OP_TYPES.get(op.name)
    if typ == ANNOTATION:
        return True
    if typ == CLIFFORD_2Q:
        targets = op.targets_copy()
        for i in range(0, len(targets), 2):
            a = targets[i]
            b = targets[i + 1]
            classical_0 = a.is_measurement_record_target or a.is_sweep_bit_target
            classical_1 = b.is_measurement_record_target or b.is_sweep_bit_target
            if not (classical_0 or classical_1):
                return False
        return True
    return False


def _split_targets_if_needed(
    op: stim.CircuitInstruction, immune_qubits: set[int]
) -> Iterator[stim.CircuitInstruction]:
    """Yield a sequence of operations whose targets interact with a single qubit."""
    typ = OP_TYPES.get(op.name)
    if typ == CLIFFORD_2Q:
        yield from _split_targets_if_needed_clifford_2q(op, immune_qubits)
    elif typ == MPP:
        yield from _split_targets_if_needed_mpp(op)
    elif typ in {JUST_MEASURE_1Q, MEASURE_RESET_1Q, JUST_RESET_1Q}:
        yield from _split_targets_if_needed_1q(op)
    elif typ in [NOISE, ANNOTATION]:
        yield op
    else:
        yield from _split_targets_if_needed_clifford_1q(op, immune_qubits)


def _split_targets_if_needed_clifford_1q(
    op: stim.CircuitInstruction, immune_qubits: set[int]
) -> Iterator[stim.CircuitInstruction]:
    """Ensure 1Q Clifford instructions address one target at a time."""
    if immune_qubits:
        args = op.gate_args_copy()
        for target in op.targets_copy():
            yield stim.CircuitInstruction(op.name, [target], args)
    else:
        yield op


def _split_targets_if_needed_clifford_2q(
    op: stim.CircuitInstruction, immune_qubits: set[int]
) -> Iterator[stim.CircuitInstruction]:
    """Ensure 2Q Clifford instructions operate on at most one pair."""
    targets = op.targets_copy()
    args = op.gate_args_copy()
    if immune_qubits or any(t.is_measurement_record_target for t in targets):
        for i in range(0, len(targets), 2):
            yield stim.CircuitInstruction(op.name, targets[i : i + 2], args)
    else:
        yield op


def _split_targets_if_needed_1q(op: stim.CircuitInstruction) -> Iterator[stim.CircuitInstruction]:
    """Split measurement/reset instructions into per-target operations."""
    args = op.gate_args_copy()
    for target in op.targets_copy():
        yield stim.CircuitInstruction(op.name, [target], args)


def _split_targets_if_needed_mpp(op: stim.CircuitInstruction) -> Iterator[stim.CircuitInstruction]:
    """Normalize multi-body parity measurements into separate instructions."""
    targets = op.targets_copy()
    args = op.gate_args_copy()
    k = 0
    start = k
    while k < len(targets):
        if k + 1 == len(targets) or not targets[k + 1].is_combiner:
            yield stim.CircuitInstruction(op.name, targets[start : k + 1], args)
            k += 1
            start = k
        else:
            k += 2
    assert k == len(targets)


def _iter_split_op_moments(
    circuit: stim.Circuit, *, immune_qubits: set[int]
) -> Iterator[stim.CircuitRepeatBlock | List[stim.CircuitInstruction]]:
    """Iterate through the circuit in moment form (split by TICKs)."""
    current: List[stim.CircuitInstruction] = []
    for op in circuit:
        if isinstance(op, stim.CircuitRepeatBlock):
            if current:
                yield current
                current = []
            yield op
        elif op.name == "TICK":
            yield current
            current = []
        else:
            current.extend(_split_targets_if_needed(op, immune_qubits=immune_qubits))
    if current:
        yield current


def _measure_basis(op: stim.CircuitInstruction) -> Optional[str]:
    """Return the measurement basis string for an instruction, if known."""
    basis = OP_MEASURE_BASES.get(op.name)
    if basis == "":
        out = ""
        for target in op.targets_copy():
            if target.is_x_target:
                out += "X"
            elif target.is_y_target:
                out += "Y"
            elif target.is_z_target:
                out += "Z"
        return out
    return basis


def _normalize_pair_value(
    value: Union[
        float,
        Tuple[str, Union[float, List[float]]],
        Mapping[str, Union[str, float, List[float]]],
        Iterable[Tuple[str, Union[float, List[float]]]],
        None,
    ]
) -> List[Tuple[str, Union[float, List[float]]]]:
    """Coerce the two-qubit noise specification into a uniform list form."""
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        if value and isinstance(value[0], tuple):
            return [(str(name), arg) for name, arg in value]  # type: ignore[arg-type]
        if value and isinstance(value[0], Mapping):
            return [
                (str(item["name"]), item.get("arg", 0.0))  # type: ignore[index]
                for item in value  # type: ignore[assignment]
            ]
        if len(value) == 2 and isinstance(value[0], str):
            return [(str(value[0]), value[1])]  # type: ignore[index]
    if isinstance(value, Mapping):
        name = str(value["name"])
        arg = value.get("arg", 0.0)
        return [(name, arg)]
    return [("DEPOLARIZE2", float(value))]
