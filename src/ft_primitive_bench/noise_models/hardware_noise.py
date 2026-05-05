"""Unified hardware-aware Stim noise engine."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from typing import (
    Callable,
    Dict,
    FrozenSet,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import stim

from ._noise_core import (
    ANNOTATION,
    CLIFFORD_1Q,
    CLIFFORD_2Q,
    COLLAPSING_OPS,
    JUST_MEASURE_1Q,
    JUST_RESET_1Q,
    MEASURE_RESET_1Q,
    MEASUREMENT_OPS,
    MPP,
    OP_TYPES,
    RESET_OPS,
    _iter_split_op_moments,
    _measure_basis,
    idle_pauli_channel_from_T1T2,
    occurs_in_classical_control_system,
)

NoiseArg = Union[float, Tuple[float, ...]]
SpatialIndex = Union[int, Tuple[int, int]]
BasisFlipInput = Union[float, Sequence[float], Mapping[str, float], None]
IdleChannelInput = Union[
    float,
    Sequence[float],
    Tuple[str, Union[float, Sequence[float]]],
    Mapping[str, object],
    str,
    Callable[[float], "RawNoiseInstruction"],
    None,
]
NamedIdleChannelInput = Union[IdleChannelInput, Mapping[str, IdleChannelInput]]
RawNoiseInstruction = Union[
    float,
    Sequence[float],
    Tuple[str, Union[float, Sequence[float]]],
    Mapping[str, object],
    str,
]
RawNoiseInstructionCollection = Union[
    RawNoiseInstruction,
    Iterable[Tuple[str, Union[float, Sequence[float]]]],
    Iterable[Mapping[str, object]],
    None,
]

_TWO_QUBIT_NOISE_NAMES = {"DEPOLARIZE2", "PAULI_CHANNEL_2"}
_PAULI_2Q_ORDER = (
    "IX",
    "IY",
    "IZ",
    "XI",
    "XX",
    "XY",
    "XZ",
    "YI",
    "YX",
    "YY",
    "YZ",
    "ZI",
    "ZX",
    "ZY",
    "ZZ",
)


@dataclass(frozen=True)
class GateDurations:
    """Wall-clock durations with optional per-gate and per-pair overrides."""

    one_qubit: float
    two_qubit: float
    measurement: float
    by_name: Mapping[str, float] = field(default_factory=dict)
    by_pair: Mapping[Tuple[int, int], float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for label, value in (
            ("one_qubit", self.one_qubit),
            ("two_qubit", self.two_qubit),
            ("measurement", self.measurement),
        ):
            if value < 0:
                raise ValueError(f"{label} duration must be non-negative, got {value}")
        object.__setattr__(self, "by_name", {str(k): float(v) for k, v in self.by_name.items()})
        object.__setattr__(
            self,
            "by_pair",
            {_normalize_pair_key(pair): float(duration) for pair, duration in self.by_pair.items()},
        )

    def for_operation(self, name: str, *, pair: Optional[Tuple[int, int]] = None) -> float:
        if pair is not None and pair in self.by_pair:
            return self.by_pair[pair]
        if name in self.by_name:
            return self.by_name[name]
        if name in MEASUREMENT_OPS or name in RESET_OPS:
            return self.measurement
        if OP_TYPES.get(name) == CLIFFORD_2Q:
            return self.two_qubit
        return self.one_qubit


_DEFAULT_CUSTOM_GATE_DURATIONS = GateDurations(
    one_qubit=40e-9, two_qubit=120e-9, measurement=200e-9
)


@dataclass(frozen=True)
class NoiseInstruction:
    name: str
    arg: NoiseArg


@dataclass(frozen=True)
class NoiseRule:
    before: Tuple[NoiseInstruction, ...] = ()
    after: Tuple[NoiseInstruction, ...] = ()
    flip_result: Optional[float] = None
    duration: Optional[float] = None


@dataclass(frozen=True)
class TwoQubitNoiseSpec:
    instructions: Tuple[NoiseInstruction, ...] = ()
    duration: Optional[float] = None


@dataclass(frozen=True)
class PerQubitRates:
    T1: float
    T2: float
    after_1q: Optional[RawNoiseInstructionCollection] = None
    after_2q: Optional[RawNoiseInstructionCollection] = None
    after_reset: BasisFlipInput = None
    before_measure: BasisFlipInput = 0.0
    idle_pauli_channel: Optional[RawNoiseInstruction] = None

    def __post_init__(self) -> None:
        if self.T1 <= 0 or self.T2 <= 0:
            raise ValueError("T1 and T2 must be positive numbers.")


@dataclass(frozen=True)
class NoiseParams:
    T1: Optional[float] = None
    T2: Optional[float] = None
    gate_error: Mapping[str, RawNoiseInstructionCollection] = field(default_factory=dict)
    spam_error: Mapping[str, BasisFlipInput] = field(default_factory=dict)
    durations: Mapping[str, float] = field(default_factory=dict)
    idle_error: NamedIdleChannelInput = None

    def __init__(
        self,
        T1: Optional[float] = None,
        T2: Optional[float] = None,
        gate_error: Optional[Mapping[str, RawNoiseInstructionCollection]] = None,
        spam_error: Optional[Mapping[str, BasisFlipInput]] = None,
        durations: Optional[Mapping[str, float]] = None,
        idle_error: NamedIdleChannelInput = None,
    ) -> None:
        if T1 is not None and T1 <= 0:
            raise ValueError("T1 must be positive when provided.")
        if T2 is not None and T2 <= 0:
            raise ValueError("T2 must be positive when provided.")
        object.__setattr__(self, "T1", T1)
        object.__setattr__(self, "T2", T2)
        object.__setattr__(self, "gate_error", _normalize_named_mapping(gate_error or {}))
        object.__setattr__(self, "spam_error", _normalize_spam_error_mapping(spam_error))
        object.__setattr__(self, "durations", _normalize_duration_mapping(durations or {}))
        object.__setattr__(self, "idle_error", idle_error)

    def merged_with(self, override: Optional["NoiseParams"]) -> "NoiseParams":
        if override is None:
            return self
        return NoiseParams(
            T1=self.T1 if override.T1 is None else override.T1,
            T2=self.T2 if override.T2 is None else override.T2,
            gate_error={**self.gate_error, **override.gate_error},
            spam_error=_merge_spam_error_mappings(self.spam_error, override.spam_error),
            durations={**self.durations, **override.durations},
            idle_error=_merge_idle_channel_inputs(self.idle_error, override.idle_error),
        )


@dataclass(frozen=True)
class RoundNoiseParams:
    qubits: Optional[NoiseParams] = None
    pairs: Optional[NoiseParams] = None


@dataclass(frozen=True)
class RoundIndexedNoiseSpec:
    global_noise: Optional[NoiseParams] = None
    qubit_overrides: Mapping[int, NoiseParams] = field(default_factory=dict)
    pair_overrides: Mapping[Tuple[int, int], NoiseParams] = field(default_factory=dict)
    round_overrides: Mapping[int, RoundNoiseParams] = field(default_factory=dict)
    qubit_round_overrides: Mapping[Tuple[int, int], NoiseParams] = field(default_factory=dict)
    pair_round_overrides: Mapping[Tuple[Tuple[int, int], int], NoiseParams] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "qubit_overrides",
            {int(qubit): value for qubit, value in self.qubit_overrides.items()},
        )
        object.__setattr__(
            self,
            "pair_overrides",
            {_normalize_pair_key(pair): value for pair, value in self.pair_overrides.items()},
        )
        object.__setattr__(
            self,
            "round_overrides",
            {int(round_idx): value for round_idx, value in self.round_overrides.items()},
        )
        object.__setattr__(
            self,
            "qubit_round_overrides",
            {
                (int(qubit), int(round_idx)): value
                for (qubit, round_idx), value in self.qubit_round_overrides.items()
            },
        )
        object.__setattr__(
            self,
            "pair_round_overrides",
            {
                (_normalize_pair_key(pair), int(round_idx)): value
                for (pair, round_idx), value in self.pair_round_overrides.items()
            },
        )

    @property
    def has_round_overrides(self) -> bool:
        return bool(self.round_overrides or self.qubit_round_overrides or self.pair_round_overrides)


@dataclass(frozen=True)
class CompiledOp:
    instruction: stim.CircuitInstruction
    op_type: str
    targets: Tuple[int, ...]
    pair_targets: Tuple[Tuple[int, int], ...]
    measurement_basis: Optional[str]
    reset_basis: Optional[str]
    duration: float = 0.0
    round_idx: Optional[int] = None


@dataclass(frozen=True)
class CompiledMoment:
    ops: Tuple[CompiledOp, ...]
    duration: float
    collapse_qubits: FrozenSet[int]
    clifford_qubits: FrozenSet[int]
    explicit_round: Optional[int] = None
    detector_round_advance: Optional[int] = None


@dataclass(frozen=True)
class CompiledRepeatBlock:
    repeat_count: int
    body: "CompiledCircuit"


CompiledBlock = Union[CompiledMoment, CompiledRepeatBlock]


@dataclass(frozen=True)
class CompiledCircuit:
    blocks: Tuple[CompiledBlock, ...]
    system_qubits: FrozenSet[int]
    immune_qubits: FrozenSet[int]


def _normalize_pair_key(pair: Tuple[int, int]) -> Tuple[int, int]:
    if len(pair) != 2:
        raise ValueError("Two-qubit pair keys must contain exactly two qubit indices.")
    a, b = int(pair[0]), int(pair[1])
    return (a, b) if a <= b else (b, a)


def _normalize_noise_arg(arg: object) -> NoiseArg:
    if isinstance(arg, (list, tuple)):
        return tuple(float(v) for v in arg)
    return float(arg)


def _normalize_instruction(spec: RawNoiseInstruction, *, default_name: str) -> NoiseInstruction:
    if isinstance(spec, Mapping):
        lower = {str(k).lower(): v for k, v in spec.items()}
        if {"px", "py", "pz"} <= lower.keys():
            return NoiseInstruction(
                name="PAULI_CHANNEL_1",
                arg=(float(lower["px"]), float(lower["py"]), float(lower["pz"])),
            )
        pauli_2q_terms = {term for term in _PAULI_2Q_ORDER if term.lower() in lower}
        if pauli_2q_terms:
            return NoiseInstruction(
                name="PAULI_CHANNEL_2",
                arg=tuple(float(lower.get(term.lower(), 0.0)) for term in _PAULI_2Q_ORDER),
            )
        return NoiseInstruction(
            name=str(spec.get("name", default_name)),
            arg=_normalize_noise_arg(spec.get("arg", 0.0)),
        )
    if isinstance(spec, (list, tuple)):
        if spec and isinstance(spec[0], str):
            arg = spec[1] if len(spec) > 1 else 0.0
            return NoiseInstruction(name=str(spec[0]), arg=_normalize_noise_arg(arg))
        return NoiseInstruction(name=default_name, arg=_normalize_noise_arg(spec))
    if isinstance(spec, str):
        return NoiseInstruction(name=spec, arg=0.0)
    return NoiseInstruction(name=default_name, arg=float(spec))


def _normalize_instruction_collection(
    value: RawNoiseInstructionCollection,
    *,
    default_name: str,
) -> Tuple[NoiseInstruction, ...]:
    if value is None:
        return ()
    if isinstance(value, Mapping):
        keys = {str(k).lower() for k in value}
        if (
            "name" in value
            or "arg" in value
            or {"px", "py", "pz"} <= keys
            or any(term.lower() in keys for term in _PAULI_2Q_ORDER)
        ):
            return (_normalize_instruction(value, default_name=default_name),)
    if isinstance(value, str):
        return (_normalize_instruction(value, default_name=default_name),)
    if isinstance(value, tuple) and value and isinstance(value[0], str):
        return (_normalize_instruction(value, default_name=default_name),)
    if isinstance(value, list) and value and isinstance(value[0], str):
        return (_normalize_instruction(value, default_name=default_name),)
    if isinstance(value, (int, float)):
        return (_normalize_instruction(value, default_name=default_name),)
    return tuple(_normalize_instruction(item, default_name=default_name) for item in value)  # type: ignore[arg-type]


def _normalize_rule(value: Union[NoiseRule, Mapping[str, object]]) -> NoiseRule:
    if isinstance(value, NoiseRule):
        return value
    before = _normalize_instruction_collection(value.get("before"), default_name="DEPOLARIZE1")
    after = _normalize_instruction_collection(value.get("after"), default_name="DEPOLARIZE1")
    flip_result = value.get("flip_result")
    duration = value.get("duration")
    return NoiseRule(
        before=before,
        after=after,
        flip_result=None if flip_result is None else float(flip_result),
        duration=None if duration is None else float(duration),
    )


def _basis_flip_map(value: BasisFlipInput) -> Dict[str, float]:
    if value is None:
        return {"X": 0.0, "Y": 0.0, "Z": 0.0}
    if isinstance(value, Mapping):
        out = {str(axis).upper(): float(prob) for axis, prob in value.items()}
        z = out.get("Z", 0.0)
        return {"Z": z, "X": out.get("X", z), "Y": out.get("Y", z)}
    if isinstance(value, (list, tuple)):
        if len(value) >= 3:
            return {"Z": float(value[0]), "X": float(value[1]), "Y": float(value[2])}
        if len(value) >= 2:
            return {"Z": float(value[0]), "X": float(value[1]), "Y": float(value[0])}
        if len(value) == 1:
            prob = float(value[0])
            return {"X": prob, "Y": prob, "Z": prob}
    prob = float(value)
    return {"X": prob, "Y": prob, "Z": prob}


def _normalize_named_mapping(value: Mapping[str, object]) -> Dict[str, object]:
    normalized: Dict[str, object] = {}
    for key, item in value.items():
        name = str(key).upper()
        normalized["*" if name == "DEFAULT" else name] = item
    return normalized


def _normalize_duration_mapping(value: Mapping[str, float]) -> Dict[str, float]:
    return {key: float(item) for key, item in _normalize_named_mapping(value).items()}


def _normalize_spam_category(value: str) -> str:
    normalized = str(value).strip().upper()
    aliases = {
        "M": "MEASURE",
        "MEAS": "MEASURE",
        "MEASURE": "MEASURE",
        "MEASUREMENT": "MEASURE",
        "READOUT": "MEASURE",
        "R": "RESET",
        "RESET": "RESET",
        "PREP": "RESET",
        "PREPARE": "RESET",
        "PREPARATION": "RESET",
    }
    if normalized not in aliases:
        raise ValueError(f"Unsupported SPAM category '{value}'. Use RESET or MEASURE.")
    return aliases[normalized]


def _normalize_spam_error_mapping(
    value: Optional[Mapping[str, BasisFlipInput]],
) -> Dict[str, Dict[str, float]]:
    if not value:
        return {}
    raw = dict(value)
    category_keys = {
        key for key in raw if str(key).strip().upper() not in {"X", "Y", "Z", "*", "DEFAULT"}
    }
    if not category_keys:
        basis_map = _basis_flip_map(raw)
        return {"RESET": dict(basis_map), "MEASURE": dict(basis_map)}
    return {_normalize_spam_category(str(key)): _basis_flip_map(item) for key, item in raw.items()}


def _merge_spam_error_mappings(
    base: Mapping[str, BasisFlipInput],
    override: Mapping[str, BasisFlipInput],
) -> Dict[str, Dict[str, float]]:
    merged = _normalize_spam_error_mapping(base)
    for category, value in _normalize_spam_error_mapping(override).items():
        current = merged.get(category)
        merged[category] = (
            _basis_flip_map(value)
            if current is None
            else _basis_flip_map(_merge_basis_flip_inputs(current, value))
        )
    return merged


def _normalize_idle_channel_mapping(value: NamedIdleChannelInput) -> Dict[str, IdleChannelInput]:
    if value is None:
        return {}
    if callable(value):
        return {"*": value}
    if isinstance(value, Mapping):
        keys = {str(key).lower() for key in value}
        if (
            "name" in keys
            or "arg" in keys
            or {"px", "py", "pz"} <= keys
            or any(term.lower() in keys for term in _PAULI_2Q_ORDER)
        ):
            return {"*": value}
        return {
            str(key).upper() if str(key).upper() != "DEFAULT" else "*": item
            for key, item in value.items()
        }
    return {"*": value}


def _merge_basis_flip_inputs(base: BasisFlipInput, override: BasisFlipInput) -> BasisFlipInput:
    if override is None:
        return base
    merged = _basis_flip_map(base)
    merged.update(_basis_flip_map(override))
    return merged


def _merge_idle_channel_inputs(
    base: NamedIdleChannelInput, override: NamedIdleChannelInput
) -> NamedIdleChannelInput:
    if override is None:
        return base
    if base is None:
        return override
    merged = _normalize_idle_channel_mapping(base)
    merged.update(_normalize_idle_channel_mapping(override))
    return merged


def _lookup_named_value(mapping: Mapping[str, object], name: str) -> Optional[object]:
    upper = str(name).upper()
    if upper in mapping:
        return mapping[upper]
    if "*" in mapping:
        return mapping["*"]
    return None


def _lookup_frozen_named_value(
    items: Tuple[Tuple[str, object], ...],
    name: str,
) -> Optional[object]:
    upper = str(name).upper()
    fallback: Optional[object] = None
    for key, value in items:
        if key == upper:
            return value
        if key == "*":
            fallback = value
    return fallback


def _freeze_instruction_mapping(
    mapping: Mapping[str, object],
    *,
    default_name: str,
) -> Tuple[Tuple[str, Tuple[NoiseInstruction, ...]], ...]:
    return tuple(
        (
            str(name).upper(),
            _normalize_instruction_collection(value, default_name=default_name),
        )
        for name, value in sorted(mapping.items())
    )


@dataclass(frozen=True)
class _ResolvedIdleSpec:
    instruction: Optional[NoiseInstruction] = None
    idle_callable: Optional[Callable[[float], RawNoiseInstruction]] = None

    def instruction_for(self, idle_duration: float) -> Optional[NoiseInstruction]:
        if self.idle_callable is not None:
            return _normalize_instruction(
                self.idle_callable(idle_duration), default_name="DEPOLARIZE1"
            )
        return self.instruction


def _resolve_idle_spec(value: IdleChannelInput) -> _ResolvedIdleSpec:
    if callable(value):
        return _ResolvedIdleSpec(idle_callable=value)
    if value is None:
        return _ResolvedIdleSpec()
    return _ResolvedIdleSpec(instruction=_normalize_instruction(value, default_name="DEPOLARIZE1"))


def _freeze_idle_mapping_items(
    value: NamedIdleChannelInput,
) -> Tuple[Tuple[str, _ResolvedIdleSpec], ...]:
    return tuple(
        (str(name).upper(), _resolve_idle_spec(item))
        for name, item in sorted(_normalize_idle_channel_mapping(value).items())
    )


def _idle_spec_for_op(
    items: Tuple[Tuple[str, _ResolvedIdleSpec], ...],
    op_name: str,
    *,
    allow_wildcard: bool,
) -> Optional[_ResolvedIdleSpec]:
    upper = str(op_name).upper()
    fallback: Optional[_ResolvedIdleSpec] = None
    for key, value in items:
        if key == upper:
            return value
        if allow_wildcard and key == "*":
            fallback = value
    return fallback


def _noise_magnitude(arg: NoiseArg) -> float:
    return float(sum(arg)) if isinstance(arg, tuple) else float(arg)


def _freeze_duration_mapping_items(mapping: Mapping[str, float]) -> Tuple[Tuple[str, float], ...]:
    return tuple((str(name).upper(), float(value)) for name, value in sorted(mapping.items()))


def _freeze_basis_flip_map(value: BasisFlipInput) -> Tuple[Tuple[str, float], ...]:
    return tuple(sorted((basis, float(prob)) for basis, prob in _basis_flip_map(value).items()))


@dataclass(frozen=True)
class _ResolvedQubitParams:
    T1: Optional[float]
    T2: Optional[float]
    gate_error: Tuple[Tuple[str, Tuple[NoiseInstruction, ...]], ...]
    reset_spam_error: Tuple[Tuple[str, float], ...]
    measure_spam_error: Tuple[Tuple[str, float], ...]
    durations: Tuple[Tuple[str, float], ...]
    idle_error: Tuple[Tuple[str, _ResolvedIdleSpec], ...]
    idle_uses_t1t2: bool

    def gate_instructions_for(self, name: str) -> Tuple[NoiseInstruction, ...]:
        value = _lookup_frozen_named_value(self.gate_error, name)
        return () if value is None else value  # type: ignore[return-value]

    def duration_for(self, name: str) -> Optional[float]:
        value = _lookup_frozen_named_value(self.durations, name)
        return None if value is None else float(value)

    def reset_probability(self, basis: str) -> float:
        reset_map = dict(self.reset_spam_error)
        return float(reset_map.get(basis, reset_map.get("Z", 0.0)))

    def measure_probability(self, basis: str) -> float:
        measure_map = dict(self.measure_spam_error)
        return float(measure_map.get(basis, measure_map.get("Z", 0.0)))

    def idle_instruction_for(
        self, op_name: str, idle_duration: float
    ) -> Optional[NoiseInstruction]:
        allow_wildcard = op_name not in MEASUREMENT_OPS and op_name not in RESET_OPS
        idle_spec = _idle_spec_for_op(self.idle_error, op_name, allow_wildcard=allow_wildcard)
        if idle_spec is not None:
            return idle_spec.instruction_for(idle_duration)
        return self.derived_idle_instruction(idle_duration)

    def derived_idle_instruction(self, idle_duration: float) -> Optional[NoiseInstruction]:
        if self.idle_uses_t1t2 and self.T1 is not None and self.T2 is not None:
            return NoiseInstruction(
                name="PAULI_CHANNEL_1",
                arg=idle_pauli_channel_from_T1T2(idle_duration, self.T1, self.T2),
            )
        return None


@dataclass(frozen=True)
class _ResolvedPairParams:
    gate_error: Tuple[Tuple[str, Tuple[NoiseInstruction, ...]], ...]
    durations: Tuple[Tuple[str, float], ...]
    idle_error: Tuple[Tuple[str, _ResolvedIdleSpec], ...]

    def gate_instructions_for(self, name: str) -> Tuple[NoiseInstruction, ...]:
        value = _lookup_frozen_named_value(self.gate_error, name)
        return () if value is None else value  # type: ignore[return-value]

    def duration_for(self, name: str) -> Optional[float]:
        value = _lookup_frozen_named_value(self.durations, name)
        return None if value is None else float(value)

    def idle_instruction_for(
        self, op_name: str, idle_duration: float
    ) -> Optional[NoiseInstruction]:
        idle_spec = _idle_spec_for_op(self.idle_error, op_name, allow_wildcard=True)
        return None if idle_spec is None else idle_spec.instruction_for(idle_duration)


@dataclass(frozen=True)
class _CompiledNoiseTables:
    mode: str
    qubit_params_pool: Tuple[_ResolvedQubitParams, ...]
    pair_params_pool: Tuple[_ResolvedPairParams, ...]
    default_qubit_param_id: int
    default_pair_param_id: int
    qubit_param_ids: Mapping[int, int] = field(default_factory=dict)
    pair_param_ids: Mapping[Tuple[int, int], int] = field(default_factory=dict)
    round_qubit_default_param_ids: Mapping[int, int] = field(default_factory=dict)
    round_pair_default_param_ids: Mapping[int, int] = field(default_factory=dict)
    round_qubit_param_ids: Mapping[int, Mapping[int, int]] = field(default_factory=dict)
    round_pair_param_ids: Mapping[int, Mapping[Tuple[int, int], int]] = field(default_factory=dict)

    @property
    def uses_round_indices(self) -> bool:
        return self.mode == "round_overrides"


def _resolve_qubit_runtime_params(params: NoiseParams) -> _ResolvedQubitParams:
    idle_uses_t1t2 = params.T1 is not None and params.T2 is not None
    return _ResolvedQubitParams(
        T1=params.T1,
        T2=params.T2,
        gate_error=_freeze_instruction_mapping(params.gate_error, default_name="DEPOLARIZE1"),
        reset_spam_error=_freeze_basis_flip_map(params.spam_error.get("RESET")),
        measure_spam_error=_freeze_basis_flip_map(params.spam_error.get("MEASURE")),
        durations=_freeze_duration_mapping_items(params.durations),
        idle_error=_freeze_idle_mapping_items(params.idle_error),
        idle_uses_t1t2=idle_uses_t1t2,
    )


def _resolve_pair_runtime_params(params: NoiseParams) -> _ResolvedPairParams:
    return _ResolvedPairParams(
        gate_error=_freeze_instruction_mapping(params.gate_error, default_name="DEPOLARIZE2"),
        durations=_freeze_duration_mapping_items(params.durations),
        idle_error=_freeze_idle_mapping_items(params.idle_error),
    )


def _compile_noise_tables(spec: RoundIndexedNoiseSpec) -> _CompiledNoiseTables:
    global_noise = spec.global_noise or NoiseParams()

    qubit_pool: List[_ResolvedQubitParams] = []
    qubit_index: Dict[_ResolvedQubitParams, int] = {}

    def intern_qubit(params: NoiseParams) -> int:
        resolved = _resolve_qubit_runtime_params(params)
        existing = qubit_index.get(resolved)
        if existing is not None:
            return existing
        idx = len(qubit_pool)
        qubit_pool.append(resolved)
        qubit_index[resolved] = idx
        return idx

    pair_pool: List[_ResolvedPairParams] = []
    pair_index: Dict[_ResolvedPairParams, int] = {}

    def intern_pair(params: NoiseParams) -> int:
        resolved = _resolve_pair_runtime_params(params)
        existing = pair_index.get(resolved)
        if existing is not None:
            return existing
        idx = len(pair_pool)
        pair_pool.append(resolved)
        pair_index[resolved] = idx
        return idx

    default_qubit_param_id = intern_qubit(global_noise)
    default_pair_param_id = intern_pair(global_noise)

    qubit_param_ids = {
        int(qubit): intern_qubit(global_noise.merged_with(override))
        for qubit, override in spec.qubit_overrides.items()
    }
    pair_param_ids = {
        _normalize_pair_key(pair): intern_pair(global_noise.merged_with(override))
        for pair, override in spec.pair_overrides.items()
    }

    round_qubit_default_param_ids: Dict[int, int] = {}
    round_qubit_param_ids: Dict[int, Dict[int, int]] = defaultdict(dict)
    for round_idx, override in spec.round_overrides.items():
        if override.qubits is not None:
            round_qubit_default_param_ids[int(round_idx)] = intern_qubit(
                global_noise.merged_with(override.qubits)
            )
            for qubit, qubit_override in spec.qubit_overrides.items():
                round_qubit_param_ids[int(round_idx)][int(qubit)] = intern_qubit(
                    global_noise.merged_with(qubit_override).merged_with(override.qubits)
                )
    for (qubit, round_idx), override in spec.qubit_round_overrides.items():
        round_base = spec.round_overrides.get(int(round_idx))
        base = global_noise
        base = base.merged_with(spec.qubit_overrides.get(int(qubit)))
        if round_base is not None and round_base.qubits is not None:
            base = base.merged_with(round_base.qubits)
        round_qubit_param_ids[int(round_idx)][int(qubit)] = intern_qubit(base.merged_with(override))

    round_pair_default_param_ids: Dict[int, int] = {}
    round_pair_param_ids: Dict[int, Dict[Tuple[int, int], int]] = defaultdict(dict)
    for round_idx, override in spec.round_overrides.items():
        if override.pairs is not None:
            round_pair_default_param_ids[int(round_idx)] = intern_pair(
                global_noise.merged_with(override.pairs)
            )
            for pair, pair_override in spec.pair_overrides.items():
                normalized_pair = _normalize_pair_key(pair)
                round_pair_param_ids[int(round_idx)][normalized_pair] = intern_pair(
                    global_noise.merged_with(pair_override).merged_with(override.pairs)
                )
    for (pair, round_idx), override in spec.pair_round_overrides.items():
        normalized_pair = _normalize_pair_key(pair)
        round_base = spec.round_overrides.get(int(round_idx))
        base = global_noise
        base = base.merged_with(spec.pair_overrides.get(normalized_pair))
        if round_base is not None and round_base.pairs is not None:
            base = base.merged_with(round_base.pairs)
        round_pair_param_ids[int(round_idx)][normalized_pair] = intern_pair(
            base.merged_with(override)
        )

    if (
        round_qubit_default_param_ids
        or round_pair_default_param_ids
        or round_qubit_param_ids
        or round_pair_param_ids
    ):
        mode = "round_overrides"
    elif qubit_param_ids or pair_param_ids:
        mode = "static_overrides"
    else:
        mode = "global_only"

    return _CompiledNoiseTables(
        mode=mode,
        qubit_params_pool=tuple(qubit_pool),
        pair_params_pool=tuple(pair_pool),
        default_qubit_param_id=default_qubit_param_id,
        default_pair_param_id=default_pair_param_id,
        qubit_param_ids=dict(qubit_param_ids),
        pair_param_ids=dict(pair_param_ids),
        round_qubit_default_param_ids=dict(round_qubit_default_param_ids),
        round_pair_default_param_ids=dict(round_pair_default_param_ids),
        round_qubit_param_ids={
            round_idx: dict(value) for round_idx, value in round_qubit_param_ids.items()
        },
        round_pair_param_ids={
            round_idx: dict(value) for round_idx, value in round_pair_param_ids.items()
        },
    )


class _CompiledNoiseEngine:
    """Private hardware-aware circuit-noise engine."""

    def __init__(
        self,
        *,
        gate_durations: GateDurations,
        default_qubit_rates: Optional[PerQubitRates] = None,
        qubit_rate_overrides: Optional[
            Mapping[int, Union[PerQubitRates, Mapping[str, object]]]
        ] = None,
        pair_rate_overrides: Optional[
            Mapping[Tuple[int, int], Union[TwoQubitNoiseSpec, Mapping[str, object]]]
        ] = None,
        two_qubit_noise_overrides: Optional[
            Mapping[
                Tuple[int, int],
                Union[RawNoiseInstructionCollection, Mapping[str, object], TwoQubitNoiseSpec, None],
            ]
        ] = None,
        gate_rules: Optional[Mapping[str, Union[NoiseRule, Mapping[str, object]]]] = None,
        measure_rules: Optional[Mapping[str, Union[NoiseRule, Mapping[str, object]]]] = None,
        reset_rules: Optional[Mapping[str, Union[NoiseRule, Mapping[str, object]]]] = None,
        any_clifford_1q_rule: Optional[Union[NoiseRule, Mapping[str, object]]] = None,
        any_clifford_2q_rule: Optional[Union[NoiseRule, Mapping[str, object]]] = None,
        verbose: bool = False,
    ) -> None:
        self._gate_durations = gate_durations
        self._default_qubit_rates = default_qubit_rates
        self._verbose = verbose

        self._qubit_rates: Dict[int, PerQubitRates] = {}
        if qubit_rate_overrides:
            for raw_qubit, raw_rates in qubit_rate_overrides.items():
                qubit = int(raw_qubit)
                if isinstance(raw_rates, PerQubitRates):
                    self._qubit_rates[qubit] = raw_rates
                else:
                    self._qubit_rates[qubit] = self._coerce_rates(
                        raw_rates, fallback=default_qubit_rates
                    )
        elif default_qubit_rates is None:
            raise ValueError(
                "_CompiledNoiseEngine requires default_qubit_rates or explicit qubit_rate_overrides."
            )

        self._default_measure_flip = (
            _basis_flip_map(default_qubit_rates.before_measure)
            if default_qubit_rates is not None
            else None
        )
        self._measure_flip_by_qubit = {
            q: _basis_flip_map(rates.before_measure) for q, rates in self._qubit_rates.items()
        }
        self._default_reset_flip = (
            _basis_flip_map(default_qubit_rates.after_reset)
            if default_qubit_rates is not None
            else None
        )
        self._reset_flip_by_qubit = {
            q: _basis_flip_map(rates.after_reset) for q, rates in self._qubit_rates.items()
        }

        self._default_after_1q = (
            _normalize_instruction_collection(
                default_qubit_rates.after_1q, default_name="DEPOLARIZE1"
            )
            if default_qubit_rates is not None
            else ()
        )
        self._after_1q_by_qubit = {
            q: _normalize_instruction_collection(rates.after_1q, default_name="DEPOLARIZE1")
            for q, rates in self._qubit_rates.items()
            if rates.after_1q is not None
        }
        self._default_idle_instruction = (
            _normalize_instruction(
                default_qubit_rates.idle_pauli_channel, default_name="PAULI_CHANNEL_1"
            )
            if default_qubit_rates is not None
            and default_qubit_rates.idle_pauli_channel is not None
            else None
        )
        self._idle_instruction_by_qubit = {
            q: _normalize_instruction(rates.idle_pauli_channel, default_name="PAULI_CHANNEL_1")
            for q, rates in self._qubit_rates.items()
            if rates.idle_pauli_channel is not None
        }

        default_pair_source = (
            default_qubit_rates.after_2q if default_qubit_rates is not None else None
        )
        self._default_pair_noise = _normalize_instruction_collection(
            default_pair_source, default_name="DEPOLARIZE2"
        )
        self._pair_noise_specs: Dict[Tuple[int, int], TwoQubitNoiseSpec] = {}
        self._load_pair_specs(pair_rate_overrides or {})
        if two_qubit_noise_overrides:
            self._load_pair_specs(two_qubit_noise_overrides)

        self._gate_rules = {
            str(name): _normalize_rule(rule) for name, rule in (gate_rules or {}).items()
        }
        self._measure_rules = {
            str(name).upper(): _normalize_rule(rule) for name, rule in (measure_rules or {}).items()
        }
        self._reset_rules = {
            str(name).upper(): _normalize_rule(rule) for name, rule in (reset_rules or {}).items()
        }
        self._any_clifford_1q_rule = (
            None if any_clifford_1q_rule is None else _normalize_rule(any_clifford_1q_rule)
        )
        self._any_clifford_2q_rule = (
            None if any_clifford_2q_rule is None else _normalize_rule(any_clifford_2q_rule)
        )

    @classmethod
    def from_scalar_rates(
        cls,
        *,
        gate_durations: GateDurations,
        T1: float,
        T2: float,
        after_1q: Optional[RawNoiseInstructionCollection] = None,
        after_2q: Optional[RawNoiseInstructionCollection] = None,
        after_reset: BasisFlipInput = None,
        before_measure: BasisFlipInput = 0.0,
        idle_pauli_channel: Optional[RawNoiseInstruction] = None,
        qubit_rate_overrides: Optional[Mapping[int, Mapping[str, object]]] = None,
        pair_rate_overrides: Optional[Mapping[Tuple[int, int], Mapping[str, object]]] = None,
        two_qubit_noise_overrides: Optional[
            Mapping[
                Tuple[int, int],
                Union[RawNoiseInstructionCollection, Mapping[str, object], TwoQubitNoiseSpec, None],
            ]
        ] = None,
        gate_rules: Optional[Mapping[str, Union[NoiseRule, Mapping[str, object]]]] = None,
        measure_rules: Optional[Mapping[str, Union[NoiseRule, Mapping[str, object]]]] = None,
        reset_rules: Optional[Mapping[str, Union[NoiseRule, Mapping[str, object]]]] = None,
        any_clifford_1q_rule: Optional[Union[NoiseRule, Mapping[str, object]]] = None,
        any_clifford_2q_rule: Optional[Union[NoiseRule, Mapping[str, object]]] = None,
        verbose: bool = False,
    ) -> "_CompiledNoiseEngine":
        default_rates = PerQubitRates(
            T1=T1,
            T2=T2,
            after_1q=after_1q,
            after_2q=after_2q,
            after_reset=after_reset,
            before_measure=before_measure,
            idle_pauli_channel=idle_pauli_channel,
        )
        overrides: Dict[int, PerQubitRates] = {}
        if qubit_rate_overrides:
            for qubit, raw_rates in qubit_rate_overrides.items():
                overrides[int(qubit)] = cls._coerce_rates(raw_rates, fallback=default_rates)
        return cls(
            gate_durations=gate_durations,
            default_qubit_rates=default_rates,
            qubit_rate_overrides=overrides if overrides else None,
            pair_rate_overrides=pair_rate_overrides,
            two_qubit_noise_overrides=two_qubit_noise_overrides,
            gate_rules=gate_rules,
            measure_rules=measure_rules,
            reset_rules=reset_rules,
            any_clifford_1q_rule=any_clifford_1q_rule,
            any_clifford_2q_rule=any_clifford_2q_rule,
            verbose=verbose,
        )

    @staticmethod
    def _coerce_rates(
        data: Mapping[str, object], *, fallback: Optional[PerQubitRates]
    ) -> PerQubitRates:
        def pick(key: str, default: Optional[object] = None) -> object:
            if key in data:
                return data[key]
            if fallback is not None:
                return getattr(fallback, key)
            if default is not None:
                return default
            raise KeyError(f"Missing required field '{key}' for PerQubitRates")

        return PerQubitRates(
            T1=float(pick("T1")),
            T2=float(pick("T2")),
            after_1q=pick("after_1q", None),
            after_2q=pick("after_2q", None),
            after_reset=pick("after_reset", None),
            before_measure=pick("before_measure", 0.0),
            idle_pauli_channel=pick("idle_pauli_channel", None),
        )

    def _load_pair_specs(
        self,
        raw_specs: Mapping[
            Tuple[int, int],
            Union[RawNoiseInstructionCollection, Mapping[str, object], TwoQubitNoiseSpec, None],
        ],
    ) -> None:
        for raw_pair, raw_value in raw_specs.items():
            pair = _normalize_pair_key(raw_pair)
            if isinstance(raw_value, TwoQubitNoiseSpec):
                self._pair_noise_specs[pair] = raw_value
                continue
            if isinstance(raw_value, Mapping) and "instructions" in raw_value:
                instructions = _normalize_instruction_collection(
                    raw_value.get("instructions"), default_name="DEPOLARIZE2"
                )
                duration_raw = raw_value.get("duration")
                self._pair_noise_specs[pair] = TwoQubitNoiseSpec(
                    instructions=instructions,
                    duration=None if duration_raw is None else float(duration_raw),
                )
                continue
            self._pair_noise_specs[pair] = TwoQubitNoiseSpec(
                instructions=_normalize_instruction_collection(
                    raw_value, default_name="DEPOLARIZE2"
                )
            )

    def _rates_for_qubit(self, qubit: int) -> PerQubitRates:
        if qubit in self._qubit_rates:
            return self._qubit_rates[qubit]
        if self._default_qubit_rates is not None:
            return self._default_qubit_rates
        raise ValueError(f"Qubit {qubit} is not specified in the noise model.")

    def compile_circuit(
        self,
        circuit: stim.Circuit,
        *,
        system_qubits: Optional[set[int]] = None,
        immune_qubits: Optional[set[int]] = None,
        moment_rounds: Optional[Sequence[int]] = None,
    ) -> CompiledCircuit:
        if system_qubits is None:
            system_qubits = set(range(circuit.num_qubits))
        if immune_qubits is None:
            immune_qubits = set()

        compiled_blocks: List[CompiledBlock] = []
        round_cursor = 0
        for moment in _iter_split_op_moments(circuit, immune_qubits=immune_qubits):
            if isinstance(moment, stim.CircuitRepeatBlock):
                if moment_rounds is not None:
                    raise ValueError(
                        "Explicit moment_rounds are not supported when REPEAT blocks are present."
                    )
                compiled_blocks.append(
                    CompiledRepeatBlock(
                        repeat_count=moment.repeat_count,
                        body=self.compile_circuit(
                            moment.body_copy(),
                            system_qubits=set(system_qubits),
                            immune_qubits=set(immune_qubits),
                        ),
                    )
                )
                continue

            explicit_round = None
            if moment_rounds is not None:
                if round_cursor >= len(moment_rounds):
                    raise ValueError("moment_rounds did not provide a round for every moment.")
                explicit_round = int(moment_rounds[round_cursor])
                round_cursor += 1
            compiled_blocks.append(self._compile_moment(moment, explicit_round=explicit_round))

        if moment_rounds is not None and round_cursor != len(moment_rounds):
            raise ValueError("moment_rounds contains unused entries.")

        return CompiledCircuit(
            blocks=tuple(compiled_blocks),
            system_qubits=frozenset(system_qubits),
            immune_qubits=frozenset(immune_qubits),
        )

    def _compile_moment(
        self, ops: List[stim.CircuitInstruction], *, explicit_round: Optional[int]
    ) -> CompiledMoment:
        compiled_ops: List[CompiledOp] = []
        collapse_qubits: List[int] = []
        clifford_qubits: List[int] = []
        durations: List[float] = []
        detector_round_advance: Optional[int] = None

        for op in ops:
            op_type = OP_TYPES.get(op.name)
            if op_type is None:
                raise ValueError(f"Unknown operation type for {op.name!r}")
            targets = tuple(
                target.value
                for target in op.targets_copy()
                if not target.is_combiner
                and (
                    target.is_qubit_target
                    or target.is_x_target
                    or target.is_y_target
                    or target.is_z_target
                )
            )
            pair_targets: Tuple[Tuple[int, int], ...] = ()
            if op_type == CLIFFORD_2Q and not occurs_in_classical_control_system(op):
                pair_targets = tuple(
                    _normalize_pair_key((targets[index], targets[index + 1]))
                    for index in range(0, len(targets), 2)
                )
            measurement_basis = (
                _measure_basis(op) if op_type in (JUST_MEASURE_1Q, MEASURE_RESET_1Q, MPP) else None
            )
            reset_basis = (
                self._reset_basis(op.name) if op_type in (JUST_RESET_1Q, MEASURE_RESET_1Q) else None
            )
            compiled_ops.append(
                CompiledOp(
                    instruction=op,
                    op_type=op_type,
                    targets=targets,
                    pair_targets=pair_targets,
                    measurement_basis=measurement_basis,
                    reset_basis=reset_basis,
                    round_idx=explicit_round,
                )
            )
            if op.name == "DETECTOR":
                detector_round = self._round_from_detector(op)
                if detector_round is not None:
                    advance = detector_round + 1
                    detector_round_advance = (
                        advance
                        if detector_round_advance is None
                        else max(detector_round_advance, advance)
                    )

            if op_type == ANNOTATION:
                continue
            if occurs_in_classical_control_system(op):
                clifford_qubits.extend(targets)
            elif op.name in COLLAPSING_OPS:
                collapse_qubits.extend(targets)
            else:
                clifford_qubits.extend(targets)
            duration = self._duration_for_operation(
                op_name=op.name,
                op_type=op_type,
                basis=measurement_basis,
                reset_basis=reset_basis,
                pair_targets=pair_targets,
                targets=targets,
                round_idx=explicit_round,
            )
            compiled_ops[-1] = CompiledOp(
                instruction=compiled_ops[-1].instruction,
                op_type=compiled_ops[-1].op_type,
                targets=compiled_ops[-1].targets,
                pair_targets=compiled_ops[-1].pair_targets,
                measurement_basis=compiled_ops[-1].measurement_basis,
                reset_basis=compiled_ops[-1].reset_basis,
                duration=duration,
                round_idx=explicit_round,
            )
            if duration > 0:
                durations.append(duration)

        usage = Counter(collapse_qubits + clifford_qubits)
        multi_use = {qubit for qubit, count in usage.items() if count != 1}
        if multi_use:
            moment = stim.Circuit()
            for op in ops:
                moment.append(op)
            raise ValueError(
                "Qubits were operated on multiple times within one moment.\n"
                f"Qubits: {sorted(multi_use)}\nMoment:\n{moment}"
            )

        return CompiledMoment(
            ops=tuple(compiled_ops),
            duration=max(durations) if durations else self._gate_durations.one_qubit,
            collapse_qubits=frozenset(collapse_qubits),
            clifford_qubits=frozenset(clifford_qubits),
            explicit_round=explicit_round,
            detector_round_advance=detector_round_advance,
        )

    def noisy_circuit(
        self,
        circuit: Union[stim.Circuit, CompiledCircuit],
        *,
        system_qubits: Optional[set[int]] = None,
        immune_qubits: Optional[set[int]] = None,
        moment_rounds: Optional[Sequence[int]] = None,
    ) -> stim.Circuit:
        compiled = (
            circuit
            if isinstance(circuit, CompiledCircuit)
            else self.compile_circuit(
                circuit,
                system_qubits=system_qubits,
                immune_qubits=immune_qubits,
                moment_rounds=moment_rounds,
            )
        )
        return self.apply_compiled(compiled)

    def apply_compiled(self, compiled: CompiledCircuit) -> stim.Circuit:
        out = stim.Circuit()
        current_round = 0
        for block in compiled.blocks:
            if not out:
                pass
            elif isinstance(block, CompiledRepeatBlock):
                pass
            elif isinstance(out[-1], stim.CircuitRepeatBlock):
                pass
            else:
                out.append("TICK", [], [])

            if isinstance(block, CompiledRepeatBlock):
                out.append(
                    stim.CircuitRepeatBlock(block.repeat_count, self.apply_compiled(block.body))
                )
            else:
                current_round = self._append_compiled_moment(
                    moment=block,
                    out=out,
                    system_qubits=set(compiled.system_qubits),
                    immune_qubits=set(compiled.immune_qubits),
                    current_round=current_round,
                )
        return out

    def _append_compiled_moment(
        self,
        *,
        moment: CompiledMoment,
        out: stim.Circuit,
        system_qubits: set[int],
        immune_qubits: set[int],
        current_round: int,
    ) -> int:
        round_idx = moment.explicit_round if moment.explicit_round is not None else current_round
        after_buf: defaultdict[Tuple[str, NoiseArg], stim.Circuit] = defaultdict(stim.Circuit)

        for entry in moment.ops:
            op = entry.instruction
            if entry.op_type == ANNOTATION:
                out.append(op)
                continue

            rules = self._rules_for_entry(entry)
            if occurs_in_classical_control_system(op):
                self._emit_rule_before(rules, entry, out, immune_qubits, round_idx)
                out.append(op)
                self._emit_rule_after(rules, entry, after_buf, immune_qubits, round_idx)
                self._queue_after_1q_calibration(entry, after_buf, immune_qubits, round_idx)
                continue

            if entry.op_type in (JUST_MEASURE_1Q, MEASURE_RESET_1Q, MPP):
                self._append_measurement(entry, rules, out, after_buf, immune_qubits, round_idx)
                continue

            if entry.op_type == JUST_RESET_1Q:
                self._emit_rule_before(rules, entry, out, immune_qubits, round_idx)
                out.append(op)
                self._emit_rule_after(rules, entry, after_buf, immune_qubits, round_idx)
                for qubit in entry.targets:
                    self._queue_reset_calibration(
                        qubit=qubit,
                        basis=entry.reset_basis or "Z",
                        after_buf=after_buf,
                        immune_qubits=immune_qubits,
                        round_idx=round_idx,
                    )
                continue

            self._emit_rule_before(rules, entry, out, immune_qubits, round_idx)
            out.append(op)
            self._emit_rule_after(rules, entry, after_buf, immune_qubits, round_idx)
            if entry.op_type == CLIFFORD_1Q:
                self._queue_after_1q_calibration(entry, after_buf, immune_qubits, round_idx)
            elif entry.op_type == CLIFFORD_2Q:
                self._queue_after_2q_calibration(entry, after_buf, immune_qubits, round_idx)

        for key in sorted(after_buf.keys(), key=lambda item: (item[0], item[1])):
            out += after_buf[key]

        self._append_idle_error(
            moment=moment,
            out=out,
            system_qubits=system_qubits,
            immune_qubits=immune_qubits,
            round_idx=round_idx,
        )

        if moment.explicit_round is not None:
            return (
                current_round
                if moment.detector_round_advance is None
                else max(current_round, moment.detector_round_advance)
            )
        if moment.detector_round_advance is not None:
            return max(round_idx, moment.detector_round_advance)
        return round_idx

    def _append_measurement(
        self,
        entry: CompiledOp,
        rules: Tuple[NoiseRule, ...],
        out: stim.Circuit,
        after_buf: defaultdict[Tuple[str, NoiseArg], stim.Circuit],
        immune_qubits: set[int],
        round_idx: int,
    ) -> None:
        self._emit_rule_before(rules, entry, out, immune_qubits, round_idx)
        op = entry.instruction
        if any(qubit in immune_qubits for qubit in entry.targets):
            out.append(op)
            return

        args = list(op.gate_args_copy())
        flip = self._flip_result_for_entry(entry, rules, round_idx)
        if flip > 0:
            if args:
                args[0] = flip
            else:
                args = [flip]
        out.append(op.name, op.targets_copy(), args)
        self._emit_rule_after(rules, entry, after_buf, immune_qubits, round_idx)

        if entry.op_type == MEASURE_RESET_1Q:
            reset_basis = entry.reset_basis or entry.measurement_basis or "Z"
            for qubit in entry.targets:
                self._queue_reset_calibration(
                    qubit=qubit,
                    basis=reset_basis,
                    after_buf=after_buf,
                    immune_qubits=immune_qubits,
                    round_idx=round_idx,
                )

    def _queue_after_1q_calibration(
        self,
        entry: CompiledOp,
        after_buf: defaultdict[Tuple[str, NoiseArg], stim.Circuit],
        immune_qubits: set[int],
        round_idx: int,
    ) -> None:
        for qubit in entry.targets:
            if qubit in immune_qubits:
                continue
            instructions = self._after_1q_by_qubit.get(qubit, self._default_after_1q)
            for instruction in instructions:
                if not self._is_zero_noise(instruction.arg):
                    self._append_noise(after_buf, instruction.name, instruction.arg, [qubit])

    def _queue_after_2q_calibration(
        self,
        entry: CompiledOp,
        after_buf: defaultdict[Tuple[str, NoiseArg], stim.Circuit],
        immune_qubits: set[int],
        round_idx: int,
    ) -> None:
        for pair in entry.pair_targets:
            a, b = pair
            if a in immune_qubits or b in immune_qubits:
                continue
            pair_spec = self._pair_noise_specs.get(
                pair, TwoQubitNoiseSpec(instructions=self._default_pair_noise)
            )
            for instruction in pair_spec.instructions:
                if not self._is_zero_noise(instruction.arg):
                    self._append_noise(after_buf, instruction.name, instruction.arg, [a, b])

    def _queue_reset_calibration(
        self,
        *,
        qubit: int,
        basis: str,
        after_buf: defaultdict[Tuple[str, NoiseArg], stim.Circuit],
        immune_qubits: set[int],
        round_idx: int,
    ) -> None:
        if qubit in immune_qubits:
            return
        flip_map = self._reset_flip_by_qubit.get(qubit, self._default_reset_flip)
        if flip_map is None:
            self._rates_for_qubit(qubit)
            return
        prob = flip_map.get(basis, flip_map.get("Z", 0.0))
        if self._is_zero_noise(prob):
            return
        if basis == "X":
            op_name = "Z_ERROR"
        elif basis == "Y":
            op_name = "Y_ERROR"
        else:
            op_name = "X_ERROR"
        self._append_noise(after_buf, op_name, prob, [qubit])

    def _append_idle_error(
        self,
        *,
        moment: CompiledMoment,
        out: stim.Circuit,
        system_qubits: set[int],
        immune_qubits: set[int],
        round_idx: int,
    ) -> None:
        if moment.duration <= 0:
            return

        active_durations: Dict[int, float] = {}
        for entry in moment.ops:
            if entry.op_type == ANNOTATION or occurs_in_classical_control_system(entry.instruction):
                continue
            for qubit in entry.targets:
                if qubit in immune_qubits:
                    continue
                active_durations[qubit] = max(active_durations.get(qubit, 0.0), entry.duration)

        grouped: Dict[Tuple[str, NoiseArg], List[int]] = defaultdict(list)
        for qubit in sorted(system_qubits - immune_qubits):
            idle_duration = max(0.0, moment.duration - active_durations.get(qubit, 0.0))
            if idle_duration <= 0.0:
                continue
            instruction = self._idle_instruction_by_qubit.get(qubit, self._default_idle_instruction)
            if instruction is None:
                rates = self._rates_for_qubit(qubit)
                instruction = NoiseInstruction(
                    name="PAULI_CHANNEL_1",
                    arg=idle_pauli_channel_from_T1T2(idle_duration, rates.T1, rates.T2),
                )
            if not self._is_zero_noise(instruction.arg):
                grouped[(instruction.name, instruction.arg)].append(qubit)

        for (name, arg), targets in sorted(
            grouped.items(), key=lambda item: (item[0][0], item[0][1])
        ):
            self._emit_instruction(out, name, targets, arg)

    def _rules_for_entry(self, entry: CompiledOp) -> Tuple[NoiseRule, ...]:
        rules: List[NoiseRule] = []
        gate_rule = self._gate_rules.get(entry.instruction.name)
        if gate_rule is not None:
            rules.append(gate_rule)
        if entry.op_type in (JUST_MEASURE_1Q, MEASURE_RESET_1Q, MPP):
            basis = (entry.measurement_basis or "Z").upper()
            measure_rule = self._measure_rules.get(basis)
            if measure_rule is not None:
                rules.append(measure_rule)
        if entry.op_type in (JUST_RESET_1Q, MEASURE_RESET_1Q):
            reset_key = (entry.reset_basis or "Z").upper()
            reset_rule = self._reset_rules.get(reset_key) or self._reset_rules.get(
                entry.instruction.name.upper()
            )
            if reset_rule is not None:
                rules.append(reset_rule)
        if entry.op_type == CLIFFORD_1Q and self._any_clifford_1q_rule is not None:
            rules.append(self._any_clifford_1q_rule)
        if entry.op_type == CLIFFORD_2Q and self._any_clifford_2q_rule is not None:
            rules.append(self._any_clifford_2q_rule)
        return tuple(rules)

    def _duration_for_operation(
        self,
        *,
        op_name: str,
        op_type: str,
        basis: Optional[str],
        reset_basis: Optional[str],
        pair_targets: Tuple[Tuple[int, int], ...],
        targets: Tuple[int, ...] = (),
        round_idx: Optional[int] = None,
    ) -> float:
        rules: List[NoiseRule] = []
        gate_rule = self._gate_rules.get(op_name)
        if gate_rule is not None:
            rules.append(gate_rule)
        if op_type in (JUST_MEASURE_1Q, MEASURE_RESET_1Q, MPP) and basis is not None:
            measure_rule = self._measure_rules.get(basis.upper())
            if measure_rule is not None:
                rules.append(measure_rule)
        if op_type in (JUST_RESET_1Q, MEASURE_RESET_1Q):
            reset_rule = self._reset_rules.get(
                (reset_basis or "Z").upper()
            ) or self._reset_rules.get(op_name.upper())
            if reset_rule is not None:
                rules.append(reset_rule)
        if op_type == CLIFFORD_1Q and self._any_clifford_1q_rule is not None:
            rules.append(self._any_clifford_1q_rule)
        if op_type == CLIFFORD_2Q and self._any_clifford_2q_rule is not None:
            rules.append(self._any_clifford_2q_rule)
        for rule in rules:
            if rule.duration is not None:
                return rule.duration
        if pair_targets:
            pair = pair_targets[0]
            pair_spec = self._pair_noise_specs.get(pair)
            if pair_spec is not None and pair_spec.duration is not None:
                return pair_spec.duration
            return self._gate_durations.for_operation(op_name, pair=pair)
        return self._gate_durations.for_operation(op_name)

    def _flip_result_for_entry(
        self, entry: CompiledOp, rules: Tuple[NoiseRule, ...], round_idx: int
    ) -> float:
        for rule in rules:
            if rule.flip_result is not None:
                return float(rule.flip_result)
        if not entry.targets:
            return 0.0
        basis = (entry.measurement_basis or "Z").upper()
        if len(entry.targets) == 1:
            flip_map = self._measure_flip_by_qubit.get(entry.targets[0], self._default_measure_flip)
            if flip_map is None:
                self._rates_for_qubit(entry.targets[0])
                return 0.0
            return flip_map.get(basis, flip_map.get("Z", 0.0))
        max_prob = 0.0
        for qubit in entry.targets:
            flip_map = self._measure_flip_by_qubit.get(qubit, self._default_measure_flip)
            if flip_map is not None:
                max_prob = max(max_prob, flip_map.get(basis, flip_map.get("Z", 0.0)))
        return max_prob

    def _emit_rule_before(
        self,
        rules: Tuple[NoiseRule, ...],
        entry: CompiledOp,
        out: stim.Circuit,
        immune_qubits: set[int],
        round_idx: int,
    ) -> None:
        for rule in rules:
            for instruction in rule.before:
                for targets, _ in self._instruction_target_groups(
                    entry, instruction.name, immune_qubits
                ):
                    if not self._is_zero_noise(instruction.arg):
                        self._emit_instruction(out, instruction.name, targets, instruction.arg)

    def _emit_rule_after(
        self,
        rules: Tuple[NoiseRule, ...],
        entry: CompiledOp,
        after_buf: defaultdict[Tuple[str, NoiseArg], stim.Circuit],
        immune_qubits: set[int],
        round_idx: int,
    ) -> None:
        for rule in rules:
            for instruction in rule.after:
                for targets, _ in self._instruction_target_groups(
                    entry, instruction.name, immune_qubits
                ):
                    if not self._is_zero_noise(instruction.arg):
                        self._append_noise(after_buf, instruction.name, instruction.arg, targets)

    def _instruction_target_groups(
        self,
        entry: CompiledOp,
        instruction_name: str,
        immune_qubits: set[int],
    ) -> Iterator[Tuple[List[int], SpatialIndex]]:
        if instruction_name in _TWO_QUBIT_NOISE_NAMES:
            if entry.pair_targets:
                for pair in entry.pair_targets:
                    if pair[0] in immune_qubits or pair[1] in immune_qubits:
                        continue
                    yield [pair[0], pair[1]], pair
                return
            if len(entry.targets) == 2 and not any(
                qubit in immune_qubits for qubit in entry.targets
            ):
                pair = _normalize_pair_key((entry.targets[0], entry.targets[1]))
                yield [pair[0], pair[1]], pair
                return
            raise ValueError(
                f"Instruction {instruction_name} requires two targets but operation {entry.instruction.name!r} "
                f"has targets {entry.targets!r}."
            )
        for qubit in entry.targets:
            if qubit in immune_qubits:
                continue
            yield [qubit], qubit

    @staticmethod
    def _round_from_detector(op: stim.CircuitInstruction) -> Optional[int]:
        coords = op.gate_args_copy()
        if not coords:
            return None
        rounded = int(round(coords[-1]))
        return rounded if abs(coords[-1] - rounded) < 1e-9 else None

    @staticmethod
    def _reset_basis(name: str) -> str:
        if name in {"RX", "MRX"}:
            return "X"
        if name in {"RY", "MRY"}:
            return "Y"
        return "Z"

    @staticmethod
    def _is_zero_noise(value: NoiseArg) -> bool:
        return all(v <= 0 for v in value) if isinstance(value, tuple) else value <= 0

    @staticmethod
    def _emit_instruction(out: stim.Circuit, name: str, targets: List[int], arg: NoiseArg) -> None:
        out.append(name, targets, list(arg) if isinstance(arg, tuple) else arg)

    @staticmethod
    def _append_noise(
        after_buf: defaultdict[Tuple[str, NoiseArg], stim.Circuit],
        op_name: str,
        arg: NoiseArg,
        targets: List[int],
    ) -> None:
        after_buf[(op_name, arg)].append(
            op_name, targets, list(arg) if isinstance(arg, tuple) else arg
        )

    def summary(self) -> str:
        mode = "uniform" if self._default_qubit_rates is not None else "strict"
        return (
            "_CompiledNoiseEngine("
            f"mode={mode}, qubit_overrides={len(self._qubit_rates)}, pair_overrides={len(self._pair_noise_specs)}, "
            f"gate_rules={len(self._gate_rules)}, measure_rules={len(self._measure_rules)}, "
            f"reset_rules={len(self._reset_rules)})"
        )

    def __repr__(self) -> str:  # pragma: no cover
        return self.summary()

    def __str__(self) -> str:  # pragma: no cover
        lines = ["_CompiledNoiseEngine:"]
        if self._default_qubit_rates is not None:
            lines.append("  Uniform: " + self._fmt(asdict(self._default_qubit_rates)))
        else:
            lines.append("  Uniform: None")
        lines.append(f"  Per-qubit overrides: {len(self._qubit_rates)}")
        lines.append(f"  Pair overrides: {len(self._pair_noise_specs)}")
        lines.append(
            "  Rule overrides: "
            f"gate={len(self._gate_rules)}, measure={len(self._measure_rules)}, reset={len(self._reset_rules)}"
        )
        return "\n".join(lines)

    @staticmethod
    def _fmt(value: object) -> str:
        if isinstance(value, float):
            return "0" if value == 0 else f"{value:.6g}"
        if isinstance(value, tuple):
            return "[" + ", ".join(_CompiledNoiseEngine._fmt(v) for v in value) + "]"
        if isinstance(value, list):
            return "[" + ", ".join(_CompiledNoiseEngine._fmt(v) for v in value) + "]"
        if isinstance(value, dict):
            return (
                "{"
                + ", ".join(f"{k}: {_CompiledNoiseEngine._fmt(v)}" for k, v in value.items())
                + "}"
            )
        return repr(value)


class NoiseModel(_CompiledNoiseEngine):
    """Structured noise model compiled into sparse fast-path lookup tables."""

    def __init__(
        self,
        *,
        spec: RoundIndexedNoiseSpec,
        gate_durations: GateDurations = _DEFAULT_CUSTOM_GATE_DURATIONS,
        verbose: bool = False,
    ) -> None:
        fallback = spec.global_noise or NoiseParams(T1=1.0, T2=1.0)
        super().__init__(
            gate_durations=gate_durations,
            default_qubit_rates=PerQubitRates(
                T1=1.0 if fallback.T1 is None else fallback.T1,
                T2=1.0 if fallback.T2 is None else fallback.T2,
            ),
            verbose=verbose,
        )
        self._spec = spec
        self._tables = _compile_noise_tables(spec)

    @property
    def spec(self) -> RoundIndexedNoiseSpec:
        return self._spec

    @property
    def mode(self) -> str:
        return self._tables.mode

    @property
    def uses_round_indices(self) -> bool:
        return self._tables.uses_round_indices

    def _set_spec(self, spec: RoundIndexedNoiseSpec) -> None:
        self._spec = spec
        self._tables = _compile_noise_tables(spec)

    def summary(self) -> str:
        round_override_count = len(
            set(self._tables.round_qubit_default_param_ids)
            | set(self._tables.round_pair_default_param_ids)
        )
        return (
            "NoiseModel("
            f"mode={self.mode}, "
            f"qubit_overrides={len(self._tables.qubit_param_ids)}, "
            f"pair_overrides={len(self._tables.pair_param_ids)}, "
            f"round_overrides={round_override_count}, "
            f"qubit_round_overrides={sum(len(items) for items in self._tables.round_qubit_param_ids.values())}, "
            f"pair_round_overrides={sum(len(items) for items in self._tables.round_pair_param_ids.values())})"
        )

    def compile_circuit(
        self,
        circuit: stim.Circuit,
        *,
        system_qubits: Optional[set[int]] = None,
        immune_qubits: Optional[set[int]] = None,
        moment_rounds: Optional[Sequence[int]] = None,
    ) -> CompiledCircuit:
        if not self.uses_round_indices:
            return super().compile_circuit(
                circuit,
                system_qubits=system_qubits,
                immune_qubits=immune_qubits,
                moment_rounds=None,
            )

        flat = circuit.flattened()
        round_indices = (
            [int(value) for value in moment_rounds]
            if moment_rounds is not None
            else infer_moment_rounds(flat, immune_qubits=immune_qubits)
        )
        return super().compile_circuit(
            flat,
            system_qubits=system_qubits,
            immune_qubits=immune_qubits,
            moment_rounds=round_indices,
        )

    def _qubit_params_for(self, qubit: int, round_idx: Optional[int]) -> _ResolvedQubitParams:
        if self.mode == "global_only":
            return self._tables.qubit_params_pool[self._tables.default_qubit_param_id]
        if round_idx is not None and self.mode == "round_overrides":
            round_params = self._tables.round_qubit_param_ids.get(int(round_idx))
            if round_params is not None:
                param_id = round_params.get(int(qubit))
                if param_id is not None:
                    return self._tables.qubit_params_pool[param_id]
            default_param_id = self._tables.round_qubit_default_param_ids.get(int(round_idx))
            if default_param_id is not None:
                return self._tables.qubit_params_pool[
                    self._tables.qubit_param_ids.get(int(qubit), default_param_id)
                ]
        param_id = self._tables.qubit_param_ids.get(int(qubit), self._tables.default_qubit_param_id)
        return self._tables.qubit_params_pool[param_id]

    def _pair_params_for(
        self, pair: Tuple[int, int], round_idx: Optional[int]
    ) -> _ResolvedPairParams:
        normalized_pair = _normalize_pair_key(pair)
        if self.mode == "global_only":
            return self._tables.pair_params_pool[self._tables.default_pair_param_id]
        if round_idx is not None and self.mode == "round_overrides":
            round_params = self._tables.round_pair_param_ids.get(int(round_idx))
            if round_params is not None:
                param_id = round_params.get(normalized_pair)
                if param_id is not None:
                    return self._tables.pair_params_pool[param_id]
            default_param_id = self._tables.round_pair_default_param_ids.get(int(round_idx))
            if default_param_id is not None:
                return self._tables.pair_params_pool[
                    self._tables.pair_param_ids.get(normalized_pair, default_param_id)
                ]
        param_id = self._tables.pair_param_ids.get(
            normalized_pair, self._tables.default_pair_param_id
        )
        return self._tables.pair_params_pool[param_id]

    def _duration_for_operation(
        self,
        *,
        op_name: str,
        op_type: str,
        basis: Optional[str],
        reset_basis: Optional[str],
        pair_targets: Tuple[Tuple[int, int], ...],
        targets: Tuple[int, ...] = (),
        round_idx: Optional[int] = None,
    ) -> float:
        if pair_targets:
            pair = pair_targets[0]
            duration = self._pair_params_for(pair, round_idx).duration_for(op_name)
            if duration is not None:
                return duration
            return self._gate_durations.for_operation(op_name, pair=pair)

        duration_values = [
            duration
            for qubit in targets
            if (duration := self._qubit_params_for(qubit, round_idx).duration_for(op_name))
            is not None
        ]
        if duration_values:
            return max(duration_values)
        return self._gate_durations.for_operation(op_name)

    def _queue_after_1q_calibration(
        self,
        entry: CompiledOp,
        after_buf: defaultdict[Tuple[str, NoiseArg], stim.Circuit],
        immune_qubits: set[int],
        round_idx: int,
    ) -> None:
        for qubit in entry.targets:
            if qubit in immune_qubits:
                continue
            for instruction in self._qubit_params_for(qubit, round_idx).gate_instructions_for(
                entry.instruction.name
            ):
                if not self._is_zero_noise(instruction.arg):
                    self._append_noise(after_buf, instruction.name, instruction.arg, [qubit])

    def _queue_after_2q_calibration(
        self,
        entry: CompiledOp,
        after_buf: defaultdict[Tuple[str, NoiseArg], stim.Circuit],
        immune_qubits: set[int],
        round_idx: int,
    ) -> None:
        for pair in entry.pair_targets:
            a, b = pair
            if a in immune_qubits or b in immune_qubits:
                continue
            for instruction in self._pair_params_for(pair, round_idx).gate_instructions_for(
                entry.instruction.name
            ):
                if not self._is_zero_noise(instruction.arg):
                    self._append_noise(after_buf, instruction.name, instruction.arg, [a, b])

    def _queue_reset_calibration(
        self,
        *,
        qubit: int,
        basis: str,
        after_buf: defaultdict[Tuple[str, NoiseArg], stim.Circuit],
        immune_qubits: set[int],
        round_idx: int,
    ) -> None:
        if qubit in immune_qubits:
            return
        prob = self._qubit_params_for(qubit, round_idx).reset_probability(basis)
        if prob <= 0.0:
            return
        op_name = "Z_ERROR" if basis == "X" else "Y_ERROR" if basis == "Y" else "X_ERROR"
        self._append_noise(after_buf, op_name, prob, [qubit])

    def _flip_result_for_entry(
        self, entry: CompiledOp, rules: Tuple[NoiseRule, ...], round_idx: int
    ) -> float:
        for rule in rules:
            if rule.flip_result is not None:
                return float(rule.flip_result)
        if not entry.targets:
            return 0.0
        basis = (entry.measurement_basis or "Z").upper()
        return max(
            self._qubit_params_for(qubit, round_idx).measure_probability(basis)
            for qubit in entry.targets
        )

    def _append_idle_error(
        self,
        *,
        moment: CompiledMoment,
        out: stim.Circuit,
        system_qubits: set[int],
        immune_qubits: set[int],
        round_idx: int,
    ) -> None:
        if moment.duration <= 0:
            return

        active_durations: Dict[int, float] = {}
        qubit_idle_sources: Dict[int, Tuple[str, int]] = {}
        pair_idle_sources: Dict[int, Tuple[str, Tuple[int, int]]] = {}
        nonactive_candidates: List[NoiseInstruction] = []
        for entry in moment.ops:
            if entry.op_type == ANNOTATION or occurs_in_classical_control_system(entry.instruction):
                continue
            if entry.pair_targets:
                for pair in entry.pair_targets:
                    pair_params = self._pair_params_for(pair, round_idx)
                    candidate = pair_params.idle_instruction_for(
                        entry.instruction.name, moment.duration
                    )
                    if candidate is not None and not self._is_zero_noise(candidate.arg):
                        nonactive_candidates.append(candidate)
                    for qubit in pair:
                        if qubit in immune_qubits:
                            continue
                        active_durations[qubit] = max(
                            active_durations.get(qubit, 0.0), entry.duration
                        )
                        pair_idle_sources[qubit] = (entry.instruction.name, pair)
                continue

            for qubit in entry.targets:
                if qubit in immune_qubits:
                    continue
                active_durations[qubit] = max(active_durations.get(qubit, 0.0), entry.duration)
                qubit_idle_sources[qubit] = (entry.instruction.name, qubit)
                candidate = self._qubit_params_for(qubit, round_idx).idle_instruction_for(
                    entry.instruction.name, moment.duration
                )
                if candidate is not None and not self._is_zero_noise(candidate.arg):
                    nonactive_candidates.append(candidate)

        largest_nonactive_candidate = (
            max(nonactive_candidates, key=lambda instruction: _noise_magnitude(instruction.arg))
            if nonactive_candidates
            else None
        )

        grouped: Dict[Tuple[str, NoiseArg], List[int]] = defaultdict(list)
        for qubit in sorted(system_qubits - immune_qubits):
            idle_duration = max(0.0, moment.duration - active_durations.get(qubit, 0.0))
            if idle_duration <= 0.0:
                continue
            if qubit in pair_idle_sources:
                op_name, pair = pair_idle_sources[qubit]
                instruction = self._pair_params_for(pair, round_idx).idle_instruction_for(
                    op_name, idle_duration
                )
            elif qubit in qubit_idle_sources:
                op_name, source_qubit = qubit_idle_sources[qubit]
                instruction = self._qubit_params_for(source_qubit, round_idx).idle_instruction_for(
                    op_name, idle_duration
                )
            else:
                # Fully idle in this moment. When the qubit has a round-
                # specific override (qubit_round_overrides or round_overrides.
                # qubits), respect its own idle spec so per-(q, r) nonuniform
                # idle scaling reaches emitted noise. Note that a resolved
                # zero-arg instruction is a legitimate spec (factor clipped to
                # 0 -> qubit emits no idle noise) and is NOT a signal to fall
                # back to the moment-max heuristic. Without an explicit round
                # override we fall through to the legacy heuristic so the
                # uniform/biased presets stay byte-identical.
                qubit_params = self._qubit_params_for(qubit, round_idx)
                has_round_specific = (
                    round_idx is not None
                    and self.mode == "round_overrides"
                    and self._tables.round_qubit_param_ids.get(int(round_idx), {}).get(int(qubit))
                    is not None
                )
                if has_round_specific:
                    instruction = qubit_params.idle_instruction_for("*", idle_duration)
                    if instruction is None:
                        instruction = qubit_params.derived_idle_instruction(idle_duration)
                elif largest_nonactive_candidate is not None:
                    instruction = largest_nonactive_candidate
                else:
                    instruction = qubit_params.derived_idle_instruction(idle_duration)
            if instruction is not None and not self._is_zero_noise(instruction.arg):
                grouped[(instruction.name, instruction.arg)].append(qubit)

        for (name, arg), targets in sorted(
            grouped.items(), key=lambda item: (item[0][0], item[0][1])
        ):
            self._emit_instruction(out, name, targets, arg)


def _as_noise_params(value: object, *, field: str) -> NoiseParams:
    if isinstance(value, NoiseParams):
        return value
    if isinstance(value, Mapping):
        return NoiseParams(**{str(k): v for k, v in value.items()})
    raise TypeError(
        f"{field} values must be NoiseParams or a dict of NoiseParams kwargs, got {type(value).__name__}"
    )


def _as_round_noise_params(value: object, *, field: str) -> RoundNoiseParams:
    if isinstance(value, RoundNoiseParams):
        return value
    if isinstance(value, Mapping):
        unexpected = set(value.keys()) - {"qubits", "pairs"}
        if unexpected:
            raise ValueError(f"{field} dict has unexpected keys {sorted(unexpected)}")
        kwargs: Dict[str, NoiseParams] = {}
        for key in ("qubits", "pairs"):
            if key in value and value[key] is not None:
                kwargs[key] = _as_noise_params(value[key], field=f"{field}[...].{key}")
        return RoundNoiseParams(**kwargs)
    raise TypeError(
        f"{field} values must be RoundNoiseParams or a dict, got {type(value).__name__}"
    )


def custom(
    spec: Optional[RoundIndexedNoiseSpec] = None,
    *,
    T1: Optional[float] = None,
    T2: Optional[float] = None,
    gate_error: Optional[Mapping[str, RawNoiseInstructionCollection]] = None,
    spam_error: Optional[Mapping[str, BasisFlipInput]] = None,
    durations: Optional[Mapping[str, float]] = None,
    idle_error: NamedIdleChannelInput = None,
    qubit_overrides: Optional[Mapping[int, object]] = None,
    pair_overrides: Optional[Mapping[Tuple[int, int], object]] = None,
    round_overrides: Optional[Mapping[int, object]] = None,
    qubit_round_overrides: Optional[Mapping[Tuple[int, int], object]] = None,
    pair_round_overrides: Optional[Mapping[Tuple[Tuple[int, int], int], object]] = None,
    gate_durations: GateDurations = _DEFAULT_CUSTOM_GATE_DURATIONS,
    verbose: bool = False,
) -> NoiseModel:
    """Build a ``NoiseModel`` from either an explicit ``RoundIndexedNoiseSpec`` or flat kwargs.

    Two styles:

    * **Flat kwargs** (common case) — pass ``T1, T2, gate_error, ...`` directly.
      Override dicts accept either ``NoiseParams``/``RoundNoiseParams`` instances
      or plain dicts of their kwargs.
    * **Explicit spec** — pass ``spec=RoundIndexedNoiseSpec(...)`` when you
      already have one assembled. In this form no other noise kwargs may be
      passed.
    """
    any_flat = any(
        v is not None
        for v in (
            T1,
            T2,
            gate_error,
            spam_error,
            durations,
            idle_error,
            qubit_overrides,
            pair_overrides,
            round_overrides,
            qubit_round_overrides,
            pair_round_overrides,
        )
    )
    if spec is not None:
        if any_flat:
            raise ValueError("Pass either `spec` or flat kwargs — not both.")
        final_spec = spec
    else:
        global_noise = None
        if any(v is not None for v in (T1, T2, gate_error, spam_error, durations, idle_error)):
            global_noise = NoiseParams(
                T1=T1,
                T2=T2,
                gate_error=gate_error,
                spam_error=spam_error,
                durations=durations,
                idle_error=idle_error,
            )

        def _convert(source, field, converter):
            if source is None:
                return {}
            return {key: converter(value, field=field) for key, value in source.items()}

        final_spec = RoundIndexedNoiseSpec(
            global_noise=global_noise,
            qubit_overrides=_convert(qubit_overrides, "qubit_overrides", _as_noise_params),
            pair_overrides=_convert(pair_overrides, "pair_overrides", _as_noise_params),
            round_overrides=_convert(round_overrides, "round_overrides", _as_round_noise_params),
            qubit_round_overrides=_convert(
                qubit_round_overrides, "qubit_round_overrides", _as_noise_params
            ),
            pair_round_overrides=_convert(
                pair_round_overrides, "pair_round_overrides", _as_noise_params
            ),
        )

    return NoiseModel(spec=final_spec, gate_durations=gate_durations, verbose=verbose)


def infer_moment_rounds(
    circuit: stim.Circuit,
    *,
    immune_qubits: Optional[set[int]] = None,
) -> List[int]:
    """Assign a stabilizer-round index to every moment in ``circuit``.

    A round is the time between two successive measurement-and-reset
    boundaries. Concretely, the round counter advances when a reset op
    (``R``/``RX``/``RY``/``RZ`` or ``MR``/``MRX``/``MRY``/``MRZ``) appears
    in a moment **and** at least one measurement op
    (``M``/``MX``/``MY``/``MZ``/``MR``/``MRX``/``MRY``/``MRZ``/
    ``MPP``/``MXX``/``MYY``/``MZZ``) has appeared since the previous round
    advance. The very first reset of the circuit also advances the counter
    (from 0 to 1) so that all initialization moments land in round 1.

    This handles the standard QEC pattern where the start of a circuit
    often contains TWO consecutive reset moments — one to initialize the
    data qubits, then a separate moment to initialize the ancillas for the
    first stabilizer cycle. Both belong to round 1; there is no
    measurement between them, so the second reset does not advance.

    Annotation-only moments (containing only ``QUBIT_COORDS``,
    ``DETECTOR``, ``OBSERVABLE_INCLUDE``, ``SHIFT_COORDS``, ``MPAD``, ``E``,
    ``TICK``, etc., or empty) do not establish a round of their own. They
    inherit the round of the surrounding non-annotation moments — the most
    recent prior non-annotation round if one exists, otherwise the round of
    the next non-annotation moment (so leading ``QUBIT_COORDS`` blocks ahead
    of the first reset get the same round as the reset, instead of a stray
    round 0).

    If no reset op appears anywhere in the circuit, every moment is assigned
    round 0 (single-round semantics).
    """
    immune = set() if immune_qubits is None else set(immune_qubits)
    moments: List[List[stim.CircuitInstruction]] = []
    for moment in _iter_split_op_moments(circuit.flattened(), immune_qubits=immune):
        if isinstance(moment, stim.CircuitRepeatBlock):
            raise ValueError(
                "infer_moment_rounds expects a flattened circuit without REPEAT blocks."
            )
        moments.append(list(moment))

    is_annotation_only = [
        len(m) == 0 or all(OP_TYPES.get(op.name) == ANNOTATION for op in m) for m in moments
    ]
    has_measurement = [
        any(OP_TYPES.get(op.name) in (JUST_MEASURE_1Q, MEASURE_RESET_1Q, MPP) for op in m)
        for m in moments
    ]
    has_reset = [
        any(OP_TYPES.get(op.name) in (JUST_RESET_1Q, MEASURE_RESET_1Q) for op in m) for m in moments
    ]

    # Pass 1: assign rounds to non-annotation-only moments.
    # `pending_advance` becomes True after a measurement (or starts True so
    # the very first reset advances). The next reset consumes it and bumps
    # the round counter; resets without an intervening measurement do not.
    rounds: List[Optional[int]] = [None] * len(moments)
    current_round = 0
    pending_advance = True
    saw_reset = False
    for i, moment in enumerate(moments):
        if is_annotation_only[i]:
            continue
        # Order matters within a moment: a measurement seen first sets pending,
        # so a measure-reset op in the same moment (MR/MRX/MRY/MRZ) bumps the
        # round.
        if has_measurement[i]:
            pending_advance = True
        if has_reset[i] and pending_advance:
            current_round += 1
            pending_advance = False
            saw_reset = True
        rounds[i] = current_round

    # Pass 2: backfill annotation-only moments. Carry-over from the most
    # recent non-annotation round; if none precedes (leading annotations),
    # adopt the round of the first following non-annotation moment.
    leading_target = next((r for r in rounds if r is not None), 0)
    last_assigned: Optional[int] = None
    for i in range(len(moments)):
        if rounds[i] is not None:
            last_assigned = rounds[i]
            continue
        rounds[i] = last_assigned if last_assigned is not None else leading_target

    if not saw_reset:
        return [0] * len(rounds)
    return [int(r) for r in rounds]


def _stim_pure_noise_names() -> FrozenSet[str]:
    """Names of Stim gates that are pure noise channels (no measurement record)."""
    out: List[str] = []
    for name in stim.gate_data():
        try:
            data = stim.gate_data(name)
        except Exception:  # pragma: no cover - defensive
            continue
        if data.is_noisy_gate and not data.produces_measurements and not data.is_reset:
            out.append(name)
    return frozenset(out)


def _stim_heralded_noise_names() -> FrozenSet[str]:
    """Names of Stim heralded noise channels (noisy AND produces measurements)."""
    out: List[str] = []
    for name in stim.gate_data():
        try:
            data = stim.gate_data(name)
        except Exception:  # pragma: no cover - defensive
            continue
        if data.is_noisy_gate and data.produces_measurements and name.startswith("HERALDED"):
            out.append(name)
    return frozenset(out)


_PURE_NOISE_NAMES: FrozenSet[str] = _stim_pure_noise_names()
_HERALDED_NOISE_NAMES: FrozenSet[str] = _stim_heralded_noise_names()


def strip_noise_channels(
    circuit: stim.Circuit,
    *,
    drop_heralded: bool = False,
    strip_measurement_flips: bool = True,
) -> stim.Circuit:
    """Return a copy of ``circuit`` with noise removed.

    Three families of "noise" appear in a Stim circuit:

    1. **Standalone noise channels** — separate ops with no measurement
       record. Always removed. In current Stim: ``DEPOLARIZE1``,
       ``DEPOLARIZE2``, ``PAULI_CHANNEL_1``, ``PAULI_CHANNEL_2``,
       ``X_ERROR``, ``Y_ERROR``, ``Z_ERROR``, ``I_ERROR``, ``II_ERROR``,
       ``E`` (alias ``CORRELATED_ERROR``), and ``ELSE_CORRELATED_ERROR``.
       The strippable set is queried from Stim metadata at import time
       (``is_noisy_gate AND not produces_measurements AND not is_reset``),
       so it tracks new pure-noise channels Stim adds.

    2. **Inline measurement flip probabilities** — Stim allows a flip
       probability as the first parens argument of any measurement op
       (``M(p)``, ``MX(p)``, ``MR(p)``, ``MPP(p)``, etc.), which flips the
       recorded outcome with probability ``p``. When
       ``strip_measurement_flips=True`` (default), those args are cleared
       so ``M(0.001) 0`` becomes ``M 0``. Pass ``False`` to keep them.

    3. **Heralded noise channels** — ``HERALDED_ERASE`` and
       ``HERALDED_PAULI_CHANNEL_1`` emit a measurement record alongside
       the noise. Removing them shifts every subsequent ``rec[-N]``
       reference, so by default they are **kept**. Pass
       ``drop_heralded=True`` to strip them too — the caller is
       responsible for any downstream record adjustments.

    Unitary gates, resets, and annotations (``TICK``, ``DETECTOR``,
    ``OBSERVABLE_INCLUDE``, ``QUBIT_COORDS``, ``SHIFT_COORDS``) pass
    through verbatim. ``REPEAT`` blocks are descended into recursively.
    """
    drop_names: set[str] = set(_PURE_NOISE_NAMES)
    if drop_heralded:
        drop_names |= _HERALDED_NOISE_NAMES

    # Measurement ops whose first parens arg is a flip probability.
    # Heralded noise also produces measurements but its arg is the noise
    # parameter itself; we don't strip those even when strip_measurement_flips
    # is True.
    flip_carrying = MEASUREMENT_OPS

    def _strip_into(source: stim.Circuit, sink: stim.Circuit) -> None:
        for op in source:
            if isinstance(op, stim.CircuitRepeatBlock):
                inner = stim.Circuit()
                _strip_into(op.body_copy(), inner)
                sink.append(stim.CircuitRepeatBlock(repeat_count=op.repeat_count, body=inner))
                continue
            if op.name in drop_names:
                continue
            if strip_measurement_flips and op.name in flip_carrying and op.gate_args_copy():
                sink.append(stim.CircuitInstruction(op.name, op.targets_copy(), []))
                continue
            sink.append(op)

    out = stim.Circuit()
    _strip_into(circuit, out)
    return out


__all__ = [
    "CompiledCircuit",
    "CompiledMoment",
    "CompiledOp",
    "CompiledRepeatBlock",
    "GateDurations",
    "NoiseModel",
    "NoiseParams",
    "NoiseInstruction",
    "NoiseRule",
    "PerQubitRates",
    "RoundNoiseParams",
    "RoundIndexedNoiseSpec",
    "SpatialIndex",
    "TwoQubitNoiseSpec",
    "custom",
    "infer_moment_rounds",
    "idle_pauli_channel_from_T1T2",
    "strip_noise_channels",
]
