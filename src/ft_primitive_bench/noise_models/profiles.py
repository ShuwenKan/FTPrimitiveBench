"""Public factories for noise models.

Public API
----------
- ``noise_model(p=..., *, p_*=..., profile=..., coherence=...)`` — canonical
  factory. Two modes:

  - **Baseline + (optional) overrides**: pass ``p`` (and optionally per-class
    rates, ``coherence``, and a ``profile`` of overrides). The ``(None, None)``
    global entry is built from the kwargs; ``profile`` entries layer on top.
  - **Fully custom profile, no defaults**: leave ``p`` as ``None`` and pass
    only ``profile``. Useful for porting calibration data.

- ``uniform_depolarizing(p, ...)``  — depolarizing baseline.
- ``pauli_biased(p, axis, bias_factor, ...)`` — Pauli-axis-biased channels.
- ``measurement_biased(p, bias_factor, ...)`` — uniform SPAM amplification.
- ``nonuniform(p, sigma, *, variant, seed, ...)`` — Gaussian per-component scatter.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import stim

from ._noise_core import (
    CLIFFORD_1Q,
    CLIFFORD_2Q,
    MEASUREMENT_OPS,
    OP_TYPES,
    RESET_OPS,
    _iter_split_op_moments,
)
from .channels import clip_probability, make_1q_biased, make_2q_biased
from .hardware_noise import (
    CompiledCircuit,
    GateDurations,
    NoiseModel,
    NoiseParams,
    RoundIndexedNoiseSpec,
    RoundNoiseParams,
    infer_moment_rounds,
)
from .noise_profile import (
    Coherence,
    NoiseProfile,
    _is_valid_pair,
    _is_valid_qubit,
    _normalize_coherence,
)

_DEFAULT_GATE_TIMES = GateDurations(one_qubit=40e-9, two_qubit=120e-9, measurement=200e-9)
_CLIFFORD_1Q_GATES = {name for name, op_type in OP_TYPES.items() if op_type == CLIFFORD_1Q}
_CLIFFORD_2Q_GATES = {name for name, op_type in OP_TYPES.items() if op_type == CLIFFORD_2Q}

# Idle-channel keys for measurement / reset moments. The engine requires explicit
# keys (no wildcard fallback) for these op classes.
_MEAS_IDLE_KEYS = MEASUREMENT_OPS | RESET_OPS

# All idle-channel keys: wildcard + every gate / measurement / reset op name.
_GATE_IDLE_KEYS = _CLIFFORD_1Q_GATES | _CLIFFORD_2Q_GATES


# ─── Mode A vs Mode B helpers ─────────────────────────────────────────────────


def _split_rate_duration(value: Any) -> Tuple[float, Optional[float]]:
    """Return (rate, duration_or_None) from a bare float or (rate, duration) tuple."""
    if isinstance(value, tuple):
        return float(value[0]), float(value[1])
    return float(value), None


# ─── NoiseProfile entry → internal NoiseParams converter ──────────────────────


def _entry_to_noise_params(entry: Mapping[str, Any]) -> NoiseParams:
    """Convert a single profile entry's rate-dict to the engine's NoiseParams.

    Per-class rates (``p_1q``, ``p_2q``, ``p_meas``, ``p_reset``) are expanded
    into the engine's ``gate_error`` / ``spam_error`` / ``durations`` dicts.
    Per-op idle rates (``p_idle``, ``p_idle_meas``) are expanded into ``idle_error``.

    Advanced fields (``gate_error`` / ``spam_error`` / ``idle_error`` set on the
    entry directly) take precedence over the per-class expansions for any
    overlapping op name.
    """
    T1 = entry.get("T1")
    T2 = entry.get("T2")

    gate_error: Dict[str, Any] = {}
    spam_error: Dict[str, Any] = {}
    idle_error: Dict[str, Any] = {}
    durations: Dict[str, float] = {}

    p_1q = entry.get("p_1q")
    if p_1q is not None:
        rate, dur = _split_rate_duration(p_1q)
        for op in sorted(_CLIFFORD_1Q_GATES):
            gate_error[op] = ("DEPOLARIZE1", rate)
            if dur is not None:
                durations[op] = dur

    p_2q = entry.get("p_2q")
    if p_2q is not None:
        rate, dur = _split_rate_duration(p_2q)
        for op in sorted(_CLIFFORD_2Q_GATES):
            gate_error[op] = ("DEPOLARIZE2", rate)
            if dur is not None:
                durations[op] = dur

    p_meas = entry.get("p_meas")
    if p_meas is not None:
        rate, dur = _split_rate_duration(p_meas)
        spam_error["MEASURE"] = {"Z": rate, "X": rate, "Y": rate}
        if dur is not None:
            for op in MEASUREMENT_OPS:
                durations[op] = dur

    p_reset = entry.get("p_reset")
    if p_reset is not None:
        rate, dur = _split_rate_duration(p_reset)
        spam_error["RESET"] = {"Z": rate, "X": rate, "Y": rate}
        if dur is not None:
            for op in RESET_OPS:
                durations[op] = dur

    p_idle = entry.get("p_idle")
    if p_idle is not None:
        idle_inst = ("DEPOLARIZE1", float(p_idle))
        idle_error["*"] = idle_inst

    p_idle_meas = entry.get("p_idle_meas")
    if p_idle_meas is not None:
        idle_inst = ("DEPOLARIZE1", float(p_idle_meas))
        for op in _MEAS_IDLE_KEYS:
            idle_error[op] = idle_inst

    # Advanced fields take precedence (override the per-class expansions).
    for op, val in (entry.get("gate_error") or {}).items():
        gate_error[op] = val
    raw_spam = entry.get("spam_error") or {}
    for category, val in raw_spam.items():
        spam_error[category] = val
    raw_idle = entry.get("idle_error") or {}
    for op, val in raw_idle.items():
        idle_error[op] = val

    return NoiseParams(
        T1=T1,
        T2=T2,
        gate_error=gate_error if gate_error else None,
        spam_error=spam_error if spam_error else None,
        durations=durations if durations else None,
        idle_error=idle_error if idle_error else None,
    )


def _profile_to_internal_spec(profile: NoiseProfile) -> RoundIndexedNoiseSpec:
    """Convert a NoiseProfile to the engine's internal RoundIndexedNoiseSpec."""
    qubit_overrides: Dict[int, NoiseParams] = {}
    pair_overrides: Dict[Tuple[int, int], NoiseParams] = {}
    round_overrides: Dict[int, RoundNoiseParams] = {}
    qubit_round_overrides: Dict[Tuple[int, int], NoiseParams] = {}
    pair_round_overrides: Dict[Tuple[Tuple[int, int], int], NoiseParams] = {}

    global_entry = profile.get((None, None), {})
    global_params = _entry_to_noise_params(global_entry)

    for (component, round_idx), entry in profile.items():
        if component is None and round_idx is None:
            continue
        params = _entry_to_noise_params(entry)
        if component is None:
            # Round-only override: the same params apply to both qubit and pair
            # resolution chains; each picks up the fields it needs.
            round_overrides[int(round_idx)] = RoundNoiseParams(qubits=params, pairs=params)
        elif _is_valid_qubit(component):
            if round_idx is None:
                qubit_overrides[int(component)] = params
            else:
                qubit_round_overrides[(int(component), int(round_idx))] = params
        elif _is_valid_pair(component):
            if round_idx is None:
                pair_overrides[component] = params
            else:
                pair_round_overrides[(component, int(round_idx))] = params
        else:
            raise ValueError(f"Unrecognized component key {component!r}.")

    return RoundIndexedNoiseSpec(
        global_noise=global_params,
        qubit_overrides=qubit_overrides,
        pair_overrides=pair_overrides,
        round_overrides=round_overrides,
        qubit_round_overrides=qubit_round_overrides,
        pair_round_overrides=pair_round_overrides,
    )


# ─── Nonuniform per-component scaling helpers ─────────────────────────────────
#
# The packaged nonuniform model perturbs EVERY physical rate — 1-qubit gate,
# measurement, reset, idle, AND 2-qubit gate — by an independent Gaussian factor
# per physical component (per qubit and per pair). The perturbed *rate* is clipped
# to [0, 0.5]. Materialization happens at compile time (see
# ``ConfiguredNoiseModel.compile_circuit``) so the sampler sees the circuit's
# actual qubits and pairs.


def _clip_factor(value: float, *, min_factor: float, max_factor: float) -> float:
    return max(float(min_factor), min(float(max_factor), float(value)))


def _scale_probability(value: float, *, factor: float, cap: float) -> float:
    return max(0.0, min(cap, float(value) * float(factor)))


def _raw_instruction(spec: Any) -> Tuple[str, Union[float, Tuple[float, ...]]]:
    """Coerce a ``("NAME", rate)`` gate/idle spec to a normalized tuple."""
    name, arg = spec
    if isinstance(arg, (list, tuple)):
        return str(name), tuple(float(value) for value in arg)
    return str(name), float(arg)


def _scale_raw_instruction(
    spec: Any,
    *,
    factor: float,
    cap: float,
) -> Tuple[str, Union[float, Tuple[float, ...]]]:
    name, arg = _raw_instruction(spec)
    if isinstance(arg, tuple):
        return name, tuple(_scale_probability(value, factor=factor, cap=cap) for value in arg)
    return name, _scale_probability(arg, factor=factor, cap=cap)


def _scale_idle_channel(value: object, *, factor: float, cap: float) -> object:
    if value is None:
        return None
    if callable(value):
        return lambda idle_t, _value=value, _factor=factor, _cap=cap: _scale_raw_instruction(
            _value(idle_t),
            factor=_factor,
            cap=_cap,
        )
    if isinstance(value, Mapping):
        keys = {str(key).lower() for key in value}
        if "name" in keys or "arg" in keys or {"px", "py", "pz"} <= keys:
            return _scale_raw_instruction(value, factor=factor, cap=cap)
        return {
            str(name): _scale_idle_channel(item, factor=factor, cap=cap)
            for name, item in value.items()
        }
    return _scale_raw_instruction(value, factor=factor, cap=cap)


def _scale_basis_rates(
    value: Mapping[str, float], *, factor: float, cap: float
) -> Dict[str, float]:
    return {
        str(basis).upper(): _scale_probability(prob, factor=factor, cap=cap)
        for basis, prob in value.items()
    }


def _pairs_in_moment(moment: Sequence[stim.CircuitInstruction]) -> List[Tuple[int, int]]:
    """Enumerate the normalized 2-qubit pairs acted on within a single moment."""
    pairs: set[Tuple[int, int]] = set()
    for op in moment:
        if OP_TYPES.get(op.name) != CLIFFORD_2Q:
            continue
        targets = [int(target.value) for target in op.targets_copy() if target.is_qubit_target]
        for i in range(0, len(targets), 2):
            if i + 1 < len(targets):
                a, b = targets[i], targets[i + 1]
                pairs.add((a, b) if a <= b else (b, a))
    return sorted(pairs)


def _sample_nonuniform_adjustments(
    *,
    config: "NoiseModelConfig",
    qubits: Sequence[int],
    round_indices: Sequence[int],
    pairs: Sequence[Tuple[int, int]],
) -> Tuple[
    Dict[Tuple[int, int], float],
    Dict[Tuple[Tuple[int, int], int], float],
    "SampledFactorSnapshot",
]:
    """Seeded per-component factor sampler.

    ``space_only``: one factor per qubit and one per pair, frozen across rounds.
    ``space_time``: an independent factor per ``(qubit, round)`` and ``(pair, round)``.
    Returns ``(qubit_round_factors, pair_round_factors, snapshot)``.
    """
    seed = 0 if config.seed is None else int(config.seed)
    sigma = float(config.sigma)
    rng = random.Random(seed)
    qubit_list = sorted({int(q) for q in qubits})
    unique_rounds = sorted({int(r) for r in round_indices})
    pair_list = sorted({_norm_pair(pair) for pair in pairs})

    def draw() -> float:
        return _clip_factor(
            1.0 + rng.gauss(0.0, sigma),
            min_factor=config.min_factor,
            max_factor=config.max_factor,
        )

    space_factors: Dict[int, float] = {}
    qubit_round_factors: Dict[Tuple[int, int], float] = {}
    pair_round_factors: Dict[Tuple[Tuple[int, int], int], float] = {}

    if config.variant == "space_only":
        space_factors = {qubit: draw() for qubit in qubit_list}
        pair_base = {pair: draw() for pair in pair_list}
        qubit_round_factors = {
            (qubit, r): space_factors[qubit] for r in unique_rounds for qubit in qubit_list
        }
        pair_round_factors = {
            (pair, r): pair_base[pair] for r in unique_rounds for pair in pair_list
        }
    else:  # space_time
        qubit_round_factors = {(qubit, r): draw() for r in unique_rounds for qubit in qubit_list}
        pair_round_factors = {(pair, r): draw() for r in unique_rounds for pair in pair_list}

    snapshot = SampledFactorSnapshot(
        variant=config.variant,
        seed=seed,
        sigma=sigma,
        space_factors=dict(space_factors),
        pair_factors=dict(pair_round_factors),
        qubit_round_factors=dict(qubit_round_factors),
    )
    return qubit_round_factors, pair_round_factors, snapshot


def _norm_pair(pair: Tuple[int, int]) -> Tuple[int, int]:
    a, b = int(pair[0]), int(pair[1])
    return (a, b) if a <= b else (b, a)


# ─── Configured NoiseModel ────────────────────────────────────────────────────


@dataclass(frozen=True)
class NoiseModelConfig:
    """Metadata for a packaged-factory-built NoiseModel."""

    model_type: str
    p: float
    bias_factor: float = 1.0
    axis: str = "Z"
    sigma: float = 0.0
    seed: Optional[int] = None
    variant: str = "space_time"
    distribution: str = "gaussian"
    min_factor: float = 0.0
    max_factor: float = 3.0


@dataclass
class SampledFactorSnapshot:
    """A snapshot of the random factors that nonuniform sampled."""

    variant: str
    seed: int
    sigma: float
    space_factors: Dict[int, float]
    pair_factors: Dict[Tuple[Tuple[int, int], int], float]
    qubit_round_factors: Dict[Tuple[int, int], float]


class ConfiguredNoiseModel(NoiseModel):
    """NoiseModel built by a packaged factory; carries metadata about its construction."""

    def __init__(
        self,
        *,
        spec: RoundIndexedNoiseSpec,
        gate_durations: GateDurations = _DEFAULT_GATE_TIMES,
        verbose: bool = False,
        profile: Optional[NoiseProfile] = None,
        config: Optional[NoiseModelConfig] = None,
        sampled_factors: Optional[SampledFactorSnapshot] = None,
    ):
        super().__init__(spec=spec, gate_durations=gate_durations, verbose=verbose)
        self._profile = profile
        self._config = config
        self._sampled_factors = sampled_factors
        # Base (unscaled) spec kept so the nonuniform scatter can be materialized
        # per-circuit at compile time (once the actual qubits AND pairs are known).
        self._base_spec = spec

    @property
    def profile(self) -> Optional[NoiseProfile]:
        return self._profile

    @property
    def config(self) -> Optional[NoiseModelConfig]:
        return self._config

    @property
    def sampled_factors(self) -> Optional[SampledFactorSnapshot]:
        return self._sampled_factors

    def summary(self) -> str:
        if self._config is None:
            return super().summary()
        cfg = self._config
        parts = [f"model_type={cfg.model_type}", f"p={cfg.p}"]
        if cfg.model_type in ("pauli_biased",):
            parts.extend([f"axis={cfg.axis}", f"bias_factor={cfg.bias_factor}"])
        elif cfg.model_type == "measurement_biased":
            parts.append(f"bias_factor={cfg.bias_factor}")
        elif cfg.model_type == "nonuniform":
            parts.extend(
                [
                    f"variant={cfg.variant}",
                    f"sigma={cfg.sigma}",
                    f"distribution={cfg.distribution}",
                    f"seed={cfg.seed if cfg.seed is not None else 0}",
                ]
            )
        return f"NoiseModel({', '.join(parts)})"

    # ── Compile-time nonuniform materialization ───────────────────────────────

    @property
    def _nonuniform_scatter_active(self) -> bool:
        cfg = self._config
        return cfg is not None and cfg.model_type == "nonuniform" and float(cfg.sigma) > 0.0

    def noisy_circuit(
        self,
        circuit: Union[stim.Circuit, CompiledCircuit],
        *,
        system_qubits: Optional[set[int]] = None,
        immune_qubits: Optional[set[int]] = None,
        moment_rounds: Optional[Sequence[int]] = None,
    ) -> stim.Circuit:
        if self._nonuniform_scatter_active and isinstance(circuit, CompiledCircuit):
            raise TypeError(
                "Packaged nonuniform models (sigma > 0) must receive a raw stim.Circuit "
                "so they can materialize per-(component, round) overrides at compile time."
            )
        return super().noisy_circuit(
            circuit,
            system_qubits=system_qubits,
            immune_qubits=immune_qubits,
            moment_rounds=moment_rounds,
        )

    def compile_circuit(
        self,
        circuit: stim.Circuit,
        *,
        system_qubits: Optional[set[int]] = None,
        immune_qubits: Optional[set[int]] = None,
        moment_rounds: Optional[Sequence[int]] = None,
    ) -> CompiledCircuit:
        if not self._nonuniform_scatter_active:
            return super().compile_circuit(
                circuit,
                system_qubits=system_qubits,
                immune_qubits=immune_qubits,
                moment_rounds=moment_rounds,
            )

        round_indices = self._resolve_round_indices(
            circuit, immune_qubits=immune_qubits, moment_rounds=moment_rounds
        )
        self._set_spec(
            self._scaled_nonuniform_spec(
                circuit,
                system_qubits=system_qubits,
                immune_qubits=immune_qubits,
                round_indices=round_indices,
            )
        )
        return super().compile_circuit(
            circuit,
            system_qubits=system_qubits,
            immune_qubits=immune_qubits,
            moment_rounds=round_indices,
        )

    def _scaled_nonuniform_spec(
        self,
        circuit: stim.Circuit,
        *,
        system_qubits: Optional[set[int]],
        immune_qubits: Optional[set[int]],
        round_indices: Sequence[int],
    ) -> RoundIndexedNoiseSpec:
        """Materialize per-(qubit, round) and per-(pair, round) scaled overrides.

        Perturbs EVERY rate — 1-qubit gate / measurement / reset / idle on the
        qubit chain, and the 2-qubit gate error on the pair chain — each by its
        own component factor, with the perturbed rate clipped to ``[0, 0.5]``.
        """
        cfg = self._config
        base_noise = self._base_spec.global_noise or NoiseParams()

        flat = circuit.flattened()
        immune = set() if immune_qubits is None else set(immune_qubits)
        qubits = sorted(
            (set(range(flat.num_qubits)) if system_qubits is None else set(system_qubits)) - immune
        )
        moments = [
            moment
            for moment in _iter_split_op_moments(flat, immune_qubits={-1})
            if not isinstance(moment, stim.CircuitRepeatBlock)
        ]
        pairs = sorted({pair for moment in moments for pair in _pairs_in_moment(moment)})

        qubit_round_factors, pair_round_factors, snapshot = _sample_nonuniform_adjustments(
            config=cfg,
            qubits=qubits,
            round_indices=round_indices,
            pairs=pairs,
        )
        self._sampled_factors = snapshot

        qubit_round_overrides: Dict[Tuple[int, int], NoiseParams] = {}
        pair_round_overrides: Dict[Tuple[Tuple[int, int], int], NoiseParams] = {}

        for (qubit, round_idx), factor in qubit_round_factors.items():
            if math.isclose(factor, 1.0, rel_tol=0.0, abs_tol=1e-12):
                continue
            qubit_round_overrides[(qubit, round_idx)] = NoiseParams(
                T1=base_noise.T1,
                T2=base_noise.T2,
                gate_error={
                    name: _scale_raw_instruction(spec, factor=factor, cap=0.5)
                    for name, spec in base_noise.gate_error.items()
                    if name in _CLIFFORD_1Q_GATES
                    or name in MEASUREMENT_OPS
                    or name in RESET_OPS
                    or name == "*"
                },
                spam_error={
                    "RESET": _scale_basis_rates(
                        base_noise.spam_error.get("RESET", {}), factor=factor, cap=0.5
                    ),
                    "MEASURE": _scale_basis_rates(
                        base_noise.spam_error.get("MEASURE", {}), factor=factor, cap=0.5
                    ),
                },
                idle_error=_scale_idle_channel(base_noise.idle_error, factor=factor, cap=0.5),
            )

        for (pair, round_idx), factor in pair_round_factors.items():
            if math.isclose(factor, 1.0, rel_tol=0.0, abs_tol=1e-12):
                continue
            pair_round_overrides[(pair, round_idx)] = NoiseParams(
                gate_error={
                    name: _scale_raw_instruction(spec, factor=factor, cap=0.5)
                    for name, spec in base_noise.gate_error.items()
                    if name in _CLIFFORD_2Q_GATES or name == "*"
                },
            )

        return RoundIndexedNoiseSpec(
            global_noise=base_noise,
            qubit_round_overrides=qubit_round_overrides,
            pair_round_overrides=pair_round_overrides,
        )

    @staticmethod
    def _resolve_round_indices(
        circuit: stim.Circuit,
        *,
        immune_qubits: Optional[set[int]],
        moment_rounds: Optional[Sequence[int]],
    ) -> List[int]:
        flat = circuit.flattened()
        moments = [
            moment
            for moment in _iter_split_op_moments(flat, immune_qubits={-1})
            if not isinstance(moment, stim.CircuitRepeatBlock)
        ]
        if moment_rounds is None:
            inferred = infer_moment_rounds(flat, immune_qubits=immune_qubits)
            if len(inferred) != len(moments):
                raise ValueError("Failed to infer one stabilizer-round index per compiled moment.")
            return inferred
        if len(moment_rounds) != len(moments):
            raise ValueError(
                f"moment_rounds length {len(moment_rounds)} does not match the circuit's "
                f"{len(moments)} compiled moments."
            )
        return [int(value) for value in moment_rounds]


# ─── Top-level baseline factory ───────────────────────────────────────────────


def noise_model(
    p: Optional[float] = None,
    *,
    p_1q: Union[float, Tuple[float, float], None] = None,
    p_2q: Union[float, Tuple[float, float], None] = None,
    p_meas: Union[float, Tuple[float, float], None] = None,
    p_reset: Union[float, Tuple[float, float], None] = None,
    p_idle: Optional[float] = None,
    p_idle_meas: Optional[float] = None,
    coherence: Union[Coherence, Tuple[float, float], None] = None,
    profile: Union[NoiseProfile, Mapping, None] = None,
    gate_durations: GateDurations = _DEFAULT_GATE_TIMES,
) -> NoiseModel:
    """Build a NoiseModel.

    Two construction modes — pick whichever fits the use case:

    1. **Baseline + (optional) overrides**: pass ``p`` (and optionally per-class
       rates and ``profile`` of overrides). The ``(None, None)`` global entry is
       built from the kwargs — each unset ``p_*`` defaults to ``p``. Any
       entries in ``profile`` layer on top.

    2. **Fully custom profile, no defaults**: leave ``p`` as ``None`` and pass
       only ``profile``. The profile is used as-is — no synthetic defaults are
       added to its ``(None, None)`` entry. Use this when porting calibration
       data and you don't want a fall-back rate for fields the calibration
       didn't measure.

    Parameters
    ----------
    p : float | None
        Baseline error rate. If set, each unset ``p_*`` defaults to ``p`` and
        the ``(None, None)`` global entry is filled in from kwargs. If ``None``,
        no defaults are populated and ``profile`` must supply everything.
    p_1q, p_2q, p_meas, p_reset : float | (rate, duration) | None
        Per-class rates. Mode A (no ``coherence``): bare floats. Mode B
        (``coherence`` set): ``(rate, duration)`` tuples.
    p_idle, p_idle_meas : float | None
        Idle rates for gate and measurement moments. Mode A only.
        ``p_idle_meas`` defaults to ``p_idle`` if unset.
    coherence : Coherence | (T1, T2) | None
        If set, switches to Mode B (PTA-derived idle from T1/T2 + durations).
    profile : NoiseProfile | dict | None
        Per-(component, round) overrides. When ``p`` is ``None``, this profile
        IS the entire spec.
    gate_durations : GateDurations
        Default per-class durations (rarely needed; Mode B bundles durations
        in the rate tuples).
    """
    coherence_obj = _normalize_coherence(coherence)
    mode_b = coherence_obj is not None

    # Mode 2 — no baseline `p`, use profile as-is.
    if p is None:
        # No defaults to populate. Reject any baseline kwarg to avoid surprise.
        for name, val in (
            ("p_1q", p_1q),
            ("p_2q", p_2q),
            ("p_meas", p_meas),
            ("p_reset", p_reset),
            ("p_idle", p_idle),
            ("p_idle_meas", p_idle_meas),
        ):
            if val is not None:
                raise ValueError(
                    f"noise_model: when p is None, {name}= and the other baseline "
                    f"kwargs are not used. Set p (and optional kwargs) to use "
                    f"the baseline mode, or put {name} into the profile's "
                    f"(None, None) entry directly."
                )
        if coherence_obj is not None:
            raise ValueError(
                "noise_model: when p is None, coherence= is not used; put T1/T2 "
                "into the profile entries directly."
            )
        if profile is None or len(profile) == 0:
            raise ValueError(
                "noise_model: requires either p (baseline mode) or a non-empty profile."
            )
        final_profile = NoiseProfile(profile)
        spec = _profile_to_internal_spec(final_profile)
        return ConfiguredNoiseModel(
            spec=spec,
            gate_durations=gate_durations,
            profile=final_profile,
            config=NoiseModelConfig(model_type="custom", p=0.0),
        )

    # Mode 1 — baseline `p` + optional kwargs / profile.
    if not (0.0 <= float(p) <= 1.0):
        raise ValueError(f"p must be in [0, 1]; got {p}.")

    global_rates: Dict[str, Any] = {}

    if mode_b:
        global_rates["T1"] = coherence_obj.T1
        global_rates["T2"] = coherence_obj.T2
        if p_idle is not None or p_idle_meas is not None:
            raise ValueError(
                "Mode B (coherence set) forbids p_idle / p_idle_meas; "
                "PTA-derived idle is computed from coherence + duration."
            )
        for name, val in (("p_1q", p_1q), ("p_2q", p_2q), ("p_meas", p_meas), ("p_reset", p_reset)):
            if val is None:
                continue
            if not (isinstance(val, tuple) and len(val) == 2):
                raise ValueError(
                    f"Mode B requires {name} as a (rate, duration) tuple; got {val!r}."
                )
            global_rates[name] = (float(val[0]), float(val[1]))
    else:
        for name, val in (("p_1q", p_1q), ("p_2q", p_2q), ("p_meas", p_meas), ("p_reset", p_reset)):
            if val is None:
                global_rates[name] = float(p)
                continue
            if isinstance(val, tuple):
                raise ValueError(
                    f"Mode A (no coherence): {name} must be a bare float; got tuple {val!r}. "
                    f"Set coherence=(T1, T2) for Mode B."
                )
            global_rates[name] = float(val)
        if p_idle_meas is not None and p_idle is None:
            raise ValueError("p_idle_meas requires p_idle to be set.")
        resolved_p_idle = float(p_idle) if p_idle is not None else float(p)
        resolved_p_idle_meas = float(p_idle_meas) if p_idle_meas is not None else resolved_p_idle
        global_rates["p_idle"] = resolved_p_idle
        global_rates["p_idle_meas"] = resolved_p_idle_meas

    final_profile = NoiseProfile(profile or {})
    user_global = final_profile.get((None, None), {})
    final_profile[None, None] = {**global_rates, **user_global}

    spec = _profile_to_internal_spec(final_profile)
    return ConfiguredNoiseModel(
        spec=spec,
        gate_durations=gate_durations,
        profile=final_profile,
        config=NoiseModelConfig(model_type="uniform_depolarizing", p=float(p)),
    )


# ─── Pre-packaged factories ───────────────────────────────────────────────────


def uniform_depolarizing(p: float, **kwargs: Any) -> NoiseModel:
    """Uniform depolarizing noise. Direct alias for ``noise_model(p, **kwargs)``."""
    return noise_model(p, **kwargs)


def pauli_biased(
    p: float,
    axis: str = "Z",
    bias_factor: float = 1.0,
    **kwargs: Any,
) -> NoiseModel:
    """Pauli-axis-biased noise.

    The chosen axis (Z or X) carries ``bias_factor × baseline`` of the Pauli
    weight; the orthogonal axes split the remaining weight evenly. Sums to the
    baseline rate. ``bias_factor == 1`` recovers depolarizing.

    SPAM is basis-asymmetric: bias-aligned readout has flip rate
    ``2·p_meas/(1+η)``; orthogonal readout has ``2·η·p_meas/(1+η)``.
    """
    axis_u = axis.upper()
    if axis_u not in {"Z", "X"}:
        raise ValueError(f"axis must be 'Z' or 'X'; got {axis!r}.")
    if bias_factor < 1.0:
        raise ValueError(f"bias_factor must be ≥ 1; got {bias_factor}. For bias < 1 swap the axis.")

    # Build the depolarizing baseline first.
    base = noise_model(p, **kwargs)
    base_profile = NoiseProfile(base.profile)  # type: ignore[attr-defined]

    # Replace the (None, None) entry's rate fields with biased channel forms.
    global_entry = dict(base_profile.get((None, None), {}))
    eta = float(bias_factor)

    # Gate channels: PAULI_CHANNEL_1 and PAULI_CHANNEL_2 with biased rates.
    p_1q_value = global_entry.get("p_1q")
    p_2q_value = global_entry.get("p_2q")
    p_meas_value = global_entry.get("p_meas")
    p_reset_value = global_entry.get("p_reset")
    p_idle_value = global_entry.get("p_idle")
    p_idle_meas_value = global_entry.get("p_idle_meas")

    # Strip per-class rates from the global entry and rewrite as gate_error / spam_error / idle_error.
    for k in ("p_1q", "p_2q", "p_meas", "p_reset", "p_idle", "p_idle_meas"):
        global_entry.pop(k, None)

    gate_error: Dict[str, Any] = dict(global_entry.get("gate_error") or {})
    spam_error: Dict[str, Any] = dict(global_entry.get("spam_error") or {})
    idle_error: Dict[str, Any] = dict(global_entry.get("idle_error") or {})
    durations: Dict[str, float] = {}

    if p_1q_value is not None:
        rate, dur = _split_rate_duration(p_1q_value)
        spec_1q = make_1q_biased(rate, axis_u, eta)
        for op in sorted(_CLIFFORD_1Q_GATES):
            gate_error[op] = spec_1q
            if dur is not None:
                durations[op] = dur

    if p_2q_value is not None:
        rate, dur = _split_rate_duration(p_2q_value)
        spec_2q = make_2q_biased(rate, axis_u, eta)
        for op in sorted(_CLIFFORD_2Q_GATES):
            gate_error[op] = spec_2q
            if dur is not None:
                durations[op] = dur

    if p_meas_value is not None:
        rate, dur = _split_rate_duration(p_meas_value)
        # bias-aligned axis (small flip), orthogonal axis (large flip)
        small = 2.0 * rate / (1.0 + eta)
        large = 2.0 * eta * rate / (1.0 + eta)
        if axis_u == "Z":
            spam_error["MEASURE"] = {"Z": small, "X": large, "Y": small}
        else:  # axis == "X"
            spam_error["MEASURE"] = {"Z": large, "X": small, "Y": small}
        if dur is not None:
            for op in MEASUREMENT_OPS:
                durations[op] = dur

    if p_reset_value is not None:
        rate, dur = _split_rate_duration(p_reset_value)
        small = 2.0 * rate / (1.0 + eta)
        large = 2.0 * eta * rate / (1.0 + eta)
        if axis_u == "Z":
            spam_error["RESET"] = {"Z": small, "X": large, "Y": small}
        else:
            spam_error["RESET"] = {"Z": large, "X": small, "Y": small}
        if dur is not None:
            for op in RESET_OPS:
                durations[op] = dur

    if p_idle_value is not None:
        spec_idle = make_1q_biased(p_idle_value, axis_u, eta)
        idle_error["*"] = spec_idle
    if p_idle_meas_value is not None:
        spec_idle = make_1q_biased(p_idle_meas_value, axis_u, eta)
        for op in _MEAS_IDLE_KEYS:
            idle_error[op] = spec_idle

    if gate_error:
        global_entry["gate_error"] = gate_error
    if spam_error:
        global_entry["spam_error"] = spam_error
    if idle_error:
        global_entry["idle_error"] = idle_error
    # durations omitted from the profile entry — the existing engine already pulls
    # durations from gate_durations when not specified per-op. For Mode B the user
    # would have written tuples themselves and they'd flow through.

    base_profile[None, None] = global_entry

    spec = _profile_to_internal_spec(base_profile)
    return ConfiguredNoiseModel(
        spec=spec,
        profile=base_profile,
        config=NoiseModelConfig(
            model_type="pauli_biased",
            p=float(p),
            axis=axis_u,
            bias_factor=eta,
        ),
    )


def measurement_biased(p: float, bias_factor: float = 1.0, **kwargs: Any) -> NoiseModel:
    """Measurement-biased noise: depolarizing baseline + uniform SPAM amplification.

    The flip rate of every reset and measurement is scaled by ``bias_factor``.
    This is a uniform amplification — there is no basis dependence in the SPAM
    flip rates, despite the function name.
    """
    if bias_factor < 1.0:
        raise ValueError(f"bias_factor must be ≥ 1; got {bias_factor}.")

    base = noise_model(p, **kwargs)
    base_profile = NoiseProfile(base.profile)  # type: ignore[attr-defined]
    global_entry = dict(base_profile.get((None, None), {}))

    eta = float(bias_factor)
    if "p_meas" in global_entry:
        rate, dur = _split_rate_duration(global_entry["p_meas"])
        scaled_rate = clip_probability(eta * rate)
        global_entry["p_meas"] = (scaled_rate, dur) if dur is not None else scaled_rate
    if "p_reset" in global_entry:
        rate, dur = _split_rate_duration(global_entry["p_reset"])
        scaled_rate = clip_probability(eta * rate)
        global_entry["p_reset"] = (scaled_rate, dur) if dur is not None else scaled_rate

    base_profile[None, None] = global_entry
    spec = _profile_to_internal_spec(base_profile)
    return ConfiguredNoiseModel(
        spec=spec,
        profile=base_profile,
        config=NoiseModelConfig(model_type="measurement_biased", p=float(p), bias_factor=eta),
    )


def nonuniform(
    p: float,
    sigma: float = 0.0,
    *,
    distribution: str = "gaussian",
    variant: str = "space_time",
    seed: Optional[int] = None,
    min_factor: float = 0.0,
    max_factor: float = 3.0,
    **kwargs: Any,
) -> NoiseModel:
    """Nonuniform noise: depolarizing baseline + Gaussian per-component scatter.

    Each component (qubit or pair) has every rate scaled by ``(1 + δ)`` with
    ``δ ~ N(0, sigma²)``, clipped to ``[min_factor, max_factor]``.

    Parameters
    ----------
    sigma : float ≥ 0
        Gaussian width. ``sigma=0`` recovers ``uniform_depolarizing(p, **kwargs)``.
    distribution : 'gaussian'
        Reserved for future variants. Only ``"gaussian"`` is supported.
    variant : 'space_only' | 'space_time'
        ``"space_only"``: one factor per qubit/pair, frozen across rounds.
        ``"space_time"``: independent factor per (qubit, round) and (pair, round).
    seed : int | None
        RNG seed for reproducibility.
    min_factor, max_factor : float
        Clipping range for the perturbation factor ``(1 + δ)``.
    """
    if sigma < 0:
        raise ValueError(f"sigma must be ≥ 0; got {sigma}.")
    if distribution != "gaussian":
        raise ValueError(
            f"only distribution='gaussian' supported in this version; got {distribution!r}."
        )
    if variant not in {"space_only", "space_time"}:
        raise ValueError(f"variant must be 'space_only' or 'space_time'; got {variant!r}.")
    if min_factor < 0:
        raise ValueError(f"min_factor must be ≥ 0; got {min_factor}.")
    if max_factor < min_factor:
        raise ValueError(f"max_factor must be ≥ min_factor; got {max_factor} < {min_factor}.")
    if kwargs.get("coherence") is not None:
        raise ValueError("nonuniform does not currently support coherence (Mode B).")

    base = noise_model(p, **kwargs)
    config = NoiseModelConfig(
        model_type="nonuniform",
        p=float(p),
        sigma=sigma,
        variant=variant,
        seed=seed,
        distribution=distribution,
        min_factor=min_factor,
        max_factor=max_factor,
    )

    # For every sigma (including 0) we hand back the unscaled baseline spec and
    # let ``ConfiguredNoiseModel.compile_circuit`` materialize the per-component
    # scatter at compile time — once the circuit's actual qubits AND pairs are
    # known. When sigma == 0 the scatter is inert and the baseline is returned
    # verbatim. This perturbs EVERY rate (1-qubit gate / measurement / reset /
    # idle AND the 2-qubit gate error), each by an independent per-component
    # factor, with the perturbed rate clipped to [0, 0.5].
    return ConfiguredNoiseModel(
        spec=base._spec,  # type: ignore[attr-defined]
        profile=base.profile,  # type: ignore[attr-defined]
        config=config,
    )


__all__ = [
    "ConfiguredNoiseModel",
    "Coherence",
    "NoiseModelConfig",
    "NoiseProfile",
    "SampledFactorSnapshot",
    "measurement_biased",
    "noise_model",
    "nonuniform",
    "pauli_biased",
    "uniform_depolarizing",
]
