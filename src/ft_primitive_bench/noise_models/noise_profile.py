"""User-facing noise profile: a dict mapping (component, round) → rate dict.

This is the canonical user-facing spec type. Users build a profile programmatically
(e.g., from a calibration JSON) and hand it to ``noise_model(profile=...)`` or
``custom(profile)``.

A NoiseProfile is just a dict subclass with validation. Each entry maps a
``(component, round)`` key to a dict of rates / coherence parameters.

    Component:  ``int`` (qubit) | ``tuple[int, int]`` (pair) | ``None`` (broadcast)
    Round:      ``int`` | ``None`` (broadcast)

The (None, None) key holds the global / baseline rates that apply when no more
specific override matches. ``noise_model(p=..., p_meas=..., profile=...)`` is
sugar that fills in (None, None) from kwargs and then merges the profile on top.

Resolution at (qubit q, round r):
    merged = profile[None, None]
          .merged_with(profile.get((q, None)))
          .merged_with(profile.get((None, r)))
          .merged_with(profile.get((q, r)))
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Tuple, Union

# ── Allowed fields in a value dict ──────────────────────────────────────────

# Rate fields that carry a probability (or (rate, duration) tuple in Mode B).
_RATE_FIELDS = frozenset(
    {
        "p_1q",
        "p_2q",
        "p_meas",
        "p_reset",
        "p_idle",
        "p_idle_meas",
    }
)

# Coherence fields (presence triggers Mode B for that entry).
_COHERENCE_FIELDS = frozenset({"T1", "T2"})

# Advanced channel-form fields (passed through to the engine verbatim).
_ADVANCED_FIELDS = frozenset({"gate_error", "spam_error", "idle_error"})

ALLOWED_FIELDS = _RATE_FIELDS | _COHERENCE_FIELDS | _ADVANCED_FIELDS

# Field-class taxonomy used for "is this field meaningful on a qubit/pair entry?".
_QUBIT_LOCAL_FIELDS = frozenset(
    {
        "T1",
        "T2",
        "p_1q",
        "p_meas",
        "p_reset",
        "p_idle",
        "p_idle_meas",
        "gate_error",
        "spam_error",
        "idle_error",
    }
)
_PAIR_LOCAL_FIELDS = frozenset({"p_2q", "gate_error"})


# ── Coherence dataclass ─────────────────────────────────────────────────────


@dataclass(frozen=True)
class Coherence:
    """Coherence parameters (T1, T2) for PTA-derived idle channels.

    A 2-tuple ``(T1, T2)`` is auto-coerced anywhere ``Coherence`` is accepted.
    """

    T1: float
    T2: float

    def __post_init__(self) -> None:
        if self.T1 <= 0 or self.T2 <= 0:
            raise ValueError(f"T1 and T2 must be positive seconds; got T1={self.T1}, T2={self.T2}")


def _normalize_coherence(value: Any) -> Coherence | None:
    """Coerce ``Coherence(T1, T2)`` or a 2-tuple ``(T1, T2)`` into ``Coherence``."""
    if value is None:
        return None
    if isinstance(value, Coherence):
        return value
    if isinstance(value, tuple) and len(value) == 2:
        return Coherence(T1=float(value[0]), T2=float(value[1]))
    raise TypeError(f"coherence must be Coherence or (T1, T2) tuple; got {type(value).__name__}.")


# ── Key / value validation ──────────────────────────────────────────────────


def _is_valid_qubit(x: Any) -> bool:
    return isinstance(x, int) and not isinstance(x, bool) and x >= 0


def _is_valid_pair(x: Any) -> bool:
    return isinstance(x, tuple) and len(x) == 2 and _is_valid_qubit(x[0]) and _is_valid_qubit(x[1])


def _validate_key(key: Any) -> None:
    """Validate a profile key has shape (component, round)."""
    if not (isinstance(key, tuple) and len(key) == 2):
        raise ValueError(f"NoiseProfile key must be a 2-tuple (component, round); got {key!r}.")
    component, round_idx = key
    if component is not None and not _is_valid_qubit(component) and not _is_valid_pair(component):
        raise ValueError(
            f"component must be a non-negative int (qubit), 2-tuple of non-negative "
            f"ints (pair), or None (broadcast); got {component!r}."
        )
    if round_idx is not None and not (
        isinstance(round_idx, int) and not isinstance(round_idx, bool) and round_idx >= 0
    ):
        raise ValueError(
            f"round must be a non-negative int or None (broadcast); got {round_idx!r}."
        )


def _validate_rate_value(field: str, value: Any) -> None:
    """Validate a single rate field's value (bare float or (rate, duration) tuple)."""
    if value is None:
        return
    if field in {"p_idle", "p_idle_meas"}:
        # Idle is always a bare rate.
        if not (isinstance(value, (int, float)) and not isinstance(value, bool)):
            raise ValueError(f"{field} must be a number; got {value!r}.")
        if not (0.0 <= float(value) <= 1.0):
            raise ValueError(f"{field} must be in [0, 1]; got {value}.")
        return
    if isinstance(value, tuple):
        if len(value) != 2:
            raise ValueError(f"{field} as a tuple must be (rate, duration); got {value!r}.")
        rate, duration = value
        if not (0.0 <= float(rate) <= 1.0):
            raise ValueError(f"{field} rate must be in [0, 1]; got {rate}.")
        if float(duration) <= 0.0:
            raise ValueError(f"{field} duration must be > 0 seconds; got {duration}.")
        return
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if not (0.0 <= float(value) <= 1.0):
            raise ValueError(f"{field} must be in [0, 1]; got {value}.")
        return
    raise ValueError(f"{field} must be a float or (rate, duration) tuple; got {value!r}.")


def _validate_value(key: Tuple[Any, Any], value: Mapping[str, Any]) -> dict:
    """Validate a full rate-dict value against the allowed-field set + key context.

    Returns a fresh dict (defensive copy).
    """
    if not isinstance(value, Mapping):
        raise TypeError(
            f"NoiseProfile value at {key!r} must be a dict; got {type(value).__name__}."
        )
    component, _ = key
    out: dict = {}
    for field, v in value.items():
        if field not in ALLOWED_FIELDS:
            raise ValueError(
                f"unknown field {field!r} in NoiseProfile entry {key!r}. "
                f"Allowed: {sorted(ALLOWED_FIELDS)}."
            )

        # Pair-keyed entries should not carry qubit-local fields like T1, p_1q,
        # p_meas etc. (they're physically per-qubit). Allow only pair-local fields.
        if _is_valid_pair(component) and field not in _PAIR_LOCAL_FIELDS:
            raise ValueError(
                f"field {field!r} is not meaningful on a pair-keyed entry "
                f"{key!r}. Pair entries should carry only "
                f"{sorted(_PAIR_LOCAL_FIELDS)}."
            )

        # Qubit-keyed entries should not carry pair-local 2Q fields.
        if _is_valid_qubit(component) and field == "p_2q":
            raise ValueError(
                f"p_2q on a qubit-keyed entry {key!r} is meaningless; put it on a "
                f"pair entry like ((q1, q2), round)."
            )

        # Validate value.
        if field in _RATE_FIELDS:
            _validate_rate_value(field, v)
        elif field in {"T1", "T2"}:
            if not (isinstance(v, (int, float)) and not isinstance(v, bool)) or float(v) <= 0:
                raise ValueError(f"{field} must be a positive float; got {v!r}.")
        elif field in _ADVANCED_FIELDS:
            if not isinstance(v, Mapping):
                raise ValueError(f"{field} must be a Mapping; got {type(v).__name__}.")

        out[field] = v

    # Per-entry sanity: T1 and T2 must be set together (or both unset).
    has_t1 = "T1" in out
    has_t2 = "T2" in out
    if has_t1 ^ has_t2:
        raise ValueError(
            f"NoiseProfile entry {key!r} sets only one of T1/T2; both are required to enable Mode B (PTA idle)."
        )
    mode_b = has_t1 and has_t2

    # If this entry sets T1/T2, it forbids constant-idle rates inside the same entry.
    if mode_b and ("p_idle" in out or "p_idle_meas" in out):
        raise ValueError(
            f"Entry {key!r} has T1/T2 (Mode B) and constant-idle rates; PTA-derived "
            f"idle is computed from T1, T2, and gate durations — drop p_idle / p_idle_meas."
        )

    # p_idle_meas requires p_idle within the same entry.
    if "p_idle_meas" in out and "p_idle" not in out:
        raise ValueError(f"p_idle_meas requires p_idle to be set in entry {key!r}.")

    # Note: cross-entry Mode A vs Mode B consistency (e.g., a pair-keyed entry
    # carrying tuple `p_2q` while the global is Mode A) is checked at engine
    # compile time once entries are merged. Per-entry, a tuple p_* is fine —
    # the engine simply ignores the duration if Mode A is active globally.

    return out


# ── NoiseProfile ────────────────────────────────────────────────────────────

ComponentKey = Union[int, Tuple[int, int], None]
RoundKey = Union[int, None]
ProfileKey = Tuple[ComponentKey, RoundKey]


class NoiseProfile(dict):
    """A noise profile: dict mapping ``(component, round)`` → rate dict.

    Mutable & dict-like — designed to be built programmatically (e.g., from a
    hardware calibration export). Keys are validated on insert.

    Examples
    --------
    >>> profile = NoiseProfile()
    >>> profile[None, None] = {"p_1q": 1e-3, "p_meas": 5e-3}     # global / baseline
    >>> profile[5, None] = {"p_meas": 1e-2}                       # qubit 5, all rounds
    >>> profile[(3, 4), None] = {"p_2q": 2e-3}                    # pair, all rounds
    >>> profile[None, 7] = {"p_1q": 2e-3, "p_2q": 5e-3}          # round 7, all components
    >>> profile[5, 7] = {"p_meas": 5e-2}                          # qubit 5 in round 7
    >>> profile[(3, 4), 7] = {"p_2q": 8e-3}                       # pair (3, 4) in round 7

    Resolution at (qubit q, round r) merges (None, None) → (q, None) → (None, r) → (q, r).
    Resolution at (pair p, round r) merges (None, None) → (p, None) → (None, r) → (p, r).
    Higher-precedence tiers' explicit fields override lower-precedence tiers'.
    """

    def __init__(self, initial: Mapping[ProfileKey, Mapping[str, Any]] | None = None) -> None:
        super().__init__()
        if initial:
            for k, v in initial.items():
                self[k] = v

    def __setitem__(self, key: ProfileKey, value: Mapping[str, Any]) -> None:
        _validate_key(key)
        super().__setitem__(key, _validate_value(key, value))

    def update(self, *args, **kwargs) -> None:  # type: ignore[override]
        # Support the standard dict.update() forms while running per-entry validation.
        if len(args) > 1:
            raise TypeError("update expected at most 1 positional argument")
        if args:
            other = args[0]
            if isinstance(other, Mapping):
                items: Iterable = other.items()
            else:
                items = other
            for k, v in items:
                self[k] = v
        for k, v in kwargs.items():
            self[k] = v

    def setdefault(self, key, default=None):  # type: ignore[override]
        if key in self:
            return self[key]
        self[key] = default if default is not None else {}
        return self[key]

    # ── Inspection / convenience ─────────────────────────────────────────────

    def has_round_overrides(self) -> bool:
        """True iff any round-keyed override (round != None) exists."""
        return any(round_idx is not None for _, round_idx in self.keys())

    def qubits(self) -> set[int]:
        """All qubit indices that appear in any per-qubit override."""
        out: set[int] = set()
        for component, _ in self.keys():
            if _is_valid_qubit(component):
                out.add(component)  # type: ignore[arg-type]
        return out

    def pairs(self) -> set[Tuple[int, int]]:
        """All pair tuples that appear in any per-pair override."""
        out: set[Tuple[int, int]] = set()
        for component, _ in self.keys():
            if _is_valid_pair(component):
                out.add(component)  # type: ignore[arg-type]
        return out

    def rounds(self) -> set[int]:
        """All round indices that appear (excluding None)."""
        out: set[int] = set()
        for _, round_idx in self.keys():
            if round_idx is not None:
                out.add(round_idx)
        return out

    def __repr__(self) -> str:
        return f"NoiseProfile({super().__repr__()})"
