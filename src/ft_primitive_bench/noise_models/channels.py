"""Typed and formula-driven noise channel builders."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

_PAULI2_ORDER = ["IX", "IY", "IZ", "XI", "XX", "XY", "XZ", "YI", "YX", "YY", "YZ", "ZI", "ZX", "ZY", "ZZ"]
_PAULI2_Z_TERMS = {"IZ", "ZI", "ZZ"}
_PAULI2_X_TERMS = {"IX", "XI", "XX"}
_PAULI2_Y_TERMS = {"IY", "YI", "YY"}
_EPSILON = 1e-12


def clip_probability(value: float) -> float:
    """Clip a probability into the physical interval [0, 1]."""
    return max(0.0, min(1.0, float(value)))


@dataclass(frozen=True)
class PauliChannel1:
    """Single-qubit Pauli channel with explicit (px, py, pz) error rates."""

    px: float
    py: float
    pz: float

    @classmethod
    def depolarizing(cls, p: float) -> "PauliChannel1":
        p = clip_probability(p)
        return cls(px=p / 3, py=p / 3, pz=p / 3)

    @classmethod
    def z_biased(cls, p: float, eta: float) -> "PauliChannel1":
        """Normalized Z-biased channel with X/Y weight 1 and Z weight eta. Alias for biased(axis='Z')."""
        return cls.biased(p, "Z", eta)

    @classmethod
    def biased(cls, p: float, axis: str, eta: float) -> "PauliChannel1":
        """Normalized Pauli-axis-biased channel.

        For axis ∈ {X, Y, Z} with bias factor η ≥ 1: the chosen-axis Pauli weight
        is ``η · p / (η+2)``; the other two are ``p / (η+2)`` each. Sums to ``p``.
        ``η == 1`` recovers depolarizing.
        """
        axis_u = axis.upper()
        p = clip_probability(p)
        denom = float(eta) + 2.0
        small = p / denom
        large = (p * float(eta)) / denom
        if axis_u == "Z":
            return cls(px=small, py=small, pz=large)
        if axis_u == "X":
            return cls(px=large, py=small, pz=small)
        if axis_u == "Y":
            return cls(px=small, py=large, pz=small)
        raise ValueError(f"axis must be 'X', 'Y', or 'Z'; got {axis!r}.")

    def to_spec(self) -> Tuple[str, List[float]]:
        return ("PAULI_CHANNEL_1", [self.px, self.py, self.pz])


@dataclass(frozen=True)
class PauliChannel2:
    """Two-qubit Pauli channel with 15-entry probability vector (IX, IY, ..., ZZ)."""

    probs: Tuple[float, ...]

    @classmethod
    def depolarizing(cls, p: float) -> "PauliChannel2":
        p = clip_probability(p)
        return cls(probs=tuple([p / 15] * 15))

    @classmethod
    def z_biased(cls, p: float, eta: float) -> "PauliChannel2":
        """Z-biased two-qubit channel. Alias for biased(axis='Z')."""
        return cls.biased(p, "Z", eta)

    @classmethod
    def biased(cls, p: float, axis: str, eta: float) -> "PauliChannel2":
        """Normalized Pauli-axis-biased two-qubit channel.

        For axis ∈ {X, Y, Z}: the three terms involving only the chosen-axis Pauli
        ({Iν, νI, νν}) get ``η · p / (12 + 3η)``; the other 12 get ``p / (12 + 3η)``.
        Sums to ``p``. ``η == 1`` recovers depolarizing.
        """
        axis_u = axis.upper()
        p = clip_probability(p)
        q = p / (12.0 + 3.0 * float(eta))
        if axis_u == "Z":
            boost = _PAULI2_Z_TERMS
        elif axis_u == "X":
            boost = _PAULI2_X_TERMS
        elif axis_u == "Y":
            boost = _PAULI2_Y_TERMS
        else:
            raise ValueError(f"axis must be 'X', 'Y', or 'Z'; got {axis!r}.")
        return cls(
            probs=tuple((float(eta) * q) if e in boost else q for e in _PAULI2_ORDER)
        )

    def to_spec(self) -> Tuple[str, List[float]]:
        return ("PAULI_CHANNEL_2", list(self.probs))


@dataclass(frozen=True)
class ResetFlip:
    """Post-reset error probabilities.

    ``z_reset_flip`` is the X_ERROR probability after a Z-basis reset.
    ``x_reset_flip`` is the Z_ERROR probability after an X-basis reset.
    The tuple order matches the ``(prob_Z, prob_X)`` convention used by
    ``PerQubitRates.after_reset``.
    """

    z_reset_flip: float
    x_reset_flip: float

    def to_spec(self) -> Tuple[float, float]:
        return (clip_probability(self.z_reset_flip), clip_probability(self.x_reset_flip))


@dataclass(frozen=True)
class MeasureFlip:
    """Pre-measurement flip probabilities.

    ``z_flip`` is the X_ERROR probability before a Z-basis measurement.
    ``x_flip`` is the Z_ERROR probability before an X-basis measurement.
    The tuple order matches the ``(prob_Z, prob_X)`` convention used by
    ``PerQubitRates.before_measure``.
    """

    z_flip: float
    x_flip: float = 0.0

    def to_spec(self) -> Tuple[float, float]:
        return (clip_probability(self.z_flip), clip_probability(self.x_flip))


def make_1q_depolarizing(p: float) -> Tuple[str, float | List[float]]:
    return ("DEPOLARIZE1", clip_probability(p))


def make_1q_z_biased(p: float, eta: float) -> Tuple[str, float | List[float]]:
    return make_1q_biased(p, "Z", eta)


def make_1q_biased(p: float, axis: str, eta: float) -> Tuple[str, float | List[float]]:
    """Build a 1Q Pauli-biased noise spec; collapses to DEPOLARIZE1 when eta == 1."""
    channel = PauliChannel1.biased(p, axis, eta)
    if abs(channel.px - channel.py) <= _EPSILON and abs(channel.py - channel.pz) <= _EPSILON:
        return ("DEPOLARIZE1", clip_probability(p))
    return channel.to_spec()


def make_2q_depolarizing(p: float) -> Tuple[str, float | List[float]]:
    return ("DEPOLARIZE2", clip_probability(p))


def make_2q_z_biased(p: float, eta: float) -> Tuple[str, float | List[float]]:
    return make_2q_biased(p, "Z", eta)


def make_2q_biased(p: float, axis: str, eta: float) -> Tuple[str, float | List[float]]:
    """Build a 2Q Pauli-biased noise spec; collapses to DEPOLARIZE2 when eta == 1."""
    channel = PauliChannel2.biased(p, axis, eta)
    if max(channel.probs) - min(channel.probs) <= _EPSILON:
        return ("DEPOLARIZE2", clip_probability(p))
    return channel.to_spec()


def make_measurement_flip_channel(p: float, *, basis: str | None = None) -> Tuple[float, float]:
    """Build a basis-aware pre-measurement flip channel spec.

    The engine stores both basis branches at once, so the optional basis is for
    auditability only. For Z-basis measurement, the X error branch is used. For
    X-basis measurement, the Z error branch is used.
    """
    prob = clip_probability(p)
    if basis is None:
        return MeasureFlip(z_flip=prob, x_flip=prob).to_spec()
    basis = basis.upper()
    if basis == "X":
        return MeasureFlip(z_flip=0.0, x_flip=prob).to_spec()
    if basis in {"Y", "Z"}:
        return MeasureFlip(z_flip=prob, x_flip=0.0).to_spec()
    raise ValueError(f"Unsupported measurement basis '{basis}'.")


def make_reset_flip_channel(p: float, *, basis: str | None = None) -> Tuple[float, float]:
    """Build a basis-aware post-reset flip channel spec."""
    prob = clip_probability(p)
    if basis is None:
        return ResetFlip(z_reset_flip=prob, x_reset_flip=prob).to_spec()
    basis = basis.upper()
    if basis == "X":
        return ResetFlip(z_reset_flip=0.0, x_reset_flip=prob).to_spec()
    if basis in {"Y", "Z"}:
        return ResetFlip(z_reset_flip=prob, x_reset_flip=0.0).to_spec()
    raise ValueError(f"Unsupported reset basis '{basis}'.")


__all__ = [
    "MeasureFlip",
    "PauliChannel1",
    "PauliChannel2",
    "ResetFlip",
    "clip_probability",
    "make_1q_depolarizing",
    "make_1q_biased",
    "make_1q_z_biased",
    "make_2q_depolarizing",
    "make_2q_biased",
    "make_2q_z_biased",
    "make_measurement_flip_channel",
    "make_reset_flip_channel",
]
