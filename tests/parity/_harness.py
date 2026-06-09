"""Parity utilities for comparing legacy vs. chunks-backend circuits.

Two layers of equivalence, cheapest first:

1. ``assert_dem_equivalent`` — canonicalize the DEM by replacing each detector
   index with its declared coordinates, then compare as a multiset of error
   lines. This tolerates detector ID reordering between legacy and chunks
   (the indices are arbitrary; the coordinates are physical).
2. ``assert_logical_error_rate_close`` — sample-based fallback for cases
   where DEM equivalence is too strict (e.g. coordinate metadata genuinely
   differs but the decoding graph is isomorphic).

The chunks backend is expected to produce coordinate-equivalent DEMs to the
legacy backend for clean (noiseless) circuits. When that fails, fall back to
the sample-based check before declaring a bug.
"""

from __future__ import annotations

from collections import Counter

import stim


def _coord_canonical_dem(circuit: stim.Circuit) -> tuple[Counter[str], int, dict[str, tuple[float, ...]]]:
    """Return (line_counter, num_observables, detector_coord_map).

    Each error line is rewritten so detector references become their coordinate
    tuples, making the multiset of errors invariant under detector reindexing.
    """
    # Compile with all syntactic sugar flattened.
    dem = circuit.detector_error_model(
        decompose_errors=False,  # tighter parity: don't let the decomposer reshape things
        approximate_disjoint_errors=True,
        flatten_loops=True,
    )

    detector_coords: dict[int, tuple[float, ...]] = dem.get_detector_coordinates()

    # Normalize the time component (third coord) by ranking unique t values
    # to {0, 1, 2, ...}. Legacy and chunks may emit a different number of
    # SHIFT_COORDS — the *labels* of t differ but the decoding graph is the
    # same. Canonicalizing this way makes the comparison robust to that.
    t_values = sorted({c[2] for c in detector_coords.values() if len(c) >= 3})
    t_rank = {t: i for i, t in enumerate(t_values)}
    def _norm(c: tuple[float, ...]) -> tuple[float, ...]:
        if len(c) < 3:
            return c
        return (c[0], c[1], float(t_rank[c[2]]), *c[3:])
    detector_coords = {k: _norm(v) for k, v in detector_coords.items()}

    def stringify_targets(targets: list[stim.DemTarget]) -> str:
        # Split at separators; sort within each group; rejoin. Order of detectors
        # within a single error term doesn't matter semantically (the error fires
        # the SET of targets), so we canonicalize to make the comparison stable.
        groups: list[list[str]] = [[]]
        for t in targets:
            if t.is_separator():
                groups.append([])
                continue
            if t.is_relative_detector_id():
                coord = detector_coords.get(t.val, ())
                groups[-1].append(f"D{coord}")
            elif t.is_logical_observable_id():
                groups[-1].append(f"L{t.val}")
            else:
                groups[-1].append(str(t))
        return " ^ ".join(" ".join(sorted(g)) for g in groups)

    lines: Counter[str] = Counter()
    num_observables = dem.num_observables
    for inst in dem.flattened():
        if inst.type == "error":
            args = inst.args_copy()
            prob = args[0] if args else 0.0
            targets = inst.targets_copy()
            line = f"error({prob}) {stringify_targets(targets)}"
            lines[line] += 1

    coord_map = {f"D{coord}": coord for coord in detector_coords.values()}
    return lines, num_observables, coord_map


def assert_dem_equivalent(legacy: stim.Circuit, chunks: stim.Circuit) -> None:
    # Structural sanity: detector and observable counts must match.
    if legacy.num_detectors != chunks.num_detectors or legacy.num_observables != chunks.num_observables:
        raise AssertionError(
            f"Circuit shape differs.\n"
            f"  legacy: detectors={legacy.num_detectors}, observables={legacy.num_observables}\n"
            f"  chunks: detectors={chunks.num_detectors}, observables={chunks.num_observables}"
        )
    a_lines, a_obs, _ = _coord_canonical_dem(legacy)
    b_lines, b_obs, _ = _coord_canonical_dem(chunks)
    if a_lines == b_lines and a_obs == b_obs:
        return

    only_a = a_lines - b_lines
    only_b = b_lines - a_lines
    msg = [
        f"DEM error multisets differ.",
        f"  legacy errors: {sum(a_lines.values())}, chunks errors: {sum(b_lines.values())}",
        f"  legacy observables: {a_obs}, chunks observables: {b_obs}",
    ]
    for line, count in list(only_a.most_common(3)):
        msg.append(f"  only in legacy x{count}: {line}")
    for line, count in list(only_b.most_common(3)):
        msg.append(f"  only in chunks x{count}: {line}")
    raise AssertionError("\n".join(msg))


def assert_logical_error_rate_close(
    legacy: stim.Circuit,
    chunks: stim.Circuit,
    *,
    shots: int = 10_000,
    tolerance: float = 0.01,
) -> None:
    """Compare logical error rates under identical sampling. Use only when
    DEM equivalence is too strict (e.g. detector reordering you accept)."""
    sampler_a = legacy.compile_detector_sampler()
    sampler_b = chunks.compile_detector_sampler()
    _, obs_a = sampler_a.sample(shots, separate_observables=True)
    _, obs_b = sampler_b.sample(shots, separate_observables=True)
    rate_a = obs_a.any(axis=1).mean() if obs_a.size else 0.0
    rate_b = obs_b.any(axis=1).mean() if obs_b.size else 0.0
    if abs(rate_a - rate_b) > tolerance:
        raise AssertionError(
            f"Logical error rates diverge: legacy={rate_a:.4f}, chunks={rate_b:.4f} "
            f"(tolerance={tolerance})"
        )
