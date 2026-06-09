"""S-gate via lattice surgery with teleported logical-Y magic state.

The surgery half is built by :func:`s_gate_surgery` (private) and emits
hand-written ``DETECTOR`` annotations following the lattice-surgery TYPE A/B/C
classification adapted for X-basis data/ancilla init. :func:`s_gate` splices in
the vendored logical-Y magic-state circuit produced by
:func:`_y_measurement.make_y_measurement_circuit`. The Y circuit's MPP preamble
AND its single memory round are stripped during splice — the memory round's
detector coordinates are used only as a lookup table for remapping the first
Y-transition round's detector-rec references back to our surgery's last
post-merge round measurements.
"""

from typing import Dict, Iterable, List, Tuple

import stim

from ._make_circuit import Backend, build_merged_patch
from ._utils import (
    append_repeated_phase,
    build_schedule,
    build_surface_code_round_circuit,
    c2xy,
    coord_sort_key,
    resolve_schedule,
)
from ._y_measurement import make_y_measurement_circuit

__all__ = ["s_gate"]


def _extract_coords(circuit: stim.Circuit) -> Dict[int, Tuple[float, ...]]:
    coords: Dict[int, Tuple[float, ...]] = {}
    for inst in circuit:
        if isinstance(inst, stim.CircuitInstruction) and inst.name == "QUBIT_COORDS":
            target = inst.targets_copy()[0]
            coords[target.value] = tuple(inst.gate_args_copy())
    return coords


def _build_base_coordinates(circuit: stim.Circuit):
    coords = _extract_coords(circuit)
    ordered = [coord for _, coord in sorted(coords.items())]
    mapping = {coord: idx for idx, coord in enumerate(ordered)}
    idx_to_coord = {idx: coord for idx, coord in enumerate(ordered)}
    return ordered, mapping, idx_to_coord


def _split_into_ticks(circuit: stim.Circuit) -> List[List[stim.CircuitInstruction]]:
    segments: List[List[stim.CircuitInstruction]] = []
    current: List[stim.CircuitInstruction] = []
    for inst in circuit:
        if inst.name == "TICK":
            if current:
                segments.append(current)
                current = []
        else:
            current.append(inst)
    if current:
        segments.append(current)
    return segments


def _is_measure_tick(segment: List[stim.CircuitInstruction]) -> bool:
    measurement_ops = {
        "M",
        "MX",
        "MY",
        "MZ",
        "MR",
        "MRX",
        "MRY",
        "MRZ",
        "DETECTOR",
        "OBSERVABLE_INCLUDE",
        "SHIFT_COORDS",
    }
    return bool(segment) and all(inst.name in measurement_ops for inst in segment)


def build_s_gate_circuit(
    distance: int,
    bridge_length: int,
    pre_rounds: int,
    merge_rounds: int,
    post_rounds: int = 1,
) -> Tuple[List[int], List[Tuple[float, float]], List[Tuple[float, float]], stim.Circuit]:
    """Build the S-gate surgery half with in-circuit detector annotations.

    Data and ancilla qubits are both X-initialized (``RX``). Ancilla data is
    destructively measured (``MX``) after the merge rounds; the final data
    measurement is left out — :func:`s_gate` splices the Y-magic circuit in
    its place.

    The S-gate primitive requires both individual patches to be square
    (``x_distance == z_distance``), so the builder takes a single ``distance``
    parameter. Detector emission mirrors :func:`lattice_surgery` (TYPE A/B/C
    classification) adapted for X-basis init — only X-type stabilizers are
    deterministic in round 0 (Z-type are random). ``post_rounds >= 1`` is
    required so the first ported Y-transition round's detector recs can be
    translated against a genuine individual-patch post-merge round via
    coordinate lookup.
    """
    if post_rounds < 1:
        raise ValueError(f"s_gate surgery requires post_rounds >= 1, got {post_rounds}.")
    if merge_rounds < 1:
        raise ValueError(f"s_gate surgery requires merge_rounds >= 1, got {merge_rounds}.")
    if (distance + bridge_length) % 2 != 0:
        raise ValueError(
            "S-gate requires (distance + bridge_length) to be even so that "
            "the merged-patch X-stabilizer checkerboard aligns with the two "
            f"individual patches. Got distance={distance}, "
            f"bridge_length={bridge_length}."
        )

    x_distance = distance
    z_distance = distance

    boundary_preset = {"top": "X", "bot": "X", "left": "Z", "right": "Z"}

    individual_patch, merged_patch = build_merged_patch(
        x_distance,
        z_distance,
        bridge_length,
        horizontal=False,
        individual_boundaries=boundary_preset,
        merged_boundaries=boundary_preset,
    )
    individual_patch._invalidate_cached_sets()

    circuit = stim.Circuit()

    merged_data = set(merged_patch.data_set)
    merged_meas = set(merged_patch.measure_set)
    individual_all = set(individual_patch.data_set) | set(individual_patch.measure_set)
    all_qubits = sorted(merged_data | merged_meas | individual_all, key=coord_sort_key)
    index_of = {coord: idx for idx, coord in enumerate(all_qubits)}
    index_to_coord = {idx: coord for idx, coord in enumerate(all_qubits)}
    for coord in all_qubits:
        circuit.append("QUBIT_COORDS", index_of[coord], list(c2xy(coord)))

    def targets(coords: Iterable[complex]) -> List[int]:
        return [index_of[c] for c in sorted(coords, key=coord_sort_key)]

    def ordered_coords(coords: Iterable[complex]) -> List[complex]:
        return sorted(coords, key=coord_sort_key)

    individual_data = set(individual_patch.data_set)
    data_init_targets = targets(individual_data)

    merged_meas_x_coords = ordered_coords(merged_patch.measure_x_set)
    merged_meas_z_coords = ordered_coords(merged_patch.measure_z_set)
    individual_meas_x_coords = ordered_coords(individual_patch.measure_x_set)
    individual_meas_z_coords = ordered_coords(individual_patch.measure_z_set)

    individual_meas_x = [index_of[c] for c in individual_meas_x_coords]
    individual_meas_z = [index_of[c] for c in individual_meas_z_coords]
    merged_meas_x = [index_of[c] for c in merged_meas_x_coords]
    merged_meas_z = [index_of[c] for c in merged_meas_z_coords]

    individual_schedule, individual_layers = build_schedule(individual_patch)
    merged_schedule, merged_layers = build_schedule(merged_patch)
    individual_resolved = resolve_schedule(individual_schedule, index_of)
    merged_resolved = resolve_schedule(merged_schedule, index_of)

    # ======== Detector bookkeeping ========
    n_indiv_x = len(individual_meas_x)
    n_indiv_z = len(individual_meas_z)
    n_indiv = n_indiv_x + n_indiv_z
    n_merged_x = len(merged_meas_x)
    n_merged_z = len(merged_meas_z)
    n_merged = n_merged_x + n_merged_z

    indiv_x_pos = {c: k for k, c in enumerate(individual_meas_x_coords)}
    indiv_z_pos = {c: k for k, c in enumerate(individual_meas_z_coords)}
    merged_x_pos = {c: k for k, c in enumerate(merged_meas_x_coords)}
    merged_z_pos = {c: k for k, c in enumerate(merged_meas_z_coords)}

    def _curr_indiv(basis: str, coord: complex) -> int:
        if basis == "X":
            return -(n_indiv_z + n_indiv_x - indiv_x_pos[coord])
        return -(n_indiv_z - indiv_z_pos[coord])

    def _curr_merged(basis: str, coord: complex) -> int:
        if basis == "X":
            return -(n_merged_z + n_merged_x - merged_x_pos[coord])
        return -(n_merged_z - merged_z_pos[coord])

    # Classify merged tiles as TYPE A/B/C against individual_patch.
    individual_tile_by_coord = {t.measurement_qubit: t for t in individual_patch.tiles}
    cls: Dict[complex, Tuple[str, frozenset, frozenset, str]] = {}
    for mt in merged_patch.tiles:
        m = mt.measurement_qubit
        D_m = frozenset(d for d in mt.data_qubits.values() if d is not None)
        basis = mt.basis
        it = individual_tile_by_coord.get(m)
        if it is None:
            cls[m] = ("C", frozenset(), D_m, basis)
        else:
            D_i = frozenset(d for d in it.data_qubits.values() if d is not None)
            extra = D_m - D_i
            cls[m] = (("A" if not extra else "B"), D_i, extra, basis)

    # ======== Round fragments ========
    def _build_pre_first(prepend_shift: bool) -> stim.Circuit:
        frag = stim.Circuit()
        if prepend_shift:
            frag.append("SHIFT_COORDS", [], (0, 0, 1))
        frag += build_surface_code_round_circuit(
            individual_resolved,
            individual_layers,
            individual_meas_x,
            individual_meas_z,
        )
        # X-stabs deterministic (+1) on |+>^n data; Z-stabs random (skip).
        for c in individual_meas_x_coords:
            frag.append(
                "DETECTOR",
                [stim.target_rec(_curr_indiv("X", c))],
                (c.real, c.imag, 0),
            )
        return frag

    def _build_pre_rest() -> stim.Circuit:
        frag = stim.Circuit()
        frag.append("SHIFT_COORDS", [], (0, 0, 1))
        frag += build_surface_code_round_circuit(
            individual_resolved,
            individual_layers,
            individual_meas_x,
            individual_meas_z,
        )
        for c in individual_meas_x_coords:
            curr = _curr_indiv("X", c)
            frag.append(
                "DETECTOR",
                [stim.target_rec(curr), stim.target_rec(curr - n_indiv)],
                (c.real, c.imag, 0),
            )
        for c in individual_meas_z_coords:
            curr = _curr_indiv("Z", c)
            frag.append(
                "DETECTOR",
                [stim.target_rec(curr), stim.target_rec(curr - n_indiv)],
                (c.real, c.imag, 0),
            )
        return frag

    def _build_merge_first(prepend_shift: bool) -> stim.Circuit:
        frag = stim.Circuit()
        if prepend_shift:
            frag.append("SHIFT_COORDS", [], (0, 0, 1))
        frag += build_surface_code_round_circuit(
            merged_resolved,
            merged_layers,
            merged_meas_x,
            merged_meas_z,
        )
        for m_coord, (ttype, _D_i, _D_anc, basis) in cls.items():
            # Only X-type stabs are deterministic in round 0 (X-basis data + ancilla).
            if basis != "X":
                continue
            curr = _curr_merged("X", m_coord)
            if pre_rounds > 0 and m_coord in indiv_x_pos:
                prev = -(n_merged + n_indiv_z + n_indiv_x - indiv_x_pos[m_coord])
                frag.append(
                    "DETECTOR",
                    [stim.target_rec(curr), stim.target_rec(prev)],
                    (m_coord.real, m_coord.imag, 0),
                )
            else:
                # TYPE A/B with pre_rounds=0, or TYPE C always: single-rec.
                frag.append(
                    "DETECTOR",
                    [stim.target_rec(curr)],
                    (m_coord.real, m_coord.imag, 0),
                )
        return frag

    def _build_merge_rest() -> stim.Circuit:
        frag = stim.Circuit()
        frag.append("SHIFT_COORDS", [], (0, 0, 1))
        frag += build_surface_code_round_circuit(
            merged_resolved,
            merged_layers,
            merged_meas_x,
            merged_meas_z,
        )
        for c in merged_meas_x_coords:
            curr = _curr_merged("X", c)
            frag.append(
                "DETECTOR",
                [stim.target_rec(curr), stim.target_rec(curr - n_merged)],
                (c.real, c.imag, 0),
            )
        for c in merged_meas_z_coords:
            curr = _curr_merged("Z", c)
            frag.append(
                "DETECTOR",
                [stim.target_rec(curr), stim.target_rec(curr - n_merged)],
                (c.real, c.imag, 0),
            )
        return frag

    # ======== Data init ========
    if data_init_targets:
        circuit.append("RX", data_init_targets)

    # ======== Pre-merge rounds ========
    is_first_round_of_circuit = True
    if pre_rounds > 0:
        pre_first = _build_pre_first(prepend_shift=False)
        pre_rest = _build_pre_rest()
        append_repeated_phase(
            circuit,
            pre_rounds,
            pre_first,
            middle_round=pre_rest,
            last_round=pre_rest,
        )
        is_first_round_of_circuit = False

    # ======== Merge setup ========
    ancilla_data = merged_data - individual_data
    ancilla_targets = targets(ancilla_data)
    ordered_ancilla_coords = sorted(ancilla_data, key=coord_sort_key)
    ancilla_pos = {c: k for k, c in enumerate(ordered_ancilla_coords)}
    n_ancilla = len(ancilla_targets)

    if ancilla_targets:
        circuit.append("RX", ancilla_targets)

    # ======== Merge rounds ========
    merge_first = _build_merge_first(prepend_shift=not is_first_round_of_circuit)
    merge_rest = _build_merge_rest()
    append_repeated_phase(
        circuit,
        merge_rounds,
        merge_first,
        middle_round=merge_rest,
        last_round=merge_rest,
    )
    is_first_round_of_circuit = False

    # ======== Destructive ancilla MX + observables + TYPE-C final detectors ========
    if ancilla_targets:
        circuit.append("MX", ancilla_targets)

    # OBSERVABLE_INCLUDE: controlled-X correction from ancilla-data destructive MX.
    controlled_x_targets: List[stim.GateTarget] = []
    for idx, qubit in enumerate(ancilla_targets):
        if index_to_coord[qubit].real == 1:
            controlled_x_targets.append(stim.target_rec(idx - n_ancilla))
    if controlled_x_targets:
        circuit.append("OBSERVABLE_INCLUDE", controlled_x_targets, 0)

    # OBSERVABLE_INCLUDE: bridge-ZZ contribution from last merge round's Z-stabs
    # in the bridge region.
    lower_mid = 0
    upper_mid = z_distance + bridge_length + 1
    bridge_zz_targets: List[stim.GateTarget] = []
    for k, c in enumerate(merged_meas_z_coords):
        if lower_mid < c.imag < upper_mid:
            # Last merge-round Z-stab rec, shifted back by destructive MX.
            bridge_zz_targets.append(stim.target_rec(-(n_ancilla + n_merged_z - k)))
    if bridge_zz_targets:
        circuit.append("OBSERVABLE_INCLUDE", bridge_zz_targets, 0)

    # TYPE-C X-stab detectors: [last_merge_stab, *ancilla_destructive_recs_in_support]
    circuit.append("SHIFT_COORDS", [], (0, 0, 1))
    for m_coord, (ttype, _D_i, D_anc, basis) in cls.items():
        if ttype != "C" or basis != "X":
            continue
        last_merge_stab = _curr_merged("X", m_coord) - n_ancilla
        ancilla_recs = [
            stim.target_rec(-(n_ancilla - ancilla_pos[d])) for d in D_anc if d in ancilla_pos
        ]
        if not ancilla_recs:
            continue
        circuit.append(
            "DETECTOR",
            [stim.target_rec(last_merge_stab)] + ancilla_recs,
            (m_coord.real, m_coord.imag, 0),
        )

    circuit.append("TICK")

    # ======== Post-merge rounds (post_rounds >= 1) ========
    def _build_post_first(prepend_shift: bool) -> stim.Circuit:
        frag = stim.Circuit()
        if prepend_shift:
            frag.append("SHIFT_COORDS", [], (0, 0, 1))
        frag += build_surface_code_round_circuit(
            individual_resolved,
            individual_layers,
            individual_meas_x,
            individual_meas_z,
        )
        for it in individual_patch.tiles:
            m = it.measurement_qubit
            basis = it.basis
            D_i = frozenset(d for d in it.data_qubits.values() if d is not None)
            ttype, _D_i_cls, D_anc, _basis_cls = cls.get(m, ("A", D_i, frozenset(), basis))
            curr = _curr_indiv(basis, m)
            if basis == "X":
                last_merge = -(n_indiv + n_ancilla + n_merged_z + n_merged_x - merged_x_pos[m])
            else:
                last_merge = -(n_indiv + n_ancilla + n_merged_z - merged_z_pos[m])
            if ttype == "A":
                frag.append(
                    "DETECTOR",
                    [stim.target_rec(curr), stim.target_rec(last_merge)],
                    (m.real, m.imag, 0),
                )
            elif ttype == "B":
                # Only X-type TYPE B can be reconciled: ancilla destructive MX
                # cancels the X-parity contribution from the extra bridge data.
                if basis != "X":
                    continue
                anc_offsets = [
                    -(n_indiv + n_ancilla - ancilla_pos[d]) for d in D_anc if d in ancilla_pos
                ]
                refs = [stim.target_rec(curr), stim.target_rec(last_merge)] + [
                    stim.target_rec(o) for o in anc_offsets
                ]
                frag.append("DETECTOR", refs, (m.real, m.imag, 0))
            # TYPE C doesn't exist in individual_patch.
        return frag

    def _build_post_rest() -> stim.Circuit:
        frag = stim.Circuit()
        frag.append("SHIFT_COORDS", [], (0, 0, 1))
        frag += build_surface_code_round_circuit(
            individual_resolved,
            individual_layers,
            individual_meas_x,
            individual_meas_z,
        )
        for c in individual_meas_x_coords:
            curr = _curr_indiv("X", c)
            frag.append(
                "DETECTOR",
                [stim.target_rec(curr), stim.target_rec(curr - n_indiv)],
                (c.real, c.imag, 0),
            )
        for c in individual_meas_z_coords:
            curr = _curr_indiv("Z", c)
            frag.append(
                "DETECTOR",
                [stim.target_rec(curr), stim.target_rec(curr - n_indiv)],
                (c.real, c.imag, 0),
            )
        return frag

    post_first = _build_post_first(prepend_shift=True)
    post_rest = _build_post_rest()
    append_repeated_phase(
        circuit,
        post_rounds,
        post_first,
        middle_round=post_rest,
        last_round=post_rest,
    )

    # Collect bridge-boundary X-stab coords (kept for legacy interface; the
    # Y splice no longer needs to augment detectors with bridge-data refs
    # because our post-merge TYPE-B detectors already reconcile the bridge.
    missing_rec_stab_upper: List[Tuple[float, float]] = []
    missing_rec_stab_lower: List[Tuple[float, float]] = []
    for coord in merged_meas_x_coords:
        x_stab_x, x_stab_y = coord.real, coord.imag
        if x_stab_y == z_distance + 0.5:
            missing_rec_stab_upper.append((x_stab_x, x_stab_y))
        elif x_stab_y == z_distance + bridge_length + 0.5:
            missing_rec_stab_lower.append((x_stab_x, x_stab_y))

    return [], missing_rec_stab_upper, missing_rec_stab_lower, circuit


def s_gate_surgery(
    distance: int,
    bridge_length: int,
    pre_rounds: int,
    merge_rounds: int,
    post_rounds: int = 1,
) -> Tuple[List[int], List[Tuple[float, float]], List[Tuple[float, float]], stim.Circuit]:
    """Private helper — ZZ lattice surgery with X-initialized data patches.

    Returns ``(final_meas, upper_x_boundary_stabs, lower_x_boundary_stabs, annotated_circuit)``.
    The returned circuit already has all surgery-side DETECTOR annotations
    in-circuit; no external annotation needed.
    """
    return build_s_gate_circuit(
        distance,
        bridge_length,
        pre_rounds,
        merge_rounds,
        post_rounds,
    )


def s_gate(
    distance: int,
    bridge_length: int,
    pre_rounds: int,
    merge_rounds: int,
    boundary_rounds: int,
    post_rounds: int = 1,
    *,
    backend: Backend = "legacy",
) -> stim.Circuit:
    """ZZ-surgery + teleported logical-Y preparation.

    Both individual patches are square (``x_distance == z_distance == distance``),
    so a single ``distance`` parameter controls the code. The surgery half is
    built via :func:`s_gate_surgery` (with ``post_rounds`` user-controllable,
    default ``1``, minimum ``1``). The Y-magic circuit is built with
    ``memory_rounds=1`` hardcoded; its MPP preamble AND memory round are
    stripped during the splice. The stripped memory round is used only as a
    coordinate lookup table so the first Y-transition round's detector refs
    can be translated back to our surgery's last post-merge round
    measurements.
    """
    if post_rounds < 1:
        raise ValueError(f"s_gate requires post_rounds >= 1, got {post_rounds}.")

    if backend == "stimflow":
        from ._stimflow.s_gate import s_gate as _s_gate_chunks

        return _s_gate_chunks(
            distance,
            bridge_length,
            pre_rounds,
            merge_rounds,
            boundary_rounds,
            post_rounds,
        )

    _, upper_x, lower_x, surgery_circuit = s_gate_surgery(
        distance,
        bridge_length,
        pre_rounds,
        merge_rounds,
        post_rounds,
    )
    Y_circ = make_y_measurement_circuit(
        boundary_rounds=boundary_rounds,
        memory_rounds=1,
        distance=distance,
    )
    mapped_patches = [(0, 0), (0, distance + bridge_length)]
    merged, _mapping = _remap_circuit(
        distance,
        Y_circ,
        surgery_circuit,
        mapped_patches,
        upper_x,
        lower_x,
    )
    return merged


def _remap_circuit(
    distance: int,
    Y_circuit: stim.Circuit,
    base_circuit: stim.Circuit,
    mapped_patches,
    upper_x,
    lower_x,
):
    """Splice the Y-magic circuit onto the surgery base, stripping Y's MPP
    preamble AND its single memory round.

    The memory-round detectors' coordinate metadata is cached as a lookup:
    for any rec index in the stripped range (preamble + memory round), the
    lookup gives the Y-circuit stabilizer coord; we resolve that coord to
    the base circuit's last measurement of the mapped qubit via
    ``base_last_meas``. This makes stripped-rec references in ported
    detectors automatically point at our post-merge round measurements.
    """
    if mapped_patches is None:
        mapped_patches = [(0.0, 0.0)]

    canonical_coords, canonical_map, idx_to_coord = _build_base_coordinates(base_circuit)
    merged = stim.Circuit()

    y_coords_raw = _extract_coords(Y_circuit)
    mappings: Dict[Tuple[float, float], Dict[int, int]] = {}

    def ensure_coord(coord):
        if coord not in canonical_map:
            canonical_map[coord] = len(canonical_coords)
            idx_to_coord[len(canonical_coords)] = coord
            canonical_coords.append(coord)
            merged.append("QUBIT_COORDS", canonical_map[coord], coord)

    for dx, dy in mapped_patches:
        mapping: Dict[int, int] = {}
        for qubit, coord in y_coords_raw.items():
            shifted = (coord[0] + 1.0 + dx, coord[1] + 1.0 + dy)
            ensure_coord(shifted)
            mapping[qubit] = canonical_map[shifted]
        mappings[(dx, dy)] = mapping

    detector_coords = base_circuit.get_detector_coordinates()
    detector_round_offset = max(
        (coords[-1] for coords in detector_coords.values() if coords),
        default=0.0,
    )

    meas_ops = {"M", "MX", "MY", "MZ", "MR", "MRX", "MRY", "MRZ"}

    # Emit the base circuit, tracking each qubit's most-recent measurement rec.
    merged_last_meas: Dict[int, int] = {}
    merged_total_meas = 0
    for inst in base_circuit:
        merged.append(inst)
        if inst.name in meas_ops:
            for t in inst.targets_copy():
                merged_last_meas[t.value] = merged_total_meas
                merged_total_meas += 1

    base_last_meas = dict(merged_last_meas)

    Y_circuit_segs = _split_into_ticks(Y_circuit.flattened())

    # Seg 0 = MPP preamble.
    n_preamble = sum(
        inst.num_measurements
        for inst in Y_circuit_segs[0]
        if inst.name == "MPP" or inst.name in meas_ops
    )

    # The first segment containing a DETECTOR marks the END of Y's memory round.
    memory_end_seg_idx = None
    for seg_idx, seg in enumerate(Y_circuit_segs):
        if any(inst.name == "DETECTOR" for inst in seg):
            memory_end_seg_idx = seg_idx
            break
    if memory_end_seg_idx is None:
        raise ValueError("Y circuit has no DETECTOR segment to locate the memory round end.")

    # All measurements in segs 1..memory_end_seg_idx (inclusive) belong to the
    # stripped memory round.
    n_memory = sum(
        inst.num_measurements
        for seg in Y_circuit_segs[1 : memory_end_seg_idx + 1]
        for inst in seg
        if inst.name == "MPP" or inst.name in meas_ops
    )
    n_stripped = n_preamble + n_memory

    # Build unified coord lookup: for each rec index < n_stripped, record the
    # Y-circuit stabilizer coord. We read this off the memory-round detectors
    # (which pair an MPP rec with a memory-round rec at the same stabilizer
    # coord).
    y_reference_coord: Dict[int, Tuple[float, float]] = {}
    y_abs_probe = n_preamble
    for seg in Y_circuit_segs[1 : memory_end_seg_idx + 1]:
        for inst in seg:
            if inst.name == "MPP" or inst.name in meas_ops:
                y_abs_probe += inst.num_measurements
            elif inst.name == "DETECTOR":
                args = inst.gate_args_copy()
                if len(args) < 2:
                    continue
                coord = (args[0], args[1])
                for rec in inst.targets_copy():
                    p_idx = y_abs_probe + rec.value
                    if 0 <= p_idx < n_stripped:
                        y_reference_coord[p_idx] = coord

    # Per-mapping map from Y absolute measurement index to merged absolute
    # measurement index, populated as we splice each Y measurement.
    y_to_merged: Dict[Tuple[float, float], Dict[int, int]] = {mk: {} for mk in mappings}

    y_meas_total = n_stripped  # start the ported walk right after the stripped phase
    y_round_offset = 0.0

    def translate_rec(rec_value: int, y_abs_here: int, mapping_key):
        y_meas_idx = y_abs_here + rec_value
        if y_meas_idx < n_stripped:
            coord = y_reference_coord.get(y_meas_idx)
            if coord is None:
                return None
            dx_, dy_ = mapping_key
            shifted_coord = (coord[0] + 1.0 + dx_, coord[1] + 1.0 + dy_)
            base_q = canonical_map.get(shifted_coord)
            if base_q is None:
                return None
            abs_idx = base_last_meas.get(base_q)
            if abs_idx is None:
                return None
            return abs_idx - merged_total_meas
        merged_abs = y_to_merged[mapping_key].get(y_meas_idx)
        if merged_abs is None:
            return None
        return merged_abs - merged_total_meas

    # Port segments from memory_end_seg_idx + 1 onwards.
    for seg in Y_circuit_segs[memory_end_seg_idx + 1 :]:
        seg_is_meas = _is_measure_tick(seg)
        seg_n_meas = sum(
            inst.num_measurements for inst in seg if inst.name in meas_ops or inst.name == "MPP"
        )

        if not seg_is_meas:
            for (_dx, _dy), mapping in mappings.items():
                for inst in seg:
                    if inst.name == "SHIFT_COORDS":
                        continue
                    new_targets = [mapping[t.value] for t in inst.targets_copy()]
                    merged.append(inst.name, new_targets, inst.gate_args_copy())
            merged.append("TICK")
            y_meas_total += seg_n_meas
            continue

        for mapping_key, mapping in mappings.items():
            dx, dy = mapping_key
            y_abs = y_meas_total
            for inst in seg:
                if inst.name == "SHIFT_COORDS":
                    continue
                if inst.name in meas_ops:
                    new_targets = [mapping[t.value] for t in inst.targets_copy()]
                    merged.append(inst.name, new_targets, inst.gate_args_copy())
                    for q in new_targets:
                        merged_last_meas[q] = merged_total_meas
                        y_to_merged[mapping_key][y_abs] = merged_total_meas
                        merged_total_meas += 1
                        y_abs += 1
                elif inst.name == "DETECTOR":
                    args = list(inst.gate_args_copy())
                    if not args:
                        continue
                    args[-1] += detector_round_offset + y_round_offset
                    shifted = (args[0] + 1.0 + dx, args[1] + 1.0 + dy)
                    args[0], args[1] = shifted
                    new_targets: List[stim.GateTarget] = []
                    for rec in inst.targets_copy():
                        translated = translate_rec(rec.value, y_abs, mapping_key)
                        if translated is None:
                            continue
                        new_targets.append(stim.target_rec(translated))
                    if new_targets:
                        merged.append("DETECTOR", new_targets, args)
                elif inst.name == "OBSERVABLE_INCLUDE":
                    obs_targets: List[stim.GateTarget] = []
                    for rec in inst.targets_copy():
                        translated = translate_rec(rec.value, y_abs, mapping_key)
                        if translated is None:
                            continue
                        obs_targets.append(stim.target_rec(translated))
                    if obs_targets:
                        merged.append(
                            "OBSERVABLE_INCLUDE",
                            obs_targets,
                            inst.gate_args_copy(),
                        )
                else:
                    new_targets = [mapping[t.value] for t in inst.targets_copy()]
                    merged.append(inst.name, new_targets, inst.gate_args_copy())

        for inst in seg:
            if inst.name == "SHIFT_COORDS":
                shift_args = inst.gate_args_copy()
                if len(shift_args) >= 3:
                    y_round_offset += shift_args[2]

        merged.append("TICK")
        y_meas_total += seg_n_meas

    return merged, mappings
