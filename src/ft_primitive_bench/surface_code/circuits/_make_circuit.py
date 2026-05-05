from typing import Dict, Iterable, List, Optional, Tuple

import stim

from ._build_patch import rectangular_surface_code_patch
from ._tile import Patch
from ._utils import (
    append_repeated_phase,
    build_schedule,
    build_surface_code_round_circuit,
    c2xy,
    coord_sort_key,
    resolve_schedule,
)

__all__ = [
    "build_merged_patch",
    "lattice_surgery",
    "memory",
    "patch_midlines",
    "transversal_h",
]


def _build_rectangular_memory_raw(
    x_distance: int,
    z_distance: int,
    rounds: int,
    meas_basis: str = "Z",
) -> stim.Circuit:
    patch = rectangular_surface_code_patch(x_distance, z_distance)
    schedule, layer_count = build_schedule(patch)

    data = set(patch.data_set)
    meas = set(patch.measure_set)

    meas_basis = meas_basis.upper()
    if meas_basis not in {"X", "Z"}:
        raise ValueError("meas_basis must be 'X' or 'Z'")

    all_coords = sorted(data | meas, key=coord_sort_key)
    index_of = {coord: idx for idx, coord in enumerate(all_coords)}

    circuit = stim.Circuit()
    for coord in all_coords:
        circuit.append("QUBIT_COORDS", index_of[coord], list(c2xy(coord)))

    def targets(coords: Iterable[complex]) -> List[int]:
        return [index_of[c] for c in sorted(coords, key=coord_sort_key)]

    ordered_data = sorted(data, key=coord_sort_key)
    data_targets = [index_of[c] for c in ordered_data]
    data_order_index = {coord: idx for idx, coord in enumerate(ordered_data)}

    data_init_gate = {"Z": "RZ", "X": "RX"}[meas_basis]
    if data_targets:
        circuit.append(data_init_gate, data_targets)
        circuit.append("TICK")

    data_meas_gate = {"Z": "MZ", "X": "MX"}[meas_basis]

    meas_x_coords = sorted(patch.measure_x_set, key=coord_sort_key)
    meas_z_coords = sorted(patch.measure_z_set, key=coord_sort_key)
    meas_x_targets = [index_of[c] for c in meas_x_coords]
    meas_z_targets = [index_of[c] for c in meas_z_coords]
    n_x = len(meas_x_targets)
    n_z = len(meas_z_targets)
    stab_per_round = n_x + n_z

    resolved = resolve_schedule(schedule, index_of)

    def _round_body() -> stim.Circuit:
        return build_surface_code_round_circuit(
            resolved,
            layer_count,
            meas_x_targets,
            meas_z_targets,
        )

    # Rec offsets at end-of-round (after MX then MZ):
    #   Z-stab k:  -(n_z - k)                  (0-indexed in meas_z_coords)
    #   X-stab k:  -(n_z + n_x - k)            (0-indexed in meas_x_coords)
    # Previous-round same stab: subtract stab_per_round.

    # First round: basis-consistent single-rec detectors only.
    first_round_fragment = _round_body()
    if meas_basis == "Z":
        for k, coord in enumerate(meas_z_coords):
            first_round_fragment.append(
                "DETECTOR",
                [stim.target_rec(-(n_z - k))],
                (coord.real, coord.imag, 0),
            )
    else:  # meas_basis == "X"
        for k, coord in enumerate(meas_x_coords):
            first_round_fragment.append(
                "DETECTOR",
                [stim.target_rec(-(n_z + n_x - k))],
                (coord.real, coord.imag, 0),
            )

    # Subsequent rounds: two-rec curr-vs-prev comparison for every stabilizer.
    compare_round_fragment = stim.Circuit()
    compare_round_fragment.append("SHIFT_COORDS", [], (0, 0, 1))
    compare_round_fragment += _round_body()
    for k, coord in enumerate(meas_z_coords):
        curr = stim.target_rec(-(n_z - k))
        prev = stim.target_rec(-(n_z - k) - stab_per_round)
        compare_round_fragment.append("DETECTOR", [curr, prev], (coord.real, coord.imag, 0))
    for k, coord in enumerate(meas_x_coords):
        curr = stim.target_rec(-(n_z + n_x - k))
        prev = stim.target_rec(-(n_z + n_x - k) - stab_per_round)
        compare_round_fragment.append("DETECTOR", [curr, prev], (coord.real, coord.imag, 0))

    if rounds > 0:
        append_repeated_phase(
            circuit,
            rounds,
            first_round_fragment,
            middle_round=compare_round_fragment,
            last_round=compare_round_fragment,
        )

    if data_targets:
        circuit.append(data_meas_gate, data_targets)

    # Final-round data-parity detectors: for each meas_basis-stabilizer,
    # XOR its last-round measurement with the data qubits in its support.
    # After the data meas, recs are (from most recent backwards):
    #   data qubit i:    -(n_data - i)    i in 0..n_data (data_order_index)
    #   last-round Z k:  -(n_data + n_z - k)
    #   last-round X k:  -(n_data + n_z + n_x - k)
    if rounds > 0 and data_targets:
        n_data = len(data_targets)
        z_index = {coord: k for k, coord in enumerate(meas_z_coords)}
        x_index = {coord: k for k, coord in enumerate(meas_x_coords)}

        circuit.append("SHIFT_COORDS", [], (0, 0, 1))
        for tile in patch.tiles:
            if tile.basis != meas_basis:
                continue
            m_coord = tile.measurement_qubit
            if meas_basis == "Z":
                k = z_index.get(m_coord)
                if k is None:
                    continue
                last_stab_rec = stim.target_rec(-(n_data + n_z - k))
            else:
                k = x_index.get(m_coord)
                if k is None:
                    continue
                last_stab_rec = stim.target_rec(-(n_data + n_z + n_x - k))

            data_recs: List[stim.GateTarget] = []
            for dc in tile.data_qubits.values():
                if dc is None:
                    continue
                di = data_order_index.get(dc)
                if di is None:
                    continue
                data_recs.append(stim.target_rec(-(n_data - di)))

            circuit.append(
                "DETECTOR",
                [last_stab_rec, *data_recs],
                (m_coord.real, m_coord.imag, 0),
            )

    def logical_strip_coords() -> List[complex]:
        if not data:
            return []
        xs = [round(c2xy(q)[0]) for q in data]
        ys = [round(c2xy(q)[1]) for q in data]
        mid_x = (min(xs) + max(xs)) // 2
        mid_y = (min(ys) + max(ys)) // 2
        if meas_basis == "Z":
            return [q for q in ordered_data if round(c2xy(q)[0]) == mid_x]
        return [q for q in ordered_data if round(c2xy(q)[1]) == mid_y]

    strip_coords = logical_strip_coords()
    if strip_coords:
        rec_targets = []
        total = len(data_targets)
        for coord in strip_coords:
            idx = data_order_index.get(coord)
            if idx is None:
                continue
            offset = total - idx
            rec_targets.append(stim.target_rec(-offset))
        if rec_targets:
            circuit.append("OBSERVABLE_INCLUDE", rec_targets, 0)

    return circuit


def memory(
    x_distance: int,
    z_distance: int,
    rounds: int,
    meas_basis: str = "Z",
) -> stim.Circuit:
    if rounds < 1:
        raise ValueError(f"memory requires rounds >= 1, got {rounds}.")
    return _build_rectangular_memory_raw(
        x_distance,
        z_distance,
        rounds,
        meas_basis=meas_basis,
    )


def _build_rectangular_lattice_surgery_raw(
    x_distance: int,
    z_distance: int,
    bridge_length: int,
    pre_rounds: int,
    merge_rounds: int,
    post_rounds: int,
    meas_basis: str = "Z",
) -> stim.Circuit:
    """Construct a lattice surgery circuit for MZZ or MXX with optional surface-code memory rounds.
    The layout will be the first plaquette will always be Z type and the top/bottom boundary will be X type
    Therefore MZZ will be vertical and MXX will be horizonal
    """
    meas_basis = meas_basis.upper()
    if meas_basis not in {"X", "Z"}:
        raise ValueError("meas_basis must be 'X' or 'Z'")

    # The merged patch is rebuilt as a single rectangular_surface_code_patch
    # over the union extent. Its bulk checkerboard re-derives every tile's
    # basis from the global (ax + ay) parity. For the two individual patches'
    # tile bases to agree with the merged checkerboard at the same coords,
    # the offset that translates patch 2 into place must preserve parity:
    #   - MZZ (meas_basis="Z", vertical stack): offset = (0, z_distance + bridge_length)
    #   - MXX (meas_basis="X", horizontal stack): offset = (x_distance + bridge_length, 0)
    # The corresponding sum must be even; otherwise basis classification in
    # the merged patch silently disagrees with the individual patches and the
    # downstream rec lookup raises KeyError.
    stack_dim = x_distance if meas_basis == "X" else z_distance
    stack_label = "x_distance" if meas_basis == "X" else "z_distance"
    if (stack_dim + bridge_length) % 2 != 0:
        raise ValueError(
            f"lattice_surgery requires ({stack_label} + bridge_length) to be "
            f"even so that the merged-patch checkerboard aligns with the two "
            f"individual patches. Got {stack_label}={stack_dim}, "
            f"bridge_length={bridge_length}, meas_basis={meas_basis!r}."
        )

    boundary_presets: Dict[str, Dict[str, str]] = {
        "Z": {"top": "X", "bot": "X", "left": "Z", "right": "Z"},
        "X": {"top": "X", "bot": "X", "left": "Z", "right": "Z"},
        # "X": {"top": "Z", "bot": "Z", "left": "X", "right": "X"},
    }

    first_patch, second_patch, individual_patch, merged_patch = _build_merged_patch_components(
        x_distance,
        z_distance,
        bridge_length,
        horizontal=(meas_basis == "X"),
        individual_boundaries=boundary_presets[meas_basis],
        merged_boundaries=boundary_presets[meas_basis],
    )

    individual_patch._invalidate_cached_sets()
    circuit = stim.Circuit()

    merged_data = set(merged_patch.data_set)
    merged_meas = set(merged_patch.measure_set)
    # Include individual patch qubits too: boundary stabilizer qubits can lie
    # outside the merged patch extent and must be reachable via index_of.
    individual_all = set(individual_patch.data_set) | set(individual_patch.measure_set)
    all_qubits = sorted(merged_data | merged_meas | individual_all, key=coord_sort_key)
    index_of = {coord: idx for idx, coord in enumerate(all_qubits)}

    for coord in all_qubits:
        circuit.append("QUBIT_COORDS", index_of[coord], list(c2xy(coord)))

    def targets(coords: Iterable[complex]) -> List[int]:
        return [index_of[c] for c in sorted(coords, key=coord_sort_key)]

    individual_data = set(individual_patch.data_set)
    data_init_gate = {"Z": "RZ", "X": "RX"}[meas_basis]
    anc_data_init_gate = {"Z": "RX", "X": "RZ"}[meas_basis]
    data_meas_gate = {"Z": "MZ", "X": "MX"}[meas_basis]
    anc_data_meas_gate = {"Z": "MX", "X": "MZ"}[meas_basis]
    data_init_targets = targets(individual_data)
    # logical qubit 1,2 data qubit is initialized
    if data_init_targets:
        circuit.append(data_init_gate, data_init_targets)
        circuit.append("TICK")

    individual_schedule, individual_layers = build_schedule(individual_patch)
    merged_schedule, merged_layers = build_schedule(merged_patch)

    # Pre-resolve coord→index lookups once for each schedule.
    individual_resolved = resolve_schedule(individual_schedule, index_of)
    merged_resolved = resolve_schedule(merged_schedule, index_of)

    def ordered_coords(coords: Iterable[complex]) -> List[complex]:
        return sorted(coords, key=coord_sort_key)

    individual_meas_x_coords = ordered_coords(individual_patch.measure_x_set)
    individual_meas_z_coords = ordered_coords(individual_patch.measure_z_set)
    merged_meas_x_coords = ordered_coords(merged_patch.measure_x_set)
    merged_meas_z_coords = ordered_coords(merged_patch.measure_z_set)

    individual_meas_x = [index_of[c] for c in individual_meas_x_coords]
    individual_meas_z = [index_of[c] for c in individual_meas_z_coords]
    merged_meas_x = [index_of[c] for c in merged_meas_x_coords]
    merged_meas_z = [index_of[c] for c in merged_meas_z_coords]
    individual_round = build_surface_code_round_circuit(
        individual_resolved,
        individual_layers,
        individual_meas_x,
        individual_meas_z,
    )
    merged_round = build_surface_code_round_circuit(
        merged_resolved,
        merged_layers,
        merged_meas_x,
        merged_meas_z,
    )

    # ======== Detector bookkeeping ========
    ancilla_basis = "X" if meas_basis == "Z" else "Z"
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
        """Rec offset (relative to end of an individual round) for stab `coord`."""
        if basis == "X":
            return -(n_indiv_z + n_indiv_x - indiv_x_pos[coord])
        return -(n_indiv_z - indiv_z_pos[coord])

    def _curr_merged(basis: str, coord: complex) -> int:
        """Rec offset (relative to end of a merged round) for stab `coord`."""
        if basis == "X":
            return -(n_merged_z + n_merged_x - merged_x_pos[coord])
        return -(n_merged_z - merged_z_pos[coord])

    # Classify merged tiles against individual_patch (TYPE A/B/C).
    individual_tile_by_coord = {t.measurement_qubit: t for t in individual_patch.tiles}
    # cls[m_coord] = (type, D_i, D_anc, basis)
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

    # ======== Phase fragments: pre-merge (individual) ========
    # pre_first: (no shift if first of circuit) + round body + basis-consistent single-rec detectors.
    # pre_rest:  shift + round body + 2-rec compare for every indiv stab.
    def _build_pre_first(prepend_shift: bool) -> stim.Circuit:
        frag = stim.Circuit()
        if prepend_shift:
            frag.append("SHIFT_COORDS", [], (0, 0, 1))
        frag += individual_round
        for c in individual_meas_z_coords if meas_basis == "Z" else individual_meas_x_coords:
            frag.append(
                "DETECTOR",
                [stim.target_rec(_curr_indiv(meas_basis, c))],
                (c.real, c.imag, 0),
            )
        return frag

    def _build_pre_rest() -> stim.Circuit:
        frag = stim.Circuit()
        frag.append("SHIFT_COORDS", [], (0, 0, 1))
        frag += individual_round
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

    # ======== Phase fragments: merge (merged schedule) ========
    # merge_first: transition detectors per TYPE A/B/C rules. Prev-round recs (if pre_rounds>0)
    # point back to pre-merge last individual round; otherwise single-rec.
    def _build_merge_first(prepend_shift: bool) -> stim.Circuit:
        frag = stim.Circuit()
        if prepend_shift:
            frag.append("SHIFT_COORDS", [], (0, 0, 1))
        frag += merged_round
        for m_coord, (ttype, _D_i, _D_anc, basis) in cls.items():
            curr = _curr_merged(basis, m_coord)
            # Prev rec back in pre-merge last indiv round (if applicable).
            prev_ok = pre_rounds > 0 and m_coord in (indiv_x_pos if basis == "X" else indiv_z_pos)
            if prev_ok:
                if basis == "X":
                    prev = -(n_merged + n_indiv_z + n_indiv_x - indiv_x_pos[m_coord])
                else:
                    prev = -(n_merged + n_indiv_z - indiv_z_pos[m_coord])
            else:
                prev = None

            if ttype == "A":
                # Deterministic if (prev_ok) OR (pre_rounds==0 and basis==meas_basis).
                if prev_ok:
                    frag.append(
                        "DETECTOR",
                        [stim.target_rec(curr), stim.target_rec(prev)],
                        (m_coord.real, m_coord.imag, 0),
                    )
                elif basis == meas_basis:
                    frag.append(
                        "DETECTOR",
                        [stim.target_rec(curr)],
                        (m_coord.real, m_coord.imag, 0),
                    )
                # else skip (non-deterministic)
            elif ttype == "B":
                # Deterministic only if basis == ancilla_basis AND prev_ok (ancilla XOR = +1).
                if prev_ok and basis == ancilla_basis:
                    frag.append(
                        "DETECTOR",
                        [stim.target_rec(curr), stim.target_rec(prev)],
                        (m_coord.real, m_coord.imag, 0),
                    )
                # else skip
            else:  # TYPE C
                # Deterministic if basis == ancilla_basis.
                if basis == ancilla_basis:
                    frag.append(
                        "DETECTOR",
                        [stim.target_rec(curr)],
                        (m_coord.real, m_coord.imag, 0),
                    )
        return frag

    def _build_merge_rest() -> stim.Circuit:
        frag = stim.Circuit()
        frag.append("SHIFT_COORDS", [], (0, 0, 1))
        frag += merged_round
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

    # ======== Pre-merge emission ========
    is_first_round_of_circuit = True
    if pre_rounds > 0:
        pre_first = _build_pre_first(prepend_shift=not is_first_round_of_circuit)
        pre_rest = _build_pre_rest()
        append_repeated_phase(
            circuit,
            pre_rounds,
            pre_first,
            middle_round=pre_rest,
            last_round=pre_rest,
        )
        is_first_round_of_circuit = False

    if merge_rounds > 0:
        ancilla_data = merged_data - individual_data
        ancilla_targets = targets(ancilla_data)
        ordered_ancilla_coords = sorted(ancilla_data, key=coord_sort_key)
        ancilla_pos = {c: k for k, c in enumerate(ordered_ancilla_coords)}
        n_ancilla = len(ancilla_targets)

        if ancilla_targets:
            circuit.append(anc_data_init_gate, ancilla_targets)
            circuit.append("TICK")

        # ======== Merge emission ========
        merge_first_frag = _build_merge_first(prepend_shift=not is_first_round_of_circuit)
        merge_rest_frag = _build_merge_rest()
        append_repeated_phase(
            circuit,
            merge_rounds,
            merge_first_frag,
            middle_round=merge_rest_frag,
            last_round=merge_rest_frag,
        )
        is_first_round_of_circuit = False

        selected_positions = _logical_bridge_measurement_positions(
            first_patch=first_patch,
            second_patch=second_patch,
            merged_meas_x_coords=merged_meas_x_coords,
            merged_meas_z_coords=merged_meas_z_coords,
            meas_basis=meas_basis,
        )

        if selected_positions:
            rec_targets: List[stim.GateTarget] = []
            if meas_basis == "X":
                trailing_z = len(merged_meas_z)
                total_x = len(merged_meas_x)
                for pos in selected_positions:
                    offset = trailing_z + (total_x - pos)
                    rec_targets.append(stim.target_rec(-offset))
            else:
                total_z = len(merged_meas_z)
                for pos in selected_positions:
                    offset = total_z - pos
                    rec_targets.append(stim.target_rec(-offset))
            if rec_targets:
                circuit.append("OBSERVABLE_INCLUDE", rec_targets, 0)

        # ======== Destructive ancilla meas + TYPE-C final detectors ========
        if ancilla_targets:
            circuit.append(anc_data_meas_gate, ancilla_targets)
            circuit.append("TICK")

        circuit.append("SHIFT_COORDS", [], (0, 0, 1))
        for m_coord, (ttype, _D_i, D_anc, basis) in cls.items():
            if ttype != "C" or basis != ancilla_basis:
                continue
            # Last-merge-round stab rec, shifted back by n_ancilla destructive recs.
            last_merge_stab = _curr_merged(basis, m_coord) - n_ancilla
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

        # ======== Post-merge emission ========
        # post_first: individual round body + (TYPE A: 2-rec compare to last merge;
        # TYPE B: 3-rec compare with last merge + ancilla destructive recs).
        def _build_post_first(prepend_shift: bool) -> stim.Circuit:
            frag = stim.Circuit()
            if prepend_shift:
                frag.append("SHIFT_COORDS", [], (0, 0, 1))
            frag += individual_round
            for it in individual_patch.tiles:
                m = it.measurement_qubit
                basis = it.basis
                D_i = frozenset(d for d in it.data_qubits.values() if d is not None)
                ttype, _D_i_cls, D_anc, _basis_cls = cls.get(m, ("A", D_i, frozenset(), basis))
                curr = _curr_indiv(basis, m)
                # Last merge-round rec of same stab (relative to end of post_first round):
                #   back by n_indiv (this round) + n_ancilla (destructive meas) + position within merged round.
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
                    anc_offsets = [
                        -(n_indiv + n_ancilla - ancilla_pos[d]) for d in D_anc if d in ancilla_pos
                    ]
                    refs = [stim.target_rec(curr), stim.target_rec(last_merge)] + [
                        stim.target_rec(o) for o in anc_offsets
                    ]
                    frag.append("DETECTOR", refs, (m.real, m.imag, 0))
                # ttype == "C" doesn't apply (TYPE C not in individual_patch)
            return frag

        def _build_post_rest() -> stim.Circuit:
            frag = stim.Circuit()
            frag.append("SHIFT_COORDS", [], (0, 0, 1))
            frag += individual_round
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

        if post_rounds > 0:
            post_first_frag = _build_post_first(prepend_shift=True)
            post_rest_frag = _build_post_rest()
            append_repeated_phase(
                circuit,
                post_rounds,
                post_first_frag,
                middle_round=post_rest_frag,
                last_round=post_rest_frag,
            )

        # ======== Final data meas + data-parity detectors ========
        if data_init_targets:
            circuit.append(data_meas_gate, data_init_targets)
            circuit.append("TICK")

        ordered_indiv_data_coords = sorted(individual_data, key=coord_sort_key)
        data_pos = {c: k for k, c in enumerate(ordered_indiv_data_coords)}
        n_data = len(data_init_targets)

        circuit.append("SHIFT_COORDS", [], (0, 0, 1))
        for it in individual_patch.tiles:
            if it.basis != meas_basis:
                continue
            m = it.measurement_qubit
            D_i = frozenset(d for d in it.data_qubits.values() if d is not None)
            ttype = cls.get(m, ("A", D_i, frozenset(), meas_basis))[0]

            if post_rounds > 0:
                last_stab = _curr_indiv(meas_basis, m) - n_data
            else:
                # Last measurement of S was in the last merge round.
                # Offset relative to end of data meas = -(n_data + n_ancilla + merged_offset_of_S).
                if meas_basis == "X":
                    last_stab = -(n_data + n_ancilla + n_merged_z + n_merged_x - merged_x_pos[m])
                else:
                    last_stab = -(n_data + n_ancilla + n_merged_z - merged_z_pos[m])
                # Skip TYPE B at post_rounds==0 (last merge measurement was random in meas_basis).
                if ttype == "B":
                    continue

            data_recs = [stim.target_rec(-(n_data - data_pos[d])) for d in D_i if d in data_pos]
            if not data_recs:
                continue
            circuit.append(
                "DETECTOR",
                [stim.target_rec(last_stab)] + data_recs,
                (m.real, m.imag, 0),
            )

    return circuit


def lattice_surgery(
    x_distance: int,
    z_distance: int,
    bridge_length: int,
    pre_rounds: int,
    merge_rounds: int,
    post_rounds: int,
    meas_basis: str = "Z",
) -> stim.Circuit:
    return _build_rectangular_lattice_surgery_raw(
        x_distance,
        z_distance,
        bridge_length,
        pre_rounds,
        merge_rounds,
        post_rounds,
        meas_basis=meas_basis,
    )


def build_merged_patch(
    x_distance: int,
    z_distance: int,
    bridge_length: int,
    *,
    horizontal: bool,
    individual_boundaries: Optional[Dict[str, str]] = None,
    merged_boundaries: Optional[Dict[str, str]] = None,
) -> Tuple[Patch, Patch]:
    """Build two individual patches and their merged lattice-surgery patch."""
    _, _, individual_patch, merged_patch = _build_merged_patch_components(
        x_distance,
        z_distance,
        bridge_length,
        horizontal=horizontal,
        individual_boundaries=individual_boundaries,
        merged_boundaries=merged_boundaries,
    )
    return individual_patch, merged_patch


def _build_merged_patch_components(
    x_distance: int,
    z_distance: int,
    bridge_length: int,
    *,
    horizontal: bool,
    individual_boundaries: Optional[Dict[str, str]] = None,
    merged_boundaries: Optional[Dict[str, str]] = None,
) -> Tuple[Patch, Patch, Patch, Patch]:
    """Return the two logical patches, their union, and the merged LS patch."""
    if x_distance <= 0 or z_distance <= 0 or bridge_length <= 0:
        raise ValueError("x_distance, z_distance, bridge_length must be positive integers")

    default_boundaries = {"top": "X", "bot": "X", "left": "Z", "right": "Z"}

    def normalize(boundaries: Optional[Dict[str, str]]) -> Dict[str, str]:
        allowed_keys = set(default_boundaries)
        base = default_boundaries.copy()
        if boundaries:
            unexpected = set(boundaries) - allowed_keys
            if unexpected:
                raise ValueError(f"unexpected boundary keys: {sorted(unexpected)}")
            for key, value in boundaries.items():
                base[key] = value.upper()
        return base

    def make_patch(patch_width, patch_height: int, boundaries: Dict[str, str]) -> Patch:
        return rectangular_surface_code_patch(
            patch_width,
            patch_height,
            top_bottom_basis=boundaries["top"],
            left_right_basis=boundaries["left"],
        )

    individual_boundaries = normalize(individual_boundaries)
    merged_boundaries = normalize(merged_boundaries)

    first_patch = make_patch(x_distance, z_distance, individual_boundaries)
    second_patch = make_patch(x_distance, z_distance, individual_boundaries)
    if horizontal:
        second_patch.offset(x_distance + bridge_length, 0)
    else:
        second_patch.offset(0, z_distance + bridge_length)

    individual_patch = Patch(first_patch.tiles)
    individual_patch.extend(second_patch.tiles)

    merged_extent = (
        (x_distance, z_distance * 2 + bridge_length)
        if not horizontal
        else (x_distance * 2 + bridge_length, z_distance)
    )
    merged_patch = make_patch(*merged_extent, merged_boundaries)

    return first_patch, second_patch, individual_patch, merged_patch


def _logical_operator_data_chain(patch: Patch, basis: str) -> List[complex]:
    """Return the canonical midline logical operator data chain for a patch."""
    basis = basis.upper()
    ordered_data = sorted(patch.data_set, key=coord_sort_key)
    if not ordered_data:
        return []

    xs = sorted({coord.real for coord in ordered_data})
    ys = sorted({coord.imag for coord in ordered_data})
    if basis == "X":
        mid_x = xs[len(xs) // 2]
        return [coord for coord in ordered_data if coord.real == mid_x]
    if basis == "Z":
        mid_y = ys[len(ys) // 2]
        return [coord for coord in ordered_data if coord.imag == mid_y]
    raise ValueError(f"Unsupported logical basis {basis!r}")


def _logical_bridge_measurement_positions(
    *,
    first_patch: Patch,
    second_patch: Patch,
    merged_meas_x_coords: List[complex],
    merged_meas_z_coords: List[complex],
    meas_basis: str,
) -> List[int]:
    """Return merged-patch stabilizer positions between the two logical operators."""
    meas_basis = meas_basis.upper()
    first_chain = _logical_operator_data_chain(first_patch, meas_basis)
    second_chain = _logical_operator_data_chain(second_patch, meas_basis)
    if not first_chain or not second_chain:
        return []

    if meas_basis == "X":
        left_x, right_x = sorted((first_chain[0].real, second_chain[0].real))
        return [
            idx for idx, coord in enumerate(merged_meas_x_coords) if left_x < coord.real < right_x
        ]
    if meas_basis == "Z":
        lower_y, upper_y = sorted((first_chain[0].imag, second_chain[0].imag))
        return [
            idx for idx, coord in enumerate(merged_meas_z_coords) if lower_y < coord.imag < upper_y
        ]
    raise ValueError(f"Unsupported logical basis {meas_basis!r}")


def patch_midlines(patch: Patch, *, horizontal: bool) -> Tuple[float, float]:
    """Return midline pairs for vertical (y-axis) or horizontal (x-axis) stacks."""
    coords = patch.measure_set
    if not coords:
        return (0.0, 0.0)

    axis_values = sorted((coord.real if horizontal else coord.imag) for coord in coords)
    if not axis_values:
        return (0.0, 0.0)

    max_gap = -1.0
    gap_idx = -1
    for i in range(len(axis_values) - 1):
        gap = axis_values[i + 1] - axis_values[i]
        if gap > max_gap:
            max_gap = gap
            gap_idx = i

    if gap_idx == -1:
        mid = (min(axis_values) + max(axis_values)) / 2
        return (mid, mid)

    lower_segment = axis_values[: gap_idx + 1]
    upper_segment = axis_values[gap_idx + 1 :]
    if not upper_segment:
        upper_segment = lower_segment

    lower_mid = (min(lower_segment) + max(lower_segment)) / 2
    upper_mid = (min(upper_segment) + max(upper_segment)) / 2
    return (lower_mid, upper_mid)


def transversal_h(
    x_distance: int,
    z_distance: int,
    pre_rounds: int,
    post_rounds: int,
    meas_basis: str = "Z",
) -> stim.Circuit:
    """Transversal logical Hadamard between ``pre_rounds`` stabilizer rounds
    before the H layer and ``post_rounds`` rounds after it (with X/Z
    stabilizer roles swapped post-H)."""
    transversal_gate = "H"
    patch = rectangular_surface_code_patch(x_distance, z_distance)
    schedule, layer_count = build_schedule(patch)

    data = set(patch.data_set)
    meas = set(patch.measure_set)

    meas_basis = meas_basis.upper()
    if meas_basis not in {"X", "Z"}:
        raise ValueError("meas_basis must be 'X' or 'Z'")

    all_coords = sorted(data | meas, key=coord_sort_key)
    index_of = {coord: idx for idx, coord in enumerate(all_coords)}
    idx2coord = {idx: coord for idx, coord in enumerate(all_coords)}

    circuit = stim.Circuit()
    for coord in all_coords:
        circuit.append("QUBIT_COORDS", index_of[coord], list(c2xy(coord)))

    def targets(coords: Iterable[complex]) -> List[int]:
        return [index_of[c] for c in sorted(coords, key=coord_sort_key)]

    ordered_data = sorted(data, key=coord_sort_key)
    data_targets = [index_of[c] for c in ordered_data]
    data_order_index = {coord: idx for idx, coord in enumerate(ordered_data)}

    data_init_gate = {"Z": "RZ", "X": "RX"}[meas_basis]
    if data_targets:
        circuit.append(data_init_gate, data_targets)
        circuit.append("TICK")

    meas_x_targets = targets(patch.measure_x_set)
    meas_z_targets = targets(patch.measure_z_set)

    # Pre-resolve coord→index lookups once; reused across all rounds.
    resolved = resolve_schedule(schedule, index_of)

    def build_deterministic_round_fragment() -> stim.Circuit:
        fragment = build_surface_code_round_circuit(
            resolved,
            layer_count,
            meas_x_targets,
            meas_z_targets,
        )
        if meas_basis == "Z":
            for i in range(len(meas_z_targets)):
                coord = idx2coord[meas_z_targets[-(i + 1)]]
                fragment.append("DETECTOR", [stim.target_rec(-i - 1)], (coord.real, coord.imag, 0))
        else:
            for i in range(len(meas_x_targets)):
                coord = idx2coord[meas_x_targets[-(i + 1)]]
                fragment.append(
                    "DETECTOR",
                    [stim.target_rec(-i - 1 - len(meas_z_targets))],
                    (coord.real, coord.imag, 0),
                )
        return fragment

    def build_compare_round_fragment(
        *, swap_xz_roles: bool = False, measure_z_first: bool = False
    ) -> stim.Circuit:
        fragment = stim.Circuit()
        fragment.append("SHIFT_COORDS", [], (0, 0, 1))
        fragment += build_surface_code_round_circuit(
            resolved,
            layer_count,
            meas_x_targets,
            meas_z_targets,
            swap_xz_roles=swap_xz_roles,
            measure_z_first=measure_z_first,
        )
        # Round body emits MX-then-MZ by default, MZ-then-MX when measure_z_first.
        # time_ordered lists qubits in measurement-time order, so time_ordered[-(i+1)]
        # is the qubit at rec[-(i+1)].
        if measure_z_first:
            time_ordered = list(meas_z_targets) + list(meas_x_targets)
        else:
            time_ordered = list(meas_x_targets) + list(meas_z_targets)
        num_meas = len(time_ordered)
        for i in range(num_meas):
            coord = idx2coord[time_ordered[-(i + 1)]]
            fragment.append(
                "DETECTOR",
                [stim.target_rec(-i - 1), stim.target_rec(-i - 1 - num_meas)],
                (coord.real, coord.imag, 0),
            )
        return fragment

    circuit += build_deterministic_round_fragment()
    append_repeated_phase(circuit, max(0, pre_rounds), build_compare_round_fragment())

    circuit.append(transversal_gate, data_targets)
    circuit.append("TICK")
    meas_basis = "Z" if meas_basis == "X" else "X"
    data_meas_gate = {"Z": "MZ", "X": "MX"}[meas_basis]
    meas_x_targets, meas_z_targets = meas_z_targets, meas_x_targets

    append_repeated_phase(
        circuit,
        max(0, post_rounds),
        build_compare_round_fragment(swap_xz_roles=True, measure_z_first=True),
    )

    circuit.append(data_meas_gate, data_targets)

    def logical_strip_coords() -> List[complex]:
        if not data:
            return []
        xs = [round(c2xy(q)[0]) for q in data]
        ys = [round(c2xy(q)[1]) for q in data]
        mid_x = (min(xs) + max(xs)) // 2
        mid_y = (min(ys) + max(ys)) // 2
        if meas_basis == "X":
            return [q for q in ordered_data if round(c2xy(q)[0]) == mid_x]
        return [q for q in ordered_data if round(c2xy(q)[1]) == mid_y]

    strip_coords = logical_strip_coords()
    if strip_coords:
        rec_targets = []
        total = len(data_targets)
        for coord in strip_coords:
            idx = data_order_index.get(coord)
            if idx is None:
                continue
            offset = total - idx
            rec_targets.append(stim.target_rec(-offset))
        if rec_targets:
            circuit.append("OBSERVABLE_INCLUDE", rec_targets, 0)

    selected = [tile for tile in patch.tiles if tile.basis == ("X" if meas_basis == "Z" else "Z")]

    total_meas = meas_z_targets + meas_x_targets + data_targets
    offset_of = {v: i for i, v in enumerate(total_meas)}
    n_total_meas = len(total_meas)
    if selected:
        circuit.append("SHIFT_COORDS", [], (0, 0, 1))
    for tile in selected:
        m_coord = tile.measurement_qubit
        total_idx = [index_of[m_coord]] + [
            index_of[coord] for coord in tile.data_qubits.values() if coord is not None
        ]
        stim_rec_targets = [stim.target_rec(-n_total_meas + offset_of[idx]) for idx in total_idx]
        circuit.append("DETECTOR", stim_rec_targets, (m_coord.real, m_coord.imag, 0))

    return circuit
