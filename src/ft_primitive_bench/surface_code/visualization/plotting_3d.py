"""Spacetime (3D) visualization of surface-code patches.

One ``Patch`` is rendered as a flat layer at each round; stacking several
segments along ``z`` gives a spacetime view of a primitive that changes
shape over time (lattice surgery, S-gate, transversal H).

Each primitive wrapper (``plot_memory_timeline``, ``plot_transversal_h_timeline``,
``plot_lattice_surgery_timeline``, ``plot_s_gate_timeline``) parses the primitive's
round structure to decide:
 - which rounds should annotate data qubits with H/S labels;
 - whether X/Z stabilizer roles swap after a transversal H.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

from ..circuits._build_patch import rectangular_surface_code_patch
from ..circuits._make_circuit import build_merged_patch
from ..circuits._tile import Patch, Tile

GateMap = Mapping[complex, str]
RoundGates = Union[GateMap, Sequence[Optional[GateMap]], None]
EdgeList = Optional[Sequence[Tuple[complex, complex]]]
# (patch, rounds)
# (patch, rounds, data_gates)
# (patch, rounds, data_gates, basis_swap)
# (patch, rounds, data_gates, basis_swap, xcy_edges)
Segment = Union[
    Tuple[Patch, int],
    Tuple[Patch, int, RoundGates],
    Tuple[Patch, int, RoundGates, bool],
    Tuple[Patch, int, RoundGates, bool, EdgeList],
]

# Stabilizer plaquette fill (semi-transparent mesh)
X_COLOR = "#d43d51"
Z_COLOR = "#3d6fd4"
Y_COLOR = "#3dc46b"
# Darker shades for stabilizer measurement-qubit nodes
X_NODE_COLOR = "#8b1a2a"
Z_NODE_COLOR = "#1d3a8a"
# CNOT spoke edges: darker than the plaquette fill, lighter than the
# measurement-node color, so the spoke reads as a distinct middle layer.
X_EDGE_COLOR = "#c26876"
Z_EDGE_COLOR = "#6787c8"

DATA_FILL = "#ffffff"
DATA_EDGE = "#111111"
WORLDLINE_COLOR = "rgba(60,60,60,0.75)"

DEFAULT_GATE_COLORS: Dict[str, str] = {
    "H": "#f5d300",
    "S": "#ff8a00",
    "S_DAG": "#ff8a00",
    "T": "#b84dff",
    "XCY": "#b84dff",
    "MY": "#3dc46b",
    "X": X_COLOR,
    "Y": Y_COLOR,
    "Z": Z_COLOR,
    # Transversal init / destructive-measurement on data qubits.
    "RX": X_COLOR,
    "RZ": Z_COLOR,
    "MX": X_COLOR,
    "MZ": Z_COLOR,
}
# Gate annotations that render with a text label on the qubit node.
# Kept empty so gates are distinguished purely by node colour + the figure
# legend (text overlays were cluttering the scene at paper scale).
TEXT_GATES: set = set()
_OTHER_GATE_COLOR = "#ff5aa0"


def _effective_basis(basis: str, *, swap: bool) -> str:
    b = basis.upper()
    if not swap:
        return b
    if b == "X":
        return "Z"
    if b == "Z":
        return "X"
    return b


def _iter_data_coords(tile: Tile) -> List[complex]:
    return [q for q in tile.data_qubits.values() if q is not None]


def _tile_polygon_corners(tile: Tile) -> List[Tuple[float, float]]:
    """2D corners of a tile's stabilizer region.

    Weight-4 tiles -> quad on the 4 data qubits. Weight-2 boundary tiles ->
    triangle (measurement qubit + 2 participating data qubits). Merged
    lattice-surgery tiles (second_measurement_qubit set) -> rectangle spanning
    both measurement qubits.
    """
    measurement = tile.measurement_qubit
    data_points = _iter_data_coords(tile)

    if tile.second_measurement_qubit is not None:
        second = tile.second_measurement_qubit
        left = min(measurement.real, second.real) - 0.5
        right = max(measurement.real, second.real) + 0.5
        top = min(measurement.imag, second.imag) - 0.5
        bottom = max(measurement.imag, second.imag) + 0.5
        return [(left, top), (right, top), (right, bottom), (left, bottom)]

    unique_data = list(dict.fromkeys(data_points))
    cx = measurement.real
    cy = measurement.imag

    if len(unique_data) >= 3:
        pts = [(q.real, q.imag) for q in unique_data]
    elif len(unique_data) == 2:
        pts = [(cx, cy)] + [(q.real, q.imag) for q in unique_data]
    else:
        pts = [(cx, cy)] + [(q.real, q.imag) for q in unique_data]
        if len(pts) < 3:
            return [
                (cx - 0.5, cy - 0.5),
                (cx + 0.5, cy - 0.5),
                (cx + 0.5, cy + 0.5),
                (cx - 0.5, cy + 0.5),
            ]

    centroid_x = sum(p[0] for p in pts) / len(pts)
    centroid_y = sum(p[1] for p in pts) / len(pts)
    pts.sort(key=lambda p: math.atan2(p[1] - centroid_y, p[0] - centroid_x))
    return pts


def _fan_triangulation(n: int) -> Tuple[List[int], List[int], List[int]]:
    i_list: List[int] = []
    j_list: List[int] = []
    k_list: List[int] = []
    for i in range(1, n - 1):
        i_list.append(0)
        j_list.append(i)
        k_list.append(i + 1)
    return i_list, j_list, k_list


def _collect_patch_geometry(patch: Patch):
    tile_polys: List[Tuple[Tile, List[Tuple[float, float]]]] = []
    for tile in patch.tiles:
        tile_polys.append((tile, _tile_polygon_corners(tile)))
    data_coords = sorted(patch.data_set, key=lambda q: (q.imag, q.real))
    meas_coords = sorted(patch.measure_set, key=lambda q: (q.imag, q.real))
    return tile_polys, data_coords, meas_coords


def _layer_mesh(
    tile_polys: Sequence[Tuple[Tile, Sequence[Tuple[float, float]]]],
    z: float,
    *,
    alpha: float,
    basis_swap: bool,
) -> List[dict]:
    traces: List[dict] = []
    for effective, color in (("X", X_COLOR), ("Z", Z_COLOR)):
        xs: List[float] = []
        ys: List[float] = []
        zs: List[float] = []
        i_idx: List[int] = []
        j_idx: List[int] = []
        k_idx: List[int] = []
        offset = 0
        for tile, corners in tile_polys:
            if _effective_basis(tile.basis, swap=basis_swap) != effective:
                continue
            n = len(corners)
            if n < 3:
                continue
            for x, y in corners:
                xs.append(x)
                ys.append(-y)
                zs.append(z)
            fi, fj, fk = _fan_triangulation(n)
            i_idx.extend(i + offset for i in fi)
            j_idx.extend(j + offset for j in fj)
            k_idx.extend(k + offset for k in fk)
            offset += n
        if not xs:
            continue
        traces.append(
            dict(
                type="mesh3d",
                x=xs,
                y=ys,
                z=zs,
                i=i_idx,
                j=j_idx,
                k=k_idx,
                color=color,
                opacity=alpha,
                flatshading=True,
                hoverinfo="skip",
                name=f"{effective} stabilizer",
                legendgroup=f"{effective}-stab",
                showlegend=False,
            )
        )
    return traces


def _layer_spokes(
    tile_polys: Sequence[Tuple[Tile, Sequence[Tuple[float, float]]]],
    z: float,
    *,
    width: float,
    basis_swap: bool,
) -> List[dict]:
    """Measurement-to-data CNOT spokes per basis, rendered in a lighter hue."""
    traces: List[dict] = []
    for effective, color in (("X", X_EDGE_COLOR), ("Z", Z_EDGE_COLOR)):
        xs: List[Optional[float]] = []
        ys: List[Optional[float]] = []
        zs: List[Optional[float]] = []
        for tile, _ in tile_polys:
            if _effective_basis(tile.basis, swap=basis_swap) != effective:
                continue
            m = tile.measurement_qubit
            for q in _iter_data_coords(tile):
                xs.extend([m.real, q.real, None])
                ys.extend([-m.imag, -q.imag, None])
                zs.extend([z, z, None])
            if tile.second_measurement_qubit is not None:
                m2 = tile.second_measurement_qubit
                xs.extend([m.real, m2.real, None])
                ys.extend([-m.imag, -m2.imag, None])
                zs.extend([z, z, None])
        if not xs:
            continue
        traces.append(
            dict(
                type="scatter3d",
                x=xs,
                y=ys,
                z=zs,
                mode="lines",
                line=dict(color=color, width=width),
                hoverinfo="skip",
                showlegend=False,
            )
        )
    return traces


def _layer_meas_points(
    tile_polys: Sequence[Tuple[Tile, Sequence[Tuple[float, float]]]],
    z: float,
    *,
    size: float,
    basis_swap: bool,
    gates: Optional[GateMap] = None,
    gate_colors: Optional[Mapping[str, str]] = None,
) -> List[dict]:
    """Stabilizer measurement nodes, colored by (possibly swapped) basis.

    When ``gates`` is supplied and contains an ancilla coordinate, the ancilla
    is rendered as a gate-annotated marker instead of the plain basis-colored
    node — this is how H on stabilizer qubits during S-gate Y-magic shows up.
    """
    palette = gate_colors if gate_colors is not None else DEFAULT_GATE_COLORS
    by_basis: Dict[str, List[complex]] = {"X": [], "Z": []}
    gated: List[Tuple[complex, str, str]] = []
    seen: set = set()

    def consider(m: complex, effective: str) -> None:
        if m in seen:
            return
        seen.add(m)
        name = None if gates is None else gates.get(m)
        if name is not None:
            color = palette.get(name.upper(), _OTHER_GATE_COLOR)
            gated.append((m, name.upper(), color))
        elif effective in by_basis:
            by_basis[effective].append(m)

    for tile, _ in tile_polys:
        effective = _effective_basis(tile.basis, swap=basis_swap)
        consider(tile.measurement_qubit, effective)
        if tile.second_measurement_qubit is not None:
            consider(tile.second_measurement_qubit, effective)

    # Lift ancilla nodes slightly above the spoke layer so the CNOT lines
    # don't bleed through the node body.
    z_node = z + 0.02

    traces: List[dict] = []
    for basis, color in (("X", X_NODE_COLOR), ("Z", Z_NODE_COLOR)):
        coords = by_basis[basis]
        if not coords:
            continue
        traces.append(
            dict(
                type="scatter3d",
                x=[q.real for q in coords],
                y=[-q.imag for q in coords],
                z=[z_node] * len(coords),
                mode="markers",
                marker=dict(
                    size=size,
                    color=color,
                    opacity=1.0,
                    line=dict(color="white", width=1.2),
                ),
                hoverinfo="skip",
                showlegend=False,
            )
        )
    for q, name, color in gated:
        show_text = name.upper() in TEXT_GATES
        trace = dict(
            type="scatter3d",
            x=[q.real],
            y=[-q.imag],
            z=[z_node],
            mode="markers+text" if show_text else "markers",
            marker=dict(
                size=size * 1.6,
                color=color,
                opacity=1.0,
                line=dict(color="white", width=1.2),
            ),
            hoverinfo="skip",
            showlegend=False,
        )
        if show_text:
            trace["text"] = [name]
            trace["textposition"] = "top center"
            trace["textfont"] = dict(size=13, color="#111")
        traces.append(trace)
    return traces


def _layer_xcy_edges(
    edges: Sequence[Tuple[complex, complex]],
    z: float,
    *,
    color: str = "#b84dff",
    width: float = 4.5,
) -> List[dict]:
    """Two-qubit XCY gate edges: a highlighted line between control and target,
    plus a small midpoint marker.
    """
    if not edges:
        return []
    xs: List[Optional[float]] = []
    ys: List[Optional[float]] = []
    zs: List[Optional[float]] = []
    mx: List[float] = []
    my: List[float] = []
    for a, b in edges:
        xs.extend([a.real, b.real, None])
        ys.extend([-a.imag, -b.imag, None])
        zs.extend([z, z, None])
        mx.append((a.real + b.real) / 2.0)
        my.append(-(a.imag + b.imag) / 2.0)
    return [
        dict(
            type="scatter3d",
            x=xs,
            y=ys,
            z=zs,
            mode="lines",
            line=dict(color=color, width=width),
            hoverinfo="skip",
            showlegend=False,
        ),
        dict(
            type="scatter3d",
            x=mx,
            y=my,
            z=[z] * len(mx),
            mode="markers",
            marker=dict(size=4, color=color, opacity=0.9),
            hoverinfo="skip",
            showlegend=False,
        ),
    ]


def _bracket_annotation(
    z_bottom: float,
    z_top: float,
    x_anchor: float,
    y_anchor: float,
    label: str,
) -> List[dict]:
    """Draw a curly '}' bracket that spans [z_bottom, z_top] along the z-axis
    at ``(x_anchor, y_anchor)``, with ``label`` shown at the bracket tip.
    """
    spine = x_anchor
    hook = x_anchor - 0.45
    tip = x_anchor + 0.45
    z_mid = (z_bottom + z_top) / 2.0
    gap = min(0.18, max(0.05, (z_top - z_bottom) * 0.08))
    xs = [hook, spine, spine, tip, spine, spine, hook]
    ys = [-y_anchor] * 7
    zs = [z_top, z_top, z_mid + gap, z_mid, z_mid - gap, z_bottom, z_bottom]
    return [
        dict(
            type="scatter3d",
            x=xs,
            y=ys,
            z=zs,
            mode="lines",
            line=dict(color="#555", width=4),
            hoverinfo="skip",
            showlegend=False,
        ),
        dict(
            type="scatter3d",
            x=[tip + 0.15],
            y=[-y_anchor],
            z=[z_mid],
            mode="text",
            text=[label],
            textposition="middle right",
            textfont=dict(size=14, color="#333"),
            hoverinfo="skip",
            showlegend=False,
        ),
    ]


def _layer_data_points(
    data_coords: Sequence[complex],
    z: float,
    *,
    size: float,
    gates: Optional[GateMap] = None,
    gate_colors: Optional[Mapping[str, str]] = None,
) -> List[dict]:
    """Data qubits: white fill + black edge; gated ones get a colored marker + label."""
    if not data_coords:
        return []
    palette = gate_colors if gate_colors is not None else DEFAULT_GATE_COLORS

    plain: List[complex] = []
    gated: List[Tuple[complex, str, str]] = []
    for q in data_coords:
        name = None if gates is None else gates.get(q)
        if name is None:
            plain.append(q)
        else:
            color = palette.get(name.upper(), _OTHER_GATE_COLOR)
            gated.append((q, name.upper(), color))

    # Small z-offset lifts data / gated nodes slightly above the CNOT-spoke
    # layer so they win the depth-buffer race and the spoke never peeks
    # through the marker interior.
    z_node = z + 0.02

    traces: List[dict] = []
    if plain:
        traces.append(
            dict(
                type="scatter3d",
                x=[q.real for q in plain],
                y=[-q.imag for q in plain],
                z=[z_node] * len(plain),
                mode="markers",
                marker=dict(
                    size=size,
                    color=DATA_FILL,
                    opacity=1.0,
                    line=dict(color=DATA_EDGE, width=1.8),
                ),
                hoverinfo="skip",
                showlegend=False,
            )
        )
    for q, name, color in gated:
        show_text = name.upper() in TEXT_GATES
        trace = dict(
            type="scatter3d",
            x=[q.real],
            y=[-q.imag],
            z=[z_node],
            mode="markers+text" if show_text else "markers",
            marker=dict(
                size=size * 1.35,
                color=color,
                opacity=1.0,
                line=dict(color=DATA_EDGE, width=1.8),
            ),
            hoverinfo="skip",
            showlegend=False,
        )
        if show_text:
            trace["text"] = [name]
            trace["textposition"] = "top center"
            trace["textfont"] = dict(size=13, color="#111")
        traces.append(trace)
    return traces


def _normalize_round_gates(round_gates: RoundGates, rounds: int) -> List[Optional[GateMap]]:
    if round_gates is None:
        return [None] * rounds
    if isinstance(round_gates, Mapping):
        return [dict(round_gates) for _ in range(rounds)]
    values = list(round_gates)
    if len(values) != rounds:
        raise ValueError(
            f"per-round data_gates length {len(values)} does not match segment rounds {rounds}"
        )
    return [None if entry is None else dict(entry) for entry in values]


def _worldlines(z_of_data: List[Tuple[float, Sequence[complex]]]) -> Optional[dict]:
    xs: List[Optional[float]] = []
    ys: List[Optional[float]] = []
    zs: List[Optional[float]] = []

    coord_to_zs: dict = {}
    for z, coords in z_of_data:
        for q in coords:
            coord_to_zs.setdefault(q, []).append(z)

    for q, zlist in coord_to_zs.items():
        if len(zlist) < 2:
            continue
        zlist = sorted(zlist)
        for a, b in zip(zlist, zlist[1:]):
            xs.extend([q.real, q.real, None])
            ys.extend([-q.imag, -q.imag, None])
            zs.extend([a, b, None])

    if not xs:
        return None
    return dict(
        type="scatter3d",
        x=xs,
        y=ys,
        z=zs,
        mode="lines",
        line=dict(color=WORLDLINE_COLOR, width=2),
        hoverinfo="skip",
        showlegend=False,
    )


def _segment_divider(
    z: float,
    x_range: Tuple[float, float],
    y_range: Tuple[float, float],
    label: Optional[str],
) -> List[dict]:
    x0, x1 = x_range
    y0, y1 = y_range
    outline = dict(
        type="scatter3d",
        x=[x0, x1, x1, x0, x0],
        y=[y0, y0, y1, y1, y0],
        z=[z] * 5,
        mode="lines",
        line=dict(color="rgba(0,0,0,0.15)", width=2, dash="dot"),
        hoverinfo="skip",
        showlegend=False,
    )
    traces = [outline]
    if label:
        traces.append(
            dict(
                type="scatter3d",
                x=[x1 + 0.6],
                y=[y0 - 0.2],
                z=[z],
                mode="text",
                text=[label],
                textfont=dict(size=14, color="#333"),
                hoverinfo="skip",
                showlegend=False,
            )
        )
    return traces


def _unpack_segment(seg: Segment, idx: int):
    if len(seg) == 2:
        patch, rounds = seg
        return patch, rounds, None, False, None
    if len(seg) == 3:
        patch, rounds, gates = seg
        return patch, rounds, gates, False, None
    if len(seg) == 4:
        patch, rounds, gates, swap = seg
        return patch, rounds, gates, bool(swap), None
    if len(seg) == 5:
        patch, rounds, gates, swap, edges = seg
        edge_list = list(edges) if edges else None
        return patch, rounds, gates, bool(swap), edge_list
    raise ValueError(
        f"segment {idx}: expected (patch, rounds[, data_gates[, basis_swap[, xcy_edges]]])"
    )


def plot_patch_timeline(
    segments: Sequence[Segment],
    *,
    segment_labels: Optional[Sequence[Optional[str]]] = None,
    layer_spacing: float = 1.0,
    alpha: float = 0.45,
    data_size: float = 5.0,
    meas_size: float = 4.5,
    edge_width: float = 2.0,
    title: Optional[str] = None,
    gate_colors: Optional[Mapping[str, str]] = None,
    brackets: Optional[Sequence[Tuple[int, int, str]]] = None,
    show: bool = True,
    width: int = 900,
    height: int = 700,
    hide_spatial_axes: bool = False,
):
    """Render segments as a 3D spacetime plot.

    Each segment is ``(patch, rounds[, data_gates[, basis_swap]])``.
    ``data_gates`` is ``None`` | ``{coord: gate_name}`` (same for every round)
    | ``Sequence[Optional[GateMap]]`` (per-round). ``basis_swap=True`` flips
    X↔Z coloring for the whole segment (used after transversal H).

    Set ``hide_spatial_axes=True`` to drop the x/y axis titles and tick
    labels and render only the ``round`` labels along the time axis.
    """
    try:
        import plotly.graph_objects as go
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ModuleNotFoundError(
            "plotly is required for surface_code.visualization.plotting_3d. "
            "Install it via `pip install plotly`."
        ) from exc

    if not segments:
        raise ValueError("segments must contain at least one (patch, rounds) entry")
    if segment_labels is not None and len(segment_labels) != len(segments):
        raise ValueError("segment_labels length must match segments length")

    traces: List[dict] = []
    z_of_data: List[Tuple[float, Sequence[complex]]] = []

    all_x: List[float] = []
    all_y: List[float] = []

    total_layers = sum(max(1, _unpack_segment(s, i)[1]) for i, s in enumerate(segments))
    z_cursor = 0.0
    segment_starts: List[float] = []
    rounds_per_segment: List[int] = []
    used_gates: Dict[str, str] = {}
    palette = gate_colors if gate_colors is not None else DEFAULT_GATE_COLORS

    for seg_idx, seg in enumerate(segments):
        patch, rounds, raw_gates, basis_swap, xcy_edges = _unpack_segment(seg, seg_idx)
        if rounds < 1:
            raise ValueError(f"segment {seg_idx}: rounds must be >= 1")
        rounds_per_segment.append(rounds)
        gates_per_round = _normalize_round_gates(raw_gates, rounds)
        for gm in gates_per_round:
            if gm is None:
                continue
            for name in gm.values():
                key = name.upper()
                used_gates.setdefault(key, palette.get(key, _OTHER_GATE_COLOR))

        tile_polys, data_coords, meas_coords = _collect_patch_geometry(patch)
        segment_starts.append(z_cursor)

        for r in range(rounds):
            z = z_cursor + r * layer_spacing
            traces.extend(_layer_mesh(tile_polys, z, alpha=alpha, basis_swap=basis_swap))
            traces.extend(_layer_spokes(tile_polys, z, width=edge_width, basis_swap=basis_swap))
            traces.extend(
                _layer_meas_points(
                    tile_polys,
                    z,
                    size=meas_size,
                    basis_swap=basis_swap,
                    gates=gates_per_round[r],
                    gate_colors=gate_colors,
                )
            )
            traces.extend(
                _layer_data_points(
                    data_coords,
                    z,
                    size=data_size,
                    gates=gates_per_round[r],
                    gate_colors=gate_colors,
                )
            )
            if xcy_edges:
                traces.extend(_layer_xcy_edges(xcy_edges, z))
                used_gates.setdefault("XCY", palette.get("XCY", _OTHER_GATE_COLOR))
            z_of_data.append((z, data_coords))

            for q in data_coords:
                all_x.append(q.real)
                all_y.append(-q.imag)
            for q in meas_coords:
                all_x.append(q.real)
                all_y.append(-q.imag)
            for _, corners in tile_polys:
                for x, y in corners:
                    all_x.append(x)
                    all_y.append(-y)

        z_cursor += rounds * layer_spacing

    wl = _worldlines(z_of_data)
    if wl is not None:
        traces.append(wl)

    if all_x:
        x_range = (min(all_x) - 0.5, max(all_x) + 0.5)
        y_range = (min(all_y) - 0.5, max(all_y) + 0.5)
    else:
        x_range = (0.0, 1.0)
        y_range = (0.0, 1.0)

    if segment_labels is not None:
        for start_z, label in zip(segment_starts, segment_labels):
            if label is None:
                continue
            traces.extend(_segment_divider(start_z, x_range, y_range, label))

    if brackets:
        # Round-count labels. Rendered on the left side of the figure (the
        # phase-label bracket used to sit on the right); each bracket range is
        # replaced by a single "N round(s)" text annotation — no spine, no
        # phase label.
        x_anchor = x_range[0] - 0.3
        y_anchor = (y_range[0] + y_range[1]) / 2.0
        for bracket in brackets:
            start_idx, end_idx = bracket[0], bracket[1]
            if start_idx < 0 or end_idx >= len(segment_starts) or start_idx > end_idx:
                continue
            z_bottom = segment_starts[start_idx]
            if end_idx + 1 < len(segment_starts):
                z_top = segment_starts[end_idx + 1]
            else:
                z_top = z_cursor
            z_mid = (z_bottom + z_top) / 2.0
            total_rounds = sum(rounds_per_segment[start_idx : end_idx + 1])
            noun = "round" if total_rounds == 1 else "rounds"
            traces.append(
                dict(
                    type="scatter3d",
                    x=[x_anchor],
                    y=[-y_anchor],
                    z=[z_mid],
                    mode="text",
                    text=[f"{total_rounds} {noun}"],
                    textposition="middle left",
                    textfont=dict(size=14, color="#333"),
                    hoverinfo="skip",
                    showlegend=False,
                )
            )

    traces.append(
        dict(
            type="scatter3d",
            x=[None],
            y=[None],
            z=[None],
            mode="markers",
            marker=dict(size=10, color=X_NODE_COLOR, line=dict(color="white", width=1)),
            name="X stabilizer",
            showlegend=True,
        )
    )
    traces.append(
        dict(
            type="scatter3d",
            x=[None],
            y=[None],
            z=[None],
            mode="markers",
            marker=dict(size=10, color=Z_NODE_COLOR, line=dict(color="white", width=1)),
            name="Z stabilizer",
            showlegend=True,
        )
    )
    traces.append(
        dict(
            type="scatter3d",
            x=[None],
            y=[None],
            z=[None],
            mode="markers",
            marker=dict(size=8, color=DATA_FILL, line=dict(color=DATA_EDGE, width=1.4)),
            name="data qubit",
            showlegend=True,
        )
    )
    for gate_name, gate_color in used_gates.items():
        traces.append(
            dict(
                type="scatter3d",
                x=[None],
                y=[None],
                z=[None],
                mode="markers",
                marker=dict(size=9, color=gate_color, line=dict(color=DATA_EDGE, width=1.4)),
                name=f"{gate_name} gate",
                showlegend=True,
            )
        )

    fig = go.Figure(data=traces)

    if hide_spatial_axes:
        xaxis_cfg = dict(visible=False)
        yaxis_cfg = dict(visible=False)
    else:
        # Keep the x/y axis lines + titles but hide ticks and numeric labels
        # entirely — the exact coordinates don't matter to the reader, only
        # the spatial layout of the patch.
        xaxis_cfg = dict(
            title="x",
            title_font=dict(size=15),
            showticklabels=False,
            ticks="",
            showbackground=False,
            showgrid=False,
            zeroline=False,
        )
        yaxis_cfg = dict(
            title="y",
            title_font=dict(size=15),
            showticklabels=False,
            ticks="",
            showbackground=False,
            showgrid=False,
            zeroline=False,
        )

    fig.update_layout(
        width=width,
        height=height,
        title=title,
        paper_bgcolor="white",
        plot_bgcolor="white",
        # Default font for axis titles / tick labels / legend / in-scene text.
        # Individual font specs that omit `family` inherit from here.
        font=dict(family="Consolas, monospace"),
        scene=dict(
            xaxis=xaxis_cfg,
            yaxis=yaxis_cfg,
            zaxis=dict(
                title="round",
                title_font=dict(size=15),
                tickfont=dict(size=12),
                showgrid=False,
                zeroline=False,
                showbackground=False,
                tickmode="array",
                tickvals=[i * layer_spacing for i in range(total_layers)],
                ticktext=[str(i + 1) for i in range(total_layers)],
            ),
            aspectmode="data",
            camera=dict(
                eye=dict(x=1.6, y=-1.6, z=1.2),
                projection=dict(type="orthographic"),
            ),
        ),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.05,
            x=0.5,
            xanchor="center",
            font=dict(size=13),
            # Plotly wraps horizontal legends per-item when widths push past
            # the available space; default itemwidth is fine but tracegroup
            # gap keeps entries visually separate when they sit on one row.
            tracegroupgap=8,
        ),
        # Constrain the 3D scene vertically so the horizontal legend below it
        # spans the full figure width (otherwise Plotly makes the scene very
        # wide and leaves the legend no room, causing single-column wrap).
        scene_domain=dict(x=[0.0, 1.0], y=[0.15, 1.0]),
        margin=dict(l=0, r=0, t=40 if title else 10, b=20),
    )

    if show:
        fig.show()
    return fig


def _init_measure_gates(
    data_coords: Iterable[complex],
    meas_basis: str,
) -> Tuple[GateMap, GateMap]:
    """Data-qubit annotations for the first (transversal R{basis}) and last
    (transversal M{basis}) rounds of a primitive that initializes / measures
    in ``meas_basis``."""
    b = meas_basis.upper()
    if b not in {"X", "Z"}:
        raise ValueError("meas_basis must be 'X' or 'Z'")
    return (
        {q: f"R{b}" for q in data_coords},
        {q: f"M{b}" for q in data_coords},
    )


def plot_memory_timeline(
    *,
    x_distance: int,
    z_distance: int,
    rounds: int,
    meas_basis: str = "Z",
    **kwargs,
):
    """3D timeline for a single-patch memory primitive.

    Data qubits are annotated with ``R{meas_basis}`` on round 1 (transversal
    initialization) and ``M{meas_basis}`` on the final round (transversal
    destructive measurement). If ``rounds == 1`` both collapse to a single
    label — the measurement wins since it closes out the primitive.
    """

    patch = rectangular_surface_code_patch(x_distance, z_distance)
    init_gates, meas_gates = _init_measure_gates(patch.data_set, meas_basis)
    per_round: List[Optional[GateMap]] = [None] * max(rounds, 1)
    if rounds >= 2:
        per_round[0] = init_gates
        per_round[-1] = meas_gates
    elif rounds == 1:
        per_round[0] = meas_gates
    return plot_patch_timeline([(patch, rounds, per_round)], **kwargs)


def plot_transversal_h_timeline(
    *,
    x_distance: int,
    z_distance: int,
    pre_rounds: int = 1,
    post_rounds: int = 1,
    meas_basis: str = "Z",
    **kwargs,
):
    """3D timeline for the transversal-H primitive.

    ``pre_rounds`` normal rounds, then a layer with H applied to every data
    qubit (annotated), then ``post_rounds`` rounds with X↔Z stabilizer roles
    swapped.

    Data-qubit annotations match the circuit builder: transversal
    ``R{meas_basis}`` on round 1, transversal ``M{swapped_basis}`` on the
    final round (since the transversal H swaps the measurement basis).
    """

    b = meas_basis.upper()
    swapped = "X" if b == "Z" else "Z"
    patch = rectangular_surface_code_patch(x_distance, z_distance)
    data_coords = patch.data_set
    init_gates = {q: f"R{b}" for q in data_coords}
    meas_gates = {q: f"M{swapped}" for q in data_coords}
    h_gates: GateMap = {q: "H" for q in data_coords}

    segments: List[Segment] = []
    brackets: List[Tuple[int, int, str]] = []
    if pre_rounds > 0:
        per_round: List[Optional[GateMap]] = [None] * pre_rounds
        per_round[0] = init_gates
        idx = len(segments)
        segments.append((patch, pre_rounds, per_round))
        brackets.append((idx, idx, "pre-H"))
        init_placed = True
    else:
        init_placed = False
    idx = len(segments)
    h_round_gates = dict(h_gates)
    if not init_placed:
        # No pre rounds — stack the init annotation onto the H-layer data qubits.
        for q, name in init_gates.items():
            h_round_gates.setdefault(q, name)
    segments.append((patch, 1, h_round_gates, False))
    brackets.append((idx, idx, "transversal H"))
    if post_rounds > 0:
        per_round_post: List[Optional[GateMap]] = [None] * post_rounds
        per_round_post[-1] = meas_gates
        idx = len(segments)
        segments.append((patch, post_rounds, per_round_post, True))
        brackets.append((idx, idx, "post-H (X/Z swapped)"))
    else:
        # No post rounds — annotate measurement on the H layer itself.
        h_round_gates.update(meas_gates)
    return plot_patch_timeline(segments, brackets=brackets, **kwargs)


def plot_lattice_surgery_timeline(
    *,
    x_distance: int,
    z_distance: int,
    bridge_length: int,
    pre_rounds: int,
    merge_rounds: int,
    post_rounds: int,
    horizontal: bool = True,
    meas_basis: str = "Z",
    **kwargs,
):
    """3D timeline: individual patches -> merged patch -> individual patches.

    Data qubits on the individual patches are annotated with transversal
    ``R{meas_basis}`` on the first round and transversal ``M{meas_basis}``
    on the final round. Ancilla data qubits introduced by the merge are
    not annotated (they're prepared / measured by the surgery itself, not
    by the user-facing primitive boundary).
    """
    individual_patch, merged_patch = build_merged_patch(
        x_distance=x_distance,
        z_distance=z_distance,
        bridge_length=bridge_length,
        horizontal=horizontal,
    )
    init_gates, meas_gates = _init_measure_gates(individual_patch.data_set, meas_basis)

    # Bridge data qubits (present in merged_patch but not in individual_patch)
    # are initialized / destructively measured in the basis opposite to the
    # individual-patch meas_basis. MZZ (meas_basis="Z") => bridge uses X.
    b = meas_basis.upper()
    bridge_basis = "X" if b == "Z" else "Z"
    bridge_coords = set(merged_patch.data_set) - set(individual_patch.data_set)
    bridge_init = {q: f"R{bridge_basis}" for q in bridge_coords}
    bridge_meas = {q: f"M{bridge_basis}" for q in bridge_coords}

    segments: List[Segment] = []
    brackets: List[Tuple[int, int, str]] = []
    init_placed = False
    if pre_rounds > 0:
        per_round: List[Optional[GateMap]] = [None] * pre_rounds
        per_round[0] = init_gates
        idx = len(segments)
        segments.append((individual_patch, pre_rounds, per_round))
        brackets.append((idx, idx, "pre-LS"))
        init_placed = True
    if merge_rounds > 0:
        per_round_m: List[Optional[GateMap]] = [None] * merge_rounds
        first_gates: GateMap = dict(bridge_init)
        last_gates: GateMap = dict(bridge_meas)
        if not init_placed:
            # Also carry individual-patch init onto the first merge round.
            first_gates.update(
                {q: name for q, name in init_gates.items() if q in merged_patch.data_set}
            )
            init_placed = True
        if merge_rounds >= 2:
            per_round_m[0] = first_gates
            per_round_m[-1] = last_gates
        else:
            # merge_rounds == 1: bridge init + measurement on the same layer;
            # measurement wins so the surgery outcome is visible.
            combined = dict(first_gates)
            combined.update(last_gates)
            per_round_m[0] = combined
        idx = len(segments)
        segments.append((merged_patch, merge_rounds, per_round_m))
        brackets.append((idx, idx, "merge"))
    if post_rounds > 0:
        per_round_post: List[Optional[GateMap]] = [None] * post_rounds
        per_round_post[-1] = meas_gates
        idx = len(segments)
        segments.append((individual_patch, post_rounds, per_round_post))
        brackets.append((idx, idx, "split"))
    else:
        # No post-LS individual rounds: attach measurement to last segment's
        # final round if possible.
        if segments:
            last = segments[-1]
            if len(last) >= 3 and isinstance(last[2], list):
                last_gates_list = list(last[2])
                idx_last = len(last_gates_list) - 1
                combined = dict(last_gates_list[idx_last] or {})
                for q, name in meas_gates.items():
                    combined[q] = name
                last_gates_list[idx_last] = combined
                segments[-1] = (last[0], last[1], last_gates_list, *last[3:])
    return plot_patch_timeline(segments, brackets=brackets, **kwargs)


_DATA_GATE_PRIORITY = {"MY": 4, "XCY": 3, "H": 2, "SQRT_X": 1, "SQRT_Y": 1}

_TILE_DIRECTIONS: Tuple[Tuple[float, float, str], ...] = (
    (-0.5, -0.5, "UL"),
    (+0.5, -0.5, "UR"),
    (-0.5, +0.5, "DL"),
    (+0.5, +0.5, "DR"),
)


def _tile_from_support(ancilla: complex, data_set: set, basis: str) -> Tile:
    """Build a ``Tile`` at ``ancilla`` whose ``data_qubits`` slots are filled
    from ``data_set`` by relative direction (UL/UR/DL/DR). Data qubits not
    adjacent to the ancilla are silently ignored so unusual stabilizer
    supports still produce a legal tile.
    """
    data_qubits: Dict[str, Optional[complex]] = {d: None for d in ("UL", "UR", "DL", "DR")}
    for dx, dy, direction in _TILE_DIRECTIONS:
        candidate = complex(ancilla.real + dx, ancilla.imag + dy)
        if candidate in data_set:
            data_qubits[direction] = candidate
    return Tile(basis=basis, measurement_qubit=ancilla, data_qubits=data_qubits)


def _parse_s_gate_rounds(
    distance: int,
    bridge_length: int,
    pre_rounds: int,
    merge_rounds: int,
    boundary_rounds: int,
) -> List[Dict]:
    """Run ``s_gate(...)`` and split its flattened program into rounds.

    Stabilizer support is taken directly from the circuit's ``CX``/``XCY``
    pairs — not from a prebuilt ``Patch`` — so unusual post-surgery shapes
    (weight-3 tiles, new ancilla positions introduced by the Y-magic phase,
    flipped stabilizer roles) are captured correctly.

    Each returned round records:

    - ``tiles``: ``List[Tile]`` built from the ancilla-data pairings observed
      between the previous and current measurement phase.
    - ``sub_phases``: ``List[Dict]`` of per-TICK gate activity on data /
      ancilla qubits (``data_gates``, ``ancilla_gates``, ``xcy_edges``). A
      round with a single sub-phase is a regular stabilizer round; a round
      with several sub-phases is a logical "Y-magic" round (H -> XCY -> H ->
      MY) that gets stacked into several 3D layers with a ``}`` bracket.
    - ``data_destroyed``: coords destructively ``MY``-measured this round.
    - ``meas_x`` / ``meas_z``: ancilla coords measured this round.
    """
    from ..circuits.S_gate import s_gate

    circuit = s_gate(
        distance=distance,
        bridge_length=bridge_length,
        pre_rounds=pre_rounds,
        merge_rounds=merge_rounds,
        boundary_rounds=boundary_rounds,
    )
    flat = circuit.flattened()

    coords: Dict[int, complex] = {}
    for inst in flat:
        if inst.name == "QUBIT_COORDS":
            q = inst.targets_copy()[0].value
            xy = tuple(inst.gate_args_copy())
            coords[q] = complex(xy[0], xy[1])

    def is_data(q: int) -> bool:
        c = coords[q]
        return c.real == int(c.real) and c.imag == int(c.imag)

    phases: List[List] = [[]]
    for inst in flat:
        if inst.name == "TICK":
            phases.append([])
        elif inst.name != "QUBIT_COORDS":
            phases[-1].append(inst)

    rounds: List[Dict] = []
    round_support: Dict[complex, set] = {}
    round_basis: Dict[complex, str] = {}
    round_destroyed: set = set()
    sub_phases: List[Dict] = []

    entangling_gates = {"CX", "CZ", "XCY", "YCX", "CY"}
    single_qubit_annotatable = {"H", "SQRT_X", "SQRT_Y", "S", "S_DAG"}

    for phase in phases:
        phase_data_gates: Dict[complex, str] = {}
        phase_ancilla_gates: Dict[complex, str] = {}
        phase_xcy_edges: List[Tuple[complex, complex]] = []
        meas_x: set = set()
        meas_z: set = set()

        for inst in phase:
            targets = [t.value for t in inst.targets_copy() if t.is_qubit_target]

            if inst.name == "RX":
                for q in targets:
                    if is_data(q):
                        # Bridge / auxiliary data qubit prepared in X basis
                        # (e.g., the extra rows introduced by the merge).
                        phase_data_gates[coords[q]] = "RX"
                    else:
                        round_basis[coords[q]] = "X"
            elif inst.name == "R":
                for q in targets:
                    if is_data(q):
                        phase_data_gates[coords[q]] = "RZ"
                    else:
                        round_basis[coords[q]] = "Z"
            elif inst.name in entangling_gates:
                for i in range(0, len(targets), 2):
                    a, b = targets[i], targets[i + 1]
                    a_data = is_data(a)
                    b_data = is_data(b)
                    if a_data and not b_data:
                        anc, data = coords[b], coords[a]
                    elif b_data and not a_data:
                        anc, data = coords[a], coords[b]
                    else:
                        continue
                    round_support.setdefault(anc, set()).add(data)
                    if inst.name == "XCY":
                        phase_data_gates[data] = "XCY"
                        phase_ancilla_gates[anc] = "XCY"
                        phase_xcy_edges.append((coords[a], coords[b]))
            elif inst.name in single_qubit_annotatable:
                for q in targets:
                    if is_data(q):
                        phase_data_gates[coords[q]] = inst.name
                    else:
                        phase_ancilla_gates[coords[q]] = inst.name
            elif inst.name == "MY":
                for q in targets:
                    if is_data(q):
                        phase_data_gates[coords[q]] = "MY"
                        round_destroyed.add(coords[q])
            elif inst.name == "MX":
                for q in targets:
                    if is_data(q):
                        # Bridge / auxiliary data qubit destructively measured
                        # in X basis mid-program.
                        phase_data_gates[coords[q]] = "MX"
                        round_destroyed.add(coords[q])
                    else:
                        meas_x.add(coords[q])
            elif inst.name == "M":
                for q in targets:
                    if is_data(q):
                        phase_data_gates[coords[q]] = "MZ"
                        round_destroyed.add(coords[q])
                    else:
                        meas_z.add(coords[q])

        if phase_data_gates or phase_ancilla_gates or phase_xcy_edges:
            sub_phases.append(
                {
                    "data_gates": phase_data_gates,
                    "ancilla_gates": phase_ancilla_gates,
                    "xcy_edges": phase_xcy_edges,
                }
            )

        if meas_x or meas_z:
            tiles: List[Tile] = []
            measured = meas_x | meas_z
            for anc in sorted(round_support.keys(), key=lambda c: (c.imag, c.real)):
                support = round_support[anc]
                if not support or anc not in measured:
                    continue
                basis = round_basis.get(anc)
                if basis is None:
                    basis = "X" if anc in meas_x else "Z"
                tiles.append(_tile_from_support(anc, support, basis))

            if not sub_phases:
                sub_phases = [
                    {
                        "data_gates": {},
                        "ancilla_gates": {},
                        "xcy_edges": [],
                    }
                ]

            rounds.append(
                {
                    "tiles": tiles,
                    "sub_phases": sub_phases,
                    "data_destroyed": frozenset(round_destroyed),
                    "meas_x": frozenset(meas_x),
                    "meas_z": frozenset(meas_z),
                }
            )

            round_support = {}
            round_basis = {}
            round_destroyed = set()
            sub_phases = []

    return rounds


def _build_round_patch_from_tiles(
    tiles: Sequence[Tile],
    destroyed_so_far: set,
) -> Patch:
    """Wrap parsed tiles in a ``Patch``, pruning data qubits already
    destructively measured (``MY``) in earlier rounds."""
    new_tiles: List[Tile] = []
    for tile in tiles:
        filtered_data = {
            d: (q if (q is None or q not in destroyed_so_far) else None)
            for d, q in tile.data_qubits.items()
        }
        new_tiles.append(
            Tile(
                basis=tile.basis,
                measurement_qubit=tile.measurement_qubit,
                data_qubits=filtered_data,
                schedule_type=tile.schedule_type,
                second_measurement_qubit=tile.second_measurement_qubit,
            )
        )
    return Patch(new_tiles)


def _tile_signature(tile: Tile) -> Tuple:
    """Hashable summary of a tile's shape (basis + meas + sorted data)."""
    data = tuple(sorted((q.real, q.imag) for q in tile.data_qubits.values() if q is not None))
    return (tile.basis, tile.measurement_qubit.real, tile.measurement_qubit.imag, data)


def _round_signature(round_info: Dict) -> frozenset:
    return frozenset(_tile_signature(t) for t in round_info["tiles"])


_NON_BOUNDARY_GATES = {"H", "XCY", "MY", "S", "S_DAG", "SQRT_X", "SQRT_Y", "T"}


def _round_has_gates(rd: Dict) -> bool:
    """True if the round has any "active" single-qubit or two-qubit gate on a
    data or ancilla qubit. Pure boundary events (``RX``/``RZ``/``MX``/``MZ``
    on data qubits) don't count — they're initializations or destructive
    measurements of bridge qubits and shouldn't trigger the Y-transition
    bracket label."""
    for sp in rd.get("sub_phases", []):
        for gate_dict in (sp.get("data_gates", {}), sp.get("ancilla_gates", {})):
            for name in gate_dict.values():
                if name.upper() in _NON_BOUNDARY_GATES:
                    return True
        if sp.get("xcy_edges"):
            return True
    return False


def _s_gate_round_label(prev: Optional[Dict], curr: Dict, idx: int) -> Optional[str]:
    """Label a round only when its stabilizer set or data-gate pattern changes."""
    if prev is None:
        return f"round {idx + 1}"
    if _round_signature(prev) != _round_signature(curr) or _round_has_gates(curr):
        return f"round {idx + 1}"
    return None


def plot_s_gate_timeline(
    *,
    distance: int,
    bridge_length: int,
    pre_rounds: int,
    merge_rounds: int,
    boundary_rounds: int = 2,
    horizontal: bool = False,
    **kwargs,
):
    """3D timeline for the S-gate primitive (parses the generated circuit).

    Runs ``s_gate(...)``, splits it into stabilizer rounds, and for each round
    reads off:

    - which stabilizers are present (a tile is rendered only if its ancilla is
      reset or measured that round — some tiles disappear post-ZZ surgery);
    - each stabilizer's basis, taken from the ancilla's reset gate (``RX`` -> X,
      ``R`` -> Z) — useful because the S-gate primitive changes stabilizer
      roles around the Y-magic phase;
    - data-qubit Clifford/Y-basis events for annotation (``H``, ``SQRT_X``,
      ``XCY``, ``MY``). The logical Y measurement is done by braiding a defect
      along the diagonal, so ``XCY`` / ``H`` / ``MY`` is the visual signature,
      not a transversal S.

    ``horizontal`` is accepted for symmetry with
    ``plot_lattice_surgery_timeline`` but ignored — the S-gate primitive
    always stacks the two patches vertically.
    """
    del horizontal

    rounds = _parse_s_gate_rounds(
        distance=distance,
        bridge_length=bridge_length,
        pre_rounds=pre_rounds,
        merge_rounds=merge_rounds,
        boundary_rounds=boundary_rounds,
    )

    segments: List[Segment] = []
    phase_of_segment: List[str] = []
    destroyed: set = set()

    y_done = False
    ls_counter = 0
    post_counter = 0
    last_sig: Optional[frozenset] = None

    for rd in rounds:
        round_patch = _build_round_patch_from_tiles(rd["tiles"], destroyed)
        if len(round_patch) == 0:
            destroyed |= rd["data_destroyed"]
            continue

        # Collapse all sub-phases of this round into a single layer:
        # last-gate-per-qubit wins (so MY naturally beats the preceding H /
        # XCY), and XCY edges from every sub-phase are drawn on this one
        # layer.
        merged_gates: Dict[complex, str] = {}
        xcy_edges: List[Tuple[complex, complex]] = []
        for sp in rd["sub_phases"]:
            merged_gates.update(sp.get("data_gates", {}))
            merged_gates.update(sp.get("ancilla_gates", {}))
            xcy_edges.extend(sp.get("xcy_edges", []))

        has_gates = _round_has_gates(rd)
        sig = _round_signature(rd)

        if has_gates:
            phase_label = "Y transition"
        elif not y_done:
            if sig != last_sig:
                ls_counter += 1
            phase_label = f"LS phase {ls_counter}"
        else:
            if sig != last_sig:
                post_counter += 1
            phase_label = f"post-Y phase {post_counter}"

        # Each round becomes its own z-layer. Consecutive rounds of the same
        # phase collapse into a single segment (for bracket labelling) with
        # ``rounds`` incremented and a per-round gate list — so data-qubit
        # annotations (RX/MX/MY) appear only on the rounds where they fire.
        # The Y transition remains a single-round segment by construction.
        if segments and phase_of_segment[-1] == phase_label and not has_gates:
            patch_prev, rounds_prev, gates_prev, swap_prev, xcy_prev = segments[-1]
            if gates_prev is None:
                gates_list: List[Optional[GateMap]] = [None] * rounds_prev
            elif isinstance(gates_prev, Mapping):
                gates_list = [dict(gates_prev) for _ in range(rounds_prev)]
            else:
                gates_list = list(gates_prev)
            gates_list.append(merged_gates or None)
            combined_xcy = list(xcy_prev) if xcy_prev else []
            combined_xcy.extend(xcy_edges)
            segments[-1] = (
                patch_prev,
                rounds_prev + 1,
                gates_list,
                swap_prev,
                combined_xcy or None,
            )
        else:
            segments.append(
                (
                    round_patch,
                    1,
                    merged_gates or None,
                    False,
                    xcy_edges or None,
                )
            )
            phase_of_segment.append(phase_label)

        last_sig = sig
        if has_gates:
            y_done = True
            # Force a fresh phase on the next non-gate round.
            last_sig = None

        destroyed |= rd["data_destroyed"]

    if not segments:
        raise RuntimeError("s_gate circuit parse produced no rounds")

    # Each collapsed phase is one segment → one layer → its own bracket.
    brackets: List[Tuple[int, int, str]] = [
        (i, i, label) for i, label in enumerate(phase_of_segment)
    ]

    return plot_patch_timeline(
        segments,
        brackets=brackets,
        **kwargs,
    )


__all__ = [
    "plot_patch_timeline",
    "plot_memory_timeline",
    "plot_transversal_h_timeline",
    "plot_lattice_surgery_timeline",
    "plot_s_gate_timeline",
]
