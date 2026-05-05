"""Plotting utilities for surface-code patches."""

from __future__ import annotations

import math
from typing import Tuple

from ..circuits._tile import Patch, Tile


def _iter_qubit_entries(entry):
    if entry is not None:
        yield entry


def _tile_data_coords(tile: Tile) -> list[complex]:
    coords: list[complex] = []
    for entry in tile.data_qubits.values():
        coords.extend(_iter_qubit_entries(entry))
    return coords


# ── Style constants (kept in sync with the 3D timeline plotter) ────────────
X_COLOR = "#d43d51"  # X-stabilizer fill (matches plotting_3d.X_COLOR)
Z_COLOR = "#3d6fd4"  # Z-stabilizer fill
Y_COLOR = "#3dc46b"  # Y-stabilizer fill
X_NODE_COLOR = "#8b1a2a"  # X ancilla node
Z_NODE_COLOR = "#1d3a8a"  # Z ancilla node
Y_NODE_COLOR = "#1a6b39"  # Y ancilla node
DATA_FILL = "#ffffff"  # data-qubit fill (white circle)
DATA_EDGE = "#111111"  # data-qubit edge


def _basis_color(basis: str) -> str:
    return {"X": X_COLOR, "Z": Z_COLOR, "Y": Y_COLOR}.get(basis.upper(), "purple")


def _basis_node_color(basis: str) -> str:
    return {"X": X_NODE_COLOR, "Z": Z_NODE_COLOR, "Y": Y_NODE_COLOR}.get(
        basis.upper(), "#444444"
    )


def plot_patch(
    patch: Patch,
    *,
    ax=None,
    show: bool = True,
    legend: bool = False,
    figsize: Tuple[float, float] = (6.0, 6.0),
    alpha: float = 0.35,
    show_axes: bool = False,
) -> Tuple["matplotlib.figure.Figure", "matplotlib.axes.Axes"]:
    """Visualize a :class:`~surface_code.circuits._tile.Patch` using matplotlib.

    Stabilizer plaquettes are filled by basis (red = X, blue = Z, green = Y).
    Bulk weight-3 / weight-4 stabilizers render as the polygon spanned by their
    data-qubit corners. **Boundary weight-2 stabilizers** render as a triangle
    whose vertices are the stabilizer measurement qubit and its two data
    qubits, making the boundary structure visually distinct from the bulk.

    Args:
        patch: The patch to visualize.
        ax: Optional matplotlib axes. When absent, a new figure/axes are created.
        show: If ``True``, call ``plt.show()`` before returning.
        legend: If ``True``, display a basis-color legend.
        figsize: Figure size when creating a new figure.
        alpha: Fill transparency for the stabilizer plaquettes.
        show_axes: If ``True``, render the X/Y axis ticks, labels, and grid.
            Default ``False`` (clean schematic style matching the 3D
            timeline plots).

    Returns:
        ``(fig, ax)`` so callers can further customise the plot.
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch as LegendPatch
        from matplotlib.patches import Polygon, Rectangle
        from matplotlib.ticker import MultipleLocator
    except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
        raise ModuleNotFoundError(
            "matplotlib is required for surface_code.visualization.plotting. "
            "Install it via `pip install matplotlib`."
        ) from exc

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    data_coords: set[complex] = set()
    meas_basis_at: dict[complex, str] = {}
    edge_x: list[float] = []
    edge_y: list[float] = []

    for tile in patch.tiles:
        meas_basis_at[tile.measurement_qubit] = tile.basis
        if tile.second_measurement_qubit is not None:
            meas_basis_at[tile.second_measurement_qubit] = tile.basis
        for dq in _tile_data_coords(tile):
            data_coords.add(dq)

    tile_draws: list[tuple[str, object, str]] = []
    for tile in patch.tiles:
        measurement = tile.measurement_qubit
        color = _basis_color(tile.basis)
        data_points = _tile_data_coords(tile)

        # Two-ancilla merged stabilizer (e.g., a lattice-surgery weight-4 cell
        # whose two ancilla qubits straddle the merged boundary).
        if tile.second_measurement_qubit is not None:
            second = tile.second_measurement_qubit
            left = min(measurement.real, second.real) - 0.5
            top = min(measurement.imag, second.imag) - 0.5
            width = abs(measurement.real - second.real) + 1.0
            height = abs(measurement.imag - second.imag) + 1.0
            tile_draws.append(("rect_bounds", (left, top, width, height), color))
            edge_x.extend([left, left + width])
            edge_y.extend([top, top + height])
            continue

        # Weight-2 boundary stabilizer: render as the triangle
        #   (measurement_qubit, data_1, data_2)
        # so the boundary plaquette visually connects the ancilla to its two
        # data qubits instead of the bulk-style square.
        if len(data_points) == 2:
            triangle = [
                (measurement.real, measurement.imag),
                (data_points[0].real, data_points[0].imag),
                (data_points[1].real, data_points[1].imag),
            ]
            tile_draws.append(("poly", triangle, color))
            for px, py in triangle:
                edge_x.append(px)
                edge_y.append(py)
            continue

        # Bulk weight-3 / weight-4 stabilizer: fill the polygon spanned by
        # the corners adjacent to each data qubit (each corner offset
        # by ±0.5 in x and y from the measurement qubit).
        corners = []
        for q in data_points:
            dx = q.real - measurement.real
            dy = q.imag - measurement.imag
            sx = 0.5 if dx > 0 else -0.5
            sy = 0.5 if dy > 0 else -0.5
            corners.append((measurement.real + sx, measurement.imag + sy))

        unique_corners = list(dict.fromkeys(corners))
        if len(unique_corners) >= 3:
            polygon_points = sorted(
                unique_corners,
                key=lambda p: math.atan2(p[1] - measurement.imag, p[0] - measurement.real),
            )
            tile_draws.append(("poly", polygon_points, color))
            for px, py in polygon_points:
                edge_x.append(px)
                edge_y.append(py)
        else:
            # Degenerate single-data case: fall back to a unit square at the
            # measurement qubit (rare; included for defensiveness).
            left = measurement.real - 0.5
            top = measurement.imag - 0.5
            tile_draws.append(("rect_bounds", (left, top, 1.0, 1.0), color))
            edge_x.extend([left, left + 1.0])
            edge_y.extend([top, top + 1.0])

    for draw_type, payload, color in tile_draws:
        if draw_type == "rect_bounds":
            left, top, width, height = payload
            ax.add_patch(
                Rectangle(
                    (left, top),
                    width,
                    height,
                    facecolor=color,
                    edgecolor=color,
                    alpha=alpha,
                    linewidth=1.5,
                )
            )
        else:
            ax.add_patch(
                Polygon(
                    payload,
                    closed=True,
                    facecolor=color,
                    edgecolor=color,
                    alpha=alpha,
                    linewidth=1.5,
                )
            )

    data_x = [q.real for q in data_coords]
    data_y = [q.imag for q in data_coords]

    if data_x:
        # Data qubits: white-filled circle with dark edge — matches the 3D
        # timeline rendering (DATA_FILL / DATA_EDGE).
        ax.scatter(
            data_x,
            data_y,
            facecolors=DATA_FILL,
            edgecolors=DATA_EDGE,
            s=70,
            linewidths=1.5,
            zorder=5,
            label="data",
        )

    # Measurement ancillas: filled circle colored by stabilizer basis (matches
    # X_NODE_COLOR / Z_NODE_COLOR / Y_NODE_COLOR in the 3D timelines).
    if meas_basis_at:
        meas_by_basis: dict[str, list[complex]] = {}
        for coord, basis in meas_basis_at.items():
            meas_by_basis.setdefault(basis.upper(), []).append(coord)
        for basis, coords in meas_by_basis.items():
            ax.scatter(
                [q.real for q in coords],
                [q.imag for q in coords],
                facecolors=_basis_node_color(basis),
                edgecolors="white",
                s=85,
                linewidths=1.2,
                zorder=6,
                label=f"{basis} ancilla",
            )

    for q in data_coords | set(meas_basis_at.keys()):
        edge_x.append(q.real)
        edge_y.append(q.imag)

    if edge_x and edge_y:
        left_edge = min(edge_x)
        right_edge = max(edge_x)
        top_edge = min(edge_y)
        bottom_edge = max(edge_y)
    else:
        left_edge = 0.0
        right_edge = 1.0
        top_edge = 0.0
        bottom_edge = 1.0

    if right_edge - left_edge <= 0:
        right_edge = left_edge + 1.0
    if bottom_edge - top_edge <= 0:
        bottom_edge = top_edge + 1.0

    ax.set_xlim(left_edge - 0.5, right_edge + 0.5)
    ax.set_ylim(top_edge - 0.5, bottom_edge + 0.5)
    ax.invert_yaxis()

    ax.set_aspect("equal", adjustable="box")
    if show_axes:
        ax.set_xlabel("X axis")
        ax.set_ylabel("Y axis (origin at top)")
        ax.grid(True, color="lightgrey", linewidth=0.5, alpha=0.4)
        ax.xaxis.set_major_locator(MultipleLocator(1))
        ax.yaxis.set_major_locator(MultipleLocator(1))
    else:
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    if legend:
        handles = [
            LegendPatch(facecolor=X_COLOR, edgecolor=X_COLOR, alpha=alpha, label="X stabilizer"),
            LegendPatch(facecolor=Z_COLOR, edgecolor=Z_COLOR, alpha=alpha, label="Z stabilizer"),
            Line2D(
                [0], [0],
                marker="o", linestyle="",
                markerfacecolor=DATA_FILL, markeredgecolor=DATA_EDGE, markersize=8,
                label="data",
            ),
            Line2D(
                [0], [0],
                marker="o", linestyle="",
                markerfacecolor=X_NODE_COLOR, markeredgecolor="white", markersize=8,
                label="X ancilla",
            ),
            Line2D(
                [0], [0],
                marker="o", linestyle="",
                markerfacecolor=Z_NODE_COLOR, markeredgecolor="white", markersize=8,
                label="Z ancilla",
            ),
        ]
        ax.legend(
            handles=handles,
            frameon=False,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.05),
            ncol=len(handles),
            handletextpad=0.4,
            columnspacing=1.2,
        )

    fig.tight_layout()
    if show:
        plt.show()
    return fig, ax
