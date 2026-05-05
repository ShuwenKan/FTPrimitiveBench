from typing import List, Literal, Tuple

from ._tile import Patch, Tile

DIRECTION_OFFSETS = {
    "UR": 0.5 - 0.5j,
    "DR": 0.5 + 0.5j,
    "UL": -0.5 - 0.5j,
    "DL": -0.5 + 0.5j,
}
BOUNDARY_ACTIVE_DIRECTIONS = {
    'bottom': {'DR', 'DL'},
    'top': {'UR', 'UL'},
    'left': {'UR', 'DR'},
    'right': {'UL', 'DL'},
}

BoundaryOrientation = Literal['top', 'bottom', 'left', 'right']
BoundaryBasis = Literal['X', 'Z']


def _schedule_type_for_basis(tile_basis: str, *, top_bottom_basis: str) -> str:
    tile_basis = tile_basis.upper()
    top_bottom_basis = top_bottom_basis.upper()
    if top_bottom_basis == "Z":
        return tile_basis
    if top_bottom_basis == "X":
        return tile_basis * 2
    raise ValueError("top_bottom_basis must be 'X' or 'Z'")


def checkerboard_weight4_patch(
    x_distance: int,
    z_distance: int,
    *,
    origin: Tuple[float, float] = (1.5, 1.5),
    start_basis: str = "Z",
) -> Patch:
    """Construct a Patch of alternating weight-4 stabilizer tiles.

    Args:
        x_distance: Number of plaquettes along the x direction (must be positive).
        z_distance: Number of plaquettes along the y direction (must be positive).
        origin: Real-space coordinate of the lower-left plaquette centre.
        start_basis: Basis for the plaquette at ``origin`` (alternates thereafter).

    Raises:
        ValueError: If dimensions are non-positive or ``start_basis`` is invalid.

    Returns:
        Patch containing ``x_distance * z_distance`` weight-4 Tiles laid out in a checkerboard pattern.
    """
    if x_distance <= 0 or z_distance <= 0:
        raise ValueError("x_distance and z_distance must be positive integers")

    start = start_basis.upper()
    if start not in {"X", "Z"}:
        raise ValueError("start_basis must be 'X' or 'Z'")

    alternate = "Z" if start == "X" else "X"
    ox, oy = origin
    tiles: List[Tile] = []

    for dx in range(x_distance):
        for dy in range(z_distance):
            mx = ox + dx
            my = oy + dy

            m = complex(mx, my)
            basis = start if (dx + dy) % 2 == 0 else alternate
            dq = {
                'UL': complex(mx - 0.5, my - 0.5),
                'UR': complex(mx + 0.5, my - 0.5),
                'DR': complex(mx + 0.5, my + 0.5),
                'DL': complex(mx - 0.5, my + 0.5),
            }
            tiles.append(
                Tile(
                    basis=basis,
                    measurement_qubit=m,
                    data_qubits=dq,
                )
            )

    return Patch(tiles)


def boundary_tiles(
    orientation: BoundaryOrientation,
    *,
    length: int,
    basis: BoundaryBasis,
    start: complex,
) -> Patch:
    """Generate boundary stabilizers with canonical data qubit ordering.

    Args:
        orientation: Which boundary direction to follow.
        length: Number of weight-2 stabilizers along the boundary (positive integer).
        basis: Stabilizer type to deploy along the boundary.
        start: Measurement coordinate for the first stabilizer (complex).

    Pattern:
        - The first stabilizer uses a weight-2 configuration.
        - Measurement coordinates advance every 2 lattice steps along the boundary,
          i.e. one coordinate is skipped between consecutive stabilizers.

    Returns:
        A Patch containing the requested boundary segment.
    """
    if length <= 0:
        raise ValueError('length must be a positive integer')

    basis = basis.upper()
    if basis not in {'X', 'Z'}:
        raise ValueError("basis must be 'X' or 'Z'")

    orientation = orientation.lower()
    if orientation not in BOUNDARY_ACTIVE_DIRECTIONS:
        raise ValueError('orientation must be one of top/bottom/left/right')

    step: complex
    if orientation in {'top', 'bottom'}:
        step = 2.0 + 0.0j
    else:
        step = 0.0 + 2.0j

    tiles: List[Tile] = []
    active_dirs = set(BOUNDARY_ACTIVE_DIRECTIONS[orientation])

    for i in range(length):
        measurement = start + i * step
        dq = {
            label: (measurement + DIRECTION_OFFSETS[label] if label in active_dirs else None)
            for label in ('UR', 'UL', 'DR', 'DL')
        }
        tiles.append(
            Tile(
                basis=basis,
                measurement_qubit=measurement,
                data_qubits=dq,
                schedule_type=basis,
            )
        )

    return Patch(tiles)


def rectangular_surface_code_patch(
    x_distance: int,
    z_distance: int,
    *,
    top_bottom_basis: str = "Z",
    left_right_basis: str = "X",
) -> Patch:
    """Construct a rectangular surface-code patch using existing helpers.

    This assembles the interior via :func:`checkerboard_weight4_patch` and the
    boundaries via :func:`boundary_tiles`, with direction-keyed data qubit dicts
    and explicit schedule_type per tile.
    """
    if x_distance <= 0 or z_distance <= 0:
        raise ValueError("x_distance and z_distance must be positive integers")

    top_bottom_basis = top_bottom_basis.upper()
    left_right_basis = left_right_basis.upper()
    if top_bottom_basis not in {"X", "Z"}:
        raise ValueError("top_bottom_basis must be 'X' or 'Z'")
    if left_right_basis not in {"X", "Z"}:
        raise ValueError("left_right_basis must be 'X' or 'Z'")

    def _checkerboard_from_indices(ax: int, ay: int) -> str:
        return "Z" if ((ax + ay) & 1) == 0 else "X"

    tiles: List[Tile] = []

    if x_distance > 1 and z_distance > 1:
        interior_patch = checkerboard_weight4_patch(
            x_distance - 1,
            z_distance - 1,
            origin=(1.5, 1.5),
            start_basis="Z",
        )
        for tile in interior_patch.tiles:
            if sum(v is not None for v in tile.data_qubits.values()) != 4:
                raise RuntimeError("Interior tiles must have weight-4 stabilizers")
            tiles.append(Tile(
                basis=tile.basis,
                measurement_qubit=tile.measurement_qubit,
                data_qubits=dict(tile.data_qubits),
                schedule_type=_schedule_type_for_basis(
                    tile.basis,
                    top_bottom_basis=top_bottom_basis,
                ),
            ))

    def _boundary_coords(orientation: str, basis: str) -> list[complex]:
        coords: list[complex] = []
        if orientation == 'bottom':
            ay = 0
            imag = 0.5
            for ax in range(1, x_distance):
                if _checkerboard_from_indices(ax, ay) == basis:
                    coords.append(complex(ax + 0.5, imag))
        elif orientation == 'top':
            ay = z_distance
            imag = z_distance + 0.5
            for ax in range(1, x_distance):
                if _checkerboard_from_indices(ax, ay) == basis:
                    coords.append(complex(ax + 0.5, imag))
        elif orientation == 'left':
            ax = 0
            real = 0.5
            for ay in range(1, z_distance):
                if _checkerboard_from_indices(ax, ay) == basis:
                    coords.append(complex(real, ay + 0.5))
        else:  # right
            ax = x_distance
            real = x_distance + 0.5
            for ay in range(1, z_distance):
                if _checkerboard_from_indices(ax, ay) == basis:
                    coords.append(complex(real, ay + 0.5))
        return coords

    for orientation, boundary_basis in (
        ('top', top_bottom_basis),
        ('bottom', top_bottom_basis),
        ('left', left_right_basis),
        ('right', left_right_basis),
    ):
        coords = _boundary_coords(orientation, boundary_basis)
        if not coords:
            continue

        boundary_patch = boundary_tiles(
            orientation=orientation,
            length=len(coords),
            basis=boundary_basis,
            start=coords[0],
        )

        for tile in boundary_patch.tiles:
            tiles.append(Tile(
                basis=boundary_basis,
                measurement_qubit=tile.measurement_qubit,
                data_qubits=dict(tile.data_qubits),
                schedule_type=_schedule_type_for_basis(
                    boundary_basis,
                    top_bottom_basis=top_bottom_basis,
                ),
            ))

    return Patch(tiles)


def create_individual_patch(x_distance: int, z_distance: int, bridge_length: int) -> Patch:
    """Build two rectangular patches stacked vertically with a gap for ancillas."""
    if x_distance <= 0 or z_distance <= 0 or bridge_length <= 0:
        raise ValueError("x_distance, z_distance, bridge_length must be positive integers")

    lower_patch = rectangular_surface_code_patch(
        x_distance,
        z_distance,
        top_bottom_basis="X",
        left_right_basis="Z",
    )
    upper_patch = rectangular_surface_code_patch(
        x_distance,
        z_distance,
        top_bottom_basis="X",
        left_right_basis="Z",
    )
    upper_patch.offset(0, z_distance + bridge_length)
    lower_patch.extend(upper_patch.tiles)
    return lower_patch
