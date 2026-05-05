import functools
from typing import Dict, FrozenSet, Iterable, Iterator, List, Optional, Sequence


DATA_DIRECTIONS: tuple[str, ...] = ("UR", "UL", "DR", "DL")
VALID_SCHEDULE_TYPES = {"X", "Z", "XX", "ZZ"}


class Tile:
    def __init__(
        self,
        *,
        basis: str,
        measurement_qubit: complex,
        data_qubits: Dict[str, Optional[complex]],
        schedule_type: Optional[str] = None,
        second_measurement_qubit: Optional[complex] = None,
    ):
        self.basis = basis.upper()
        if self.basis not in {"X", "Z"}:
            raise ValueError("basis must be 'X' or 'Z'")
        if not isinstance(measurement_qubit, complex):
            raise TypeError("measurement_qubit must be a complex coordinate")
        if second_measurement_qubit is not None and not isinstance(second_measurement_qubit, complex):
            raise TypeError("second_measurement_qubit must be a complex coordinate")

        unexpected_keys = set(data_qubits) - set(DATA_DIRECTIONS)
        if unexpected_keys:
            raise ValueError(f"unexpected data_qubit keys: {sorted(unexpected_keys)}")

        normalized_data: Dict[str, Optional[complex]] = {}
        for direction in DATA_DIRECTIONS:
            coord = data_qubits.get(direction)
            if coord is not None and not isinstance(coord, complex):
                raise TypeError("data_qubit entries must be complex coordinates or None")
            normalized_data[direction] = coord

        self.schedule_type = (schedule_type or self.basis).upper()
        if self.schedule_type not in VALID_SCHEDULE_TYPES:
            raise ValueError(f"schedule_type must be one of {sorted(VALID_SCHEDULE_TYPES)}")

        self.measurement_qubit = measurement_qubit
        self.second_measurement_qubit = second_measurement_qubit
        self.data_qubits = normalized_data

    def __str__(self):
        return (
            f"Tile(basis={self.basis!r}, schedule_type={self.schedule_type!r}, "
            f"m={self.measurement_qubit!r}, data={self.data_qubits!r})"
        )

    def __repr__(self):
        return (
            f"Tile(basis={self.basis!r}, schedule_type={self.schedule_type!r}, "
            f"m={self.measurement_qubit!r}, data={self.data_qubits!r})"
        )


class Patch:
    """Patch variant that enforces unique measurement coordinates.

    Appending a tile whose measurement qubit already exists in the patch will
    replace the older tile at that coordinate instead of creating a duplicate.
    """

    def __init__(
        self,
        tiles: Optional[Iterable[Tile]] = None,
    ):
        self._tiles: List[Tile] = []
        self._index_by_measure: Dict[complex, int] = {}
        if tiles is not None:
            self.extend(tiles)

    def __iter__(self) -> Iterator[Tile]:
        return iter(self._tiles)

    def __len__(self) -> int:
        return len(self._tiles)

    def __getitem__(self, item):
        return self._tiles[item]

    @property
    def tiles(self) -> Sequence[Tile]:
        """Read-only view of current tiles."""
        return tuple(self._tiles)

    def append(self, tile: Tile) -> None:
        """Insert a tile, replacing any tile with the same measurement coordinate."""
        self._add_or_replace(tile)

    def extend(self, tiles: Iterable[Tile]) -> None:
        """Append many tiles, respecting unique measurement coordinates.

        Invalidates cached sets once after all tiles are inserted rather than
        once per tile, which avoids redundant cache thrashing for large batches.
        """
        for tile in tiles:
            self._add_or_replace_impl(tile)
        self._invalidate_cached_sets()

    def _add_or_replace(self, tile: Tile) -> None:
        self._add_or_replace_impl(tile)
        self._invalidate_cached_sets()

    def _add_or_replace_impl(self, tile: Tile) -> None:
        """Core insertion logic without cache invalidation."""
        m = tile.measurement_qubit
        if not isinstance(m, complex):
            raise TypeError("measurement_qubit must be a complex coordinate")

        secondary = getattr(tile, "second_measurement_qubit", None)
        insert_at: Optional[int] = None
        if secondary is not None:
            if not isinstance(secondary, complex):
                raise TypeError("second_measurement_qubit must be a complex coordinate")
            secondary_idx = self._index_by_measure.get(secondary)
            if secondary_idx is not None:
                self._tiles.pop(secondary_idx)
                if secondary == m:
                    insert_at = secondary_idx
                self._index_by_measure = {
                    existing_tile.measurement_qubit: i
                    for i, existing_tile in enumerate(self._tiles)
                }

        existing = self._index_by_measure.get(m)
        if existing is None:
            if insert_at is not None and insert_at <= len(self._tiles):
                self._tiles.insert(insert_at, tile)
            else:
                self._tiles.append(tile)
        else:
            self._tiles[existing] = tile

        self._index_by_measure = {
            existing_tile.measurement_qubit: i
            for i, existing_tile in enumerate(self._tiles)
        }

    def offset(self, offset_x: float, offset_y: float) -> None:
        """Translate all coordinates by the given offsets."""

        def _shift_complex(coord: complex) -> complex:
            return complex(coord.real + offset_x, coord.imag + offset_y)

        for tile in self._tiles:
            tile.measurement_qubit = _shift_complex(tile.measurement_qubit)
            if getattr(tile, "second_measurement_qubit", None) is not None:
                tile.second_measurement_qubit = _shift_complex(
                    tile.second_measurement_qubit
                )
            tile.data_qubits = {
                d: (_shift_complex(coord) if coord is not None else None)
                for d, coord in tile.data_qubits.items()
            }

        self._index_by_measure = {
            tile.measurement_qubit: i for i, tile in enumerate(self._tiles)
        }
        self._invalidate_cached_sets()

    @functools.cached_property
    def data_set(self) -> FrozenSet[complex]:
        s = set()
        for tile in self._tiles:
            for q in tile.data_qubits.values():
                if q is not None:
                    s.add(q)
        return frozenset(s)

    @functools.cached_property
    def measure_set(self) -> FrozenSet[complex]:
        return frozenset(tile.measurement_qubit for tile in self._tiles)

    @functools.cached_property
    def measure_x_set(self) -> FrozenSet[complex]:
        """All measurement qubits for X-basis tiles."""
        return frozenset(t.measurement_qubit for t in self._tiles if t.basis == 'X')

    @functools.cached_property
    def measure_z_set(self) -> FrozenSet[complex]:
        """All measurement qubits for Z-basis tiles."""
        return frozenset(t.measurement_qubit for t in self._tiles if t.basis == 'Z')

    def _invalidate_cached_sets(self) -> None:
        for attr in ("data_set", "measure_set", "measure_x_set", "measure_z_set"):
            self.__dict__.pop(attr, None)

    def __repr__(self) -> str:
        return f"Patch(tiles={len(self._tiles)})"

    def __str__(self) -> str:
        nx = len(self.measure_x_set)
        nz = len(self.measure_z_set)
        summary = (
            f"Patch with {len(self._tiles)} tiles "
            f"({nx} X-tiles, {nz} Z-tiles), "
            f"{len(self.data_set)} data, {len(self.measure_set)} ancillas."
        )
        tiles = "\n".join(
            f"{i:>3}. {str(tile)}" for i, tile in enumerate(self._tiles, start=1)
        )
        return "\n".join([summary, tiles]) if tiles else summary
