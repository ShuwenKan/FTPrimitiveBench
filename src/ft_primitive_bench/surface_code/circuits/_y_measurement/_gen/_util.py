"""Low-level helpers for complex-number qubit coordinates and circuit rewriting.

Ported verbatim (with relative imports only) from the ``midout.gen._util`` module of
the "Inplace Access to the Surface Code Y Basis" codebase (Craig Gidney, 2023).
"""

from typing import Any, Callable, Iterable, List, TypeVar

import stim

TItem = TypeVar("TItem")


def complex_key(c: complex) -> Any:
    return c.real != int(c.real), c.real, c.imag


def sorted_complex(
    values: Iterable[TItem], *, key: Callable[[TItem], Any] = lambda e: e
) -> List[TItem]:
    return sorted(values, key=lambda e: complex_key(key(e)))


def not_nones(vs) -> List[Any]:
    return [v for v in vs if v is not None]


def stim_circuit_with_transformed_coords(
    circuit: stim.Circuit, transform: Callable[[complex], complex]
) -> stim.Circuit:
    """Returns an equivalent circuit, but with the qubit and detector position metadata modified.
    The "position" is assumed to be the first two coordinates. These are mapped to the real and
    imaginary values of a complex number which is then transformed.

    Note that `SHIFT_COORDS` instructions that modify the first two coordinates are not supported.
    This is because supporting them requires flattening loops, or promising that the given
    transformation is affine.

    Args:
        circuit: The circuit with qubits to reposition.
        transform: The transformation to apply to the positions. The positions are given one by one
            to this method, as complex numbers. The method returns the new complex number for the
            position.

    Returns:
        The transformed circuit.
    """
    result = stim.Circuit()
    for instruction in circuit:
        if isinstance(instruction, stim.CircuitInstruction):
            if instruction.name == "QUBIT_COORDS" or instruction.name == "DETECTOR":
                args = list(instruction.gate_args_copy())
                while len(args) < 2:
                    args.append(0)
                c = transform(args[0] + args[1] * 1j)
                args[0] = c.real
                args[1] = c.imag
                result.append(instruction.name, instruction.targets_copy(), args)
                continue
            if instruction.name == "SHIFT_COORDS":
                args = instruction.gate_args_copy()
                if any(args[:2]):
                    raise NotImplementedError(f"Shifting first two coords: {instruction=}")

        if isinstance(instruction, stim.CircuitRepeatBlock):
            result.append(
                stim.CircuitRepeatBlock(
                    repeat_count=instruction.repeat_count,
                    body=stim_circuit_with_transformed_coords(instruction.body_copy(), transform),
                )
            )
            continue

        result.append(instruction)
    return result
