"""Generate interactive HTML circuit viewers for the chunks-backend primitives.

Uses ``stimflow.stim_circuit_html_viewer`` to produce single-file HTML that
embeds a navigable 3D / time-sliced view of each compiled circuit. Open the
generated ``.html`` files directly in a browser — no server needed.

Run from the repo root::

    python figures/generate_stimflow_html.py

Outputs land in ``figures/stimflow_*.html``.
"""

from __future__ import annotations

from pathlib import Path

import stimflow as sf

from ft_primitive_bench.surface_code.circuits._stimflow.lattice_surgery import (
    build_lattice_surgery_chunks,
)
from ft_primitive_bench.surface_code.circuits._stimflow.memory import build_memory_chunks
from ft_primitive_bench.surface_code.circuits._stimflow.transversal_h import (
    build_transversal_h_chunks,
)


def _compile(chunks):
    compiler = sf.ChunkCompiler()
    for c in chunks:
        compiler.append(c)
    return compiler.finish_circuit()


def main() -> None:
    out_dir = Path(__file__).parent
    out_dir.mkdir(exist_ok=True)

    targets = [
        (
            "stimflow_memory_d3_r3_Z.html",
            "memory(x_distance=3, z_distance=3, rounds=3, meas_basis='Z')",
            _compile(build_memory_chunks(3, 3, 3, "Z")),
        ),
        (
            "stimflow_memory_d5_r3_Z.html",
            "memory(x_distance=5, z_distance=5, rounds=3, meas_basis='Z')",
            _compile(build_memory_chunks(5, 5, 3, "Z")),
        ),
        (
            "stimflow_transversal_h_d3_p1q1_Z.html",
            "transversal_h(x_distance=3, z_distance=3, pre=1, post=1, meas_basis='Z')",
            _compile(build_transversal_h_chunks(3, 3, 1, 1, "Z")),
        ),
        (
            "stimflow_lattice_surgery_d3_b1_1m2p1_Z.html",
            "lattice_surgery(d=3, bridge=1, pre=1, merge=2, post=1, meas_basis='Z')",
            _compile(build_lattice_surgery_chunks(3, 3, 1, 1, 2, 1, "Z")),
        ),
        (
            "stimflow_lattice_surgery_d5_b1_2m3p2_Z.html",
            "lattice_surgery(d=5, bridge=1, pre=2, merge=3, post=2, meas_basis='Z')",
            _compile(build_lattice_surgery_chunks(5, 5, 1, 2, 3, 2, "Z")),
        ),
    ]

    for filename, label, circuit in targets:
        path = out_dir / filename
        viewer = sf.stim_circuit_html_viewer(circuit, width=900, height=700)
        path.write_text(str(viewer), encoding="utf-8")
        print(f"  wrote {path.relative_to(Path.cwd())}  ({len(circuit)} instructions, {circuit.num_detectors} detectors)  — {label}")

    print(f"\nDone. Open the .html files in figures/ in any modern browser.")


if __name__ == "__main__":
    main()
