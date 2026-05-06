"""Render a representative gallery of figures into ``figures/``.

Run from the repo root:

    python figures/generate_figures.py

Produces, for each 2D patch:
    figures/patch_<NxN>.png

For each primitive, both an interactive HTML figure and a 2D PNG snapshot
rendered from the same plotly figure with the default camera view:
    figures/<primitive>_timeline.html
    figures/<primitive>_timeline.png

Primitives covered: ``memory``, ``transversal_h``, ``lattice_surgery``,
``s_gate``.

The PNG snapshots require ``kaleido`` (install via the ``[viz]`` extra).
HTML figures are always written.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from surface_code.circuits import (
    build_merged_patch,
    rectangular_surface_code_patch,
)
from surface_code.visualization import (
    plot_lattice_surgery_timeline,
    plot_memory_timeline,
    plot_patch,
    plot_s_gate_timeline,
    plot_transversal_h_timeline,
)

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "figures"
FIGURES.mkdir(exist_ok=True)

# Default camera for every 3D timeline figure.
CAMERA_EYE = dict(x=0.8, y=-2.4, z=0.4)


def _apply_camera(fig) -> None:
    """Set the same default 3D camera on every scene in the figure."""
    fig.update_layout(scene_camera=dict(eye=CAMERA_EYE))
    # Some primitives use multiple subplots (scene, scene2, ...).
    for scene_name in [k for k in fig.layout if k.startswith("scene")]:
        fig.layout[scene_name].camera = dict(eye=CAMERA_EYE)


def _save_mpl(fig, name: str) -> None:
    out = FIGURES / name
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out.relative_to(ROOT)}")


def _save_plotly(fig, name: str) -> None:
    """Save both an interactive 3D HTML and a static 2D PNG snapshot.

    The PNG is rendered from the same plotly figure with the default camera
    (``CAMERA_EYE``) so it gives a consistent angle across primitives.
    """
    _apply_camera(fig)

    html_path = FIGURES / f"{name}.html"
    fig.write_html(html_path, include_plotlyjs="cdn")
    print(f"wrote {html_path.relative_to(ROOT)}")

    png_path = FIGURES / f"{name}.png"
    try:
        fig.write_image(png_path, width=900, height=700, scale=2)
        print(f"wrote {png_path.relative_to(ROOT)}")
    except (ValueError, ImportError) as exc:  # kaleido missing
        print(f"kaleido missing ({exc}); skipped {png_path.relative_to(ROOT)}")


def main() -> None:
    # ─── 2D patches (matplotlib) ─────────────────────────────────────────
    for d in (3, 5):
        patch = rectangular_surface_code_patch(x_distance=d, z_distance=d)
        fig, _ = plot_patch(patch, show=False, legend=True)
        _save_mpl(fig, f"patch_{d}x{d}.png")

    rect = rectangular_surface_code_patch(x_distance=3, z_distance=5)
    fig, _ = plot_patch(rect, show=False, legend=True)
    _save_mpl(fig, "patch_3x5.png")

    # Pre-merge layout: two patches separated by a bridge.
    individual, _ = build_merged_patch(
        x_distance=3, z_distance=3, bridge_length=1, horizontal=False
    )
    fig, _ = plot_patch(individual, show=False, legend=True)
    _save_mpl(fig, "merged_patch.png")

    # ─── 3D spacetime timelines (plotly) ─────────────────────────────────
    fig = plot_memory_timeline(x_distance=3, z_distance=3, rounds=3, meas_basis="Z")
    _save_plotly(fig, "memory_timeline")

    fig = plot_transversal_h_timeline(
        x_distance=3, z_distance=3, pre_rounds=2, post_rounds=2, meas_basis="Z"
    )
    _save_plotly(fig, "transversal_h_timeline")

    fig = plot_lattice_surgery_timeline(
        x_distance=3, z_distance=3, bridge_length=1,
        pre_rounds=1, merge_rounds=2, post_rounds=1,
    )
    _save_plotly(fig, "lattice_surgery_timeline")

    fig = plot_s_gate_timeline(
        distance=3, bridge_length=1, pre_rounds=1, merge_rounds=2,
        boundary_rounds=1,
    )
    _save_plotly(fig, "s_gate_timeline")


if __name__ == "__main__":
    main()
