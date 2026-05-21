"""Equal-size subplot grid for synchronized heatmap panels."""
from __future__ import annotations

import math

import plotly.graph_objects as go

PANEL_PX = 232
PANEL_GAP = 0.055
ROW_GAP = 0.11


def optimal_grid(n_panels: int) -> tuple[int, int]:
    if n_panels <= 0:
        return 1, 1
    if n_panels == 1:
        return 1, 1
    if n_panels == 2:
        return 1, 2
    if n_panels == 3:
        return 1, 3
    if n_panels == 4:
        return 2, 2
    if n_panels <= 6:
        return 2, 3
    if n_panels <= 8:
        return 2, 4
    if n_panels <= 9:
        return 3, 3
    cols = math.ceil(math.sqrt(n_panels))
    rows = math.ceil(n_panels / cols)
    return rows, cols


def figure_dimensions(n_rows: int, n_cols: int, *, playback_dock: bool = True) -> tuple[int, int]:
    width = int(n_cols * PANEL_PX + 72)
    plot_height = int(width * n_rows / max(n_cols, 1))
    dock = 158 if playback_dock else 48
    title_band = 52 + (12 if n_rows > 1 else 0)
    height = plot_height + title_band + dock
    return width, height


def _axis_suffix(row: int, col: int, n_cols: int) -> str:
    idx = (row - 1) * n_cols + col
    return '' if idx == 1 else str(idx)


def apply_uniform_heatmap_axes(
    fig: go.Figure,
    positions: list[tuple[int, int]],
    n_rows: int,
    n_cols: int,
    *,
    ny: int,
    nx: int,
) -> None:
    x_range = [0, max(nx - 1, 1)]
    y_range = [0, max(ny - 1, 1)]
    occupied = set(positions)
    for row in range(1, n_rows + 1):
        for col in range(1, n_cols + 1):
            if (row, col) not in occupied:
                fig.update_xaxes(visible=False, row=row, col=col)
                fig.update_yaxes(visible=False, row=row, col=col)
                continue
            suffix = _axis_suffix(row, col, n_cols)
            fig.update_xaxes(visible=False, constrain='domain', range=x_range, row=row, col=col)
            fig.update_yaxes(
                visible=False,
                constrain='domain',
                range=y_range,
                scaleanchor=f'x{suffix}',
                scaleratio=1,
                row=row,
                col=col,
            )


def subplot_spacing(n_rows: int, n_cols: int) -> tuple[float, float]:
    return PANEL_GAP, ROW_GAP if n_rows > 1 else PANEL_GAP
