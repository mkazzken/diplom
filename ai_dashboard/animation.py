"""Client-side Plotly animations — synchronized panels, single frame clock."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .charts import THEME
from .panel_layout import (
    PANEL_PX,
    apply_uniform_heatmap_axes,
    figure_dimensions,
    optimal_grid,
    subplot_spacing,
)


@dataclass
class AnimationPanel:
    z: np.ndarray
    title: str
    colorscale: str = 'Viridis'
    zmin: float | None = None
    zmax: float | None = None


def _heatmap_trace(z: np.ndarray, panel: AnimationPanel) -> go.Heatmap:
    ny, nx = z.shape[-2], z.shape[-1]
    return go.Heatmap(
        z=z,
        x=list(range(nx)),
        y=list(range(ny)),
        colorscale=panel.colorscale,
        zmin=panel.zmin,
        zmax=panel.zmax,
        showscale=False,
        zsmooth=False,
        hovertemplate='x=%{x}<br>y=%{y}<br>v=%{z:.4f}<extra></extra>',
    )


def _playback_margins(n_rows: int) -> dict[str, int]:
    return dict(l=44, r=44, t=52 + (12 if n_rows > 1 else 0), b=158 if n_rows > 1 else 138)


def _playback_updatemenus(frame_duration_ms: int) -> list[dict]:
    play_pause = [
        dict(
            label='▶  Play',
            method='animate',
            args=[None, dict(frame=dict(duration=frame_duration_ms, redraw=True), fromcurrent=True, mode='immediate', transition=dict(duration=0))],
        ),
        dict(
            label='⏸  Pause',
            method='animate',
            args=[[None], dict(frame=dict(duration=0, redraw=False), mode='immediate', transition=dict(duration=0))],
        ),
    ]
    return [
        dict(
            type='buttons',
            showactive=False,
            direction='right',
            x=0.5,
            xanchor='center',
            y=-0.04,
            yanchor='top',
            bgcolor='rgba(14, 20, 40, 0.92)',
            bordercolor='rgba(100, 140, 220, 0.45)',
            borderwidth=1,
            font=dict(color='#e8eefc', size=13),
            buttons=play_pause,
        )
    ]


def _playback_slider(n_frames: int) -> list[dict]:
    steps = [
        dict(
            method='animate',
            args=[[str(t)], dict(frame=dict(duration=0, redraw=True), mode='immediate', transition=dict(duration=0))],
            label=str(t + 1),
        )
        for t in range(n_frames)
    ]
    return [
        dict(
            active=0,
            x=0.5,
            xanchor='center',
            len=0.62,
            y=-0.20,
            yanchor='top',
            pad=dict(t=52, b=12),
            bgcolor='rgba(14, 20, 40, 0.55)',
            bordercolor='rgba(100, 140, 220, 0.3)',
            currentvalue=dict(prefix='Frame  ', visible=True, font=dict(color='#00d4aa', size=13)),
            steps=steps,
        )
    ]


def build_synchronized_animation(
    panels: list[AnimationPanel],
    *,
    n_cols: int | None = None,
    n_rows: int | None = None,
    frame_duration_ms: int = 450,
    height: int | None = None,
    width: int | None = None,
    uirevision: str = 'sync-playback',
) -> go.Figure:
    if not panels:
        raise ValueError('panels must not be empty')

    n_frames = panels[0].z.shape[0]
    ny, nx = int(panels[0].z.shape[-2]), int(panels[0].z.shape[-1])
    for panel in panels:
        if panel.z.shape[0] != n_frames:
            raise ValueError('All panel sequences must have the same number of frames')
        if panel.z.shape[-2:] != (ny, nx):
            raise ValueError('All panels must share the same grid shape')

    n_panels = len(panels)
    auto_rows, auto_cols = optimal_grid(n_panels)
    n_rows = n_rows or auto_rows
    n_cols = n_cols or auto_cols
    hspace, vspace = subplot_spacing(n_rows, n_cols)
    titles = [panel.title for panel in panels] + [''] * (n_rows * n_cols - n_panels)

    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=titles[: n_rows * n_cols],
        horizontal_spacing=hspace,
        vertical_spacing=vspace,
        row_heights=[1.0] * n_rows,
        column_widths=[1.0] * n_cols,
    )

    positions: list[tuple[int, int]] = []
    for idx in range(n_panels):
        row = idx // n_cols + 1
        col = idx % n_cols + 1
        positions.append((row, col))
        fig.add_trace(_heatmap_trace(panels[idx].z[0], panels[idx]), row=row, col=col)

    fig.frames = [
        go.Frame(
            data=[_heatmap_trace(panels[i].z[t], panels[i]) for i in range(n_panels)],
            name=str(t),
        )
        for t in range(n_frames)
    ]

    fig_w, fig_h = figure_dimensions(n_rows, n_cols)
    theme = {key: value for key, value in THEME.items() if key != 'margin'}
    fig.update_layout(
        width=width or fig_w,
        height=height or fig_h,
        autosize=False,
        uirevision=uirevision,
        showlegend=False,
        margin=_playback_margins(n_rows),
        updatemenus=_playback_updatemenus(frame_duration_ms),
        sliders=_playback_slider(n_frames),
        **theme,
    )

    for annotation in fig.layout.annotations:
        if annotation.text:
            annotation.font = dict(color='#c5d0ea', size=12)

    apply_uniform_heatmap_axes(fig, positions, n_rows, n_cols, ny=ny, nx=nx)
    return fig


def duration_from_speed(speed: float, base_ms: int = 450) -> int:
    return max(80, int(base_ms / max(speed, 0.1)))
