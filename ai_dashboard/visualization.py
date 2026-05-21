"""Plotly visualization helpers for AI comparison."""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

PANEL_SIZE = 380
COLORBAR_WIDTH = 90
HORIZONTAL_SPACING = 0.06


def _mm_axes(x, y):
    x_mm = np.asarray(x, dtype=np.float64) * 1000.0
    y_mm = np.asarray(y, dtype=np.float64) * 1000.0
    return x_mm, y_mm


def _axis_ref(col_idx: int, axis: str) -> str:
    return axis if col_idx == 1 else f'{axis}{col_idx}'


def _make_heatmap_trace(field, x_mm, y_mm, name, zmin, zmax, showscale: bool = False):
    return go.Heatmap(
        z=field,
        x=x_mm,
        y=y_mm,
        colorscale='viridis',
        zmin=zmin,
        zmax=zmax,
        coloraxis='coloraxis',
        showscale=showscale,
        name=name,
        hovertemplate=(
            f'{name}<br>x=%{{x:.1f}} mm<br>y=%{{y:.1f}} mm<br>'
            'value=%{{z:.4f}}<extra></extra>'
        ),
    )


def _make_contour_trace(field, x_mm, y_mm, level=0.5):
    return go.Contour(
        z=field,
        x=x_mm,
        y=y_mm,
        contours=dict(start=level, end=level, size=1),
        line=dict(color='red', width=2),
        showscale=False,
        hoverinfo='skip',
    )


def _apply_equal_panel_layout(fig, n_cols: int, x_mm, y_mm):
    x_range = [float(x_mm[0]), float(x_mm[-1])]
    y_range = [float(y_mm[-1]), float(y_mm[0])]

    fig.update_layout(
        template='plotly_white',
        height=PANEL_SIZE,
        width=PANEL_SIZE * n_cols + COLORBAR_WIDTH,
        margin=dict(l=48, r=COLORBAR_WIDTH + 12, t=56, b=48),
        coloraxis=dict(
            colorscale='viridis',
            colorbar=dict(title='Value', len=0.75, thickness=16),
        ),
    )

    for col_idx in range(1, n_cols + 1):
        x_ref = _axis_ref(col_idx, 'x')
        y_ref = _axis_ref(col_idx, 'y')
        fig.update_xaxes(
            range=x_range,
            title_text='x [mm]' if col_idx == 1 else '',
            constrain='domain',
            row=1,
            col=col_idx,
        )
        fig.update_yaxes(
            range=y_range,
            title_text='y [mm]' if col_idx == 1 else '',
            scaleanchor=x_ref,
            scaleratio=1,
            constrain='domain',
            row=1,
            col=col_idx,
        )


def create_single_heatmap_figure(field, x, y, title, zmin=None, zmax=None, show_contour: bool = False):
    if zmin is None:
        zmin = float(np.nanmin(field))
    if zmax is None:
        zmax = float(np.nanmax(field))

    x_mm, y_mm = _mm_axes(x, y)
    fig = go.Figure()
    fig.add_trace(_make_heatmap_trace(field, x_mm, y_mm, title, zmin, zmax, showscale=True))
    if show_contour:
        fig.add_trace(_make_contour_trace(field, x_mm, y_mm, level=0.5))

    fig.update_layout(
        title=title,
        template='plotly_white',
        height=PANEL_SIZE,
        width=PANEL_SIZE + COLORBAR_WIDTH,
        margin=dict(l=48, r=COLORBAR_WIDTH + 12, t=56, b=48),
        coloraxis=dict(colorscale='viridis', cmin=zmin, cmax=zmax),
    )
    fig.update_xaxes(title_text='x [mm]', range=[x_mm[0], x_mm[-1]], constrain='domain')
    fig.update_yaxes(
        title_text='y [mm]',
        range=[y_mm[-1], y_mm[0]],
        scaleanchor='x',
        scaleratio=1,
        constrain='domain',
    )
    return fig


def create_comparison_heatmap_figure(fields, x, y, titles, zmin, zmax, show_contour=False):
    n_cols = len(fields)
    x_mm, y_mm = _mm_axes(x, y)

    fig = make_subplots(
        rows=1,
        cols=n_cols,
        subplot_titles=titles,
        horizontal_spacing=HORIZONTAL_SPACING,
        column_widths=[1] * n_cols,
    )

    for idx, field in enumerate(fields, start=1):
        showscale = idx == n_cols
        fig.add_trace(
            _make_heatmap_trace(field, x_mm, y_mm, titles[idx - 1], zmin, zmax, showscale=showscale),
            row=1,
            col=idx,
        )
        if show_contour:
            fig.add_trace(_make_contour_trace(field, x_mm, y_mm, level=0.5), row=1, col=idx)

    fig.update_layout(showlegend=False, title_text=' | '.join(titles))
    fig.update_layout(coloraxis=dict(cmin=zmin, cmax=zmax))
    _apply_equal_panel_layout(fig, n_cols, x_mm, y_mm)
    return fig


def create_three_way_comparison_figure(gt, pure, hybrid, x, y, zmin, zmax, show_contour=False):
    return create_comparison_heatmap_figure(
        [gt, pure, hybrid],
        x,
        y,
        ['Ground truth', 'Pure AI', 'Hybrid'],
        zmin,
        zmax,
        show_contour=show_contour,
    )
