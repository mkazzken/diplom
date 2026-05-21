"""Plotly visualization helpers for AI comparison."""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def _make_heatmap_trace(field, x, y, name, zmin, zmax):
    return go.Heatmap(
        z=field,
        x=np.asarray(x) * 1000.0,
        y=np.asarray(y) * 1000.0,
        colorscale='viridis',
        zmin=zmin,
        zmax=zmax,
        coloraxis='coloraxis',
        colorbar=dict(title='Value'),
        name=name,
        hovertemplate=f'{name}<br>x=%{{x:.1f}} mm<br>y=%{{y:.1f}} mm<br>value=%{{z:.4f}}<extra></extra>'
    )


def _make_contour_trace(field, x, y, level=0.5):
    return go.Contour(
        z=field,
        x=np.asarray(x) * 1000.0,
        y=np.asarray(y) * 1000.0,
        contours=dict(start=level, end=level, size=1),
        line=dict(color='red', width=2),
        showscale=False,
        hoverinfo='skip'
    )


def create_single_heatmap_figure(field, x, y, title, zmin=None, zmax=None, show_contour: bool = False):
    if zmin is None:
        zmin = float(np.nanmin(field))
    if zmax is None:
        zmax = float(np.nanmax(field))

    fig = go.Figure()
    fig.add_trace(_make_heatmap_trace(field, x, y, title, zmin, zmax))
    if show_contour:
        fig.add_trace(_make_contour_trace(field, x, y, level=0.5))

    fig.update_layout(
        title=title,
        xaxis_title='x [mm]',
        yaxis_title='y [mm]',
        yaxis_autorange='reversed',
        template='plotly_white'
    )
    fig.update_yaxes(scaleanchor='x', scaleratio=1)
    return fig


def create_comparison_heatmap_figure(fields, x, y, titles, zmin, zmax, show_contour=False):
    fig = make_subplots(rows=1, cols=len(fields), subplot_titles=titles, shared_yaxes=True)
    for idx, field in enumerate(fields, start=1):
        fig.add_trace(_make_heatmap_trace(field, x, y, titles[idx - 1], zmin, zmax), row=1, col=idx)
        if show_contour:
            fig.add_trace(_make_contour_trace(field, x, y, level=0.5), row=1, col=idx)
    fig.update_layout(
        template='plotly_white',
        coloraxis=dict(colorscale='viridis', cmin=zmin, cmax=zmax),
        showlegend=False,
        title_text=' | '.join(titles)
    )
    fig.update_yaxes(scaleanchor='x', scaleratio=1)
    return fig


def create_three_way_comparison_figure(gt, pure, hybrid, x, y, zmin, zmax, show_contour=False):
    return create_comparison_heatmap_figure(
        [gt, pure, hybrid],
        x,
        y,
        ['Ground truth', 'Pure AI', 'Hybrid'],
        zmin,
        zmax,
        show_contour=show_contour
    )
