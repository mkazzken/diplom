"""Visualization utilities for the Stefan problem sandbox."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go

FIELD_PX = 520


def _mm_coords(x, y):
    x_mm = np.asarray(x, dtype=np.float64) * 1000.0
    y_mm = np.asarray(y, dtype=np.float64) * 1000.0
    return x_mm, y_mm


def _apply_square_field_layout(fig, x, y, title=''):
    """Keep heatmaps square (1:1 mm/mm) and readable on the dark Streamlit theme."""
    x_mm, y_mm = _mm_coords(x, y)
    fig.update_layout(
        title=dict(text=title, font=dict(color='#f0f4ff', size=14)),
        xaxis_title='x [mm]',
        yaxis_title='y [mm]',
        width=FIELD_PX,
        height=FIELD_PX,
        autosize=False,
        paper_bgcolor='rgba(8, 12, 24, 0.55)',
        plot_bgcolor='rgba(10, 14, 28, 0.6)',
        font=dict(color='#e8eefc', family='Inter, Segoe UI, sans-serif', size=12),
        margin=dict(l=52, r=88, t=56, b=52),
    )
    fig.update_xaxes(
        range=[float(x_mm[0]), float(x_mm[-1])],
        constrain='domain',
        color='#c5d0ea',
    )
    fig.update_yaxes(
        autorange='reversed',
        scaleanchor='x',
        scaleratio=1,
        constrain='domain',
        color='#c5d0ea',
    )


def create_temperature_figure(field, x, y, title='', zmin=None, zmax=None):
    if zmin is None:
        zmin = np.min(field)
    if zmax is None:
        zmax = np.max(field)

    x_mm, y_mm = _mm_coords(x, y)
    fig = go.Figure(
        data=go.Heatmap(
            z=field,
            x=x_mm,
            y=y_mm,
            colorscale='hot',
            zmin=zmin,
            zmax=zmax,
            colorbar=dict(title='°C', len=0.75, thickness=14),
        )
    )
    _apply_square_field_layout(fig, x, y, title)
    return fig


def create_liquid_fraction_figure(field, x, y, title=''):
    x_mm, y_mm = _mm_coords(x, y)
    fig = go.Figure()
    fig.add_trace(
        go.Heatmap(
            z=field,
            x=x_mm,
            y=y_mm,
            colorscale='RdBu',
            zmin=0.0,
            zmax=1.0,
            colorbar=dict(title='Liquid fraction', len=0.75, thickness=14),
        )
    )
    fig.add_trace(
        go.Contour(
            z=field,
            x=x_mm,
            y=y_mm,
            contours=dict(start=0.5, end=0.5, size=1),
            line=dict(color='#f0f4ff', width=2),
            showscale=False,
        )
    )
    _apply_square_field_layout(fig, x, y, title)
    return fig


def create_heat_source_figure(Q_map, x, y, centers, sigma, title=''):
    x_mm, y_mm = _mm_coords(x, y)
    fig = go.Figure(
        data=go.Heatmap(
            z=Q_map,
            x=x_mm,
            y=y_mm,
            colorscale='Inferno',
            colorbar=dict(title='W/m³', len=0.75, thickness=14),
        )
    )
    for cx, cy in centers:
        fig.add_trace(
            go.Scatter(
                x=[cx * 1000.0],
                y=[cy * 1000.0],
                mode='markers',
                marker=dict(color='#00d4aa', size=12, symbol='x', line=dict(width=2)),
                name='Source center',
            )
        )
    _apply_square_field_layout(fig, x, y, f'{title}\nσ={sigma * 1000:.1f} mm')
    fig.update_layout(showlegend=False)
    return fig


def create_history_line_figure(time, values, title, yaxis_title, color):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=time, y=values, mode='lines', line=dict(color=color), name=title)
    )
    fig.update_layout(
        title=title,
        xaxis_title='Time [s]',
        yaxis_title=yaxis_title,
        template='plotly_dark',
        paper_bgcolor='rgba(8, 12, 24, 0.55)',
        plot_bgcolor='rgba(10, 14, 28, 0.6)',
        font=dict(color='#e8eefc'),
    )
    return fig


def build_metrics_history_dataframe(
    time,
    max_temp,
    melt_radius_mm,
    velocity_mm_s,
    flux_mw_m2,
    liquid_frac_max,
) -> pd.DataFrame:
    """Tabular metrics matching main.py console output."""
    return pd.DataFrame(
        {
            'Time [s]': np.asarray(time, dtype=float),
            'Max T [°C]': np.asarray(max_temp, dtype=float),
            'Melt rad [mm]': np.asarray(melt_radius_mm, dtype=float),
            'v_n [mm/s]': np.asarray(velocity_mm_s, dtype=float),
            'Flux [MW/m²]': np.asarray(flux_mw_m2, dtype=float),
            'Liquid frac max': np.asarray(liquid_frac_max, dtype=float),
        }
    )


def create_multiple_metrics_figure(time, max_temp, mean_radius, velocity, T_m):
    velocity_arr = np.asarray(velocity, dtype=float) * 1000.0
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=time, y=max_temp, mode='lines', name='Max temp [°C]', line=dict(color='#ff6b6b', width=2)))
    fig.add_trace(go.Scatter(x=time, y=mean_radius, mode='lines', name='Melt radius [mm]', line=dict(color='#3d8bfd', width=2)))
    fig.add_trace(go.Scatter(x=time, y=velocity_arr, mode='lines', name='Velocity [mm/s]', line=dict(color='#00d4aa', width=2)))
    fig.add_vline(x=0, line_dash='dash', line_color='rgba(200,210,240,0.4)')
    fig.update_layout(
        title=dict(text='Evolution of key metrics', font=dict(color='#f0f4ff')),
        xaxis_title='Time [s]',
        yaxis_title='Value',
        template='plotly_dark',
        paper_bgcolor='rgba(8, 12, 24, 0.55)',
        plot_bgcolor='rgba(10, 14, 28, 0.6)',
        font=dict(color='#e8eefc'),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        xaxis=dict(gridcolor='rgba(120, 150, 220, 0.2)', color='#c5d0ea'),
        yaxis=dict(gridcolor='rgba(120, 150, 220, 0.2)', color='#c5d0ea'),
    )
    return fig
