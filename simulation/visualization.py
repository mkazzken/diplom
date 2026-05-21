"""Visualization utilities for the Stefan problem sandbox."""

import numpy as np
import plotly.graph_objects as go


def create_temperature_figure(field, x, y, title='', zmin=None, zmax=None):
    if zmin is None:
        zmin = np.min(field)
    if zmax is None:
        zmax = np.max(field)

    fig = go.Figure(
        data=go.Heatmap(
            z=field,
            x=x * 1000.0,
            y=y * 1000.0,
            colorscale='hot',
            zmin=zmin,
            zmax=zmax,
            colorbar=dict(title='°C')
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title='x [mm]',
        yaxis_title='y [mm]',
        yaxis_autorange='reversed',
        template='plotly_white'
    )
    return fig


def create_liquid_fraction_figure(field, x, y, title=''):
    fig = go.Figure()
    fig.add_trace(
        go.Heatmap(
            z=field,
            x=x * 1000.0,
            y=y * 1000.0,
            colorscale='RdBu',
            zmin=0.0,
            zmax=1.0,
            colorbar=dict(title='Liquid fraction')
        )
    )
    fig.add_trace(
        go.Contour(
            z=field,
            x=x * 1000.0,
            y=y * 1000.0,
            contours=dict(start=0.5, end=0.5, size=1),
            line=dict(color='black', width=2),
            showscale=False
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title='x [mm]',
        yaxis_title='y [mm]',
        yaxis_autorange='reversed',
        template='plotly_white'
    )
    return fig


def create_heat_source_figure(Q_map, x, y, centers, sigma, title=''):
    fig = go.Figure(
        data=go.Heatmap(
            z=Q_map,
            x=x * 1000.0,
            y=y * 1000.0,
            colorscale='Inferno',
            colorbar=dict(title='W/m³')
        )
    )
    for cx, cy in centers:
        fig.add_trace(
            go.Scatter(
                x=[cx * 1000.0],
                y=[cy * 1000.0],
                mode='markers',
                marker=dict(color='cyan', size=10, symbol='x'),
                name='Source center'
            )
        )
    fig.update_layout(
        title=f'{title}\nσ={sigma*1000:.1f} mm',
        xaxis_title='x [mm]',
        yaxis_title='y [mm]',
        yaxis_autorange='reversed',
        template='plotly_white',
        showlegend=False
    )
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
        template='plotly_white'
    )
    return fig


def create_multiple_metrics_figure(time, max_temp, mean_radius, velocity, T_m):
    velocity_arr = np.asarray(velocity, dtype=float) * 1000.0
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=time, y=max_temp, mode='lines', name='Max temp [°C]', line=dict(color='firebrick')))
    fig.add_trace(go.Scatter(x=time, y=mean_radius, mode='lines', name='Melt radius [mm]', line=dict(color='navy')))
    fig.add_trace(go.Scatter(x=time, y=velocity_arr, mode='lines', name='Velocity [mm/s]', line=dict(color='green')))
    fig.add_vline(x=0, line_dash='dash', line_color='gray')
    fig.update_layout(
        title='Evolution of key metrics',
        xaxis_title='Time [s]',
        yaxis_title='Value',
        template='plotly_white',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )
    return fig
