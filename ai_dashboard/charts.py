"""Plotly chart builders for the AI dashboard."""
from __future__ import annotations

import plotly.graph_objects as go

THEME = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(10,14,28,0.6)',
    font=dict(color='#e8eefc', family='Inter, Segoe UI, sans-serif', size=12),
    margin=dict(l=40, r=20, t=50, b=40),
)


def loss_curves(train: list[float], val: list[float], title: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=train, mode='lines', name='Train', line=dict(color='#00d4aa', width=2)))
    fig.add_trace(go.Scatter(y=val, mode='lines', name='Validation', line=dict(color='#ff6b9d', width=2)))
    fig.update_layout(title=title, xaxis_title='Epoch', yaxis_title='Loss', **THEME, height=360)
    return fig
