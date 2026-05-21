"""Metrics dashboard and research-grade visualizations for AI comparison."""

import numpy as np
import plotly.graph_objects as go

DARK_THEME = dict(
    template='plotly_dark',
    paper_bgcolor='rgba(8, 12, 24, 0.55)',
    plot_bgcolor='rgba(16, 22, 42, 0.9)',
    font=dict(color='#e8eefc', family='Inter, Segoe UI, sans-serif', size=12),
)

AXIS_STYLE = dict(
    gridcolor='rgba(120, 150, 220, 0.28)',
    zerolinecolor='rgba(120, 150, 220, 0.45)',
    linecolor='rgba(120, 150, 220, 0.5)',
    tickcolor='#aab4d4',
    color='#c5d0ea',
)


def _apply_dark_axes(fig: go.Figure) -> None:
    fig.update_xaxes(**AXIS_STYLE)
    fig.update_yaxes(**AXIS_STYLE)


def create_metric_cards(pure_metrics: dict, hybrid_metrics: dict) -> list:
    cards = []
    if pure_metrics is None or hybrid_metrics is None:
        return cards
    for label in ['MSE', 'RMSE', 'MAE', 'Max error', 'Relative error']:
        pure_val = pure_metrics.get(label.lower().replace(' ', '_'), None)
        hybrid_val = hybrid_metrics.get(label.lower().replace(' ', '_'), None)
        if pure_val is None or hybrid_val is None:
            continue
        cards.append({
            'label': label,
            'pure': float(pure_val),
            'hybrid': float(hybrid_val),
            'improvement': float((pure_val - hybrid_val) / np.maximum(abs(pure_val), 1e-8) * 100.0),
        })
    return cards


def create_metric_comparison_figure(time: np.ndarray, pure: np.ndarray, hybrid: np.ndarray, name: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=time, y=pure, mode='lines', name=f'Pure AI {name}', line=dict(color='#3d8bfd', width=2)))
    fig.add_trace(go.Scatter(x=time, y=hybrid, mode='lines', name=f'Hybrid {name}', line=dict(color='#00d4aa', width=2)))
    fig.update_layout(
        title=f'{name} vs Time',
        xaxis_title='Time [s]',
        yaxis_title=name,
        height=360,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        **DARK_THEME,
    )
    _apply_dark_axes(fig)
    return fig


def create_error_histogram(gt: np.ndarray, pure: np.ndarray, hybrid: np.ndarray) -> go.Figure:
    abs_pure = np.abs(pure - gt).ravel()
    abs_hybrid = np.abs(hybrid - gt).ravel()
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=abs_pure, name='Pure AI abs error', opacity=0.75, nbinsx=60, marker_color='#3d8bfd'))
    fig.add_trace(go.Histogram(x=abs_hybrid, name='Hybrid abs error', opacity=0.75, nbinsx=60, marker_color='#00d4aa'))
    fig.update_layout(
        title='Error distribution histogram',
        xaxis_title='Absolute error',
        yaxis_title='Count',
        barmode='overlay',
        height=360,
        **DARK_THEME,
    )
    _apply_dark_axes(fig)
    return fig


def create_scatter_comparison_figure(gt: np.ndarray, pred: np.ndarray, label: str) -> go.Figure:
    limit = float(max(gt.max(), pred.max(), 1e-6))
    fig = go.Figure()
    fig.add_trace(
        go.Scattergl(
            x=gt.ravel(),
            y=pred.ravel(),
            mode='markers',
            marker=dict(size=4, opacity=0.45, color='#5eb3ff'),
            name=label,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[0, limit],
            y=[0, limit],
            mode='lines',
            line=dict(color='#f0f4ff', dash='dash', width=2),
            name='y = x',
        )
    )
    fig.update_layout(
        title=f'GT vs {label}',
        xaxis_title='Ground truth',
        yaxis_title=label,
        width=480,
        height=480,
        autosize=False,
        showlegend=True,
        legend=dict(bgcolor='rgba(8,12,24,0.8)', font=dict(color='#e8eefc')),
        **DARK_THEME,
    )
    fig.update_xaxes(range=[0, limit], constrain='domain', **AXIS_STYLE)
    fig.update_yaxes(range=[0, limit], scaleanchor='x', scaleratio=1, constrain='domain', **AXIS_STYLE)
    return fig


def create_correlation_figure(gt: np.ndarray, pred: np.ndarray, label: str) -> go.Figure:
    gt_flat = gt.ravel()
    pred_flat = pred.ravel()
    covariance = np.cov(gt_flat, pred_flat)
    correlation = covariance[0, 1] / np.maximum(np.sqrt(covariance[0, 0] * covariance[1, 1]), 1e-8)
    limit = float(max(gt_flat.max(), pred_flat.max(), 1e-6))

    fig = go.Figure()
    fig.add_trace(
        go.Scattergl(
            x=gt_flat,
            y=pred_flat,
            mode='markers',
            marker=dict(size=4, opacity=0.45, color='#00d4aa'),
            name=label,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[0, limit],
            y=[0, limit],
            mode='lines',
            line=dict(color='#f0f4ff', dash='dash', width=2),
            name='y = x',
        )
    )
    fig.update_layout(
        title=f'Pixel-wise correlation: {label} (ρ = {correlation:.3f})',
        xaxis_title='Ground truth',
        yaxis_title=label,
        width=480,
        height=480,
        autosize=False,
        showlegend=True,
        legend=dict(bgcolor='rgba(8,12,24,0.8)', font=dict(color='#e8eefc')),
        **DARK_THEME,
    )
    fig.update_xaxes(range=[0, limit], constrain='domain', **AXIS_STYLE)
    fig.update_yaxes(range=[0, limit], scaleanchor='x', scaleratio=1, constrain='domain', **AXIS_STYLE)
    return fig


def create_improvement_figure(time: np.ndarray, improvement: np.ndarray, name: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=time, y=improvement, mode='lines', name=f'{name} improvement', line=dict(color='#00d4aa', width=2)))
    fig.update_layout(
        title=f'{name} improvement over time',
        xaxis_title='Time [s]',
        yaxis_title='Improvement [%]',
        height=360,
        **DARK_THEME,
    )
    _apply_dark_axes(fig)
    return fig


def create_training_history_figure(history: dict) -> go.Figure:
    fig = go.Figure()
    if history is None:
        return fig
    if 'train_loss' in history:
        fig.add_trace(go.Scatter(x=list(range(len(history['train_loss']))), y=history['train_loss'], mode='lines', name='Train loss', line=dict(color='#00d4aa', width=2)))
    if 'val_loss' in history:
        fig.add_trace(go.Scatter(x=list(range(len(history['val_loss']))), y=history['val_loss'], mode='lines', name='Validation loss', line=dict(color='#ff6b9d', width=2)))
    if 'train_mae' in history:
        fig.add_trace(go.Scatter(x=list(range(len(history['train_mae']))), y=history['train_mae'], mode='lines', name='Train MAE', line=dict(color='#3d8bfd', width=2)))
    if 'val_mae' in history:
        fig.add_trace(go.Scatter(x=list(range(len(history['val_mae']))), y=history['val_mae'], mode='lines', name='Validation MAE', line=dict(color='#ff9f43', width=2)))
    fig.update_layout(
        title='Training history',
        xaxis_title='Epoch',
        yaxis_title='Value',
        height=360,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        **DARK_THEME,
    )
    _apply_dark_axes(fig)
    return fig
