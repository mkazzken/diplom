"""Metrics dashboard and research-grade visualizations for AI comparison."""

import numpy as np
import plotly.graph_objects as go


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
            'improvement': float((pure_val - hybrid_val) / np.maximum(abs(pure_val), 1e-8) * 100.0)
        })
    return cards


def create_metric_comparison_figure(time: np.ndarray, pure: np.ndarray, hybrid: np.ndarray, name: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=time, y=pure, mode='lines', name=f'Pure AI {name}', line=dict(color='#1f77b4')))
    fig.add_trace(go.Scatter(x=time, y=hybrid, mode='lines', name=f'Hybrid {name}', line=dict(color='#ff7f0e')))
    fig.update_layout(
        title=f'{name} vs Time',
        xaxis_title='Time [s]',
        yaxis_title=name,
        template='plotly_white',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )
    return fig


def create_error_histogram(gt: np.ndarray, pure: np.ndarray, hybrid: np.ndarray) -> go.Figure:
    abs_pure = np.abs(pure - gt).ravel()
    abs_hybrid = np.abs(hybrid - gt).ravel()
    bins = np.linspace(0.0, max(abs_pure.max(), abs_hybrid.max(), 1e-3), 60)
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=abs_pure, name='Pure AI abs error', opacity=0.7, nbinsx=60))
    fig.add_trace(go.Histogram(x=abs_hybrid, name='Hybrid abs error', opacity=0.7, nbinsx=60))
    fig.update_layout(
        title='Error distribution histogram',
        xaxis_title='Absolute error',
        yaxis_title='Count',
        barmode='overlay',
        template='plotly_white'
    )
    return fig


def create_scatter_comparison_figure(gt: np.ndarray, pred: np.ndarray, label: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scattergl(
        x=gt.ravel(),
        y=pred.ravel(),
        mode='markers',
        marker=dict(size=2, opacity=0.4),
        name=label
    ))
    limit = max(gt.max(), pred.max())
    fig.add_trace(go.Scatter(
        x=[0, limit],
        y=[0, limit],
        mode='lines',
        line=dict(color='black', dash='dash'),
        name='y = x'
    ))
    fig.update_layout(
        title=f'GT vs {label}',
        xaxis_title='Ground truth',
        yaxis_title=label,
        template='plotly_white'
    )
    return fig


def create_correlation_figure(gt: np.ndarray, pred: np.ndarray, label: str) -> go.Figure:
    gt_flat = gt.ravel()
    pred_flat = pred.ravel()
    covariance = np.cov(gt_flat, pred_flat)
    correlation = covariance[0, 1] / np.maximum(np.sqrt(covariance[0, 0] * covariance[1, 1]), 1e-8)
    fig = go.Figure()
    fig.add_trace(go.Scattergl(x=gt_flat, y=pred_flat, mode='markers', marker=dict(size=2, opacity=0.4)))
    fig.update_layout(
        title=f'Pixel-wise correlation: {label} (ρ = {correlation:.3f})',
        xaxis_title='Ground truth',
        yaxis_title=label,
        template='plotly_white'
    )
    return fig


def create_improvement_figure(time: np.ndarray, improvement: np.ndarray, name: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=time, y=improvement, mode='lines', name=f'{name} improvement', line=dict(color='green')))
    fig.update_layout(
        title=f'{name} improvement over time',
        xaxis_title='Time [s]',
        yaxis_title='Improvement [%]',
        template='plotly_white'
    )
    return fig


def create_training_history_figure(history: dict) -> go.Figure:
    fig = go.Figure()
    if history is None:
        return fig
    if 'train_loss' in history:
        fig.add_trace(go.Scatter(x=list(range(len(history['train_loss']))), y=history['train_loss'], mode='lines', name='Train loss'))
    if 'val_loss' in history:
        fig.add_trace(go.Scatter(x=list(range(len(history['val_loss']))), y=history['val_loss'], mode='lines', name='Validation loss'))
    if 'train_mae' in history:
        fig.add_trace(go.Scatter(x=list(range(len(history['train_mae']))), y=history['train_mae'], mode='lines', name='Train MAE'))
    if 'val_mae' in history:
        fig.add_trace(go.Scatter(x=list(range(len(history['val_mae']))), y=history['val_mae'], mode='lines', name='Validation MAE'))
    fig.update_layout(
        title='Training history',
        xaxis_title='Epoch',
        yaxis_title='Value',
        template='plotly_white',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )
    return fig
