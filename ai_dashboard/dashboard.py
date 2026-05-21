"""AI Inference Dashboard — same pipeline as visualization_app (discovery → live inference → sync playback)."""
from __future__ import annotations

import io

import numpy as np
import pandas as pd
import streamlit as st

from ._torch_bootstrap import TORCH_ERROR, ensure_torch
from .components.comparison_panel import (
    render_error_panel,
    render_gt_hybrid_panel,
    render_gt_pure_panel,
    render_three_way_panel,
)
from .components.layout import (
    apply_theme,
    init_runs,
    render_hero,
    sidebar_dataset_selector,
    sidebar_pair_selector,
)
from .data_loader import load_npz_sample, summarize_sample
from .discovery import format_timestamp
from .error_analysis import compute_improvement_metrics, compute_temporal_metrics
from .inference_engine import predict_from_sample
from .metrics_dashboard import (
    create_correlation_figure,
    create_error_histogram,
    create_improvement_figure,
    create_metric_comparison_figure,
    create_scatter_comparison_figure,
)
from .metrics_parser import flat_metrics, training_losses
from .charts import loss_curves


def render_ai_inference_dashboard() -> None:
    apply_theme()
    render_hero(
        'AI Inference Dashboard',
        'Live inference from model_best.pt · synchronized Plotly playback · GT vs Pure vs Hybrid',
    )

    if not ensure_torch():
        st.error(
            'PyTorch is required for live inference. Install dependencies and restart.\n'
            f'Details: {TORCH_ERROR}'
        )
        return

    runs = init_runs()
    pure_run, hybrid_run = sidebar_pair_selector()
    _, sample_path = sidebar_dataset_selector()

    if sample_path is None:
        st.warning('Add simulation samples under `dataset/dirichlet` or `dataset/neumann`.')
        return

    if pure_run is None and hybrid_run is None:
        st.warning('No trained experiments with `model_best.pt` found under `experiments/`.')
        return

    run_button = st.sidebar.button('▶ Run live inference', type='primary', key='ai_run_live')

    cache_key = (
        sample_path,
        pure_run.name if pure_run else None,
        hybrid_run.name if hybrid_run else None,
    )
    if run_button or st.session_state.get('ai_pred_cache_key') != cache_key:
        with st.spinner('Running live inference from model_best.pt…'):
            preds = predict_from_sample(
                sample_path,
                pure_run.path if pure_run else None,
                hybrid_run.path if hybrid_run else None,
            )
        st.session_state.ai_predictions = preds
        st.session_state.ai_pred_cache_key = cache_key
    elif 'ai_predictions' not in st.session_state:
        st.info('Click **Run live inference** in the sidebar to load models and predict.')
        return

    preds = st.session_state.ai_predictions
    gt = preds['ground_truth']
    pure = preds.get('pure')
    hybrid = preds.get('hybrid')
    times = preds['times']
    n_frames = gt.shape[0]

    summary = summarize_sample(load_npz_sample(sample_path))
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric('Frames', summary['frames'])
    c2.metric('Grid', summary['grid'])
    c3.metric('Aligned frames', n_frames)
    if preds.get('pure_ms'):
        c4.metric('Pure AI', f"{preds['pure_ms']:.2f} ms/frame")
    if preds.get('hybrid_ms'):
        c5.metric('Hybrid AI', f"{preds['hybrid_ms']:.2f} ms/frame")
    st.markdown('</div>', unsafe_allow_html=True)

    if pure_run:
        metrics = flat_metrics(pure_run.metrics)
        st.caption(
            f"Pure run `{pure_run.name}` ({format_timestamp(pure_run.timestamp)}) · "
            f"MAE={metrics.get('mae', metrics.get('MAE', 0)):.4f} · "
            f"RMSE={metrics.get('rmse', metrics.get('RMSE', 0)):.4f}"
        )
    if hybrid_run:
        metrics = flat_metrics(hybrid_run.metrics)
        st.caption(
            f"Hybrid run `{hybrid_run.name}` ({format_timestamp(hybrid_run.timestamp)}) · "
            f"MAE={metrics.get('mae', metrics.get('MAE', 0)):.4f} · "
            f"RMSE={metrics.get('rmse', metrics.get('RMSE', 0)):.4f}"
        )

    tabs = st.tabs([
        'GT vs Pure AI',
        'GT vs Hybrid',
        'GT vs Pure AI vs Hybrid',
        'Error Analysis',
        'Metrics Dashboard',
    ])

    with tabs[0]:
        if pure is None:
            st.info('Select a Pure AI experiment (`run_*`) in the sidebar.')
        else:
            render_gt_pure_panel(gt, pure, key_prefix='tab_pure')

    with tabs[1]:
        if hybrid is None:
            st.info('Select a Hybrid experiment (`inspired_surrogate_*`) in the sidebar.')
        else:
            render_gt_hybrid_panel(gt, hybrid, key_prefix='tab_hybrid')

    with tabs[2]:
        if pure is None or hybrid is None:
            st.info('Select both Pure AI and Hybrid runs for three-way comparison.')
        else:
            render_three_way_panel(gt, pure, hybrid, key_prefix='tab_three')

    with tabs[3]:
        if pure is None and hybrid is None:
            st.info('Run inference with at least one model to view error maps.')
        else:
            render_error_panel(gt, pure, hybrid, key_prefix='tab_errors')
            current = n_frames // 2
            gt_slice = gt[current]
            pure_slice = pure[current] if pure is not None else None
            hybrid_slice = hybrid[current] if hybrid is not None else None
            cols = st.columns(4)
            if pure_slice is not None:
                err = np.abs(gt_slice - pure_slice)
                cols[0].metric('Pure RMSE (frame)', f"{float(np.sqrt(np.mean(err ** 2))):.4f}")
                cols[1].metric('Pure MAE (frame)', f"{float(np.mean(err)):.4f}")
                cols[2].metric('Pure max error', f"{float(np.max(err)):.4f}")
            if hybrid_slice is not None:
                err = np.abs(gt_slice - hybrid_slice)
                cols[3].metric('Hybrid RMSE (frame)', f"{float(np.sqrt(np.mean(err ** 2))):.4f}")

    with tabs[4]:
        _render_metrics_tab(gt, pure, hybrid, times, pure_run, hybrid_run)

    with st.expander('Export aligned predictions'):
        buffer = io.BytesIO()
        np.savez_compressed(
            buffer,
            ground_truth=gt,
            pure_prediction=pure if pure is not None else np.array([]),
            hybrid_prediction=hybrid if hybrid is not None else np.array([]),
            times=times,
            sample_path=sample_path,
        )
        buffer.seek(0)
        st.download_button('Download NPZ', data=buffer, file_name='live_inference.npz', mime='application/octet-stream')


def _render_metrics_tab(gt, pure, hybrid, times, pure_run, hybrid_run) -> None:
    pure_metrics = compute_temporal_metrics(gt, pure) if pure is not None else None
    hybrid_metrics = compute_temporal_metrics(gt, hybrid) if hybrid is not None else None

    rows = []
    for run, label in ((pure_run, 'pure_ai'), (hybrid_run, 'hybrid')):
        if run is None:
            continue
        metrics = flat_metrics(run.metrics)
        rows.append({
            'Run': run.name,
            'Type': label,
            'MAE': metrics.get('mae', metrics.get('MAE')),
            'RMSE': metrics.get('rmse', metrics.get('RMSE')),
            'R²': metrics.get('r2', metrics.get('R2')),
            'Dice': metrics.get('dice', metrics.get('Dice')),
        })
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    if pure_metrics is not None and hybrid_metrics is not None:
        c1, c2, c3 = st.columns(3)
        c1.metric('Pure avg RMSE', f"{float(np.mean(pure_metrics['rmse'])):.4f}")
        c2.metric('Hybrid avg RMSE', f"{float(np.mean(hybrid_metrics['rmse'])):.4f}")
        improvement = compute_improvement_metrics(pure_metrics, hybrid_metrics)
        c3.metric('RMSE improvement %', f"{float(np.mean(improvement.get('rmse', [0]))):.2f}")

        st.plotly_chart(
            create_metric_comparison_figure(times, pure_metrics['rmse'], hybrid_metrics['rmse'], 'RMSE'),
            use_container_width=True,
        )
        st.plotly_chart(
            create_metric_comparison_figure(times, pure_metrics['mae'], hybrid_metrics['mae'], 'MAE'),
            use_container_width=True,
        )
        st.plotly_chart(
            create_metric_comparison_figure(times, pure_metrics['max_error'], hybrid_metrics['max_error'], 'Max error'),
            use_container_width=True,
        )
        improvement = compute_improvement_metrics(pure_metrics, hybrid_metrics)
        if 'mae' in improvement:
            st.plotly_chart(create_improvement_figure(times, improvement['mae'], 'MAE'), use_container_width=True)

        mid = len(gt) // 2
        st.plotly_chart(
            create_error_histogram(gt[mid], pure[mid], hybrid[mid]),
            use_container_width=True,
        )
        st.plotly_chart(create_scatter_comparison_figure(gt[mid], pure[mid], 'Pure AI'), use_container_width=True)
        st.plotly_chart(create_scatter_comparison_figure(gt[mid], hybrid[mid], 'Hybrid'), use_container_width=True)
        st.plotly_chart(create_correlation_figure(gt[mid], hybrid[mid], 'Hybrid'), use_container_width=True)

    col1, col2 = st.columns(2)
    if pure_run and pure_run.history:
        train, val = training_losses(pure_run.history)
        if train:
            col1.plotly_chart(loss_curves(train, val, 'Pure AI training'), use_container_width=True)
    if hybrid_run and hybrid_run.history:
        train, val = training_losses(hybrid_run.history)
        if train:
            col2.plotly_chart(loss_curves(train, val, 'Hybrid training'), use_container_width=True)
