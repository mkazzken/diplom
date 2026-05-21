"""AI comparison UI for the Stefan problem sandbox."""

import io
import json
import os
from datetime import datetime

import imageio
import numpy as np
import plotly.graph_objects as go
import streamlit as st

from .experiment_loader import (
    build_time_axis,
    classify_experiment,
    find_experiment_directories,
    load_experiment
)
from .error_analysis import (
    compute_error_maps,
    compute_improvement_metrics,
    compute_temporal_metrics
)
from .metrics_dashboard import (
    create_correlation_figure,
    create_error_histogram,
    create_improvement_figure,
    create_metric_comparison_figure,
    create_scatter_comparison_figure,
    create_training_history_figure
)
from .playback_controller import render_playback_controls
from .visualization import (
    create_comparison_heatmap_figure,
    create_single_heatmap_figure,
    create_three_way_comparison_figure
)


def _squeeze_field(field):
    if field is None:
        return None
    arr = np.asarray(field)
    if arr.ndim == 4 and arr.shape[1] == 1:
        arr = arr[:, 0, ...]
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0, ...]
    return arr


def _safe_array(arr):
    if arr is None:
        return None
    return np.asarray(arr, dtype=np.float32)


def _frame_count_for(*arrays):
    lengths = [arr.shape[0] for arr in arrays if arr is not None and arr.ndim >= 3]
    return int(min(lengths)) if lengths else 0


def _build_axes(config: dict, shape: tuple):
    length = 0.05
    if config is not None:
        length = float(config.get('length', config.get('domain_size', length)))
    nx = shape[-1]
    ny = shape[-2]
    x = np.linspace(0.0, length, nx)
    y = np.linspace(0.0, length, ny)
    return x, y


def _create_png(fig):
    return fig.to_image(format='png', engine='kaleido')


def _create_gif(frames, x, y, title, duration=0.12):
    images = []
    for frame in frames:
        fig = create_single_heatmap_figure(frame, x, y, title, zmin=np.min(frame), zmax=np.max(frame), show_contour=False)
        img_bytes = fig.to_image(format='png', engine='kaleido')
        images.append(imageio.v2.imread(io.BytesIO(img_bytes)))
    buffer = io.BytesIO()
    imageio.mimsave(buffer, images, format='GIF', duration=duration)
    buffer.seek(0)
    return buffer


def _create_npz_export(gt, pure, hybrid, errors, metadata):
    buffer = io.BytesIO()
    np.savez(
        buffer,
        ground_truth=gt,
        pure_prediction=pure,
        hybrid_prediction=hybrid,
        errors=errors,
        metadata=json.dumps(metadata)
    )
    buffer.seek(0)
    return buffer


def _render_experiment_sidebar(experiments):
    st.sidebar.markdown('## AI Comparison Controls')
    if not experiments:
        st.sidebar.warning('No AI experiment directories were detected.')
        return None, None, None, 0.01, False

    exp_names = [exp.name for exp in experiments]
    selected_gt = st.sidebar.selectbox('Ground truth experiment', exp_names, index=0, key='ai_gt_experiment')

    pure_names = [exp.name for exp in experiments if classify_experiment(exp.name) == 'pure']
    hybrid_names = [exp.name for exp in experiments if classify_experiment(exp.name) == 'hybrid']

    selected_pure = st.sidebar.selectbox(
        'Pure AI experiment', pure_names if pure_names else ['(none)'], index=0, key='ai_pure_experiment'
    )
    selected_hybrid = st.sidebar.selectbox(
        'Hybrid experiment', hybrid_names if hybrid_names else ['(none)'], index=0, key='ai_hybrid_experiment'
    )
    time_per_frame = st.sidebar.number_input('Seconds per frame', min_value=0.0, value=0.02, step=0.01, key='ai_time_step')
    show_contour = st.sidebar.checkbox('Show melting front contour', value=True, key='ai_contour_toggle')
    return selected_gt, selected_pure, selected_hybrid, time_per_frame, show_contour


def _load_selected_experiments(experiments, gt_name, pure_name, hybrid_name):
    experiment_map = {exp.name: exp for exp in experiments}
    gt_data = load_experiment(experiment_map[gt_name]) if gt_name in experiment_map else None
    pure_data = load_experiment(experiment_map[pure_name]) if pure_name in experiment_map else None
    hybrid_data = load_experiment(experiment_map[hybrid_name]) if hybrid_name in experiment_map else None
    return gt_data, pure_data, hybrid_data


def render_ai_comparison_tab():
    st.header('🤖 AI Comparison Dashboard')
    st.markdown(
        'Explore ground truth simulation data, pure AI predictions, and hybrid AI predictions with synchronized playback, error analysis, ' 
        'research metrics, and experiment metadata.'
    )

    experiments = find_experiment_directories()
    selected_gt, selected_pure, selected_hybrid, time_per_frame, show_contour = _render_experiment_sidebar(experiments)
    experiment_map = {exp.name: exp for exp in experiments}

    if not experiments:
        st.warning('No experiments were found in the configured experiment directories.')
        return

    gt_data, pure_data, hybrid_data = _load_selected_experiments(experiments, selected_gt, selected_pure, selected_hybrid)
    if gt_data is None:
        st.warning('No ground truth experiment is selected or ground truth data is missing.')
        return

    gt_field = _squeeze_field(gt_data.get('ground_truth'))
    pure_field = _squeeze_field(pure_data.get('predictions')) if pure_data is not None else None
    hybrid_field = _squeeze_field(hybrid_data.get('predictions')) if hybrid_data is not None else None

    frame_count = _frame_count_for(gt_field, pure_field, hybrid_field)
    if frame_count == 0:
        st.warning('Selected experiments do not contain valid temporal arrays.')
        return

    current_frame = render_playback_controls(frame_count, time_per_frame, prefix='ai')
    current_frame = min(current_frame, frame_count - 1)

    x, y = _build_axes(gt_data.get('config'), gt_field.shape)
    time_axis = build_time_axis(gt_data.get('config'), frame_count, default_dt=time_per_frame)
    frame_time = float(time_axis[current_frame]) if len(time_axis) > current_frame else current_frame * time_per_frame

    st.markdown(f'**Frame {current_frame + 1} / {frame_count}**  \n**Time = {frame_time:.3f} s**')

    gt_slice = gt_field[current_frame] if gt_field is not None else None
    pure_slice = pure_field[current_frame] if pure_field is not None else None
    hybrid_slice = hybrid_field[current_frame] if hybrid_field is not None else None

    zmin = np.nanmin([arr.min() for arr in [gt_slice, pure_slice, hybrid_slice] if arr is not None])
    zmax = np.nanmax([arr.max() for arr in [gt_slice, pure_slice, hybrid_slice] if arr is not None])

    tab_names = [
        'GT vs Pure AI',
        'GT vs Hybrid',
        'GT vs Pure AI vs Hybrid',
        'Error Analysis',
        'Metrics Dashboard',
        'Training History',
        'Experiment Info'
    ]
    sub_tabs = st.tabs(tab_names)

    with sub_tabs[0]:
        st.subheader('Ground Truth vs Pure AI Prediction')
        if pure_slice is None:
            st.info('Select a pure AI experiment to compare against ground truth.')
        else:
            fig = create_comparison_heatmap_figure(
                [gt_slice, pure_slice],
                x,
                y,
                ['Ground truth', 'Pure AI'],
                zmin,
                zmax,
                show_contour=show_contour
            )
            st.plotly_chart(fig, use_container_width=True, key=f'gt_pure_{current_frame}')

    with sub_tabs[1]:
        st.subheader('Ground Truth vs Hybrid Prediction')
        if hybrid_slice is None:
            st.info('Select a hybrid experiment to compare against ground truth.')
        else:
            fig = create_comparison_heatmap_figure(
                [gt_slice, hybrid_slice],
                x,
                y,
                ['Ground truth', 'Hybrid'],
                zmin,
                zmax,
                show_contour=show_contour
            )
            st.plotly_chart(fig, use_container_width=True, key=f'gt_hybrid_{current_frame}')

    with sub_tabs[2]:
        st.subheader('Ground Truth, Pure AI, and Hybrid Comparison')
        if pure_slice is None or hybrid_slice is None:
            st.info('Choose both a pure AI experiment and a hybrid experiment to enable three-way comparison.')
        else:
            fig = create_three_way_comparison_figure(
                gt_slice,
                pure_slice,
                hybrid_slice,
                x,
                y,
                zmin,
                zmax,
                show_contour=show_contour
            )
            st.plotly_chart(fig, use_container_width=True, key=f'gt_pure_hybrid_{current_frame}')

    with sub_tabs[3]:
        st.subheader('Error Analysis')
        if pure_slice is None and hybrid_slice is None:
            st.info('Choose at least one AI experiment to show error maps.')
        else:
            if pure_slice is not None:
                pure_errors = compute_error_maps(gt_slice, pure_slice)
                fig = create_comparison_heatmap_figure(
                    [pure_errors['abs_error'], pure_errors['sq_error']],
                    x,
                    y,
                    ['Pure AI absolute error', 'Pure AI squared error'],
                    0.0,
                    np.nanmax([pure_errors['abs_error'].max(), pure_errors['sq_error'].max()]),
                    show_contour=False
                )
                st.plotly_chart(fig, use_container_width=True, key=f'pure_error_maps_{current_frame}')
            if hybrid_slice is not None:
                hybrid_errors = compute_error_maps(gt_slice, hybrid_slice)
                fig = create_comparison_heatmap_figure(
                    [hybrid_errors['abs_error'], hybrid_errors['sq_error']],
                    x,
                    y,
                    ['Hybrid absolute error', 'Hybrid squared error'],
                    0.0,
                    np.nanmax([hybrid_errors['abs_error'].max(), hybrid_errors['sq_error'].max()]),
                    show_contour=False
                )
                st.plotly_chart(fig, use_container_width=True, key=f'hybrid_error_maps_{current_frame}')
            if pure_slice is not None and hybrid_slice is not None:
                pure_errors = compute_error_maps(gt_slice, pure_slice)
                hybrid_errors = compute_error_maps(gt_slice, hybrid_slice)
                diff = hybrid_errors['abs_error'] - pure_errors['abs_error']
                fig = create_comparison_heatmap_figure(
                    [pure_errors['difference'], hybrid_errors['difference'], diff],
                    x,
                    y,
                    ['Pure AI difference', 'Hybrid difference', 'Hybrid - Pure abs error'],
                    np.nanmin([pure_errors['difference'].min(), hybrid_errors['difference'].min(), diff.min()]),
                    np.nanmax([pure_errors['difference'].max(), hybrid_errors['difference'].max(), diff.max()]),
                    show_contour=False
                )
                st.plotly_chart(fig, use_container_width=True, key=f'error_diff_maps_{current_frame}')

    with sub_tabs[4]:
        st.subheader('Metrics Dashboard')
        pure_metrics = compute_temporal_metrics(gt_field, pure_field) if pure_field is not None else None
        hybrid_metrics = compute_temporal_metrics(gt_field, hybrid_field) if hybrid_field is not None else None

        cols = st.columns(3)
        if pure_metrics is not None and hybrid_metrics is not None:
            avg_pure_rmse = float(np.mean(pure_metrics['rmse']))
            avg_hybrid_rmse = float(np.mean(hybrid_metrics['rmse']))
            cols[0].metric('Pure AI average RMSE', f'{avg_pure_rmse:.4f}')
            cols[1].metric('Hybrid average RMSE', f'{avg_hybrid_rmse:.4f}')
            improvement = 100.0 * max(0.0, (avg_pure_rmse - avg_hybrid_rmse) / max(avg_pure_rmse, 1e-8))
            cols[2].metric('Average improvement', f'{improvement:.2f} %')

        if pure_metrics is not None and hybrid_metrics is not None:
            st.plotly_chart(create_metric_comparison_figure(time_axis, pure_metrics['rmse'], hybrid_metrics['rmse'], 'RMSE'), use_container_width=True, key='rmse_time')
            st.plotly_chart(create_metric_comparison_figure(time_axis, pure_metrics['mae'], hybrid_metrics['mae'], 'MAE'), use_container_width=True, key='mae_time')
            st.plotly_chart(create_metric_comparison_figure(time_axis, pure_metrics['max_error'], hybrid_metrics['max_error'], 'Max error'), use_container_width=True, key='maxerr_time')
            improvement = compute_improvement_metrics(pure_metrics, hybrid_metrics)
            if 'mae' in improvement:
                st.plotly_chart(create_improvement_figure(time_axis, improvement['mae'], 'MAE'), use_container_width=True, key='improv_mae')
            st.plotly_chart(create_error_histogram(gt_slice, pure_slice if pure_slice is not None else gt_slice, hybrid_slice if hybrid_slice is not None else gt_slice), use_container_width=True, key='error_histogram')
            if pure_slice is not None:
                st.plotly_chart(create_scatter_comparison_figure(gt_slice, pure_slice, 'Pure AI'), use_container_width=True, key='scatter_pure')
            if hybrid_slice is not None:
                st.plotly_chart(create_scatter_comparison_figure(gt_slice, hybrid_slice, 'Hybrid'), use_container_width=True, key='scatter_hybrid')
            if hybrid_slice is not None:
                st.plotly_chart(create_correlation_figure(gt_slice, hybrid_slice, 'Hybrid'), use_container_width=True, key='corr_hybrid')

    with sub_tabs[5]:
        st.subheader('Training History')
        history_fig = create_training_history_figure(gt_data.get('history'))
        if history_fig.data:
            st.plotly_chart(history_fig, use_container_width=True, key='training_history')
        else:
            st.info('Training history is not available for this experiment.')

    with sub_tabs[6]:
        st.subheader('Experiment Info')
        info_cols = st.columns(2)
        info_cols[0].markdown('**Selected ground truth experiment**')
        info_cols[0].write({'name': gt_data['name'], 'path': gt_data['path'], 'type': gt_data['classification']})
        info_cols[1].markdown('**Selected pure AI experiment**')
        info_cols[1].write({'name': pure_data['name'] if pure_data is not None else None, 'type': pure_data['classification'] if pure_data is not None else None})
        st.markdown('**Configuration**')
        st.json(gt_data.get('config') or {})
        st.markdown('**Metrics file**')
        st.json(gt_data.get('metrics') or {})

    with st.expander('Export current comparison', expanded=False):
        if st.button('Download current comparison PNG'):
            fig = create_three_way_comparison_figure(
                gt_slice if gt_slice is not None else np.zeros((x.size, y.size), dtype=np.float32),
                pure_slice if pure_slice is not None else np.zeros((x.size, y.size), dtype=np.float32),
                hybrid_slice if hybrid_slice is not None else np.zeros((x.size, y.size), dtype=np.float32),
                x,
                y,
                zmin,
                zmax,
                show_contour=show_contour
            )
            png = _create_png(fig)
            st.download_button('Download PNG', data=png, file_name=f'ai_comparison_{current_frame}.png', mime='image/png')
        if st.button('Download current state npz'):
            npz = _create_npz_export(gt_slice, pure_slice, hybrid_slice, None, {'frame': current_frame})
            st.download_button('Download NPZ', data=npz, file_name=f'ai_comparison_{current_frame}.npz', mime='application/octet-stream')
        if st.button('Download GIF of current experiment'):
            frames = []
            if gt_field is not None:
                frames = [gt_field[i] for i in range(min(30, frame_count))]
            if frames:
                gif = _create_gif(frames, x, y, 'Ground truth sequence')
                st.download_button('Download GIF', data=gif, file_name='ai_comparison.gif', mime='image/gif')
            else:
                st.warning('No frames available to export.')
