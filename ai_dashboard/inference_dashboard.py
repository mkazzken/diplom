"""AI Inference Dashboard with live model inference on dataset samples."""

from pathlib import Path

import numpy as np
import streamlit as st

from .dataset_loader import build_spatial_axes, list_dataset_samples, load_dataset_sample
from .error_analysis import compute_error_maps, compute_improvement_metrics, compute_temporal_metrics
from .experiment_loader import classify_experiment, find_trained_experiments
from .inference_engine import run_live_inference
from .metrics_dashboard import (
    create_correlation_figure,
    create_error_histogram,
    create_improvement_figure,
    create_metric_comparison_figure,
    create_scatter_comparison_figure,
    create_training_history_figure,
)
from .playback_controller import render_playback_controls
from .visualization import create_comparison_heatmap_figure, create_three_way_comparison_figure


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _render_controls(root: Path):
    with st.expander('Model & dataset selection', expanded=True):
        col1, col2, col3, col4 = st.columns([2, 2, 2, 1])

        experiments = find_trained_experiments(root)
        if not experiments:
            st.warning('No trained experiments with model_best.pt were found.')
            return None

        pure_exps = [exp for exp in experiments if classify_experiment(exp.name) == 'pure']
        hybrid_exps = [exp for exp in experiments if classify_experiment(exp.name) == 'hybrid']

        pure_names = [exp.name for exp in pure_exps]
        hybrid_names = [exp.name for exp in hybrid_exps]

        selected_pure = col1.selectbox(
            'Pure AI model',
            pure_names if pure_names else ['(none)'],
            index=0,
            key='ai_pure_model',
        )
        selected_hybrid = col2.selectbox(
            'Hybrid model',
            hybrid_names if hybrid_names else ['(none)'],
            index=0,
            key='ai_hybrid_model',
        )

        samples = list_dataset_samples(root / 'dataset')
        if not samples:
            st.warning(f'No dataset samples found in {root / "dataset"}')
            return None

        sample_labels = [sample['display_name'] for sample in samples]
        selected_sample_label = col3.selectbox(
            'Dataset sample',
            sample_labels,
            index=0,
            key='ai_dataset_sample',
        )

        show_contour = col4.checkbox('Contour', value=True, key='ai_contour_toggle')
        run_button = col4.button('Run live inference', type='primary', key='ai_run_inference')

        sample_map = {sample['display_name']: sample for sample in samples}
        exp_map = {exp.name: exp for exp in experiments}

        pure_dir = exp_map.get(selected_pure) if selected_pure in exp_map else None
        hybrid_dir = exp_map.get(selected_hybrid) if selected_hybrid in exp_map else None
        sample_entry = sample_map.get(selected_sample_label)

        return pure_dir, hybrid_dir, sample_entry, run_button, show_contour


def _cache_key(pure_dir, hybrid_dir, sample_entry) -> str:
    pure_name = pure_dir.name if pure_dir is not None else 'none'
    hybrid_name = hybrid_dir.name if hybrid_dir is not None else 'none'
    sample_name = sample_entry['id'] if sample_entry is not None else 'none'
    return f'{pure_name}|{hybrid_name}|{sample_name}'


def _ensure_inference(pure_dir, hybrid_dir, sample_entry, run_button: bool):
    if pure_dir is None and hybrid_dir is None:
        return None
    if sample_entry is None:
        return None

    cache_key = _cache_key(pure_dir, hybrid_dir, sample_entry)
    cached = st.session_state.get('ai_inference_result')
    cached_key = st.session_state.get('ai_inference_cache_key')

    if cached_key != cache_key and not run_button:
        st.info('Selection changed. Click **Run live inference** to update predictions.')
        return None

    if cached is None and not run_button:
        return None

    if run_button or cached_key != cache_key:
        with st.spinner('Loading models and running live inference...'):
            sample = load_dataset_sample(sample_entry['path'])
            result = run_live_inference(
                sample=sample,
                pure_exp_dir=pure_dir,
                hybrid_exp_dir=hybrid_dir,
            )
            result['sample'] = sample
            result['cache_key'] = cache_key
            st.session_state['ai_inference_result'] = result
            st.session_state['ai_inference_cache_key'] = cache_key
            st.session_state['ai_frame_index'] = 0
            st.session_state['ai_frame_slider'] = 0
            st.session_state['ai_is_playing'] = False
            st.session_state['ai_playback_advance'] = False
            return result

    return cached


def render_ai_inference_dashboard():
    st.header('🤖 AI Inference Dashboard')
    st.markdown(
        'Select trained models and a dataset sample, then run **live inference** from `model_best.pt`. '
        'Ground truth comes from the dataset; predictions are computed on the fly and visualized with '
        'synchronized playback across all comparison tabs.'
    )

    root = _project_root()
    controls = _render_controls(root)
    if controls is None:
        st.warning('Configure experiments and dataset paths to enable live inference.')
        return

    pure_dir, hybrid_dir, sample_entry, run_button, show_contour = controls
    if pure_dir is None and hybrid_dir is None:
        st.warning('Select at least one trained model (pure AI or hybrid).')
        return

    inference = _ensure_inference(pure_dir, hybrid_dir, sample_entry, run_button)
    if inference is None:
        st.info('Click **Run live inference** in the sidebar to load models and generate predictions.')
        return

    gt_field = inference['ground_truth']
    pure_field = inference.get('pure_prediction')
    hybrid_field = inference.get('hybrid_prediction')
    times = inference.get('times')

    frame_count = gt_field.shape[0] if gt_field is not None else 0
    if frame_count == 0:
        st.error('Inference produced no frames. Check the selected sample and model configuration.')
        return

    default_dt = 0.01
    if times is not None and len(times) > 1:
        default_dt = float(np.median(np.diff(times)))

    current_frame = render_playback_controls(frame_count, default_dt, prefix='ai')
    current_frame = min(current_frame, frame_count - 1)

    config = inference.get('pure_config') or inference.get('hybrid_config') or {}
    x, y = build_spatial_axes(gt_field.shape, config)
    frame_time = float(times[current_frame]) if times is not None and len(times) > current_frame else current_frame * default_dt

    st.markdown(f'**Frame {current_frame + 1} / {frame_count}** — **Time = {frame_time:.3f} s**')
    st.caption(
        f'Device: {inference.get("device", "cpu")} · '
        f'Sample: {sample_entry["display_name"]} · '
        f'Pure: {pure_dir.name if pure_dir else "—"} · '
        f'Hybrid: {hybrid_dir.name if hybrid_dir else "—"}'
    )

    gt_slice = gt_field[current_frame]
    pure_slice = pure_field[current_frame] if pure_field is not None else None
    hybrid_slice = hybrid_field[current_frame] if hybrid_field is not None else None

    zmin = float(np.nanmin([arr.min() for arr in [gt_slice, pure_slice, hybrid_slice] if arr is not None]))
    zmax = float(np.nanmax([arr.max() for arr in [gt_slice, pure_slice, hybrid_slice] if arr is not None]))

    tab_names = [
        'GT vs Pure AI',
        'GT vs Hybrid',
        'GT vs Pure AI vs Hybrid',
        'Error Analysis',
        'Metrics Dashboard',
    ]
    sub_tabs = st.tabs(tab_names)

    with sub_tabs[0]:
        st.subheader('Ground Truth vs Pure AI (live)')
        if pure_slice is None:
            st.info('Select a pure AI model and run inference.')
        else:
            fig = create_comparison_heatmap_figure(
                [gt_slice, pure_slice],
                x, y, ['Ground truth', 'Pure AI'],
                zmin, zmax, show_contour=show_contour,
            )
            st.plotly_chart(fig, use_container_width=True, key=f'live_gt_pure_{current_frame}')

    with sub_tabs[1]:
        st.subheader('Ground Truth vs Hybrid (live)')
        if hybrid_slice is None:
            st.info('Select a hybrid model and run inference.')
        else:
            fig = create_comparison_heatmap_figure(
                [gt_slice, hybrid_slice],
                x, y, ['Ground truth', 'Hybrid'],
                zmin, zmax, show_contour=show_contour,
            )
            st.plotly_chart(fig, use_container_width=True, key=f'live_gt_hybrid_{current_frame}')

    with sub_tabs[2]:
        st.subheader('Three-way comparison (live)')
        if pure_slice is None or hybrid_slice is None:
            st.info('Select both pure AI and hybrid models to enable three-way comparison.')
        else:
            fig = create_three_way_comparison_figure(
                gt_slice, pure_slice, hybrid_slice,
                x, y, zmin, zmax, show_contour=show_contour,
            )
            st.plotly_chart(fig, use_container_width=True, key=f'live_gt_pure_hybrid_{current_frame}')

    with sub_tabs[3]:
        st.subheader('Error Analysis')
        metric_cols = st.columns(4)

        if pure_slice is not None:
            pure_errors = compute_error_maps(gt_slice, pure_slice)
            pure_rmse = float(np.sqrt(np.mean(pure_errors['sq_error'])))
            pure_mae = float(np.mean(pure_errors['abs_error']))
            pure_max = float(np.max(pure_errors['abs_error']))
            metric_cols[0].metric('Pure AI RMSE', f'{pure_rmse:.4f}')
            metric_cols[1].metric('Pure AI MAE', f'{pure_mae:.4f}')
            metric_cols[2].metric('Pure AI max error', f'{pure_max:.4f}')
            err_max = max(pure_errors['abs_error'].max(), 1e-6)
            fig = create_comparison_heatmap_figure(
                [pure_errors['abs_error'], pure_errors['sq_error']],
                x, y,
                ['Pure AI |error|', 'Pure AI squared error'],
                0.0, err_max, show_contour=False,
            )
            st.plotly_chart(fig, use_container_width=True, key=f'live_pure_err_{current_frame}')

        if hybrid_slice is not None:
            hybrid_errors = compute_error_maps(gt_slice, hybrid_slice)
            hybrid_rmse = float(np.sqrt(np.mean(hybrid_errors['sq_error'])))
            hybrid_mae = float(np.mean(hybrid_errors['abs_error']))
            hybrid_max = float(np.max(hybrid_errors['abs_error']))
            hcols = st.columns(3)
            hcols[0].metric('Hybrid RMSE', f'{hybrid_rmse:.4f}')
            hcols[1].metric('Hybrid MAE', f'{hybrid_mae:.4f}')
            hcols[2].metric('Hybrid max error', f'{hybrid_max:.4f}')
            err_max = max(hybrid_errors['abs_error'].max(), 1e-6)
            fig = create_comparison_heatmap_figure(
                [hybrid_errors['abs_error'], hybrid_errors['sq_error']],
                x, y,
                ['Hybrid |error|', 'Hybrid squared error'],
                0.0, err_max, show_contour=False,
            )
            st.plotly_chart(fig, use_container_width=True, key=f'live_hybrid_err_{current_frame}')

        if pure_slice is not None and hybrid_slice is not None:
            pure_errors = compute_error_maps(gt_slice, pure_slice)
            hybrid_errors = compute_error_maps(gt_slice, hybrid_slice)
            diff = hybrid_errors['abs_error'] - pure_errors['abs_error']
            dmin = float(np.nanmin(diff))
            dmax = float(np.nanmax(diff))
            fig = create_comparison_heatmap_figure(
                [diff],
                x, y,
                ['Hybrid − Pure |error|'],
                dmin, dmax, show_contour=False,
            )
            st.plotly_chart(fig, use_container_width=True, key=f'live_err_diff_{current_frame}')

    with sub_tabs[4]:
        st.subheader('Metrics Dashboard')
        time_axis = times if times is not None else np.arange(frame_count, dtype=np.float32) * default_dt
        pure_metrics = compute_temporal_metrics(gt_field, pure_field) if pure_field is not None else None
        hybrid_metrics = compute_temporal_metrics(gt_field, hybrid_field) if hybrid_field is not None else None

        cols = st.columns(3)
        if pure_metrics is not None:
            cols[0].metric('Pure AI avg RMSE', f'{float(np.mean(pure_metrics["rmse"])):.4f}')
            cols[1].metric('Pure AI avg MAE', f'{float(np.mean(pure_metrics["mae"])):.4f}')
        if hybrid_metrics is not None:
            cols[2].metric('Hybrid avg RMSE', f'{float(np.mean(hybrid_metrics["rmse"])):.4f}')

        if pure_metrics is not None and hybrid_metrics is not None:
            st.plotly_chart(
                create_metric_comparison_figure(time_axis, pure_metrics['rmse'], hybrid_metrics['rmse'], 'RMSE'),
                use_container_width=True, key='live_rmse_time',
            )
            st.plotly_chart(
                create_metric_comparison_figure(time_axis, pure_metrics['mae'], hybrid_metrics['mae'], 'MAE'),
                use_container_width=True, key='live_mae_time',
            )
            st.plotly_chart(
                create_metric_comparison_figure(time_axis, pure_metrics['max_error'], hybrid_metrics['max_error'], 'Max error'),
                use_container_width=True, key='live_maxerr_time',
            )
            improvement = compute_improvement_metrics(pure_metrics, hybrid_metrics)
            if 'mae' in improvement:
                st.plotly_chart(
                    create_improvement_figure(time_axis, improvement['mae'], 'MAE'),
                    use_container_width=True, key='live_improv_mae',
                )
            st.plotly_chart(
                create_error_histogram(
                    gt_slice,
                    pure_slice if pure_slice is not None else gt_slice,
                    hybrid_slice if hybrid_slice is not None else gt_slice,
                ),
                use_container_width=True, key='live_error_hist',
            )
            if pure_slice is not None:
                st.plotly_chart(
                    create_scatter_comparison_figure(gt_slice, pure_slice, 'Pure AI'),
                    use_container_width=True, key='live_scatter_pure',
                )
            if hybrid_slice is not None:
                st.plotly_chart(
                    create_scatter_comparison_figure(gt_slice, hybrid_slice, 'Hybrid'),
                    use_container_width=True, key='live_scatter_hybrid',
                )
                st.plotly_chart(
                    create_correlation_figure(gt_slice, hybrid_slice, 'Hybrid'),
                    use_container_width=True, key='live_corr_hybrid',
                )

        with st.expander('Training history (saved experiments)'):
            if pure_dir is not None:
                from .experiment_loader import load_experiment
                pure_exp = load_experiment(pure_dir)
                fig = create_training_history_figure(pure_exp.get('history'))
                if fig.data:
                    st.plotly_chart(fig, use_container_width=True, key='live_train_pure')
            if hybrid_dir is not None:
                from .experiment_loader import load_experiment
                hybrid_exp = load_experiment(hybrid_dir)
                fig = create_training_history_figure(hybrid_exp.get('history'))
                if fig.data:
                    st.plotly_chart(fig, use_container_width=True, key='live_train_hybrid')
