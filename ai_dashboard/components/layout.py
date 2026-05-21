"""Shared layout and sidebar controls."""
from __future__ import annotations

import streamlit as st

from ..discovery import ExperimentRun, format_timestamp, latest_by_kind, scan_experiments
from ..paths import EXPERIMENTS_ROOT, STYLES_DIR


def apply_theme() -> None:
    css_path = STYLES_DIR / 'theme.css'
    if css_path.is_file():
        st.markdown(f'<style>{css_path.read_text(encoding="utf-8")}</style>', unsafe_allow_html=True)


def render_hero(title: str, subtitle: str) -> None:
    st.markdown(f'<p class="hero-title">{title}</p>', unsafe_allow_html=True)
    st.markdown(f'<p class="hero-sub">{subtitle}</p>', unsafe_allow_html=True)


def init_runs() -> list[ExperimentRun]:
    if 'ai_runs' not in st.session_state:
        st.session_state.ai_runs = scan_experiments(EXPERIMENTS_ROOT)
    return st.session_state.ai_runs


def sidebar_pair_selector() -> tuple[ExperimentRun | None, ExperimentRun | None]:
    runs = init_runs()
    pure_list = [run for run in runs if run.kind == 'pure_ai']
    hybrid_list = [run for run in runs if run.kind == 'hybrid']

    st.sidebar.markdown('#### Model runs')
    if st.sidebar.button('🔄 Refresh experiments', key='ai_refresh_runs'):
        st.session_state.ai_runs = scan_experiments(EXPERIMENTS_ROOT)
        st.rerun()

    pure_default = latest_by_kind(runs, 'pure_ai')
    hybrid_default = latest_by_kind(runs, 'hybrid')

    pure_name = st.sidebar.selectbox(
        'Pure AI run',
        [run.name for run in pure_list] or ['—'],
        index=0 if pure_list else 0,
        key='ai_pure_run_select',
    )
    hybrid_name = st.sidebar.selectbox(
        'Hybrid run',
        [run.name for run in hybrid_list] or ['—'],
        index=0 if hybrid_list else 0,
        key='ai_hybrid_run_select',
    )

    pure = next((run for run in pure_list if run.name == pure_name), pure_default)
    hybrid = next((run for run in hybrid_list if run.name == hybrid_name), hybrid_default)
    return pure, hybrid


def sidebar_dataset_selector() -> tuple[str | None, str | None]:
    from ..discovery import discover_dataset_samples

    datasets = discover_dataset_samples()
    if not datasets:
        st.sidebar.warning('No dataset samples found in `dataset/`.')
        return None, None

    ds_label = st.sidebar.selectbox('Dataset split', list(datasets.keys()), key='ai_dataset_split')
    files = datasets[ds_label]
    sample_path = st.sidebar.selectbox(
        'Sample',
        files,
        format_func=lambda path: path.name,
        key='ai_dataset_sample_select',
    )
    return ds_label, str(sample_path) if sample_path else None
