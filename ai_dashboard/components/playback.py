"""Plotly-native playback toolbar (browser-side animation)."""
from __future__ import annotations

import streamlit as st

import plotly.graph_objects as go

from ..animation import duration_from_speed


def animation_speed_control(key_prefix: str = 'pb', default: float = 1.0) -> int:
    speed = st.select_slider(
        'Speed',
        options=[0.25, 0.5, 1.0, 1.5, 2.0, 3.0],
        value=default,
        key=f'{key_prefix}_speed',
        label_visibility='collapsed',
    )
    return duration_from_speed(float(speed))


def render_playback_toolbar(key_prefix: str, n_frames: int, *, title: str = 'Synchronized playback') -> int:
    st.markdown(
        f'<div class="playback-toolbar"><span class="playback-toolbar-title">{title}</span></div>',
        unsafe_allow_html=True,
    )
    left, center, right = st.columns([1.35, 2.3, 1.35], gap='medium')
    with left:
        st.markdown('<span class="playback-label">Speed</span>', unsafe_allow_html=True)
        duration_ms = animation_speed_control(key_prefix)
    with center:
        st.markdown(
            '<p class="playback-toolbar-hint">'
            'Use <strong>Play</strong> and <strong>Pause</strong> under the heatmaps · scrub with the frame slider'
            '</p>',
            unsafe_allow_html=True,
        )
    with right:
        st.markdown(
            f'<p class="playback-toolbar-meta"><span>{n_frames}</span> frames</p>',
            unsafe_allow_html=True,
        )
    return duration_ms


def show_animated_figure(
    fig: go.Figure,
    *,
    caption: str | None = None,
    n_frames: int | None = None,
    key_prefix: str = 'viz',
    toolbar_title: str = 'Synchronized playback',
    show_toolbar: bool = True,
) -> None:
    if show_toolbar and n_frames is not None:
        render_playback_toolbar(key_prefix, n_frames, title=toolbar_title)

    st.markdown('<div class="viz-stage viz-stage--fixed">', unsafe_allow_html=True)
    st.plotly_chart(
        fig,
        use_container_width=False,
        config={
            'displayModeBar': True,
            'scrollZoom': False,
            'displaylogo': False,
            'responsive': False,
            'modeBarButtonsToRemove': ['lasso2d', 'select2d', 'autoScale2d'],
        },
    )
    st.markdown('</div>', unsafe_allow_html=True)

    if caption:
        st.markdown(f'<p class="viz-caption">{caption}</p>', unsafe_allow_html=True)
