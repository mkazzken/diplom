"""Playback controller for synchronized AI dashboard timeline navigation."""

import time

import streamlit as st


def _init_state(prefix: str, total_frames: int) -> None:
    state = st.session_state
    state.setdefault(f'{prefix}_frame_index', 0)
    state.setdefault(f'{prefix}_is_playing', False)
    state.setdefault(f'{prefix}_playback_speed', 1.0)
    state.setdefault(f'{prefix}_loop_playback', True)
    state[f'{prefix}_total_frames'] = total_frames

    max_frame = max(0, total_frames - 1)
    state[f'{prefix}_frame_index'] = min(max(0, int(state[f'{prefix}_frame_index'])), max_frame)


def _advance_frame(state, prefix: str, max_frame: int) -> None:
    next_frame = int(state[f'{prefix}_frame_index']) + 1
    if next_frame > max_frame:
        if state.get(f'{prefix}_loop_playback', True):
            next_frame = 0
        else:
            next_frame = max_frame
            state[f'{prefix}_is_playing'] = False
    state[f'{prefix}_frame_index'] = next_frame


def render_playback_controls(
    total_frames: int,
    time_per_frame: float = 0.01,
    prefix: str = 'ai',
) -> int:
    _init_state(prefix, total_frames)
    state = st.session_state
    max_frame = max(0, total_frames - 1)
    index_key = f'{prefix}_frame_index'

    slider_key = f'{prefix}_frame_slider'
    if state.get(f'{prefix}_playback_advance', False):
        _advance_frame(state, prefix, max_frame)
        state[f'{prefix}_playback_advance'] = False
        state[slider_key] = int(state[index_key])

    with st.expander('Playback controls', expanded=True):
        row1 = st.columns([1, 1, 1, 1, 3, 2])

        if row1[0].button('⏮️ Previous', key=f'{prefix}_prev'):
            new_frame = max(0, int(state[index_key]) - 1)
            state[index_key] = new_frame
            state[slider_key] = new_frame
            state[f'{prefix}_is_playing'] = False

        if row1[1].button('▶️ Play', key=f'{prefix}_play'):
            state[f'{prefix}_is_playing'] = True

        if row1[2].button('⏹️ Stop', key=f'{prefix}_stop'):
            state[index_key] = 0
            state[slider_key] = 0
            state[f'{prefix}_is_playing'] = False

        if row1[3].button('⏭️ Next', key=f'{prefix}_next'):
            new_frame = min(max_frame, int(state[index_key]) + 1)
            state[index_key] = new_frame
            state[slider_key] = new_frame
            state[f'{prefix}_is_playing'] = False

        current_frame = int(
            row1[4].slider(
                'Frame',
                min_value=0,
                max_value=max_frame,
                value=int(state[index_key]),
                key=slider_key,
            )
        )
        state[index_key] = current_frame

        row1[5].select_slider(
            'Speed',
            options=[0.25, 0.5, 1.0, 1.5, 2.0, 3.0],
            value=float(state.get(f'{prefix}_playback_speed', 1.0)),
            key=f'{prefix}_playback_speed',
        )

        st.checkbox(
            'Loop playback',
            value=state.get(f'{prefix}_loop_playback', True),
            key=f'{prefix}_loop_playback',
        )

        frame_time = current_frame * time_per_frame
        if total_frames > 0 and prefix == 'ai' and 'ai_inference_result' in state:
            times = state['ai_inference_result'].get('times')
            if times is not None and len(times) > current_frame:
                frame_time = float(times[current_frame])

        st.markdown(
            f'**Frame:** {current_frame + 1} / {max(total_frames, 1)}  \n'
            f'**Time:** {frame_time:.3f} s'
        )

    if state[f'{prefix}_is_playing'] and total_frames > 1:
        interval = max(0.05, 0.35 / float(state[f'{prefix}_playback_speed']))
        time.sleep(interval)
        state[f'{prefix}_playback_advance'] = True
        st.rerun()

    return int(state[index_key])
