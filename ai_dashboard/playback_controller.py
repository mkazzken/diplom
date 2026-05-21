"""Playback controller for AI comparison timeline navigation."""

import time
import streamlit as st


def _init_state(prefix: str, total_frames: int) -> None:
    state = st.session_state
    # Initialize only missing keys to avoid overwriting values bound to widgets
    state.setdefault(f'{prefix}_total_frames', total_frames)
    # Always keep the total frame count up-to-date
    state[f'{prefix}_total_frames'] = total_frames
    state.setdefault(f'{prefix}_current_frame', 0)
    state.setdefault(f'{prefix}_is_playing', False)
    state.setdefault(f'{prefix}_playback_speed', 1.0)
    state.setdefault(f'{prefix}_loop_playback', False)
    state.setdefault(f'{prefix}_autoplay', False)


def render_playback_controls(total_frames: int, time_per_frame: float = 0.01, prefix: str = 'ai') -> int:
    _init_state(prefix, total_frames)
    state = st.session_state
    current_frame = int(state[f'{prefix}_current_frame'])

    with st.expander('Playback controls', expanded=True):
        cols = st.columns([1, 1, 1, 1, 2, 2])
        if cols[0].button('⏮️ Previous', key=f'{prefix}_prev'):
            current_frame = max(0, current_frame - 1)
            state[f'{prefix}_is_playing'] = False
        if cols[1].button('▶️ Play', key=f'{prefix}_play'):
            state[f'{prefix}_is_playing'] = True
        if cols[2].button('⏸️ Pause', key=f'{prefix}_pause'):
            state[f'{prefix}_is_playing'] = False
        if cols[3].button('⏹️ Stop', key=f'{prefix}_stop'):
            current_frame = 0
            state[f'{prefix}_is_playing'] = False

        # Use widget keys that match our session keys so Streamlit manages state safely.
        current_frame = int(cols[4].slider(
            'Frame',
            min_value=0,
            max_value=max(0, total_frames - 1),
            value=current_frame,
            key=f'{prefix}_current_frame'
        ))

        cols[5].select_slider(
            'Speed',
            options=[0.25, 0.5, 1.0, 1.5, 2.0, 3.0],
            value=float(state.get(f'{prefix}_playback_speed', 1.0)),
            key=f'{prefix}_playback_speed'
        )

        col2a, col2b = st.columns([2, 3])
        col2a.checkbox('Loop playback', value=state.get(f'{prefix}_loop_playback', False), key=f'{prefix}_loop_playback')
        col2b.checkbox('Autoplay', value=state.get(f'{prefix}_autoplay', False), key=f'{prefix}_autoplay')

        if state[f'{prefix}_is_playing'] and state[f'{prefix}_autoplay']:
            interval = max(0.05, 1.0 / float(state[f'{prefix}_playback_speed']))
            time.sleep(interval)
            next_frame = current_frame + 1
            if next_frame >= total_frames:
                if state[f'{prefix}_loop_playback']:
                    next_frame = 0
                else:
                    next_frame = total_frames - 1
                    state[f'{prefix}_is_playing'] = False
            state[f'{prefix}_current_frame'] = next_frame
            st.experimental_rerun()

    st.markdown(f'**Frame:** {current_frame + 1} / {total_frames}  \n**Time:** {current_frame * time_per_frame:.3f} s')
    return current_frame
