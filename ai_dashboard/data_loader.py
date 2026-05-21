"""Dataset sample loading for Stefan surrogate live inference."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np
import streamlit as st


@st.cache_data(show_spinner=False)
def load_npz_sample(path: str) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as data:
        T_norm = np.asarray(data['T_norm'], dtype=np.float32)
        liquid_frac = np.asarray(data['liquid_frac'], dtype=np.float32)
        if 'times' in data.files:
            times = np.asarray(data['times'], dtype=np.float32)
        else:
            times = np.arange(T_norm.shape[0], dtype=np.float32)
    return {
        'T_norm': T_norm,
        'liquid_frac': liquid_frac,
        'times': times,
    }


def start_frame_index(config: Optional[dict]) -> int:
    if config is None:
        return 1
    if config.get('use_T_prev', True) or config.get('use_delta_T', True):
        return 1
    return 0


def build_model_inputs(sample: dict[str, np.ndarray], config: Optional[dict]) -> np.ndarray:
    T_norm = sample['T_norm']
    n_frames, height, width = T_norm.shape

    use_T_t = bool(config.get('use_T_t', True)) if config else True
    use_T_prev = bool(config.get('use_T_prev', True)) if config else True
    use_delta_T = bool(config.get('use_delta_T', True)) if config else True

    channel_names: list[str] = []
    if use_T_t:
        channel_names.append('T_t')
    if use_T_prev:
        channel_names.append('T_prev')
    if use_delta_T:
        channel_names.append('delta_T')
    if not channel_names:
        channel_names = ['T_t']

    inputs = np.zeros((n_frames, len(channel_names), height, width), dtype=np.float32)
    for idx in range(n_frames):
        current = T_norm[idx]
        previous = T_norm[idx - 1] if idx > 0 else T_norm[idx]
        delta = current - previous
        maps = []
        for name in channel_names:
            if name == 'T_t':
                maps.append(current)
            elif name == 'T_prev':
                maps.append(previous)
            elif name == 'delta_T':
                maps.append(delta)
        inputs[idx] = np.stack(maps, axis=0)
    return inputs


def align_ground_truth(sample: dict[str, np.ndarray], config: Optional[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Return GT phase sequence and times aligned with model prediction frames."""
    start = start_frame_index(config)
    gt = sample['liquid_frac'][start:].astype(np.float32)
    times = sample['times'][start:].astype(np.float32)
    return gt, times


def summarize_sample(sample: dict[str, np.ndarray]) -> dict[str, Any]:
    temp = sample['T_norm']
    phase = sample['liquid_frac']
    return {
        'frames': int(temp.shape[0]),
        'grid': f'{temp.shape[1]}×{temp.shape[2]}',
        't_min': float(temp.min()),
        't_max': float(temp.max()),
        'phase_max': float(phase.max()),
    }
