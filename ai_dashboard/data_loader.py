"""Dataset sample loading for Stefan surrogate live inference."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np
import streamlit as st

DOMAIN_LENGTH_M = 0.05


def _parse_scalar(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        if value.shape == ():
            value = value.item()
        else:
            return value
    if isinstance(value, bytes):
        return value.decode('utf-8')
    return value


@st.cache_data(show_spinner=False)
def load_npz_sample(path: str) -> dict[str, Any]:
    with np.load(path, allow_pickle=True) as data:
        T_norm = np.asarray(data['T_norm'], dtype=np.float32)
        liquid_frac = np.asarray(data['liquid_frac'], dtype=np.float32)
        if 'times' in data.files:
            times = np.asarray(data['times'], dtype=np.float32)
        else:
            times = np.arange(T_norm.shape[0], dtype=np.float32)

        metadata: dict[str, Any] = {
            'path': path,
            'filename': Path(path).name,
            'split': Path(path).parent.name,
        }
        for key in ('bc_type', 'Q0', 'sigma', 'center_x', 'center_y', 'time_total'):
            if key in data.files:
                metadata[key] = _parse_scalar(data[key])

    return {
        'T_norm': T_norm,
        'liquid_frac': liquid_frac,
        'times': times,
        'metadata': metadata,
    }


def build_sample_info(sample: dict[str, Any], config: Optional[dict] = None) -> dict[str, Any]:
    """Human-readable simulation parameters for the dashboard."""
    meta = sample.get('metadata', {})
    times = np.asarray(sample['times'], dtype=np.float32)
    n_frames = int(sample['T_norm'].shape[0])
    ny, nx = int(sample['T_norm'].shape[1]), int(sample['T_norm'].shape[2])

    start = start_frame_index(config)
    aligned_times = times[start:]
    dt_steps = float(np.median(np.diff(times))) if len(times) > 1 else 0.0

    center_x = float(meta.get('center_x', 0.0))
    center_y = float(meta.get('center_y', 0.0))
    sigma_m = float(meta.get('sigma', 0.0))
    q0 = float(meta.get('Q0', 0.0))
    time_total = float(meta.get('time_total', float(times[-1]) if len(times) else 0.0))

    return {
        'sample_id': f"{meta.get('split', '—')}/{meta.get('filename', '—')}",
        'bc_type': str(meta.get('bc_type', 'unknown')).capitalize(),
        'time_total_s': time_total,
        'time_start_s': float(times[0]) if len(times) else 0.0,
        'time_end_s': float(times[-1]) if len(times) else 0.0,
        'n_frames_raw': n_frames,
        'n_frames_aligned': max(0, n_frames - start),
        'inference_start_index': start,
        'dt_approx_s': dt_steps,
        'aligned_time_start_s': float(aligned_times[0]) if len(aligned_times) else 0.0,
        'aligned_time_end_s': float(aligned_times[-1]) if len(aligned_times) else 0.0,
        'Q0_W_m3': q0,
        'sigma_mm': sigma_m * 1000.0,
        'center_x_mm': center_x * 1000.0,
        'center_y_mm': center_y * 1000.0,
        'domain_mm': DOMAIN_LENGTH_M * 1000.0,
        'grid': f'{ny}×{nx}',
        'dx_mm': DOMAIN_LENGTH_M * 1000.0 / max(nx - 1, 1),
        'T_norm_min': float(sample['T_norm'].min()),
        'T_norm_max': float(sample['T_norm'].max()),
        'phase_max': float(sample['liquid_frac'].max()),
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


def summarize_sample(sample: dict[str, Any]) -> dict[str, Any]:
    info = build_sample_info(sample)
    return {
        'frames': info['n_frames_raw'],
        'grid': info['grid'],
        't_min': info['T_norm_min'],
        't_max': info['T_norm_max'],
        'phase_max': info['phase_max'],
    }
