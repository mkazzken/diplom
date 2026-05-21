"""Cached live inference using project TemporalUNet / InspiredSurrogate checkpoints."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

import numpy as np
import streamlit as st

from .data_loader import align_ground_truth, build_model_inputs, load_npz_sample, start_frame_index


def _ensure_torch():
    from ._torch_bootstrap import ensure_torch

    return ensure_torch()


def _extract_state_dict(checkpoint) -> dict:
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            return checkpoint['model_state_dict']
        if 'state_dict' in checkpoint:
            return checkpoint['state_dict']
    return checkpoint


def _count_channels(config: dict) -> int:
    count = 0
    if config.get('use_T_t', True):
        count += 1
    if config.get('use_T_prev', True):
        count += 1
    if config.get('use_delta_T', True):
        count += 1
    return max(count, 1)


@st.cache_resource(show_spinner='Loading neural surrogate…')
def load_surrogate(kind: str, experiment_path: str, config_json: str):
    if not _ensure_torch():
        return None

    from ._torch_bootstrap import InspiredSurrogate, TemporalUNet, torch

    config = json.loads(config_json) if config_json else {}
    exp_path = Path(experiment_path)
    in_channels = _count_channels(config)

    if kind == 'hybrid':
        model = InspiredSurrogate(
            in_channels=in_channels,
            features=config.get('features', [8, 16, 32, 64]),
            dropout=float(config.get('dropout', 0.05)),
        )
    else:
        model = TemporalUNet(
            in_channels=in_channels,
            n_classes=1,
            features=config.get('features', [8, 16, 32, 64]),
            dropout=float(config.get('dropout', 0.05)),
        )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    weights = exp_path / 'model_best.pt'
    checkpoint = torch.load(str(weights), map_location=device, weights_only=False)
    model.load_state_dict(_extract_state_dict(checkpoint))
    model.to(device)
    model.eval()
    return model, str(device)


@st.cache_data(show_spinner=False)
def run_model_inference(kind: str, experiment_path: str, config_json: str, sample_path: str) -> tuple[np.ndarray, float]:
    if not _ensure_torch():
        raise RuntimeError('PyTorch is not available')

    from ._torch_bootstrap import torch

    loaded = load_surrogate(kind, experiment_path, config_json)
    if loaded is None:
        raise RuntimeError(f'Failed to load {kind} model')
    model, device_name = loaded
    device = torch.device(device_name)

    sample = load_npz_sample(sample_path)
    config = json.loads(config_json) if config_json else {}
    inputs = build_model_inputs(sample, config)
    start = start_frame_index(config)
    inputs = inputs[start:]

    model.eval()
    t0 = time.perf_counter()
    with torch.no_grad():
        tensor = torch.from_numpy(inputs).to(device)
        outputs = model(tensor).detach().cpu().numpy()
    elapsed_ms = (time.perf_counter() - t0) * 1000.0 / max(len(inputs), 1)

    if outputs.ndim == 4 and outputs.shape[1] == 1:
        outputs = outputs[:, 0, ...]
    return outputs.astype(np.float32), float(elapsed_ms)


def predict_from_sample(
    sample_path: str,
    pure_run: Optional[Path],
    hybrid_run: Optional[Path],
) -> dict:
    sample = load_npz_sample(sample_path)
    config = {}
    if pure_run is not None:
        config_path = pure_run / 'config.json'
        if config_path.is_file():
            config = json.loads(config_path.read_text(encoding='utf-8'))
    elif hybrid_run is not None:
        config_path = hybrid_run / 'config.json'
        if config_path.is_file():
            config = json.loads(config_path.read_text(encoding='utf-8'))

    gt, times = align_ground_truth(sample, config)
    result = {
        'ground_truth': gt,
        'times': times,
        'pure': None,
        'hybrid': None,
        'pure_ms': None,
        'hybrid_ms': None,
        'sample_summary': sample,
    }

    if pure_run is not None:
        cfg_text = (pure_run / 'config.json').read_text(encoding='utf-8') if (pure_run / 'config.json').is_file() else '{}'
        pred, ms = run_model_inference('pure', str(pure_run), cfg_text, sample_path)
        result['pure'] = _trim_to_length(pred, len(gt))
        result['pure_ms'] = ms

    if hybrid_run is not None:
        cfg_text = (hybrid_run / 'config.json').read_text(encoding='utf-8') if (hybrid_run / 'config.json').is_file() else '{}'
        pred, ms = run_model_inference('hybrid', str(hybrid_run), cfg_text, sample_path)
        result['hybrid'] = _trim_to_length(pred, len(gt))
        result['hybrid_ms'] = ms

    return result


def _trim_to_length(pred: np.ndarray, length: int) -> np.ndarray:
    if pred.shape[0] == length:
        return pred
    return pred[:length]
