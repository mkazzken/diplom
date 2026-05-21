"""Live model loading and inference for the AI Inference Dashboard."""

import json
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch

from hubrid_unet import InspiredSurrogate
from unet import TemporalUNet


def _count_input_channels(config: dict) -> int:
    channels = 0
    if config.get('use_T_t', True):
        channels += 1
    if config.get('use_T_prev', True):
        channels += 1
    if config.get('use_delta_T', True):
        channels += 1
    return max(channels, 1)


def _start_frame_index(config: dict) -> int:
    if config.get('use_T_prev', True) or config.get('use_delta_T', True):
        return 1
    return 0


def build_frame_input(T_seq: np.ndarray, frame_idx: int, config: dict) -> np.ndarray:
    channels = []
    if config.get('use_T_t', True):
        channels.append(T_seq[frame_idx])
    if config.get('use_T_prev', True) and frame_idx > 0:
        channels.append(T_seq[frame_idx - 1])
    if config.get('use_delta_T', True) and frame_idx > 0:
        channels.append(T_seq[frame_idx] - T_seq[frame_idx - 1])
    if not channels:
        channels.append(T_seq[frame_idx])
    return np.stack(channels, axis=0).astype(np.float32)


def load_config(exp_dir: Path) -> dict:
    config_path = exp_dir / 'config.json'
    with config_path.open('r', encoding='utf-8') as handle:
        return json.load(handle)


def _extract_state_dict(checkpoint) -> dict:
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            return checkpoint['model_state_dict']
        if 'state_dict' in checkpoint:
            return checkpoint['state_dict']
    return checkpoint


def load_pure_model(exp_dir: Path, device: torch.device) -> Tuple[TemporalUNet, dict]:
    config = load_config(exp_dir)
    model = TemporalUNet(
        in_channels=_count_input_channels(config),
        n_classes=1,
        features=config.get('features', [8, 16, 32, 64]),
        dropout=config.get('dropout', 0.05),
    )
    weights_path = exp_dir / 'model_best.pt'
    checkpoint = torch.load(weights_path, map_location=device)
    model.load_state_dict(_extract_state_dict(checkpoint))
    model.to(device)
    model.eval()
    return model, config


def load_hybrid_model(exp_dir: Path, device: torch.device) -> Tuple[InspiredSurrogate, dict]:
    config = load_config(exp_dir)
    model = InspiredSurrogate(
        in_channels=_count_input_channels(config),
        n_classes=1,
        features=config.get('features', [8, 16, 32, 64]),
        dropout=config.get('dropout', 0.05),
    )
    weights_path = exp_dir / 'model_best.pt'
    checkpoint = torch.load(weights_path, map_location=device)
    model.load_state_dict(_extract_state_dict(checkpoint))
    model.to(device)
    model.eval()
    return model, config


@torch.no_grad()
def predict_sequence(
    model: torch.nn.Module,
    T_seq: np.ndarray,
    config: dict,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run frame-by-frame inference; returns predictions, frame indices, times indices."""
    start_idx = _start_frame_index(config)
    n_frames = T_seq.shape[0]
    predictions = []
    frame_indices = []

    for frame_idx in range(start_idx, n_frames):
        X = build_frame_input(T_seq, frame_idx, config)
        tensor = torch.tensor(X, dtype=torch.float32, device=device).unsqueeze(0)
        pred = model(tensor).squeeze().detach().cpu().numpy().astype(np.float32)
        predictions.append(pred)
        frame_indices.append(frame_idx)

    if not predictions:
        empty = np.zeros((0, T_seq.shape[1], T_seq.shape[2]), dtype=np.float32)
        return empty, np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.float32)

    return (
        np.stack(predictions, axis=0),
        np.asarray(frame_indices, dtype=np.int32),
        np.asarray(frame_indices, dtype=np.int32),
    )


def run_live_inference(
    sample: dict,
    pure_exp_dir: Optional[Path],
    hybrid_exp_dir: Optional[Path],
    device: Optional[torch.device] = None,
) -> dict:
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    T_seq = sample['T_norm']
    liquid_frac = sample['liquid_frac']
    times = sample['times']

    start_idx = 0
    gt_frames = liquid_frac
    time_axis = times

    pure_pred = None
    hybrid_pred = None
    pure_config = None
    hybrid_config = None

    if pure_exp_dir is not None:
        pure_model, pure_config = load_pure_model(pure_exp_dir, device)
        pure_pred, frame_indices, _ = predict_sequence(pure_model, T_seq, pure_config, device)
        start_idx = int(frame_indices[0]) if len(frame_indices) else _start_frame_index(pure_config)
        gt_frames = liquid_frac[frame_indices]
        time_axis = times[frame_indices]

    if hybrid_exp_dir is not None:
        hybrid_model, hybrid_config = load_hybrid_model(hybrid_exp_dir, device)
        hybrid_pred, hybrid_indices, _ = predict_sequence(hybrid_model, T_seq, hybrid_config, device)
        if pure_pred is None:
            start_idx = int(hybrid_indices[0]) if len(hybrid_indices) else _start_frame_index(hybrid_config)
            gt_frames = liquid_frac[hybrid_indices]
            time_axis = times[hybrid_indices]
        else:
            common_count = min(len(gt_frames), len(hybrid_pred))
            gt_frames = gt_frames[:common_count]
            time_axis = time_axis[:common_count]
            pure_pred = pure_pred[:common_count]
            hybrid_pred = hybrid_pred[:common_count]

    if pure_pred is not None and hybrid_pred is not None:
        common_count = min(len(gt_frames), len(pure_pred), len(hybrid_pred))
        gt_frames = gt_frames[:common_count]
        pure_pred = pure_pred[:common_count]
        hybrid_pred = hybrid_pred[:common_count]
        time_axis = time_axis[:common_count]

    return {
        'ground_truth': gt_frames.astype(np.float32),
        'pure_prediction': pure_pred,
        'hybrid_prediction': hybrid_pred,
        'times': time_axis.astype(np.float32),
        'pure_config': pure_config,
        'hybrid_config': hybrid_config,
        'device': str(device),
        'start_frame_index': start_idx,
    }
