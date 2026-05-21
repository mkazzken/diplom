"""Error analysis utilities for AI comparison."""

import numpy as np


def compute_error_maps(gt: np.ndarray, pred: np.ndarray) -> dict:
    diff = pred - gt
    abs_error = np.abs(diff)
    sq_error = diff**2
    return {
        'abs_error': abs_error,
        'sq_error': sq_error,
        'difference': diff
    }


def compute_temporal_metrics(gt: np.ndarray, pred: np.ndarray) -> dict:
    gt = np.asarray(gt, dtype=np.float32)
    pred = np.asarray(pred, dtype=np.float32)
    diff = pred - gt
    axis = tuple(range(1, gt.ndim))

    mse = np.mean(diff**2, axis=axis)
    mae = np.mean(np.abs(diff), axis=axis)
    rmse = np.sqrt(mse)
    max_error = np.max(np.abs(diff), axis=axis)
    rel_error = np.mean(np.abs(diff) / np.maximum(np.abs(gt), 1e-8), axis=axis)

    return {
        'mse': mse,
        'mae': mae,
        'rmse': rmse,
        'max_error': max_error,
        'rel_error': rel_error
    }


def compute_improvement_metrics(pure_metrics: dict, hybrid_metrics: dict) -> dict:
    improvement = {}
    for key in ['mse', 'mae', 'rmse', 'max_error', 'rel_error']:
        if key in pure_metrics and key in hybrid_metrics:
            pure = np.asarray(pure_metrics[key], dtype=np.float32)
            hybrid = np.asarray(hybrid_metrics[key], dtype=np.float32)
            improvement[key] = 100.0 * np.maximum(0.0, (pure - hybrid) / np.maximum(pure, 1e-8))
    return improvement
