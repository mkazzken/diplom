"""Load experiment data and metadata for AI comparison."""

import json
from pathlib import Path
from typing import List, Optional

import numpy as np

SEARCH_DIRS = ['experiments', Path('internal') / 'experiments']


def _safe_load_json(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        with path.open('r', encoding='utf-8') as handle:
            return json.load(handle)
    except Exception:
        return None


def _safe_load_npz(path: Path) -> Optional[np.ndarray]:
    if not path.exists():
        return None
    try:
        with np.load(path, allow_pickle=True) as data:
            key = 'arr' if 'arr' in data.files else data.files[0]
            arr = np.asarray(data[key])
            if arr.ndim == 4 and arr.shape[1] == 1:
                arr = arr[:, 0, ...]
            if arr.ndim == 4 and arr.shape[0] == 1:
                arr = arr[0, ...]
            return arr.astype(np.float32)
    except Exception:
        return None


def find_experiment_directories(root: Optional[Path] = None) -> List[Path]:
    root = Path(root) if root is not None else Path.cwd()
    experiments = []
    for candidate in SEARCH_DIRS:
        path = root / candidate
        if path.exists() and path.is_dir():
            for sub in sorted(path.iterdir()):
                if sub.is_dir():
                    experiments.append(sub)
    return experiments


def classify_experiment(name: str) -> str:
    key = name.lower()
    if 'hybrid' in key or 'surrogate' in key or 'inspired' in key:
        return 'hybrid'
    if key.startswith('run_') or 'pure' in key:
        return 'pure'
    return 'other'


def build_time_axis(config: Optional[dict], num_frames: int, default_dt: float = 0.01) -> np.ndarray:
    if num_frames <= 0:
        return np.zeros(0, dtype=np.float32)
    if config is None:
        dt = default_dt
    else:
        dt = config.get('time_step') or config.get('dt') or config.get('delta_t') or default_dt
    try:
        dt = float(dt)
    except Exception:
        dt = default_dt
    return np.arange(num_frames, dtype=np.float32) * dt


def load_experiment(path: Path) -> dict:
    data = {
        'name': path.name,
        'path': str(path),
        'classification': classify_experiment(path.name),
        'ground_truth': _safe_load_npz(path / 'ground_truth.npz'),
        'predictions': _safe_load_npz(path / 'predictions.npz'),
        'errors': _safe_load_npz(path / 'errors.npz'),
        'metrics': _safe_load_json(path / 'metrics.json'),
        'history': _safe_load_json(path / 'history.json'),
        'config': _safe_load_json(path / 'config.json')
    }
    if data['ground_truth'] is None and data['predictions'] is not None:
        data['ground_truth'] = data['predictions'].copy()
    return data
