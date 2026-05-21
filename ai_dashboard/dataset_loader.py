"""Load dataset simulation samples for live AI inference."""

import glob
import json
from pathlib import Path
from typing import List, Optional

import numpy as np


def get_dataset_root(root: Optional[Path] = None) -> Path:
    base = Path(root) if root is not None else Path.cwd()
    return base / 'dataset'


def list_dataset_samples(data_dir: Optional[Path] = None) -> List[dict]:
    data_dir = Path(data_dir) if data_dir is not None else get_dataset_root()
    samples = []

    for bc_name in ('dirichlet', 'neumann'):
        bc_dir = data_dir / bc_name
        if not bc_dir.is_dir():
            continue
        for filepath in sorted(bc_dir.glob('*.npz')):
            samples.append({
                'id': f'{bc_name}/{filepath.stem}',
                'display_name': f'{bc_name} / {filepath.stem}',
                'path': filepath,
                'bc_type': bc_name,
            })
    return samples


def load_dataset_sample(path: Path) -> dict:
    with np.load(path, allow_pickle=True) as data:
        T_norm = np.asarray(data['T_norm'], dtype=np.float32)
        liquid_frac = np.asarray(data['liquid_frac'], dtype=np.float32)
        times = np.asarray(data['times'], dtype=np.float32)

        metadata = {
            'bc_type': str(data['bc_type']) if 'bc_type' in data.files else path.parent.name,
            'sim_name': path.stem,
            'Q0': float(data['Q0']) if 'Q0' in data.files else 0.0,
            'sigma': float(data['sigma']) if 'sigma' in data.files else 0.0,
            'center_x': float(data['center_x']) if 'center_x' in data.files else 0.0,
            'center_y': float(data['center_y']) if 'center_y' in data.files else 0.0,
            'time_total': float(data['time_total']) if 'time_total' in data.files else float(times[-1] if len(times) else 0.0),
        }

    return {
        'T_norm': T_norm,
        'liquid_frac': liquid_frac,
        'times': times,
        'metadata': metadata,
        'path': str(path),
    }


def build_spatial_axes(shape: tuple, config: Optional[dict] = None) -> tuple:
    length = 0.05
    if config is not None:
        length = float(config.get('length', config.get('domain_size', length)))
    ny, nx = shape[-2], shape[-1]
    x = np.linspace(0.0, length, nx)
    y = np.linspace(0.0, length, ny)
    return x, y
