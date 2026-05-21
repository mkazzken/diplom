from pathlib import Path
import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

ROOT = Path(__file__).resolve().parent


def is_branch_available(root: Path, branch_name: str) -> bool:
    return (root / branch_name).is_dir()


def get_branch_paths(root: Path):
    branches = {}
    for name in ['internal', 'boundary']:
        branch_path = root / name
        if branch_path.exists() and branch_path.is_dir():
            branches[name] = branch_path
    return branches


def safe_json_load(path: Path):
    try:
        with open(path, 'r', encoding='utf-8') as handle:
            return json.load(handle)
    except Exception:
        return {}


def parse_metadata_value(value):
    if isinstance(value, np.ndarray):
        if value.shape == ():
            value = value.item()
        else:
            value = value.tolist()
    if isinstance(value, bytes):
        value = value.decode('utf-8')
    if isinstance(value, str):
        try:
            return json.loads(value)
        except Exception:
            return value
    return value


def load_npz_array(path: Path):
    with np.load(path, allow_pickle=True) as z:
        if len(z.files) == 0:
            return np.array([])
        if 'arr' in z.files and len(z.files) == 1:
            return z['arr']
        return z[z.files[0]]


def list_internal_samples(root: Path):
    internal_dir = root / 'internal'
    dataset_dir = internal_dir / 'dataset'
    samples = []
    if not dataset_dir.exists():
        return samples

    for bc_dir in sorted(dataset_dir.iterdir()):
        if not bc_dir.is_dir():
            continue
        for sample_file in sorted(bc_dir.glob('*.npz')):
            samples.append({
                'branch': 'internal',
                'category': bc_dir.name,
                'id': f"{bc_dir.name}/{sample_file.stem}",
                'display_name': f"{bc_dir.name} / {sample_file.stem}",
                'path': sample_file,
                'source_type': 'internal',
                'boundary_condition': bc_dir.name,
            })
    return samples


def list_boundary_samples(root: Path):
    boundary_dir = root / 'boundary'
    dataset_dir = boundary_dir / 'dataset_v3'
    samples = []
    if not dataset_dir.exists():
        return samples
    for sample_file in sorted(dataset_dir.glob('sample_*.npz')):
        samples.append({
            'branch': 'boundary',
            'category': 'boundary',
            'id': sample_file.stem,
            'display_name': sample_file.stem,
            'path': sample_file,
            'source_type': 'boundary',
            'boundary_condition': 'boundary heating',
        })
    return samples


def load_internal_sample(path: Path):
    with np.load(path, allow_pickle=True) as z:
        T = np.array(z['T_norm'], dtype=np.float32)
        phi = np.array(z['liquid_frac'], dtype=np.float32)
        times = np.array(z['times'], dtype=np.float32)
        metadata = {
            'bc_type': parse_metadata_value(z.get('bc_type', 'unknown')),
            'Q0': float(parse_metadata_value(z.get('Q0', 0.0))),
            'sigma': float(parse_metadata_value(z.get('sigma', 0.0))),
            'center_x': float(parse_metadata_value(z.get('center_x', 0.0))),
            'center_y': float(parse_metadata_value(z.get('center_y', 0.0))),
            'time_total': float(parse_metadata_value(z.get('time_total', 0.0))),
            'source_type': 'internal heat source',
            'boundary_condition': str(parse_metadata_value(z.get('bc_type', 'unknown'))),
        }
    return {
        'temperature': T,
        'liquid_fraction': phi,
        'times': times,
        'metadata': metadata,
        'initial_temperature': T[0],
        'initial_liquid_fraction': phi[0],
    }


def load_boundary_sample(path: Path):
    with np.load(path, allow_pickle=True) as z:
        T = np.array(z['temperature'], dtype=np.float32)
        phi = np.array(z['liquid_fraction'], dtype=np.float32)
        raw_meta = parse_metadata_value(z.get('metadata', {}))
        if isinstance(raw_meta, str):
            try:
                raw_meta = json.loads(raw_meta)
            except Exception:
                raw_meta = {'note': raw_meta}
        metadata = raw_meta if isinstance(raw_meta, dict) else {}
        n_frames = T.shape[0]
        physical_time = float(metadata.get('physical_time', n_frames - 1))
        times = np.linspace(0.0, physical_time, n_frames, dtype=np.float32)
        metadata['source_type'] = 'boundary heating'
        metadata['boundary_condition'] = metadata.get('pair', 'boundary heating')
    return {
        'temperature': T,
        'liquid_fraction': phi,
        'times': times,
        'metadata': metadata,
        'initial_temperature': T[0],
        'initial_liquid_fraction': phi[0],
    }


def discover_experiments(root: Path):
    experiments = []
    for branch_name in ['internal', 'boundary']:
        branch_dir = root / branch_name
        exp_dir = branch_dir / 'experiments'
        if not exp_dir.exists() or not exp_dir.is_dir():
            continue
        for child in sorted(exp_dir.iterdir()):
            if not child.is_dir():
                continue
            metrics = safe_json_load(child / 'metrics.json')
            config = safe_json_load(child / 'config.json')
            history = safe_json_load(child / 'history.json')
            model_file = child / 'model_final.pt'
            model_size_mb = model_file.stat().st_size / 1_000_000 if model_file.exists() else None
            experiments.append({
                'branch': branch_name,
                'name': child.name,
                'path': child,
                'config': config,
                'metrics': metrics,
                'history': history,
                'model_size_mb': model_size_mb,
                'has_ground_truth': (child / 'ground_truth.npz').exists(),
                'has_predictions': (child / 'predictions.npz').exists(),
                'has_errors': (child / 'errors.npz').exists(),
            })
    return experiments


def load_experiment_arrays(exp_path: Path):
    result = {}
    if (exp_path / 'ground_truth.npz').exists():
        result['ground_truth'] = load_npz_array(exp_path / 'ground_truth.npz')
    if (exp_path / 'predictions.npz').exists():
        result['predictions'] = load_npz_array(exp_path / 'predictions.npz')
    if (exp_path / 'errors.npz').exists():
        result['errors'] = load_npz_array(exp_path / 'errors.npz')
    return result


def normalize_field(field, clip_percentile=2):
    if field.size == 0:
        return field
    vmin = np.nanpercentile(field, clip_percentile)
    vmax = np.nanpercentile(field, 100 - clip_percentile)
    if np.isclose(vmin, vmax):
        return field, float(np.nanmin(field)), float(np.nanmax(field))
    return field, float(vmin), float(vmax)


def make_heatmap(field, title='', colorscale='viridis', zmin=None, zmax=None):
    fig = go.Figure(go.Heatmap(
        z=field,
        colorscale=colorscale,
        zmin=zmin,
        zmax=zmax,
        colorbar=dict(title='Value', len=0.75, thickness=15),
        hoverinfo='z'
    ))
    fig.update_layout(
        template='plotly_dark',
        title=title,
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=420,
    )
    return fig


def make_flowchart_figure():
    fig = go.Figure()
    boxes = [
        {'label': 'Physical model', 'x': 0.12, 'y': 0.85},
        {'label': 'Boundary / Initial conditions', 'x': 0.35, 'y': 0.85},
        {'label': 'Numerical simulation', 'x': 0.60, 'y': 0.85},
        {'label': 'Dataset generation', 'x': 0.85, 'y': 0.85},
        {'label': 'AI surrogate', 'x': 0.5, 'y': 0.45},
        {'label': 'Prediction comparison', 'x': 0.5, 'y': 0.15},
    ]
    for box in boxes:
        fig.add_shape(
            type='rect', x0=box['x'] - 0.14, y0=box['y'] - 0.08,
            x1=box['x'] + 0.14, y1=box['y'] + 0.08,
            line=dict(color='white', width=2), fillcolor='rgba(255,255,255,0.04)'
        )
        fig.add_annotation(x=box['x'], y=box['y'], text=box['label'], showarrow=False,
                           font=dict(color='white', size=12))
    arrows = [
        ((0.26, 0.85), (0.46, 0.85)),
        ((0.74, 0.85), (0.86, 0.85)),
        ((0.5, 0.77), (0.5, 0.53)),
        ((0.5, 0.37), (0.5, 0.23)),
    ]
    for start, end in arrows:
        fig.add_annotation(x=end[0], y=end[1], ax=start[0], ay=start[1], xref='x', yref='y',
                           axref='x', ayref='y', showarrow=True, arrowhead=3, arrowsize=1,
                           arrowwidth=2, arrowcolor='white')
    fig.update_xaxes(range=[0, 1], visible=False)
    fig.update_yaxes(range=[0, 1], visible=False)
    fig.update_layout(template='plotly_dark', margin=dict(l=10, r=10, t=10, b=10), height=420)
    return fig


def make_domain_diagram():
    fig = go.Figure()
    fig.add_shape(type='rect', x0=0, y0=0, x1=1, y1=1,
                  line=dict(color='white', width=3), fillcolor='rgba(255,255,255,0.05)')
    fig.add_annotation(x=0.5, y=0.95, text='2D rectangular domain', showarrow=False,
                       font=dict(size=14, color='white'))
    boundaries = [
        {'x0': 0, 'y0': 0, 'x1': 1, 'y1': 0, 'text': 'y=0', 'pos': (0.5, -0.1)},
        {'x0': 0, 'y0': 1, 'x1': 1, 'y1': 1, 'text': 'y=L', 'pos': (0.5, 1.08)},
        {'x0': 0, 'y0': 0, 'x1': 0, 'y1': 1, 'text': 'x=0', 'pos': (-0.08, 0.5)},
        {'x0': 1, 'y0': 0, 'x1': 1, 'y1': 1, 'text': 'x=L', 'pos': (1.08, 0.5)}
    ]
    for boundary in boundaries:
        fig.add_annotation(x=boundary['pos'][0], y=boundary['pos'][1], text=boundary['text'],
                           showarrow=False, font=dict(color='white'))
    fig.update_xaxes(range=[-0.2, 1.2], visible=False)
    fig.update_yaxes(range=[-0.2, 1.2], visible=False)
    fig.update_layout(template='plotly_dark', margin=dict(l=5, r=5, t=5, b=5), height=380)
    return fig


def make_heat_source_diagram():
    fig = go.Figure()
    fig.add_shape(type='rect', x0=0, y0=0, x1=1, y1=1,
                  line=dict(color='white', width=3), fillcolor='rgba(255,255,255,0.04)')
    fig.add_shape(type='circle', x0=0.35, y0=0.35, x1=0.65, y1=0.65,
                  fillcolor='rgba(255,80,80,0.5)', line=dict(color='red', width=2))
    fig.add_annotation(x=0.5, y=0.8, text='Internal heat source Q(x,y,t)', showarrow=False,
                       font=dict(color='white', size=13))
    arrows = [((0.5, 0.65), (0.7, 0.9)), ((0.5, 0.65), (0.85, 0.5)), ((0.5, 0.65), (0.7, 0.1))]
    for start, end in arrows:
        fig.add_annotation(x=end[0], y=end[1], ax=start[0], ay=start[1], showarrow=True,
                           arrowhead=3, arrowcolor='white')
    fig.update_xaxes(range=[0, 1], visible=False)
    fig.update_yaxes(range=[0, 1], visible=False)
    fig.update_layout(template='plotly_dark', margin=dict(l=10, r=10, t=10, b=10), height=370)
    return fig


def make_boundary_heat_diagram(pair='right+bottom'):
    fig = go.Figure()
    fig.add_shape(type='rect', x0=0, y0=0, x1=1, y1=1,
                  line=dict(color='white', width=3), fillcolor='rgba(255,255,255,0.04)')
    walls = []
    if 'left' in pair:
        walls.append({'x0':0,'y0':0,'x1':0.06,'y1':1})
    if 'right' in pair:
        walls.append({'x0':0.94,'y0':0,'x1':1,'y1':1})
    if 'top' in pair:
        walls.append({'x0':0,'y0':0.94,'x1':1,'y1':1})
    if 'bottom' in pair:
        walls.append({'x0':0,'y0':0,'x1':1,'y1':0.06})
    for wall in walls:
        fig.add_shape(type='rect', x0=wall['x0'], y0=wall['y0'], x1=wall['x1'], y1=wall['y1'],
                      fillcolor='rgba(255,150,0,0.5)', line=dict(color='rgba(255,150,0,0.8)', width=2))
    if walls:
        fig.add_annotation(x=0.5, y=0.9, text='Boundary heating region', showarrow=False,
                           font=dict(color='white', size=13))
    fig.update_xaxes(range=[0, 1], visible=False)
    fig.update_yaxes(range=[0, 1], visible=False)
    fig.update_layout(template='plotly_dark', margin=dict(l=10, r=10, t=10, b=10), height=370)
    return fig


def make_fdm_diagram():
    fig = go.Figure()
    for x in np.linspace(0.1, 0.9, 5):
        fig.add_shape(type='line', x0=x, y0=0.1, x1=x, y1=0.9, line=dict(color='white', width=1))
    for y in np.linspace(0.1, 0.9, 5):
        fig.add_shape(type='line', x0=0.1, y0=y, x1=0.9, y1=y, line=dict(color='white', width=1))
    fig.add_shape(type='circle', x0=0.46, y0=0.46, x1=0.54, y1=0.54,
                  fillcolor='rgba(0,255,255,0.7)', line=dict(color='cyan', width=2))
    offsets = [(-0.12, 0), (0.12, 0), (0, -0.12), (0, 0.12)]
    for dx, dy in offsets:
        fig.add_annotation(x=0.5 + dx, y=0.5 + dy, ax=0.5, ay=0.5,
                           xref='x', yref='y', axref='x', ayref='y', showarrow=True,
                           arrowhead=3, arrowsize=1, arrowcolor='white')
    fig.add_annotation(x=0.5, y=0.95, text='Finite difference stencil', showarrow=False,
                       font=dict(color='white', size=14))
    fig.add_annotation(x=0.5, y=0.03, text='Δx and Δy on a uniform grid', showarrow=False,
                       font=dict(color='white', size=12))
    fig.update_xaxes(range=[0, 1], visible=False)
    fig.update_yaxes(range=[0, 1], visible=False)
    fig.update_layout(template='plotly_dark', margin=dict(l=5, r=5, t=5, b=5), height=420)
    return fig


def make_unet_diagram():
    fig = go.Figure()
    x = [0.1, 0.2, 0.35, 0.5, 0.65, 0.8]
    widths = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
    heights = [0.5, 0.4, 0.3, 0.3, 0.4, 0.5]
    labels = [
        'Input\n(T_t, T_{t-1}, ΔT)',
        'Encoder\nlevel 1',
        'Encoder\nlevel 2',
        'Bottleneck',
        'Decoder\nlevel 2',
        'Output\nφ_t',
    ]
    for i, (cx, h) in enumerate(zip(x, heights)):
        fig.add_shape(type='rect', x0=cx - 0.07, y0=0.5 - h / 2, x1=cx + 0.07, y1=0.5 + h / 2,
                      fillcolor='rgba(80,120,255,0.2)', line=dict(color='white', width=2))
        fig.add_annotation(x=cx, y=0.5, text=labels[i], showarrow=False,
                           font=dict(color='white', size=11))
    for i in range(len(x) - 1):
        fig.add_annotation(x=x[i+1] - 0.07, y=0.5, ax=x[i] + 0.07, ay=0.5,
                           xref='x', yref='y', axref='x', ayref='y', showarrow=True,
                           arrowhead=3, arrowsize=1, arrowcolor='white')
    fig.update_xaxes(range=[0, 1], visible=False)
    fig.update_yaxes(range=[0, 1], visible=False)
    fig.update_layout(template='plotly_dark', margin=dict(l=5, r=5, t=5, b=5), height=420)
    return fig


def build_experiment_dataframe(experiments):
    rows = []
    for exp in experiments:
        metrics = exp.get('metrics', {})
        config = exp.get('config', {})
        rows.append({
            'Branch': exp['branch'],
            'Experiment': exp['name'],
            'MAE': metrics.get('MAE'),
            'RMSE': metrics.get('RMSE'),
            'R2': metrics.get('R2'),
            'MSE': metrics.get('MSE'),
            'Epochs': config.get('epochs'),
            'Model size [MB]': exp.get('model_size_mb'),
            'Dataset': config.get('data_dir'),
            'Source': config.get('task_type'),
        })
    return pd.DataFrame(rows)


def safe_format(value, precision=4):
    if value is None:
        return 'N/A'
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, float):
        return f"{value:.{precision}g}"
    return str(value)
