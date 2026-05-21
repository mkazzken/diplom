"""Automatic experiment discovery for Stefan surrogate runs."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from .paths import EXPERIMENTS_ROOT, PROJECT_ROOT

TIMESTAMP_RE = re.compile(r'(\d{8}_\d{6})')


@dataclass
class ExperimentRun:
    path: Path
    name: str
    kind: str  # pure_ai | hybrid
    timestamp: str | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    config: dict[str, Any] = field(default_factory=dict)
    history: dict[str, Any] = field(default_factory=dict)
    artifacts: list[str] = field(default_factory=list)

    @property
    def mtime(self) -> float:
        return self.path.stat().st_mtime

    def artifact(self, *names: str) -> Path | None:
        for name in names:
            candidate = self.path / name
            if candidate.is_file():
                return candidate
        return None


def _extract_timestamp(name: str) -> str | None:
    match = TIMESTAMP_RE.search(name)
    return match.group(1) if match else None


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        with path.open(encoding='utf-8') as handle:
            return json.load(handle)
    except (json.JSONDecodeError, OSError):
        return {}


def _classify_run_dir(path: Path) -> str | None:
    name = path.name.lower()
    if name.startswith('run_'):
        return 'pure_ai'
    if 'hybrid' in name or 'inspired_surrogate' in name:
        return 'hybrid'
    return None


def _list_artifacts(path: Path) -> list[str]:
    extensions = {'.png', '.gif', '.pt', '.json', '.npz'}
    return sorted(
        item.name for item in path.iterdir()
        if item.is_file() and item.suffix.lower() in extensions
    )


def scan_experiments(root: Path | None = None) -> list[ExperimentRun]:
    root = root or EXPERIMENTS_ROOT
    runs: list[ExperimentRun] = []

    if not root.is_dir():
        return runs

    for entry in sorted(root.iterdir()):
        if not entry.is_dir():
            continue
        kind = _classify_run_dir(entry)
        if kind is None:
            continue
        if not (entry / 'model_best.pt').is_file():
            continue
        runs.append(
            ExperimentRun(
                path=entry,
                name=entry.name,
                kind=kind,
                timestamp=_extract_timestamp(entry.name),
                metrics=_load_json(entry / 'metrics.json'),
                config=_load_json(entry / 'config.json'),
                history=_load_json(entry / 'history.json'),
                artifacts=_list_artifacts(entry),
            )
        )

    runs.sort(key=lambda run: run.mtime, reverse=True)
    return runs


def latest_by_kind(runs: list[ExperimentRun], kind: str) -> ExperimentRun | None:
    filtered = [run for run in runs if run.kind == kind]
    return filtered[0] if filtered else None


def discover_dataset_samples(root: Path | None = None) -> dict[str, list[Path]]:
    root = root or (PROJECT_ROOT / 'dataset')
    grouped: dict[str, list[Path]] = {}
    if not root.is_dir():
        return grouped

    for bc_dir in sorted(root.iterdir()):
        if not bc_dir.is_dir():
            continue
        files = sorted(bc_dir.glob('*.npz'))
        if files:
            grouped[bc_dir.name] = files
    return grouped


def format_timestamp(timestamp: str | None) -> str:
    if not timestamp:
        return '—'
    try:
        return datetime.strptime(timestamp, '%Y%m%d_%H%M%S').strftime('%Y-%m-%d %H:%M')
    except ValueError:
        return timestamp
