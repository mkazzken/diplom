"""Resolve project paths for the AI visualization dashboard."""
from __future__ import annotations

from pathlib import Path

APP_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = APP_ROOT.parent
EXPERIMENTS_ROOT = PROJECT_ROOT / 'experiments'
DATASET_ROOT = PROJECT_ROOT / 'dataset'
STYLES_DIR = APP_ROOT / 'styles'
