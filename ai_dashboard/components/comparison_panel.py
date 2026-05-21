"""GT | Pure | Hybrid — synchronized Plotly animation with equal panels."""
from __future__ import annotations

import numpy as np

from ..animation import AnimationPanel, build_synchronized_animation
from ..panel_layout import figure_dimensions, optimal_grid
from .playback import render_playback_toolbar, show_animated_figure


PHASE_ZMIN = 0.0
PHASE_ZMAX = 1.0


def _phase_panels(
    gt: np.ndarray,
    pure: np.ndarray | None,
    hybrid: np.ndarray | None,
    *,
    include_errors: bool,
) -> list[AnimationPanel]:
    panels = [AnimationPanel(gt, 'Ground Truth', 'Viridis', PHASE_ZMIN, PHASE_ZMAX)]
    if pure is not None:
        panels.append(AnimationPanel(pure, 'Pure AI', 'Viridis', PHASE_ZMIN, PHASE_ZMAX))
    if hybrid is not None:
        panels.append(AnimationPanel(hybrid, 'Hybrid AI', 'Viridis', PHASE_ZMIN, PHASE_ZMAX))
    if include_errors:
        if pure is not None:
            panels.append(AnimationPanel(np.abs(gt - pure), 'Pure |error|', 'Magma', 0.0, 0.35))
        if hybrid is not None:
            panels.append(AnimationPanel(np.abs(gt - hybrid), 'Hybrid |error|', 'Magma', 0.0, 0.35))
    return panels


def render_comparison_panel(
    gt: np.ndarray,
    pure: np.ndarray | None,
    hybrid: np.ndarray | None,
    key_prefix: str = 'cmp',
    *,
    include_errors: bool = False,
    title: str = 'Model comparison',
) -> None:
    n_frames = gt.shape[0]
    panels = _phase_panels(gt, pure, hybrid, include_errors=include_errors)
    n_rows, n_cols = optimal_grid(len(panels))
    fig_w, fig_h = figure_dimensions(n_rows, n_cols)
    duration_ms = render_playback_toolbar(key_prefix, n_frames, title=title)

    fig = build_synchronized_animation(
        panels,
        n_rows=n_rows,
        n_cols=n_cols,
        frame_duration_ms=duration_ms,
        width=fig_w,
        height=fig_h,
        uirevision=f'{key_prefix}-sync',
    )
    show_animated_figure(
        fig,
        caption=f'{len(panels)} equal-size panels · shared color scale [0, 1]',
        show_toolbar=False,
    )


def render_gt_pure_panel(gt: np.ndarray, pure: np.ndarray, key_prefix: str = 'gt_pure') -> None:
    render_comparison_panel(gt, pure, None, key_prefix=key_prefix, title='Ground Truth vs Pure AI')


def render_gt_hybrid_panel(gt: np.ndarray, hybrid: np.ndarray, key_prefix: str = 'gt_hybrid') -> None:
    render_comparison_panel(gt, None, hybrid, key_prefix=key_prefix, title='Ground Truth vs Hybrid')


def render_three_way_panel(
    gt: np.ndarray,
    pure: np.ndarray,
    hybrid: np.ndarray,
    key_prefix: str = 'gt_pure_hybrid',
) -> None:
    render_comparison_panel(gt, pure, hybrid, key_prefix=key_prefix, title='GT · Pure AI · Hybrid')


def render_error_panel(
    gt: np.ndarray,
    pure: np.ndarray | None,
    hybrid: np.ndarray | None,
    key_prefix: str = 'errors',
) -> None:
    render_comparison_panel(gt, pure, hybrid, key_prefix=key_prefix, include_errors=True, title='Error analysis')
