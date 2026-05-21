"""AI comparison package for the Stefan problem sandbox."""

from .ai_comparison import render_ai_comparison_tab
from .experiment_loader import find_experiment_directories, load_experiment
from .playback_controller import render_playback_controls
from .error_analysis import compute_error_maps, compute_temporal_metrics, compute_improvement_metrics
from .metrics_dashboard import (
    create_metric_cards,
    create_metric_comparison_figure,
    create_error_histogram,
    create_scatter_comparison_figure,
    create_correlation_figure,
    create_improvement_figure,
    create_training_history_figure
)
from .visualization import (
    create_comparison_heatmap_figure,
    create_three_way_comparison_figure,
    create_single_heatmap_figure
)
