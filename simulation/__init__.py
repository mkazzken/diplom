"""Simulation package for the Stefan problem sandbox."""

from .materials import PREDEFINED_MATERIALS
from .heat_sources import build_heat_source_map, random_source_centers
from .boundary_conditions import (
    apply_boundary_conditions,
    apply_enthalpy_boundary_conditions
)
from .solver import run_simulation, simulate, create_grid
from .visualization import (
    create_temperature_figure,
    create_liquid_fraction_figure,
    create_heat_source_figure,
    create_multiple_metrics_figure
)
