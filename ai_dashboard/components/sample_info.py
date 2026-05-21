"""Display physical simulation metadata for the selected dataset sample."""
from __future__ import annotations

from typing import Any

import streamlit as st


def render_sample_info(info: dict[str, Any]) -> None:
    st.markdown('#### Selected sample')
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)

    st.markdown(f"**`{info['sample_id']}`**")

    row1 = st.columns(4)
    row1[0].metric('Boundary condition', info['bc_type'])
    row1[1].metric('Total simulation time', f"{info['time_total_s']:.3f} s")
    row1[2].metric('Frames (dataset)', info['n_frames_raw'])
    row1[3].metric('Frames (inference)', info['n_frames_aligned'])

    row2 = st.columns(4)
    row2[0].metric('Time span', f"{info['time_start_s']:.3f} – {info['time_end_s']:.3f} s")
    row2[1].metric('Step Δt (approx.)', f"{info['dt_approx_s']:.3f} s")
    row2[2].metric('Aligned window', f"{info['aligned_time_start_s']:.3f} – {info['aligned_time_end_s']:.3f} s")
    row2[3].metric('Inference from frame', info['inference_start_index'] + 1)

    st.markdown('**Heat source (Gaussian internal heating)**')
    row3 = st.columns(4)
    row3[0].metric('Peak strength Q₀', f"{info['Q0_W_m3']:.3e} W/m³")
    row3[1].metric('Width σ', f"{info['sigma_mm']:.2f} mm")
    row3[2].metric('Center X', f"{info['center_x_mm']:.2f} mm")
    row3[3].metric('Center Y', f"{info['center_y_mm']:.2f} mm")

    st.markdown('**Domain & fields**')
    row4 = st.columns(4)
    row4[0].metric('Domain size', f"{info['domain_mm']:.1f} × {info['domain_mm']:.1f} mm")
    row4[1].metric('Grid', info['grid'])
    row4[2].metric('Cell size Δx', f"{info['dx_mm']:.3f} mm")
    row4[3].metric('T_norm range', f"{info['T_norm_min']:.3f} – {info['T_norm_max']:.3f}")

    row5 = st.columns(2)
    row5[0].metric('Max liquid fraction (GT)', f"{info['phase_max']:.4f}")
    row5[1].caption(
        'Normalized temperature `T_norm` and liquid fraction `liquid_frac` are loaded from the dataset. '
        'Inference uses the same temporal inputs as training (Tₜ, Tₜ₋₁, ΔT when enabled).'
    )

    st.markdown('</div>', unsafe_allow_html=True)
