import io
from datetime import datetime

import imageio
import numpy as np
import plotly.graph_objects as go
import streamlit as st

from simulation import boundary_conditions, heat_sources, materials, solver, visualization


def format_source_centers(centers):
    return [f"({cx * 1000:.1f}, {cy * 1000:.1f}) mm" for cx, cy in centers]


def create_downloadable_npz(results):
    buffer = io.BytesIO()

    np.savez(
        buffer,
        bc_type=results['bc_type'],
        T_final=results['T_final'],
        liquid_frac_final=results['liquid_frac_final'],
        Q_map=results['Q_map'],
        time=results['time'],
        max_temp=results['max_temp'],
        melt_radius=results['melt_radius'],
        velocity=results['velocity'],
        liq_frac_max=results['liq_frac_max'],
        source_centers=np.array(results['source_info']['centers']),
        sigma=results['source_info']['sigma'],
        Q0=results['source_info']['Q0'],
        material_name=results['source_info']['material_name'],
        bc_type_label=results['bc_type'],
        params=results['params']
    )

    buffer.seek(0)
    return buffer


def create_plot_png(fig):
    return fig.to_image(format='png', engine='kaleido')


def create_gif(frames, x, y, title, duration=0.15):
    images = []

    for temperature in frames:
        fig = visualization.create_temperature_figure(
            temperature,
            x,
            y,
            title=title,
            zmin=np.min(temperature),
            zmax=np.max(temperature)
        )

        img_bytes = fig.to_image(format='png', engine='kaleido')
        images.append(imageio.v2.imread(io.BytesIO(img_bytes)))

    gif_buffer = io.BytesIO()

    imageio.mimsave(
        gif_buffer,
        images,
        format='GIF',
        duration=duration
    )

    gif_buffer.seek(0)
    return gif_buffer


def main():
    st.set_page_config(
        page_title='Stefan Problem Simulation Sandbox',
        layout='wide',
        initial_sidebar_state='expanded'
    )

    st.title('🔬 Stefan Problem Simulation Sandbox')

    st.markdown(
        'Interactive scientific sandbox for the 2D one-phase Stefan problem '
        'with enthalpy formulation, Dirichlet/Neumann boundary conditions, '
        'and Gaussian internal heating.'
    )

    sidebar = st.sidebar
    sidebar.title('Simulation Controls')

    material_name = sidebar.selectbox(
        'Material selection',
        list(materials.PREDEFINED_MATERIALS.keys()),
        index=list(materials.PREDEFINED_MATERIALS.keys()).index('Lead')
    )

    material = materials.PREDEFINED_MATERIALS[material_name]

    bc_type = sidebar.selectbox(
        'Boundary condition',
        ['dirichlet', 'neumann']
    )

    sidebar.markdown('---')

    sidebar.subheader('Domain & grid')

    length_mm = sidebar.slider(
        'Domain size [mm]',
        10,
        100,
        50,
        step=1
    )

    length = length_mm / 1000.0

    nx = sidebar.slider(
        'Grid points (nx)',
        40,
        200,
        100,
        step=10
    )

    ny = sidebar.slider(
        'Grid points (ny)',
        40,
        200,
        100,
        step=10
    )

    sidebar.subheader('Time settings')

    dt = sidebar.number_input(
        'Time step dt [s]',
        min_value=1e-5,
        max_value=1e-2,
        value=5e-4,
        format='%.6f'
    )

    time_total = sidebar.number_input(
        'Total simulation time [s]',
        min_value=0.1,
        max_value=10.0,
        value=4.0,
        step=0.1
    )

    output_interval = sidebar.slider(
        'Visualization update every N steps',
        1,
        200,
        100,
        step=1
    )

    sidebar.subheader('Initial condition')

    T_initial = sidebar.number_input(
        'Initial temperature [°C]',
        min_value=-50.0,
        max_value=150.0,
        value=25.0,
        step=1.0
    )

    sidebar.markdown('---')

    sidebar.subheader('Heat source')

    Q0 = sidebar.number_input(
        'Peak source strength Q0 [W/m³]',
        min_value=1e7,
        max_value=2e9,
        value=5e8,
        format='%.0f',
        step=1e7
    )

    sigma_mm = sidebar.slider(
        'Gaussian width σ [mm]',
        1.0,
        15.0,
        6.0,
        step=0.5
    )

    sigma = sigma_mm / 1000.0

    random_position = sidebar.checkbox(
        'Random source position',
        value=True
    )

    num_sources = sidebar.slider(
        'Number of sources',
        1,
        3,
        1
    )

    seed = sidebar.number_input(
        'Random source seed',
        min_value=0,
        max_value=9999,
        value=42,
        step=1
    )

    manual_centers = []

    if not random_position:
        sidebar.markdown('**Manual source positions [mm]**')

        for idx in range(num_sources):
            cols = sidebar.columns(2)

            x_val = cols[0].number_input(
                f'Source {idx + 1} X [mm]',
                min_value=1.0,
                max_value=float(length_mm - 1.0),
                value=float(length_mm / 2),
                step=1.0,
                key=f'x_{idx}'
            )

            y_val = cols[1].number_input(
                f'Source {idx + 1} Y [mm]',
                min_value=1.0,
                max_value=float(length_mm - 1.0),
                value=float(length_mm / 2),
                step=1.0,
                key=f'y_{idx}'
            )

            manual_centers.append(
                (x_val / 1000.0, y_val / 1000.0)
            )

    sidebar.markdown('---')

    run_button = sidebar.button('Run simulation')

    sidebar.info(
        'Run the simulation to see live field plots, metrics and export options.'
    )

    if 'simulation_result' not in st.session_state:
        st.session_state.simulation_result = None
        st.session_state.animation_frames = []

    placeholder_status = st.empty()
    placeholder_progress = st.sidebar.empty()

    metrics_col1, metrics_col2, metrics_col3 = st.columns(3)

    placeholder_time = metrics_col1.empty()
    placeholder_temp = metrics_col2.empty()
    placeholder_radius = metrics_col3.empty()

    placeholder_velocity = st.empty()

    tab1, tab2, tab3, tab4 = st.tabs(
        [
            'Temperature',
            'Liquid fraction',
            'Heat source',
            'Metrics'
        ]
    )

    temp_plot = tab1.empty()
    liquid_plot = tab2.empty()
    source_plot = tab3.empty()
    metrics_plot = tab4.empty()

    if run_button:
        source_centers = []

        if random_position:
            source_centers = heat_sources.random_source_centers(
                length=length,
                num_sources=num_sources,
                margin=0.008,
                seed=int(seed)
            )

        else:
            source_centers = manual_centers or [
                (length / 2.0, length / 2.0)
            ]

        x, y, X, Y, dx, dy = solver.create_grid(
            length=length,
            nx=nx,
            ny=ny
        )

        Q_map = heat_sources.build_heat_source_map(
            nx=nx,
            ny=ny,
            length=length,
            Q0=Q0,
            sigma=sigma,
            centers=source_centers
        )

        status_text = 'Simulation starting...'

        placeholder_status.info(status_text)

        progress_bar = placeholder_progress.progress(0)

        st.session_state.animation_frames = []

        simulation = solver.simulate(
            material=material,
            bc_type=bc_type,
            length=length,
            nx=nx,
            ny=ny,
            dt=dt,
            time_total=time_total,
            T_initial=T_initial,
            Q0=Q0,
            sigma=sigma,
            source_centers=source_centers,
            output_every=output_interval,
            random_seed=int(seed)
        )

        final_result = None
        frame_limit = 30

        for state in simulation:
            step = state['step']
            current_time = state['time']
            T_field = state['T']
            liquid_frac = state['liquid_frac']
            max_temp = state['max_temp']

            melt_radius = state.get(
                'mean_radius',
                state.get('melt_radius', 0.0)
            )

            interface_velocity = state['velocity']

            progress = int(
                (step + 1) / max(1, state['n_steps']) * 100
            )

            placeholder_status.info(
                f'Step {step + 1} / {state["n_steps"]} — '
                f'Time = {current_time:.3f} s'
            )

            progress_bar.progress(progress)

            placeholder_time.metric(
                'Time [s]',
                f'{current_time:.3f}'
            )

            placeholder_temp.metric(
                'Max temperature [°C]',
                f'{max_temp:.2f}'
            )

            placeholder_radius.metric(
                'Mean melt radius [mm]',
                f'{melt_radius:.2f}'
            )

            placeholder_velocity.metric(
                'Interface velocity [mm/s]',
                f'{interface_velocity * 1000:.3f}'
            )

            with tab1:
                temp_fig = visualization.create_temperature_figure(
                    T_field,
                    x,
                    y,
                    title=f'Temperature field at t = {current_time:.3f} s',
                    zmin=T_initial,
                    zmax=max(max_temp, T_initial + 1)
                )

                temp_plot.plotly_chart(
                    temp_fig,
                    use_container_width=True,
                    key=f'temp_plot_{step}'
                )

            with tab2:
                liq_fig = visualization.create_liquid_fraction_figure(
                    liquid_frac,
                    x,
                    y,
                    title=f'Liquid fraction at t = {current_time:.3f} s'
                )

                liquid_plot.plotly_chart(
                    liq_fig,
                    use_container_width=True,
                    key=f'liq_plot_{step}'
                )

            with tab3:
                source_fig = visualization.create_heat_source_figure(
                    Q_map,
                    x,
                    y,
                    source_centers,
                    sigma,
                    title='Internal heat source distribution'
                )

                source_plot.plotly_chart(
                    source_fig,
                    use_container_width=True,
                    key=f'source_plot_{step}'
                )

            with tab4:
                metrics_fig = visualization.create_multiple_metrics_figure(
                    state['time_history'],
                    state['max_temp_history'],
                    state['mean_radius_history'],
                    state['velocity_history'],
                    T_m=material['T_m']
                )

                metrics_plot.plotly_chart(
                    metrics_fig,
                    use_container_width=True,
                    key=f'metrics_plot_{step}'
                )

            if len(st.session_state.animation_frames) < frame_limit:
                st.session_state.animation_frames.append(
                    T_field.copy()
                )

            final_result = state

        if final_result is not None:
            st.session_state.simulation_result = {
                'bc_type': bc_type,
                'T_final': final_result['T'],
                'liquid_frac_final': final_result['liquid_frac'],
                'Q_map': Q_map,
                'time': np.array(final_result['time_history']),
                'max_temp': np.array(final_result['max_temp_history']),
                'melt_radius': np.array(final_result['mean_radius_history']),
                'velocity': np.array(final_result['velocity_history']),
                'liq_frac_max': np.array(final_result['liquid_frac_history']),
                'source_info': {
                    'centers': source_centers,
                    'sigma': sigma,
                    'Q0': Q0,
                    'material_name': material_name
                },
                'params': {
                    'nx': nx,
                    'ny': ny,
                    'length': length,
                    'dt': dt,
                    'time_total': time_total,
                    'T_initial': T_initial,
                    'T_m': material['T_m'],
                    'rho': material['rho'],
                    'c_p': material['c_p'],
                    'k': material['k'],
                    'L': material['L']
                }
            }

            progress_bar.progress(100)

            placeholder_status.success(
                'Simulation completed successfully.'
            )

    if st.session_state.simulation_result is not None:
        st.sidebar.markdown('---')

        st.sidebar.subheader('Export results')

        buffer_npz = create_downloadable_npz(
            st.session_state.simulation_result
        )

        sidebar.download_button(
            label='Download results (.npz)',
            data=buffer_npz,
            file_name=f'stefan_simulation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.npz',
            mime='application/octet-stream'
        )

        if st.button('Download last temperature plot (.png)'):
            fig = visualization.create_temperature_figure(
                st.session_state.simulation_result['T_final'],
                np.linspace(
                    0,
                    st.session_state.simulation_result['params']['length'],
                    st.session_state.simulation_result['params']['nx']
                ),
                np.linspace(
                    0,
                    st.session_state.simulation_result['params']['length'],
                    st.session_state.simulation_result['params']['ny']
                ),
                title='Final temperature field',
                zmin=st.session_state.simulation_result['params']['T_initial'],
                zmax=np.max(
                    st.session_state.simulation_result['T_final']
                )
            )

            png_bytes = create_plot_png(fig)

            st.download_button(
                label='Download PNG file',
                data=png_bytes,
                file_name=f'stefan_temperature_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png',
                mime='image/png'
            )

        if st.button('Download animation GIF'):
            if st.session_state.animation_frames:
                gif_buffer = create_gif(
                    st.session_state.animation_frames,
                    np.linspace(
                        0,
                        st.session_state.simulation_result['params']['length'],
                        st.session_state.simulation_result['params']['nx']
                    ),
                    np.linspace(
                        0,
                        st.session_state.simulation_result['params']['length'],
                        st.session_state.simulation_result['params']['ny']
                    ),
                    title='Temperature evolution'
                )

                st.download_button(
                    label='Download GIF file',
                    data=gif_buffer,
                    file_name=f'stefan_animation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.gif',
                    mime='image/gif'
                )

            else:
                st.warning(
                    'No animation frames were generated during the simulation.'
                )

    st.sidebar.markdown('---')

    st.sidebar.write('Run instructions:')

    st.sidebar.code('pip install -r requirements.txt')
    st.sidebar.code('streamlit run streamlit_app.py')


if __name__ == '__main__':
    main()