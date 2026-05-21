"""Solver implementation for the Stefan problem sandbox."""

import numpy as np

from .boundary_conditions import (
    apply_boundary_conditions,
    apply_enthalpy_boundary_conditions
)


def create_grid(length, nx, ny):
    x = np.linspace(0.0, length, nx)
    y = np.linspace(0.0, length, ny)
    X, Y = np.meshgrid(x, y)
    dx = length / (nx - 1)
    dy = length / (ny - 1)
    return x, y, X, Y, dx, dy


def enthalpy_to_temperature(H_field, material):
    rho = material['rho']
    c_p = material['c_p']
    T_m = material['T_m']
    L = material['L']

    H_solid = rho * c_p * T_m
    H_liquid = H_solid + rho * L

    T_field = np.zeros_like(H_field)
    liquid_frac_field = np.zeros_like(H_field)

    mask_solid = H_field <= H_solid
    T_field[mask_solid] = H_field[mask_solid] / (rho * c_p)
    liquid_frac_field[mask_solid] = 0.0

    mask_mushy = (H_field > H_solid) & (H_field < H_liquid)
    T_field[mask_mushy] = T_m
    liquid_frac_field[mask_mushy] = (H_field[mask_mushy] - H_solid) / (rho * L)

    mask_liquid = H_field >= H_liquid
    T_field[mask_liquid] = T_m + (H_field[mask_liquid] - H_liquid) / (rho * c_p)
    liquid_frac_field[mask_liquid] = 1.0

    return T_field, liquid_frac_field


def compute_laplacian(T_field, dx, dy):
    laplacian = np.zeros_like(T_field)
    laplacian[1:-1, 1:-1] = (
        (T_field[2:, 1:-1] - 2.0 * T_field[1:-1, 1:-1] + T_field[:-2, 1:-1]) / dy**2 +
        (T_field[1:-1, 2:] - 2.0 * T_field[1:-1, 1:-1] + T_field[1:-1, :-2]) / dx**2
    )
    return laplacian


def find_interface_position(liquid_frac_field, x, y, centers, threshold=0.5):
    values = np.stack(
        [
            liquid_frac_field[:-1, :-1],
            liquid_frac_field[1:, :-1],
            liquid_frac_field[:-1, 1:],
            liquid_frac_field[1:, 1:]
        ],
        axis=-1
    )
    min_vals = np.min(values, axis=-1)
    max_vals = np.max(values, axis=-1)
    interface_mask = (min_vals < threshold) & (max_vals > threshold)

    if not np.any(interface_mask):
        return None, 0.0

    x_centers = (x[:-1] + x[1:]) / 2.0
    y_centers = (y[:-1] + y[1:]) / 2.0
    XC, YC = np.meshgrid(x_centers, y_centers)
    interface_points = np.column_stack((XC[interface_mask], YC[interface_mask]))

    centers_arr = np.asarray(centers, dtype=float)
    if centers_arr.ndim == 1:
        centers_arr = centers_arr.reshape(1, 2)

    distances = np.min(
        np.sqrt(
            (interface_points[:, 0, np.newaxis] - centers_arr[np.newaxis, :, 0])**2 +
            (interface_points[:, 1, np.newaxis] - centers_arr[np.newaxis, :, 1])**2
        ),
        axis=1
    )
    mean_radius = float(np.mean(distances))

    return interface_points, mean_radius


def compute_stefan_velocity_and_flux(T_field, liquid_frac_field, centers, material, dx, dy):
    rho = material['rho']
    k = material['k']
    L = material['L']

    interface_points, mean_radius = find_interface_position(liquid_frac_field, material['x'], material['y'], centers)
    if interface_points is None or len(interface_points) < 10:
        return 0.0, 0.0, mean_radius

    grad_T_x = np.gradient(T_field, dx, axis=1)
    grad_T_y = np.gradient(T_field, dy, axis=0)
    dT_dn_values = []

    for pt in interface_points[:100]:
        i = int(pt[0] / dx)
        j = int(pt[1] / dy)
        i = np.clip(i, 1, T_field.shape[1] - 2)
        j = np.clip(j, 1, T_field.shape[0] - 2)

        center = np.array(centers[0], dtype=float)
        differences = np.sqrt((center[0] - pt[0])**2 + (center[1] - pt[1])**2)
        dx_rad = pt[0] - center[0]
        dy_rad = pt[1] - center[1]
        norm = np.sqrt(dx_rad**2 + dy_rad**2) + 1e-10
        nx_rad = dx_rad / norm
        ny_rad = dy_rad / norm

        dT_dn = grad_T_x[j, i] * nx_rad + grad_T_y[j, i] * ny_rad
        if dT_dn < 0.0:
            dT_dn_values.append(-dT_dn)

    if len(dT_dn_values) == 0:
        return 0.0, 0.0, mean_radius

    avg_dT_dn = float(np.mean(dT_dn_values))
    v_n = (k / (rho * L)) * avg_dT_dn
    q_interface = k * avg_dT_dn
    return v_n, q_interface, mean_radius


def simulate(
    material,
    bc_type,
    length,
    nx,
    ny,
    dt,
    time_total,
    T_initial,
    Q0,
    sigma,
    source_centers,
    output_every=100,
    random_seed=42
):
    x, y, X, Y, dx, dy = create_grid(length=length, nx=nx, ny=ny)
    material = material.copy()
    material['x'] = x
    material['y'] = y
    Q_map = np.zeros((ny, nx))
    if len(source_centers) > 0:
        from .heat_sources import build_heat_source_map
        Q_map = build_heat_source_map(nx=nx, ny=ny, length=length, Q0=Q0, sigma=sigma, centers=source_centers)

    T = np.ones((ny, nx), dtype=float) * T_initial
    liquid_frac = np.zeros((ny, nx), dtype=float)
    H_solid_ref = material['rho'] * material['c_p'] * T_initial
    H = np.ones((ny, nx), dtype=float) * H_solid_ref

    history_time = []
    history_max_temp = []
    history_radius = []
    history_velocity = []
    history_flux = []
    history_liq_frac = []

    n_steps = int(time_total / dt)
    if n_steps < 1:
        n_steps = 1

    for step in range(n_steps):
        current_time = step * dt
        T_with_bc = apply_boundary_conditions(T.copy(), bc_type, T_initial)
        laplacian = compute_laplacian(T_with_bc, dx, dy)
        dHdt = material['k'] * laplacian + Q_map
        H = H + dt * dHdt
        H = apply_enthalpy_boundary_conditions(H, bc_type, material['rho'], material['c_p'], T_initial)
        T, liquid_frac = enthalpy_to_temperature(H, material)
        T = apply_boundary_conditions(T, bc_type, T_initial)

        if step % output_every == 0 or step == n_steps - 1:
            v_n, q_interface, mean_radius = compute_stefan_velocity_and_flux(
                T, liquid_frac, source_centers, material, dx, dy
            )
            max_temp = float(np.max(T))
            max_liq = float(np.max(liquid_frac))
            history_time.append(current_time)
            history_max_temp.append(max_temp)
            history_radius.append(float(mean_radius * 1000.0))
            history_velocity.append(float(v_n * 1000.0))
            history_flux.append(float(q_interface / 1e6))
            history_liq_frac.append(max_liq)

            yield {
                'step': step,
                'n_steps': n_steps,
                'time': current_time,
                'T': T.copy(),
                'liquid_frac': liquid_frac.copy(),
                'Q_map': Q_map,
                'max_temp': max_temp,
                'mean_radius': float(mean_radius * 1000.0),
                'melt_radius': float(mean_radius * 1000.0),
                'velocity': float(v_n),
                'time_history': history_time.copy(),
                'max_temp_history': history_max_temp.copy(),
                'mean_radius_history': history_radius.copy(),
                'melt_radius_history': history_radius.copy(),
                'velocity_history': history_velocity.copy(),
                'flux_history': history_flux.copy(),
                'liquid_frac_history': history_liq_frac.copy(),
            }


def run_simulation(*args, **kwargs):
    final_state = None
    for state in simulate(*args, **kwargs):
        final_state = state
    return final_state
