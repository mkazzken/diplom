import numpy as np
import matplotlib.pyplot as plt
import warnings
import os
from datetime import datetime

warnings.filterwarnings('ignore')

# ============================================================
# TWO-DIMENSIONAL ONE-PHASE STEFAN PROBLEM
# COMPARISON OF DIRICHLET vs NEUMANN BOUNDARY CONDITIONS
# ============================================================
# ENTHALPY FORMULATION: ∂H/∂t = k∇²T + Q(x,y)
# Stefan condition: ρL v_n = -k ∂T/∂n on Γ(t)
# Enthalpy-porosity method for fixed grid
# ============================================================

# ============================================================
# SIMULATION MODE SELECTION
# ============================================================
# Choose boundary condition type:
#   "dirichlet" - fixed wall temperature (heat loss to environment)
#   "neumann"   - insulated walls (heat is trapped inside)
BC_TYPE = "dirichlet"   # CHANGE THIS TO "neumann" FOR COMPARISON

# ============================================================
# MATERIAL PROPERTIES (Lead - Pb)
# ============================================================
rho = 11340.0          # density [kg/m³]
c_p = 128.0            # specific heat capacity [J/(kg·K)]
k = 35.0               # thermal conductivity [W/(m·K)]
T_m = 327.5            # melting temperature [°C]
L = 23000.0            # latent heat of fusion [J/kg]

# ============================================================
# COMPUTATIONAL DOMAIN
# ============================================================
length = 0.05          # domain size [m] (50 mm)
nx = 100               # number of grid points in x-direction
ny = 100               # number of grid points in y-direction

dx = length / (nx - 1)
dy = length / (ny - 1)

# Create grid
x = np.linspace(0, length, nx)
y = np.linspace(0, length, ny)
X, Y = np.meshgrid(x, y)

# ============================================================
# SIMULATION PARAMETERS
# ============================================================
time_total = 4.0      # total simulation time [s]
dt = 0.0005            # time step [s]

n_steps = int(time_total / dt)
print_every = 100      # print progress every N steps

# ============================================================
# INTERNAL HEAT SOURCE — ONE LOCALIZED GAUSSIAN AT RANDOM POSITION
# ============================================================
np.random.seed(42)          # фиксируем seed для воспроизводимости

# Уменьшенная мощность для реалистичного перегрева (было 1e9 -> 5e8)
Q0 = 5.0e8                  # пиковая мощность [W/m³]
sigma = 0.006               # ширина пятна [m] (6 мм)

# Случайные координаты источника (с отступом от края)
margin = 0.008              # отступ от края 8 мм
center_x = np.random.uniform(margin, length - margin)
center_y = np.random.uniform(margin, length - margin)

Q_map = np.zeros((ny, nx))

for i in range(nx):
    for j in range(ny):
        r2 = (x[i] - center_x)**2 + (y[j] - center_y)**2
        Q_map[j, i] = Q0 * np.exp(-r2 / (2 * sigma**2))

# ============================================================
# INITIAL CONDITION
# ============================================================
T_initial = 25.0       # [°C] - well below melting point

# ============================================================
# BOUNDARY CONDITIONS FUNCTIONS
# ============================================================

def apply_dirichlet_bc(T_field):
    """
    Dirichlet boundary condition: fixed temperature at walls.
    Physical meaning: The walls are maintained at constant temperature (T_initial).
    Heat can escape from the domain to the environment.
    This represents a cooled/controlled boundary.
    """
    T_field[0, :] = T_initial      # top wall
    T_field[-1, :] = T_initial     # bottom wall
    T_field[:, 0] = T_initial      # left wall
    T_field[:, -1] = T_initial     # right wall
    return T_field

def apply_neumann_bc(T_field):
    """
    Neumann boundary condition: zero heat flux (insulated walls).
    ∂T/∂n = 0 at all boundaries.
    Physical meaning: Walls are perfectly insulated. Heat cannot escape.
    Any heat generated stays inside the domain.
    This represents an adiabatic/isolated boundary.
    
    Numerical implementation: The temperature at the boundary is set equal
    to the temperature of the adjacent interior cell.
    """
    T_field[0, :] = T_field[1, :]      # top
    T_field[-1, :] = T_field[-2, :]    # bottom
    T_field[:, 0] = T_field[:, 1]      # left
    T_field[:, -1] = T_field[:, -2]    # right
    return T_field

def apply_boundary_conditions(T_field, bc_type):
    """Apply the selected boundary condition type."""
    if bc_type == "dirichlet":
        return apply_dirichlet_bc(T_field)
    elif bc_type == "neumann":
        return apply_neumann_bc(T_field)
    else:
        raise ValueError(f"Unknown bc_type: {bc_type}. Use 'dirichlet' or 'neumann'")

def apply_enthalpy_neumann_bc(H_field):
    """
    Apply Neumann (zero-gradient) boundary conditions to enthalpy field.
    ∂H/∂n = 0 at all boundaries.
    This is physically consistent with insulated walls.
    """
    H_field[0, :] = H_field[1, :]      # top
    H_field[-1, :] = H_field[-2, :]    # bottom
    H_field[:, 0] = H_field[:, 1]      # left
    H_field[:, -1] = H_field[:, -2]    # right
    return H_field

def apply_enthalpy_dirichlet_bc(H_field):
    """
    Apply Dirichlet boundary conditions to enthalpy field.
    Fixed enthalpy corresponding to T_initial at walls.
    """
    H_boundary = rho * c_p * T_initial
    H_field[0, :] = H_boundary
    H_field[-1, :] = H_boundary
    H_field[:, 0] = H_boundary
    H_field[:, -1] = H_boundary
    return H_field

def apply_enthalpy_boundary_conditions(H_field, bc_type):
    """Apply boundary conditions to enthalpy field based on BC type."""
    if bc_type == "dirichlet":
        return apply_enthalpy_dirichlet_bc(H_field)
    elif bc_type == "neumann":
        return apply_enthalpy_neumann_bc(H_field)
    else:
        raise ValueError(f"Unknown bc_type: {bc_type}")

# ============================================================
# ENTHALPY METHOD FUNCTIONS
# ============================================================

def enthalpy_to_temperature(H_field):
    """
    Convert enthalpy field to temperature and liquid fraction.
    
    H_solid = ρc_p T_m (enthalpy at melting point, fully solid)
    H_liquid = ρc_p T_m + ρL (enthalpy at melting point, fully liquid)
    """
    H_solid = rho * c_p * T_m
    H_liquid = rho * c_p * T_m + rho * L
    
    T_field = np.zeros_like(H_field)
    liquid_frac_field = np.zeros_like(H_field)
    
    # Solid region (H <= H_solid)
    mask_solid = H_field <= H_solid
    T_field[mask_solid] = H_field[mask_solid] / (rho * c_p)
    liquid_frac_field[mask_solid] = 0.0
    
    # Mushy/phase-change region (H_solid < H < H_liquid)
    mask_mushy = (H_field > H_solid) & (H_field < H_liquid)
    T_field[mask_mushy] = T_m
    liquid_frac_field[mask_mushy] = (H_field[mask_mushy] - H_solid) / (rho * L)
    
    # Liquid region (H >= H_liquid)
    mask_liquid = H_field >= H_liquid
    T_field[mask_liquid] = T_m + (H_field[mask_liquid] - H_liquid) / (rho * c_p)
    liquid_frac_field[mask_liquid] = 1.0
    
    return T_field, liquid_frac_field

def compute_laplacian(T_field):
    """Compute Laplacian using second-order central differences."""
    laplacian = np.zeros_like(T_field)
    
    laplacian[1:-1, 1:-1] = (
        (T_field[2:, 1:-1] - 2*T_field[1:-1, 1:-1] + T_field[:-2, 1:-1]) / dy**2 +
        (T_field[1:-1, 2:] - 2*T_field[1:-1, 1:-1] + T_field[1:-1, :-2]) / dx**2
    )
    
    return laplacian

def find_interface_position(liquid_frac_field, threshold=0.5):
    """Find coordinates of the melting front where liquid_fraction = threshold."""
    interface_points = []
    
    for i in range(nx-1):
        for j in range(ny-1):
            values = np.array([
                liquid_frac_field[j, i],
                liquid_frac_field[j, i+1],
                liquid_frac_field[j+1, i],
                liquid_frac_field[j+1, i+1]
            ])
            
            if np.min(values) < threshold < np.max(values):
                interface_points.append((
                    (x[i] + x[i+1]) / 2,
                    (y[j] + y[j+1]) / 2
                ))
    
    interface_points = np.array(interface_points) if interface_points else None
    
    if interface_points is not None and len(interface_points) > 0:
        radii = np.sqrt((interface_points[:,0] - center_x)**2 + 
                        (interface_points[:,1] - center_y)**2)
        mean_radius = np.mean(radii)
    else:
        mean_radius = 0.0
    
    return interface_points, mean_radius

def compute_stefan_velocity_and_flux(T_field, liquid_frac_field):
    """Compute interface velocity from the Stefan condition."""
    interface_points, mean_radius = find_interface_position(liquid_frac_field, threshold=0.5)
    
    if interface_points is None or len(interface_points) < 10:
        return 0.0, 0.0, mean_radius
    
    grad_T_x = np.gradient(T_field, dx, axis=1)
    grad_T_y = np.gradient(T_field, dy, axis=0)
    
    dT_dn_values = []
    
    for pt in interface_points[:100]:
        i = int(pt[0] / dx)
        j = int(pt[1] / dy)
        i = np.clip(i, 1, nx-2)
        j = np.clip(j, 1, ny-2)
        
        dx_rad = pt[0] - center_x
        dy_rad = pt[1] - center_y
        norm = np.sqrt(dx_rad**2 + dy_rad**2) + 1e-10
        nx_rad = dx_rad / norm
        ny_rad = dy_rad / norm
        
        dT_dn = grad_T_x[j, i] * nx_rad + grad_T_y[j, i] * ny_rad
        
        if dT_dn < 0:
            dT_dn_values.append(-dT_dn)
    
    if len(dT_dn_values) == 0:
        return 0.0, 0.0, mean_radius
    
    avg_dT_dn = np.mean(dT_dn_values)
    v_n = (k / (rho * L)) * avg_dT_dn
    q_interface = k * avg_dT_dn
    
    return v_n, q_interface, mean_radius

# ============================================================
# RUN SIMULATION FUNCTION
# ============================================================

def run_simulation(bc_type, save_results=True):
    """
    Run the Stefan problem simulation with specified boundary conditions.
    
    Parameters:
    -----------
    bc_type : str
        "dirichlet" or "neumann"
    save_results : bool
        Whether to save results to .npz file
    
    Returns:
    --------
    dict with all simulation results
    """
    print("=" * 70)
    print(f"RUNNING SIMULATION WITH {bc_type.upper()} BOUNDARY CONDITIONS")
    print("=" * 70)
    print(f"BC Type: {bc_type.upper()}")
    if bc_type == "dirichlet":
        print("Physical meaning: Walls at fixed temperature T = T_initial = 25°C")
        print("Effect: Heat escapes from the domain → cooling")
    else:
        print("Physical meaning: Insulated walls (∂T/∂n = 0)")
        print("Effect: Heat is trapped inside → higher temperatures")
    print(f"Heat source: Q0 = {Q0:.2e} W/m³, σ = {sigma*1000:.1f} mm")
    print(f"Source center: ({center_x*1000:.1f}, {center_y*1000:.1f}) mm")
    print("=" * 70)
    
    # Initialize fields
    T = np.ones((ny, nx)) * T_initial
    liquid_frac = np.zeros((ny, nx))
    H_solid_ref = rho * c_p * T_initial
    H = np.ones((ny, nx)) * H_solid_ref
    
    # Storage for history
    history_time = []
    history_max_temp = []
    history_radius = []
    history_velocity = []
    history_liq_frac_max = []
    
    print(f"\n{'Time [s]':>10} | {'Max T [°C]':>12} | {'Melt rad [mm]':>13} | {'v_n [mm/s]':>12} | {'Flux [MW/m²]':>13} | {'Liquid frac max':>15}")
    print("-" * 85)
    
    for step in range(n_steps):
        current_time = step * dt
        
        # Apply boundary conditions to temperature
        T_with_bc = T.copy()
        T_with_bc = apply_boundary_conditions(T_with_bc, bc_type)
        
        # Compute Laplacian
        laplacian = compute_laplacian(T_with_bc)
        
        # Update enthalpy: H^{n+1} = H^n + dt * (k∇²T + Q)
        dHdt = k * laplacian + Q_map
        H = H + dt * dHdt
        
        # Apply boundary conditions to enthalpy
        H = apply_enthalpy_boundary_conditions(H, bc_type)
        
        # Convert enthalpy to temperature and liquid fraction
        T, liquid_frac = enthalpy_to_temperature(H)
        
        # Apply boundary conditions to temperature again (ensure consistency)
        T = apply_boundary_conditions(T, bc_type)
        
        # Store history every print_every steps
        if step % print_every == 0 or step == n_steps - 1:
            v_n, q_interface, mean_radius = compute_stefan_velocity_and_flux(T, liquid_frac)
            
            v_n_mms = v_n * 1000
            mean_radius_mm = mean_radius * 1000
            q_mwm2 = q_interface / 1e6
            
            max_temp = np.max(T)
            max_liq = np.max(liquid_frac)
            
            history_time.append(current_time)
            history_max_temp.append(max_temp)
            history_radius.append(mean_radius_mm)
            history_velocity.append(v_n_mms)
            history_liq_frac_max.append(max_liq)
            
            print(f"{current_time:10.3f} | {max_temp:12.1f} | {mean_radius_mm:13.2f} | {v_n_mms:12.3f} | {q_mwm2:13.2f} | {max_liq:15.3f}")
    
    print("\n" + "=" * 70)
    print(f"SIMULATION COMPLETED with {bc_type.upper()} BC")
    print(f"Final maximum temperature: {np.max(T):.1f} °C")
    print(f"Final maximum liquid fraction: {np.max(liquid_frac):.3f}")
    print(f"Final melt radius from source: {history_radius[-1] if history_radius else 0:.2f} mm")
    print("=" * 70)
    
    # Prepare results
    results = {
        'bc_type': bc_type,
        'T_final': T,
        'liquid_frac_final': liquid_frac,
        'Q_map': Q_map,
        'time': np.array(history_time),
        'max_temp': np.array(history_max_temp),
        'melt_radius': np.array(history_radius),
        'velocity': np.array(history_velocity),
        'liq_frac_max': np.array(history_liq_frac_max),
        'source_info': {
            'center_x': center_x,
            'center_y': center_y,
            'sigma': sigma,
            'Q0': Q0
        },
        'params': {
            'nx': nx, 'ny': ny, 'length': length,
            'dt': dt, 'time_total': time_total,
            'T_initial': T_initial, 'T_m': T_m,
            'rho': rho, 'c_p': c_p, 'k': k, 'L': L
        }
    }
    
    # Save results if requested
    if save_results:
        filename = f"stefan_results_{bc_type}.npz"
        np.savez(filename,
                 bc_type=bc_type,
                 T_final=T,
                 liquid_frac_final=liquid_frac,
                 Q_map=Q_map,
                 time=np.array(history_time),
                 max_temp=np.array(history_max_temp),
                 melt_radius=np.array(history_radius),
                 velocity=np.array(history_velocity),
                 liq_frac_max=np.array(history_liq_frac_max),
                 center_x=center_x,
                 center_y=center_y,
                 sigma=sigma,
                 Q0=Q0,
                 nx=nx, ny=ny, length=length,
                 dt=dt, time_total=time_total,
                 T_initial=T_initial, T_m=T_m,
                 rho=rho, c_p=c_p, k=k, L=L)
        print(f"\nResults saved to {filename}")
    
    return results

# ============================================================
# COMPARISON PLOTTING FUNCTION
# ============================================================

def plot_comparison(results_dirichlet, results_neumann):
    """
    Create side-by-side comparison plots for Dirichlet vs Neumann BC.
    """
    # Extract results
    T_dir = results_dirichlet['T_final']
    T_neu = results_neumann['T_final']
    liq_dir = results_dirichlet['liquid_frac_final']
    liq_neu = results_neumann['liquid_frac_final']
    Q_map_data = results_dirichlet['Q_map']
    
    time_dir = results_dirichlet['time']
    time_neu = results_neumann['time']
    radius_dir = results_dirichlet['melt_radius']
    radius_neu = results_neumann['melt_radius']
    temp_dir = results_dirichlet['max_temp']
    temp_neu = results_neumann['max_temp']
    vel_dir = results_dirichlet['velocity']
    vel_neu = results_neumann['velocity']
    
    source_x = results_dirichlet['source_info']['center_x'] * 1000
    source_y = results_dirichlet['source_info']['center_y'] * 1000
    sigma_val = results_dirichlet['source_info']['sigma'] * 1000
    Q0_val = results_dirichlet['source_info']['Q0']
    
    # Determine common colorbar limits
    temp_max = max(np.max(T_dir), np.max(T_neu))
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))
    
    # 1. Final Temperature Fields (side by side)
    ax1 = plt.subplot(3, 4, 1)
    im1 = ax1.imshow(T_dir, extent=[0, length*1000, 0, length*1000], 
                     origin='lower', cmap='hot', vmin=25, vmax=temp_max)
    ax1.set_title(f'Dirichlet BC: Temperature\nT_max = {np.max(T_dir):.0f}°C')
    ax1.set_xlabel('x [mm]')
    ax1.set_ylabel('y [mm]')
    plt.colorbar(im1, ax=ax1, label='°C')
    
    ax2 = plt.subplot(3, 4, 2)
    im2 = ax2.imshow(T_neu, extent=[0, length*1000, 0, length*1000], 
                     origin='lower', cmap='hot', vmin=25, vmax=temp_max)
    ax2.set_title(f'Neumann BC: Temperature\nT_max = {np.max(T_neu):.0f}°C')
    ax2.set_xlabel('x [mm]')
    ax2.set_ylabel('y [mm]')
    plt.colorbar(im2, ax=ax2, label='°C')
    
    # 2. Final Liquid Fraction Fields
    ax3 = plt.subplot(3, 4, 5)
    im3 = ax3.imshow(liq_dir, extent=[0, length*1000, 0, length*1000], 
                     origin='lower', cmap='coolwarm', vmin=0, vmax=1)
    ax3.set_title(f'Dirichlet BC: Liquid Fraction')
    ax3.set_xlabel('x [mm]')
    ax3.set_ylabel('y [mm]')
    ax3.contour(liq_dir, levels=[0.5], colors='red', linewidths=2,
                extent=[0, length*1000, 0, length*1000])
    plt.colorbar(im3, ax=ax3, label='Liquid fraction')
    
    ax4 = plt.subplot(3, 4, 6)
    im4 = ax4.imshow(liq_neu, extent=[0, length*1000, 0, length*1000], 
                     origin='lower', cmap='coolwarm', vmin=0, vmax=1)
    ax4.set_title(f'Neumann BC: Liquid Fraction')
    ax4.set_xlabel('x [mm]')
    ax4.set_ylabel('y [mm]')
    ax4.contour(liq_neu, levels=[0.5], colors='red', linewidths=2,
                extent=[0, length*1000, 0, length*1000])
    plt.colorbar(im4, ax=ax4, label='Liquid fraction')
    
    # 3. Heat Source Distribution
    ax5 = plt.subplot(3, 4, 3)
    im5 = ax5.imshow(Q_map_data, extent=[0, length*1000, 0, length*1000], 
                     origin='lower', cmap='hot')
    ax5.set_title(f'Internal Heat Source Q(x,y)\nCenter: ({source_x:.1f}, {source_y:.1f}) mm')
    ax5.set_xlabel('x [mm]')
    ax5.set_ylabel('y [mm]')
    plt.colorbar(im5, ax=ax5, label='W/m³')
    
    # Empty subplot for spacing (4th position in row 1)
    ax6 = plt.subplot(3, 4, 4)
    ax6.axis('off')
    ax6.text(0.5, 0.5, f'Comparison: Dirichlet vs Neumann BC\n'
             f'Source: Gaussian (σ={sigma_val:.1f}mm)\n'
             f'Q0 = {Q0_val:.2e} W/m³\n'
             f'Material: Lead (T_m = {T_m}°C)',
             ha='center', va='center', fontsize=12, transform=ax6.transAxes)
    
    # 4. Melt Radius vs Time
    ax7 = plt.subplot(3, 4, 9)
    ax7.plot(time_dir, radius_dir, 'b-', linewidth=2, label='Dirichlet')
    ax7.plot(time_neu, radius_neu, 'r-', linewidth=2, label='Neumann')
    ax7.set_xlabel('Time [s]')
    ax7.set_ylabel('Melt Radius [mm]')
    ax7.set_title('Melting Front Position vs Time')
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    
    # 5. Maximum Temperature vs Time
    ax8 = plt.subplot(3, 4, 10)
    ax8.plot(time_dir, temp_dir, 'b-', linewidth=2, label='Dirichlet')
    ax8.plot(time_neu, temp_neu, 'r-', linewidth=2, label='Neumann')
    ax8.axhline(y=T_m, color='g', linestyle='--', linewidth=1, label=f'Melting point ({T_m}°C)')
    ax8.set_xlabel('Time [s]')
    ax8.set_ylabel('Maximum Temperature [°C]')
    ax8.set_title('Maximum Temperature vs Time')
    ax8.legend()
    ax8.grid(True, alpha=0.3)
    
    # 6. Interface Velocity vs Time
    ax9 = plt.subplot(3, 4, 11)
    ax9.plot(time_dir, vel_dir, 'b-', linewidth=2, label='Dirichlet')
    ax9.plot(time_neu, vel_neu, 'r-', linewidth=2, label='Neumann')
    ax9.set_xlabel('Time [s]')
    ax9.set_ylabel('Interface Velocity [mm/s]')
    ax9.set_title('Stefan Interface Velocity vs Time')
    ax9.legend()
    ax9.grid(True, alpha=0.3)  # FIXED: was alpha=3, now alpha=0.3
    
    # 7. Summary comparison (text)
    ax10 = plt.subplot(3, 4, 12)
    ax10.axis('off')
    
    # Calculate max possible radius (distance to nearest wall)
    max_radius_dir = min(source_x, source_y, length*1000 - source_x, length*1000 - source_y)
    max_radius_neu = min(source_x, source_y, length*1000 - source_x, length*1000 - source_y)
    
    summary_text = (
        f"COMPARISON SUMMARY:\n\n"
        f"{'Parameter':<25} {'Dirichlet':>15} {'Neumann':>15}\n"
        f"{'-'*55}\n"
        f"{'Max temperature [°C]':<25} {np.max(T_dir):>15.1f} {np.max(T_neu):>15.1f}\n"
        f"{'Final melt radius [mm]':<25} {radius_dir[-1] if len(radius_dir)>0 else 0:>15.2f} {radius_neu[-1] if len(radius_neu)>0 else 0:>15.2f}\n"
        f"{'Max velocity [mm/s]':<25} {np.max(vel_dir) if len(vel_dir)>0 else 0:>15.2f} {np.max(vel_neu) if len(vel_neu)>0 else 0:>15.2f}\n\n"
        f"PHYSICAL INTERPRETATION:\n"
        f"• Dirichlet: Walls at fixed T=25°C\n"
        f"  → Heat escapes → lower temperatures\n"
        f"  → Slower melting\n\n"
        f"• Neumann: Insulated walls (∂T/∂n=0)\n"
        f"  → Heat trapped → higher temperatures\n"
        f"  → Faster melting\n"
        f"  → Higher final T and larger melt radius"
    )
    
    ax10.text(0.05, 0.95, summary_text, ha='left', va='top', fontsize=10, 
              transform=ax10.transAxes, fontfamily='monospace')
    
    plt.suptitle(f'Comparison of Dirichlet vs Neumann Boundary Conditions\n'
                 f'2D Stefan Problem with Internal Gaussian Heat Source\n'
                 f'Time = {time_total:.1f}s, dt = {dt:.4f}s, Grid = {nx}×{ny}',
                 fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('stefan_bc_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return fig

def plot_individual_results(results, bc_type):
    """
    Plot individual results for a single boundary condition type.
    """
    T_final = results['T_final']
    liq_final = results['liquid_frac_final']
    Q_map_data = results['Q_map']
    time_data = results['time']
    radius_data = results['melt_radius']
    temp_data = results['max_temp']
    
    source_x = results['source_info']['center_x'] * 1000
    source_y = results['source_info']['center_y'] * 1000
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Temperature field
    im1 = axes[0, 0].imshow(T_final, extent=[0, length*1000, 0, length*1000], 
                            origin='lower', cmap='hot')
    axes[0, 0].set_title(f'Temperature Field ({bc_type.upper()} BC)\nMax = {np.max(T_final):.0f}°C')
    axes[0, 0].set_xlabel('x [mm]')
    axes[0, 0].set_ylabel('y [mm]')
    plt.colorbar(im1, ax=axes[0, 0], label='°C')
    
    # Liquid fraction
    im2 = axes[0, 1].imshow(liq_final, extent=[0, length*1000, 0, length*1000], 
                            origin='lower', cmap='coolwarm', vmin=0, vmax=1)
    axes[0, 1].contour(liq_final, levels=[0.5], colors='red', linewidths=2,
                       extent=[0, length*1000, 0, length*1000])
    axes[0, 1].set_title(f'Liquid Fraction ({bc_type.upper()} BC)')
    axes[0, 1].set_xlabel('x [mm]')
    axes[0, 1].set_ylabel('y [mm]')
    plt.colorbar(im2, ax=axes[0, 1], label='Liquid fraction')
    
    # Melt radius evolution
    if len(time_data) > 0 and len(radius_data) > 0:
        axes[1, 0].plot(time_data, radius_data, 'b-', linewidth=2)
    axes[1, 0].set_xlabel('Time [s]')
    axes[1, 0].set_ylabel('Melt Radius [mm]')
    axes[1, 0].set_title('Melting Front Position vs Time')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Temperature evolution
    if len(time_data) > 0 and len(temp_data) > 0:
        axes[1, 1].plot(time_data, temp_data, 'r-', linewidth=2)
    axes[1, 1].axhline(y=T_m, color='g', linestyle='--', label=f'Melting point ({T_m}°C)')
    axes[1, 1].set_xlabel('Time [s]')
    axes[1, 1].set_ylabel('Maximum Temperature [°C]')
    axes[1, 1].set_title('Maximum Temperature vs Time')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle(f'2D Stefan Problem Results: {bc_type.upper()} Boundary Conditions\n'
                 f'Heat source at ({source_x:.1f}, {source_y:.1f}) mm, σ={results["source_info"]["sigma"]*1000:.1f}mm',
                 fontsize=12)
    plt.tight_layout()
    plt.savefig(f'stefan_results_{bc_type}.png', dpi=150, bbox_inches='tight')
    plt.show()

# ============================================================
# MAIN EXECUTION
# ============================================================

if __name__ == "__main__":
    print("=" * 70)
    print("2D ONE-PHASE STEFAN PROBLEM")
    print("COMPARISON OF DIRICHLET vs NEUMANN BOUNDARY CONDITIONS")
    print("=" * 70)
    print(f"\nBC_TYPE selected: {BC_TYPE.upper()}")
    print("To compare both BC types, change BC_TYPE and run again.")
    print("=" * 70)
    
    # Run simulation with selected BC type
    results = run_simulation(BC_TYPE, save_results=True)
    
    # Plot individual results
    plot_individual_results(results, BC_TYPE)
    
    # If both result files exist, load and compare
    if os.path.exists('stefan_results_dirichlet.npz') and os.path.exists('stefan_results_neumann.npz'):
        print("\n" + "=" * 70)
        print("LOADING BOTH RESULTS FOR COMPARISON...")
        print("=" * 70)
        
        # Load results
        dir_data = np.load('stefan_results_dirichlet.npz', allow_pickle=True)
        neu_data = np.load('stefan_results_neumann.npz', allow_pickle=True)
        
        # Convert to dictionary format
        results_dir = {
            'T_final': dir_data['T_final'],
            'liquid_frac_final': dir_data['liquid_frac_final'],
            'Q_map': dir_data['Q_map'],
            'time': dir_data['time'],
            'max_temp': dir_data['max_temp'],
            'melt_radius': dir_data['melt_radius'],
            'velocity': dir_data['velocity'],
            'source_info': {
                'center_x': float(dir_data['center_x']),
                'center_y': float(dir_data['center_y']),
                'sigma': float(dir_data['sigma']),
                'Q0': float(dir_data['Q0'])
            }
        }
        
        results_neu = {
            'T_final': neu_data['T_final'],
            'liquid_frac_final': neu_data['liquid_frac_final'],
            'Q_map': neu_data['Q_map'],
            'time': neu_data['time'],
            'max_temp': neu_data['max_temp'],
            'melt_radius': neu_data['melt_radius'],
            'velocity': neu_data['velocity'],
            'source_info': {
                'center_x': float(neu_data['center_x']),
                'center_y': float(neu_data['center_y']),
                'sigma': float(neu_data['sigma']),
                'Q0': float(neu_data['Q0'])
            }
        }
        
        # Plot comparison
        plot_comparison(results_dir, results_neu)
        
        # Print summary comparison
        print("\n" + "=" * 70)
        print("COMPARISON SUMMARY")
        print("=" * 70)
        print(f"{'Parameter':<30} {'Dirichlet':>15} {'Neumann':>15} {'Difference':>15}")
        print("-" * 75)
        print(f"{'Max temperature [°C]':<30} {np.max(dir_data['T_final']):>15.1f} {np.max(neu_data['T_final']):>15.1f} {np.max(neu_data['T_final']) - np.max(dir_data['T_final']):>+15.1f}")
        
        dir_radius_last = dir_data['melt_radius'][-1] if len(dir_data['melt_radius']) > 0 else 0
        neu_radius_last = neu_data['melt_radius'][-1] if len(neu_data['melt_radius']) > 0 else 0
        print(f"{'Final melt radius [mm]':<30} {dir_radius_last:>15.2f} {neu_radius_last:>15.2f} {neu_radius_last - dir_radius_last:>+15.2f}")
        
        dir_vel_max = np.max(dir_data['velocity']) if len(dir_data['velocity']) > 0 else 0
        neu_vel_max = np.max(neu_data['velocity']) if len(neu_data['velocity']) > 0 else 0
        print(f"{'Max velocity [mm/s]':<30} {dir_vel_max:>15.2f} {neu_vel_max:>15.2f} {neu_vel_max - dir_vel_max:>+15.2f}")
        print("=" * 70)
        print("\nPHYSICAL INTERPRETATION:")
        print("  • Dirichlet BC: Fixed wall temperature at T_initial")
        print("    → Heat escapes from domain → lower temperatures, slower melting")
        print("  • Neumann BC: Insulated walls (∂T/∂n = 0)")
        print("    → Heat is trapped inside → higher temperatures, faster melting")
        print("=" * 70)
    
    else:
        print("\n" + "=" * 70)
        print("TO SEE COMPARISON, RUN SIMULATION WITH BOTH BC TYPES:")
        print("  1. Set BC_TYPE = 'dirichlet' and run")
        print("  2. Set BC_TYPE = 'neumann' and run")
        print("  3. Then run again with either BC type to see comparison")
        print("=" * 70)