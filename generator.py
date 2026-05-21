import numpy as np
import os
import time
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# FAST DATASET GENERATOR FOR U-NET TRAINING
# ONE NPZ FILE PER SIMULATION (CONTAINS ALL FRAMES)
# ============================================================

# ============================================================
# FIXED PHYSICAL PARAMETERS
# ============================================================
rho = 11340.0
c_p = 128.0
k = 35.0
T_m = 327.5
L = 23000.0
T_initial = 25.0

# ============================================================
# COMPUTATIONAL DOMAIN
# ============================================================
length = 0.05
nx = 100
ny = 100

dx = length / (nx - 1)
dy = length / (ny - 1)

x = np.linspace(0, length, nx)
y = np.linspace(0, length, ny)
X, Y = np.meshgrid(x, y)

# ============================================================
# SIMULATION PARAMETERS
# ============================================================
dt = 0.0005

# Randomization rangess
Q0_range = (5e8, 2e9)
sigma_range = (0.003, 0.008)
time_range = (2.0, 4.0)
margin = 0.008

N_FRAMES = 10

# ============================================================
# BOUNDARY CONDITIONS
# ============================================================

def apply_dirichlet_bc(T_field):
    T_field[0, :] = T_initial
    T_field[-1, :] = T_initial
    T_field[:, 0] = T_initial
    T_field[:, -1] = T_initial
    return T_field

def apply_neumann_bc(T_field):
    T_field[0, :] = T_field[1, :]
    T_field[-1, :] = T_field[-2, :]
    T_field[:, 0] = T_field[:, 1]
    T_field[:, -1] = T_field[:, -2]
    return T_field

def apply_boundary_conditions(T_field, bc_type):
    if bc_type == "dirichlet":
        return apply_dirichlet_bc(T_field)
    return apply_neumann_bc(T_field)

def apply_enthalpy_dirichlet_bc(H_field):
    H_boundary = rho * c_p * T_initial
    H_field[0, :] = H_boundary
    H_field[-1, :] = H_boundary
    H_field[:, 0] = H_boundary
    H_field[:, -1] = H_boundary
    return H_field

def apply_enthalpy_neumann_bc(H_field):
    H_field[0, :] = H_field[1, :]
    H_field[-1, :] = H_field[-2, :]
    H_field[:, 0] = H_field[:, 1]
    H_field[:, -1] = H_field[:, -2]
    return H_field

def apply_enthalpy_boundary_conditions(H_field, bc_type):
    if bc_type == "dirichlet":
        return apply_enthalpy_dirichlet_bc(H_field)
    return apply_enthalpy_neumann_bc(H_field)

# ============================================================
# PHYSICS FUNCTIONS
# ============================================================

def enthalpy_to_temperature(H_field):
    H_solid = rho * c_p * T_m
    H_liquid = rho * c_p * T_m + rho * L
    
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

def compute_laplacian(T_field):
    laplacian = np.zeros_like(T_field)
    laplacian[1:-1, 1:-1] = (
        (T_field[2:, 1:-1] - 2*T_field[1:-1, 1:-1] + T_field[:-2, 1:-1]) / dy**2 +
        (T_field[1:-1, 2:] - 2*T_field[1:-1, 1:-1] + T_field[1:-1, :-2]) / dx**2
    )
    return laplacian

def create_gaussian_source(center_x, center_y, sigma, Q0):
    """VECTORIZED - much faster than double loop"""
    r2 = (X - center_x)**2 + (Y - center_y)**2
    return Q0 * np.exp(-r2 / (2 * sigma**2))

def normalize_temperature(T):
    return (T - T_initial) / (T_m - T_initial)

# ============================================================
# SINGLE SIMULATION (returns all frames at once)
# ============================================================

def run_single_simulation(params, frame_times):
    bc_type = params['bc_type']
    Q0 = params['Q0']
    sigma = params['sigma']
    center_x = params['center_x']
    center_y = params['center_y']
    time_total = params['time_total']
    
    Q_map = create_gaussian_source(center_x, center_y, sigma, Q0)
    
    T = np.ones((ny, nx)) * T_initial
    H = np.ones((ny, nx)) * (rho * c_p * T_initial)
    
    actual_frame_times = [t * time_total for t in frame_times]
    n_steps = int(time_total / dt)
    
    frames = []
    next_frame_idx = 0
    
    for step in range(n_steps):
        current_time = step * dt
        
        T_with_bc = T.copy()
        T_with_bc = apply_boundary_conditions(T_with_bc, bc_type)
        laplacian = compute_laplacian(T_with_bc)
        
        H = H + dt * (k * laplacian + Q_map)
        H = apply_enthalpy_boundary_conditions(H, bc_type)
        
        T, liquid_frac = enthalpy_to_temperature(H)
        T = apply_boundary_conditions(T, bc_type)
        
        if next_frame_idx < len(actual_frame_times):
            if current_time >= actual_frame_times[next_frame_idx]:
                frames.append({
                    'T_norm': normalize_temperature(T).astype(np.float32).copy(),
                    'liquid_frac': liquid_frac.astype(np.float32).copy(),
                    'time': current_time
                })
                next_frame_idx += 1
        
        if next_frame_idx >= len(actual_frame_times):
            break
    
    # Stack all frames into single arrays
    if frames:
        all_T = np.stack([f['T_norm'] for f in frames])
        all_liq = np.stack([f['liquid_frac'] for f in frames])
        all_times = np.array([f['time'] for f in frames], dtype=np.float32)
    else:
        all_T = np.zeros((N_FRAMES, ny, nx), dtype=np.float32)
        all_liq = np.zeros((N_FRAMES, ny, nx), dtype=np.float32)
        all_times = np.zeros(N_FRAMES, dtype=np.float32)
    
    return all_T, all_liq, all_times

# ============================================================
# DATASET GENERATION
# ============================================================

def generate_dataset(n_dirichlet=400, n_neumann=400, base_dir="dataset"):
    total_sims = n_dirichlet + n_neumann
    
    print("=" * 70)
    print("U-NET DATASET GENERATOR")
    print("2D One-Phase Stefan Problem")
    print("=" * 70)
    print(f"Dirichlet simulations: {n_dirichlet}")
    print(f"Neumann simulations: {n_neumann}")
    print(f"Total simulations: {total_sims}")
    print(f"Frames per simulation: {N_FRAMES}")
    print(f"Total training samples: {total_sims * N_FRAMES}")
    print("=" * 70)
    print(f"\nRandomization:")
    print(f"  Q0: {Q0_range[0]:.2e} - {Q0_range[1]:.2e} W/m³")
    print(f"  sigma: {sigma_range[0]*1000:.1f} - {sigma_range[1]*1000:.1f} mm")
    print(f"  time: {time_range[0]:.1f} - {time_range[1]:.1f} s")
    print("=" * 70)
    
    frame_times = np.linspace(0.1, 1.0, N_FRAMES)
    
    dirichlet_dir = os.path.join(base_dir, "dirichlet")
    neumann_dir = os.path.join(base_dir, "neumann")
    os.makedirs(dirichlet_dir, exist_ok=True)
    os.makedirs(neumann_dir, exist_ok=True)
    
    np.random.seed(42)
    start_time = time.time()
    
    # ========== DIRICHLET ==========
    print("\n" + "=" * 70)
    print("GENERATING DIRICHLET SIMULATIONS")
    print("=" * 70)
    
    for sim_idx in range(n_dirichlet):
        Q0 = np.random.uniform(*Q0_range)
        sigma = np.random.uniform(*sigma_range)
        time_total = np.random.uniform(*time_range)
        center_x = np.random.uniform(margin, length - margin)
        center_y = np.random.uniform(margin, length - margin)
        
        params = {
            'bc_type': 'dirichlet',
            'Q0': Q0, 'sigma': sigma,
            'center_x': center_x, 'center_y': center_y,
            'time_total': time_total
        }
        
        T_norm, liquid_frac, times = run_single_simulation(params, frame_times)
        
        sim_dir = os.path.join(dirichlet_dir, f"sample_{sim_idx+1:04d}")
        os.makedirs(sim_dir, exist_ok=True)
        
        np.savez(
            os.path.join(sim_dir, "simulation.npz"),
            T_norm=T_norm,
            liquid_frac=liquid_frac,
            times=times,
            bc_type='dirichlet',
            Q0=np.float32(Q0),
            sigma=np.float32(sigma),
            center_x=np.float32(center_x),
            center_y=np.float32(center_y),
            time_total=np.float32(time_total)
        )
        
        if (sim_idx + 1) % 20 == 0 or sim_idx + 1 == n_dirichlet:
            elapsed = time.time() - start_time
            rate = (sim_idx + 1) / elapsed
            eta = (total_sims - (sim_idx + 1)) / rate
            print(f"  Dirichlet: {sim_idx+1:3d}/{n_dirichlet} | "
                  f"Rate: {rate:.2f} sim/s | ETA: {eta:.0f}s")
    
    # ========== NEUMANN ==========
    print("\n" + "=" * 70)
    print("GENERATING NEUMANN SIMULATIONS")
    print("=" * 70)
    
    for sim_idx in range(n_neumann):
        Q0 = np.random.uniform(*Q0_range)
        sigma = np.random.uniform(*sigma_range)
        time_total = np.random.uniform(*time_range)
        center_x = np.random.uniform(margin, length - margin)
        center_y = np.random.uniform(margin, length - margin)
        
        params = {
            'bc_type': 'neumann',
            'Q0': Q0, 'sigma': sigma,
            'center_x': center_x, 'center_y': center_y,
            'time_total': time_total
        }
        
        T_norm, liquid_frac, times = run_single_simulation(params, frame_times)
        
        sim_dir = os.path.join(neumann_dir, f"sample_{sim_idx+1:04d}")
        os.makedirs(sim_dir, exist_ok=True)
        
        np.savez(
            os.path.join(sim_dir, "simulation.npz"),
            T_norm=T_norm,
            liquid_frac=liquid_frac,
            times=times,
            bc_type='neumann',
            Q0=np.float32(Q0),
            sigma=np.float32(sigma),
            center_x=np.float32(center_x),
            center_y=np.float32(center_y),
            time_total=np.float32(time_total)
        )
        
        sims_done = n_dirichlet + sim_idx + 1
        if (sim_idx + 1) % 20 == 0 or sim_idx + 1 == n_neumann:
            elapsed = time.time() - start_time
            rate = sims_done / elapsed
            eta = (total_sims - sims_done) / rate
            print(f"  Neumann: {sim_idx+1:3d}/{n_neumann} | "
                  f"Rate: {rate:.2f} sim/s | ETA: {eta:.0f}s")
    
    total_time = time.time() - start_time
    print("\n" + "=" * 70)
    print("DATASET GENERATION COMPLETED")
    print("=" * 70)
    print(f"Total time: {total_time:.2f} s ({total_time/60:.2f} min)")
    print(f"Average: {total_time/total_sims:.3f} s/sim")
    print(f"Output: {base_dir}/")
    print("=" * 70)

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    generate_dataset(n_dirichlet=400, n_neumann=400, base_dir="dataset")