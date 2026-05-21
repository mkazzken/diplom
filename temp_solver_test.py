from simulation import solver, materials, heat_sources
material = materials.PREDEFINED_MATERIALS['Lead']
centers = heat_sources.random_source_centers(length=0.05, num_sources=1, margin=0.008, seed=42)
for state in solver.simulate(
    material=material,
    bc_type='dirichlet',
    length=0.05,
    nx=20,
    ny=20,
    dt=0.001,
    time_total=0.01,
    T_initial=25.0,
    Q0=5e8,
    sigma=0.006,
    source_centers=centers,
    output_every=1,
    random_seed=42
):
    print('step', state['step'], 'mean_radius', state['mean_radius'], 'max_temp', state['max_temp'])
