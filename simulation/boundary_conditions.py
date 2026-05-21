"""Boundary condition helpers for the Stefan problem sandbox."""


def apply_dirichlet_bc(T_field, T_initial):
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


def apply_boundary_conditions(T_field, bc_type, T_initial):
    if bc_type == 'dirichlet':
        return apply_dirichlet_bc(T_field, T_initial)
    elif bc_type == 'neumann':
        return apply_neumann_bc(T_field)
    raise ValueError(f'Unknown bc_type: {bc_type}. Use "dirichlet" or "neumann"')


def apply_enthalpy_dirichlet_bc(H_field, rho, c_p, T_initial):
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


def apply_enthalpy_boundary_conditions(H_field, bc_type, rho, c_p, T_initial):
    if bc_type == 'dirichlet':
        return apply_enthalpy_dirichlet_bc(H_field, rho, c_p, T_initial)
    elif bc_type == 'neumann':
        return apply_enthalpy_neumann_bc(H_field)
    raise ValueError(f'Unknown bc_type: {bc_type}. Use "dirichlet" or "neumann"')
