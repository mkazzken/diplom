"""Predefined materials for the Stefan problem simulation."""

PREDEFINED_MATERIALS = {
    'Lead': {
        'rho': 11340.0,
        'c_p': 128.0,
        'k': 35.0,
        'T_m': 327.5,
        'L': 23000.0
    },
    'Aluminum': {
        'rho': 2700.0,
        'c_p': 897.0,
        'k': 205.0,
        'T_m': 660.0,
        'L': 397000.0
    },
    'Copper': {
        'rho': 8960.0,
        'c_p': 385.0,
        'k': 401.0,
        'T_m': 1085.0,
        'L': 205000.0
    },
    'Steel': {
        'rho': 7850.0,
        'c_p': 470.0,
        'k': 45.0,
        'T_m': 1500.0,
        'L': 272000.0
    },
    'Paraffin PCM': {
        'rho': 900.0,
        'c_p': 2000.0,
        'k': 0.2,
        'T_m': 45.0,
        'L': 170000.0
    }
}
