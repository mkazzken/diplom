"""Heat source generation for the Stefan problem sandbox."""

import numpy as np


def random_source_centers(length, num_sources=1, margin=0.008, seed=42):
    rng = np.random.default_rng(seed)
    centers = []
    low = margin
    high = length - margin
    for _ in range(num_sources):
        cx = float(rng.uniform(low, high))
        cy = float(rng.uniform(low, high))
        centers.append((cx, cy))
    return centers


def build_heat_source_map(nx, ny, length, Q0, sigma, centers):
    x = np.linspace(0.0, length, nx)
    y = np.linspace(0.0, length, ny)
    X, Y = np.meshgrid(x, y)

    centers_arr = np.asarray(centers, dtype=float)
    if centers_arr.ndim == 1:
        centers_arr = centers_arr.reshape(1, 2)

    dx = X[:, :, np.newaxis] - centers_arr[np.newaxis, np.newaxis, :, 0]
    dy = Y[:, :, np.newaxis] - centers_arr[np.newaxis, np.newaxis, :, 1]
    r2 = dx**2 + dy**2
    Q_map = np.sum(Q0 * np.exp(-r2 / (2.0 * sigma**2)), axis=-1)

    return Q_map
