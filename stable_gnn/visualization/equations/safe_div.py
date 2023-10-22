import numpy as np


def safe_div(a: np.ndarray, b: np.ndarray, jitter_scale: float = 0.000001):
    mask = b == 0
    b[mask] = 1
    inv_b = 1.0 / b
    res = a * inv_b
    if mask.sum() > 0:
        res[mask.repeat(2, 2)] = np.random.randn(mask.sum() * 2) * jitter_scale
    return res
