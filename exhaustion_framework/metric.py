import numpy as np
from scipy.special import erf
from typing import Tuple, Union


def phi_gaussian_cdf(x: np.ndarray) -> np.ndarray:
    """
    Computes the standard normal CDF \Phi(x) using the error function.
    Optimized via scipy.special.erf for vectorization.
    """
    return 0.5 * (1.0 + erf(x / np.sqrt(2.0)))


def weighted_kolmogorov_error(
        z_samples: np.ndarray,
        q: float,
        t_grid: np.ndarray
) -> Tuple[float, float]:
    """
    Computes the Weighted Kolmogorov Metric as defined in Petrosyan (2025), Def 2.1.

    Args:
        z_samples: Shape (B,), i.i.d. draws of the normalized sum Z_n.
        q: The weight exponent for w_q(t) = (1 + |t|)^(-q).
        t_grid: The evaluation grid in R.

    Returns:
        (weighted_error, uniform_error) pair.
    """
    z_sorted = np.sort(z_samples)
    n_samples = z_sorted.size

    counts = np.searchsorted(z_sorted, t_grid, side="right")
    f_emp = counts / n_samples

    f_ref = phi_gaussian_cdf(t_grid)

    diff = np.abs(f_emp - f_ref)

    weights = 1.0 / (1.0 + np.abs(t_grid)) ** q

    weighted_err = np.max(weights * diff)
    uniform_err = np.max(diff)

    return weighted_err, uniform_err