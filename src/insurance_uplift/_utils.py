"""Shared helpers for insurance-uplift.

Internal module — not part of the public API.
"""

from __future__ import annotations

import warnings
from typing import Union

import numpy as np
import polars as pl


ArrayLike = Union[np.ndarray, pl.Series, list]


def to_numpy(x: ArrayLike) -> np.ndarray:
    """Convert ArrayLike to a 1-D float64 numpy array."""
    if isinstance(x, pl.Series):
        return x.to_numpy().astype(np.float64)
    if isinstance(x, list):
        return np.asarray(x, dtype=np.float64)
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"Expected 1-D array, got shape {arr.shape}.")
    return arr


def to_numpy_2d(x: Union[pl.DataFrame, np.ndarray]) -> np.ndarray:
    """Convert a Polars DataFrame or 2-D array to a float64 numpy array."""
    if isinstance(x, pl.DataFrame):
        return x.to_numpy().astype(np.float64)
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return arr


def log_price_ratio(renewal_premium: ArrayLike, expiring_premium: ArrayLike) -> np.ndarray:
    """Compute the continuous treatment: log(renewal / expiring).

    Parameters
    ----------
    renewal_premium:
        Renewal offer premium for each policy.
    expiring_premium:
        Expiring (prior year) premium for each policy.

    Returns
    -------
    np.ndarray
        Log price ratio, centred around zero. Positive values indicate a price
        increase; negative values indicate a price reduction.

    Raises
    ------
    ValueError
        If any expiring premium is non-positive.
    """
    r = to_numpy(renewal_premium)
    e = to_numpy(expiring_premium)
    if np.any(e <= 0):
        raise ValueError("expiring_premium must be strictly positive for all policies.")
    if np.any(r <= 0):
        raise ValueError("renewal_premium must be strictly positive for all policies.")
    return np.log(r / e)


def binarise_treatment(treatment: np.ndarray, method: str = "median") -> np.ndarray:
    """Binarise a continuous treatment vector at its median.

    Used for Qini evaluation when the treatment is log price ratio rather than
    a discrete A/B assignment. Policies with treatment <= median are labelled
    control (0); those above median are labelled treated (1).

    Parameters
    ----------
    treatment:
        Continuous treatment values (e.g. log price ratio).
    method:
        Only 'median' is supported in v0.1.

    Returns
    -------
    np.ndarray
        Binary array: 1 for above-median (treated), 0 for below-median.
    """
    if method != "median":
        raise ValueError(f"Unsupported binarisation method '{method}'. Use 'median'.")
    median_val = np.median(treatment)
    binary = (treatment > median_val).astype(np.int32)
    n_treated = binary.sum()
    n_total = len(binary)
    if n_treated < 50:
        warnings.warn(
            f"Only {n_treated} policies are above-median treatment. "
            "Qini curves may be noisy. Consider collecting more data.",
            UserWarning,
            stacklevel=3,
        )
    return binary


def check_r_available() -> bool:
    """Return True if rpy2 and the required R packages are importable."""
    try:
        import rpy2.robjects  # noqa: F401
        from rpy2.robjects.packages import importr  # noqa: F401

        importr("grf")
        return True
    except Exception:
        return False


def validate_panel_columns(df: pl.DataFrame, required: list[str]) -> None:
    """Raise ValueError if any required columns are absent from df."""
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Panel is missing required columns: {missing}. "
            f"Available columns: {df.columns}."
        )


def validate_min_samples(n: int, min_n: int = 200, context: str = "") -> None:
    """Warn if the dataset is smaller than the recommended minimum."""
    if n < min_n:
        warnings.warn(
            f"{context}Only {n} observations available. "
            f"Causal forest requires at least {min_n} for stable CATE estimates.",
            UserWarning,
            stacklevel=3,
        )


def segment_label(tau: float, threshold: float = 0.0) -> str:
    """Map a scalar CATE to the Guelman et al. four-customer taxonomy label.

    This is a pure-tau mapping used internally. For the full taxonomy that
    also conditions on outcome, see :func:`~insurance_uplift.evaluate.segment_types`.

    Parameters
    ----------
    tau:
        Individual CATE estimate τ̂(x). Negative tau means price increase reduces
        renewal probability (customer is price-sensitive).
    threshold:
        Absolute threshold around zero for "approximately zero" effect.

    Returns
    -------
    str
        One of 'Persuadable', 'Do Not Disturb', 'Near Zero'.
    """
    if tau < -abs(threshold):
        return "Persuadable"
    if tau > abs(threshold):
        return "Do Not Disturb"
    return "Near Zero"


def safe_divide(numerator: float, denominator: float, fill: float = 0.0) -> float:
    """Return numerator / denominator, or fill if denominator is zero."""
    if denominator == 0:
        return fill
    return numerator / denominator
