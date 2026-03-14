"""Targeting quality evaluation for uplift models.

Provides Qini curves, AUUC, uplift at k, and the Guelman four-customer
taxonomy. All metrics support continuous price treatment via median
binarisation.

The core insight from Rößler & Schoder (2022) is that standard ML metrics
(MSE on τ̂, R²) do not measure targeting quality. A model with high CATE
MSE can still produce better campaign outcomes than a model with lower MSE,
if it correctly ranks the customers who matter most. Qini and AUUC measure
targeting quality directly.

Usage
-----
::

    from insurance_uplift.evaluate import qini_curve, auuc, segment_types

    fractions, gains = qini_curve(y_true, treatment, tau_hat)
    score = auuc(y_true, treatment, tau_hat)
    segments = segment_types(y_true, treatment, tau_hat)
"""

from __future__ import annotations

import warnings
from typing import Optional, Union

import numpy as np

# NumPy 2.0 removed numpy.trapz; use numpy.trapezoid if available
_trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))
import polars as pl

from ._utils import ArrayLike, binarise_treatment, to_numpy


def qini_curve(
    y_true: ArrayLike,
    treatment: ArrayLike,
    tau_hat: ArrayLike,
    *,
    n_buckets: int = 100,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the Qini curve for uplift model evaluation.

    The Qini curve plots, for each fraction of the customer base targeted
    in descending order of predicted τ̂(x) (highest first), the incremental
    number of renewals attributable to targeting (relative to no targeting).

    Customers are ranked by τ̂(x) descending. Higher τ̂(x) = stronger
    positive treatment response = higher priority for intervention.

    For continuous treatment, the treatment vector is binarised at its
    median: customers with above-median price increases are "treated",
    below-median are "control". This is the standard adaptation when no
    explicit A/B assignment exists.

    Parameters
    ----------
    y_true:
        Binary renewal outcome (1=renewed, 0=lapsed).
    treatment:
        Treatment indicator (1=treated, 0=control) or continuous treatment
        (binarised automatically at median if values outside {0, 1}).
    tau_hat:
        Predicted CATE from :meth:`~insurance_uplift.fit.RetentionUpliftModel.cate`.
    n_buckets:
        Number of points on the Qini curve.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(fraction_targeted, incremental_renewals)`` where both arrays have
        length ``n_buckets + 1`` and the first point is (0, 0).

    Notes
    -----
    The Qini coefficient (area between the Qini curve and the random
    targeting diagonal) is returned by :func:`auuc`.
    """
    y = to_numpy(y_true)
    t = to_numpy(treatment)
    tau = to_numpy(tau_hat)

    if len(y) != len(t) or len(y) != len(tau):
        raise ValueError(
            f"y_true, treatment, and tau_hat must have the same length. "
            f"Got {len(y)}, {len(t)}, {len(tau)}."
        )

    # Binarise continuous treatment
    unique_t = np.unique(t)
    if not (len(unique_t) == 2 and set(unique_t).issubset({0, 1, 0.0, 1.0})):
        t = binarise_treatment(t).astype(float)

    n = len(y)
    n_treated = int(t.sum())
    n_control = n - n_treated

    if n_treated == 0 or n_control == 0:
        raise ValueError(
            "Qini curve requires both treated and control observations. "
            "Check that treatment binarisation produced non-empty groups."
        )

    # Sort customers by predicted tau descending (highest priority first)
    order = np.argsort(-tau)
    y_sorted = y[order]
    t_sorted = t[order]

    fractions = np.zeros(n_buckets + 1)
    gains = np.zeros(n_buckets + 1)

    # Cumulative treated and control counts at each bucket boundary
    cum_t = np.cumsum(t_sorted)
    cum_c = np.cumsum(1 - t_sorted)
    cum_y_t = np.cumsum(y_sorted * t_sorted)
    cum_y_c = np.cumsum(y_sorted * (1 - t_sorted))

    for i in range(1, n_buckets + 1):
        cutoff = int(np.ceil(i * n / n_buckets)) - 1
        cutoff = min(cutoff, n - 1)

        n_t_at_k = cum_t[cutoff]
        n_c_at_k = cum_c[cutoff]

        # Qini gain: renewals in treated group minus expected renewals from control rate
        if n_t_at_k > 0 and n_c_at_k > 0:
            renewal_rate_control = cum_y_c[cutoff] / n_c_at_k if n_c_at_k > 0 else 0.0
            gain = cum_y_t[cutoff] - n_t_at_k * renewal_rate_control
        elif n_t_at_k > 0:
            gain = cum_y_t[cutoff]
        else:
            gain = 0.0

        fractions[i] = (cutoff + 1) / n
        gains[i] = gain

    return fractions, gains


def auuc(
    y_true: ArrayLike,
    treatment: ArrayLike,
    tau_hat: ArrayLike,
    *,
    n_buckets: int = 100,
) -> float:
    """Compute the Area Under the Uplift Curve (AUUC) — Qini coefficient.

    The AUUC is the area between the Qini curve and the random targeting
    baseline (the diagonal). Higher values indicate better targeting quality.
    A positive AUUC means the model beats random assignment.

    Parameters
    ----------
    y_true:
        Binary renewal outcome.
    treatment:
        Treatment indicator or continuous treatment (binarised at median).
    tau_hat:
        Predicted CATE.
    n_buckets:
        Granularity of the Qini curve approximation.

    Returns
    -------
    float
        AUUC (Qini coefficient). Units: cumulative incremental renewals per
        customer targeted, normalised to [−1, 1] approximately.
    """
    fractions, gains = qini_curve(y_true, treatment, tau_hat, n_buckets=n_buckets)

    # Random targeting baseline: linear interpolation from (0,0) to (1, total_gain)
    total_gain = gains[-1]
    random_baseline = fractions * total_gain

    # Area between curve and baseline, using trapezoid rule
    auuc_val = float(_trapz(gains - random_baseline, fractions))
    return auuc_val


def uplift_at_k(
    y_true: ArrayLike,
    treatment: ArrayLike,
    tau_hat: ArrayLike,
    k: float = 0.3,
) -> float:
    """Uplift at top-k%: what fraction of achievable retention gain is captured?

    Answers: "If I offer a discount to the top k% of customers by predicted
    tau, what fraction of the maximum possible incremental renewals do I
    achieve?"

    The "maximum achievable gain" is the peak of the Qini curve — the point
    where additional targeting yields no incremental benefit. This is used
    rather than the gain at 100% targeting (which can decline as lower-tau
    customers are added to the treatment pool).

    Parameters
    ----------
    y_true:
        Binary renewal outcome.
    treatment:
        Treatment indicator or continuous treatment.
    tau_hat:
        Predicted CATE.
    k:
        Fraction of customers to target (0 < k <= 1). Default 0.3 (top 30%).

    Returns
    -------
    float
        Fraction of achievable gain captured in the top-k bucket. A value of
        0.8 means that targeting the top 30% captures 80% of the total possible
        incremental renewals.

    Raises
    ------
    ValueError
        If k is not in (0, 1].
    """
    if not (0 < k <= 1):
        raise ValueError(f"k must be in (0, 1], got {k}.")

    fractions, gains = qini_curve(y_true, treatment, tau_hat, n_buckets=1000)

    # Find gain at fraction = k
    idx = np.searchsorted(fractions, k, side="right") - 1
    idx = max(0, min(idx, len(gains) - 1))
    gain_at_k = gains[idx]

    # Peak Qini gain across all targeting fractions is the denominator.
    # Using gains[-1] (100% targeting) fails when the curve peaks before k=1,
    # which is common in mixed populations with DND customers. Using the peak
    # gives a well-defined [0, 1] metric: "what fraction of the best achievable
    # targeting outcome does this k% capture?"
    max_gain = float(gains.max())
    if max_gain <= 0:
        warnings.warn(
            "Total Qini gain is non-positive. uplift_at_k is not meaningful.",
            UserWarning,
            stacklevel=2,
        )
        return 0.0

    return float(min(gain_at_k / max_gain, 1.0))


def segment_types(
    y_true: ArrayLike,
    treatment: ArrayLike,
    tau_hat: ArrayLike,
    threshold: float = 0.0,
) -> pl.DataFrame:
    """Classify customers into the Guelman et al. four-customer taxonomy.

    The four types are defined by the combination of observed outcome and
    predicted treatment effect. The partition is exhaustive — every customer
    belongs to exactly one segment.

    - **Persuadable**: τ̂(x) < −threshold and Y=0 (lapsed; price-sensitive
      enough that a discount would have changed their decision). Primary
      discount targets.
    - **Do Not Disturb**: τ̂(x) > threshold (positive: inelastic or
      comparison-shopping risk). Any price reduction wasted or
      counterproductive. Avoid intervention.
    - **Sure Thing**: not DND, Y=1 (renewed regardless of price sensitivity).
      Discounting wastes margin — they were going to renew anyway.
    - **Lost Cause**: not DND, not Persuadable, Y=0 (lapsed despite not
      being strongly price-sensitive). No realistic intervention effect.

    Note on sign convention: in this library, τ̂(x) is the effect of a +1 unit
    increase in log price on renewal probability. A negative τ̂(x) means the
    customer is price-sensitive (price increase → lapse). A Persuadable customer
    has τ̂(x) < −threshold combined with having actually lapsed.

    Parameters
    ----------
    y_true:
        Binary renewal outcome.
    treatment:
        Treatment indicator or continuous treatment (unused in classification,
        retained for API consistency).
    tau_hat:
        Predicted CATE (τ̂(x)).
    threshold:
        Sensitivity threshold. Only customers with τ̂(x) < −threshold are
        classified as Persuadable; only customers with τ̂(x) > threshold are
        classified as Do Not Disturb. Default 0 classifies all non-zero-tau
        customers by sign.

    Returns
    -------
    pl.DataFrame
        Columns: ``[segment_type, n, fraction, avg_tau, min_tau, max_tau]``
        One row per segment type. Fractions sum to 1.0.
    """
    y = to_numpy(y_true)
    tau = to_numpy(tau_hat)

    thresh = abs(threshold)

    # Exhaustive, non-overlapping partition:
    do_not_disturb = tau > thresh
    persuadable = (tau < -thresh) & (y == 0)
    # Sure Thing: not DND (tau <= thresh), and renewed
    sure_thing = (~do_not_disturb) & (y == 1)
    # Lost Cause: not DND, not Persuadable (tau >= -thresh), and lapsed
    lost_cause = (~do_not_disturb) & (~persuadable) & (y == 0)

    n = len(y)
    rows = []
    for label, mask in [
        ("Persuadable", persuadable),
        ("Sure Thing", sure_thing),
        ("Lost Cause", lost_cause),
        ("Do Not Disturb", do_not_disturb),
    ]:
        n_seg = int(mask.sum())
        tau_seg = tau[mask]
        rows.append(
            {
                "segment_type": label,
                "n": n_seg,
                "fraction": n_seg / n if n > 0 else 0.0,
                "avg_tau": float(np.mean(tau_seg)) if n_seg > 0 else 0.0,
                "min_tau": float(np.min(tau_seg)) if n_seg > 0 else 0.0,
                "max_tau": float(np.max(tau_seg)) if n_seg > 0 else 0.0,
            }
        )

    return pl.DataFrame(rows)


def persuadable_rate(tau_hat: ArrayLike, threshold: float = 0.0) -> float:
    """Return the fraction of customers classified as Persuadable.

    A customer is Persuadable if τ̂(x) < −threshold, meaning they are
    sufficiently price-sensitive that a discount is expected to change their
    renewal decision.

    Parameters
    ----------
    tau_hat:
        Predicted CATE.
    threshold:
        Sensitivity threshold. Only customers with τ̂(x) < −threshold are
        classified as persuadable. Default 0 (any negative tau).

    Returns
    -------
    float
        Fraction of customers who are Persuadable.
    """
    tau = to_numpy(tau_hat)
    return float(np.mean(tau < -abs(threshold)))


def plot_qini(
    y_true: ArrayLike,
    treatment: ArrayLike,
    tau_hat: ArrayLike,
    ax=None,
    *,
    label: str = "Model",
    n_buckets: int = 100,
):
    """Plot the Qini curve against the random targeting baseline.

    Parameters
    ----------
    y_true:
        Binary renewal outcome.
    treatment:
        Treatment indicator or continuous treatment.
    tau_hat:
        Predicted CATE.
    ax:
        Matplotlib axes to plot on. If ``None``, a new figure is created.
    label:
        Legend label for the model curve.
    n_buckets:
        Granularity of the Qini curve.

    Returns
    -------
    matplotlib.axes.Axes
        The axes object for further customisation.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib is required for plot_qini. Install it with pip.")

    fractions, gains = qini_curve(y_true, treatment, tau_hat, n_buckets=n_buckets)
    auuc_score = auuc(y_true, treatment, tau_hat, n_buckets=n_buckets)

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))

    ax.plot(fractions, gains, label=f"{label} (AUUC={auuc_score:.4f})", linewidth=2)
    ax.plot(
        [0, 1],
        [0, gains[-1]],
        linestyle="--",
        color="grey",
        alpha=0.7,
        label="Random targeting",
    )
    ax.fill_between(
        fractions,
        gains,
        fractions * gains[-1],
        alpha=0.12,
        label=f"AUUC area = {auuc_score:.4f}",
    )

    ax.set_xlabel("Fraction of customers targeted (descending τ̂)")
    ax.set_ylabel("Incremental renewals (Qini gain)")
    ax.set_title("Qini Curve — Uplift Model Evaluation")
    ax.legend(loc="upper left")
    ax.set_xlim(0, 1)
    ax.grid(alpha=0.3)

    return ax
