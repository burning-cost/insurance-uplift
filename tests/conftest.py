"""Shared fixtures for insurance-uplift tests.

Uses a synthetic DGP with known CATE so tests can verify correctness
rather than just runtime behaviour.

DGP specification
-----------------
- N=2000 policies
- Features: age (20-80), ncd (0-5), vehicle_age (0-15), region (A-E)
- Treatment: log price ratio ~ N(0, 0.1), slightly correlated with age
- True CATE: tau(x) = -1.5 * (age/50 - 1) + 0.3 * ncd
  - Young customers (age<50) are price sensitive (tau < 0 = Persuadable)
  - Older customers (age>50) are inelastic (tau > 0 = Sure Thing / DND)
- Outcome: logistic model with E[Y|X,T] = sigma(X_effect + tau(X)*T)
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest


def make_synthetic_panel(
    n: int = 2000,
    seed: int = 42,
    censored_frac: float = 0.08,
) -> pl.DataFrame:
    """Generate a synthetic renewal panel with known CATE."""
    rng = np.random.default_rng(seed)

    age = rng.uniform(20, 80, n)
    ncd = rng.integers(0, 6, n).astype(float)
    vehicle_age = rng.uniform(0, 15, n)
    region = rng.choice(["A", "B", "C", "D", "E"], n)
    region_effect = {"A": 0.1, "B": -0.05, "C": 0.2, "D": -0.1, "E": 0.0}

    # Continuous treatment: log price ratio
    treatment = rng.normal(0.03, 0.08, n)  # avg 3% increase, some variation

    # True CATE: negative means price-sensitive
    tau_true = -1.5 * (age / 50 - 1) + 0.3 * ncd

    # Base renewal propensity
    base_logit = (
        1.5
        + 0.02 * ncd
        - 0.005 * vehicle_age
        + np.array([region_effect[r] for r in region])
        - 0.01 * (age - 50)
    )

    # Observed renewal: base + CATE * treatment + noise
    logit = base_logit + tau_true * treatment
    prob_renew = 1 / (1 + np.exp(-logit))
    renewed = rng.binomial(1, prob_renew, n).astype(float)

    # Premiums: reconstruct from treatment
    expiring = rng.uniform(300, 1200, n)
    renewal = expiring * np.exp(treatment)

    # ENBP: approximately equal to expiring premium for most customers
    enbp = expiring * (1 + rng.uniform(-0.05, 0.05, n))
    # Ensure some policies have enbp < renewal (will be clipped)
    enbp = np.where(renewal > enbp, enbp * 0.99, enbp)

    # Dates
    start_dates = [date(2023, 1, 1) + timedelta(days=int(rng.uniform(0, 180))) for _ in range(n)]
    end_dates = [sd + timedelta(days=365) for sd in start_dates]

    # Mark some as censored by pushing end_date past censor_date
    censor_date = date(2024, 6, 30)
    censored_indices = rng.choice(n, int(n * censored_frac), replace=False)
    for i in censored_indices:
        end_dates[i] = date(2024, 8, 1)  # after censor date

    # For censored policies, renewal indicator is null
    renewed_nullable = [None if i in set(censored_indices) else float(renewed[i]) for i in range(n)]

    policy_ids = [f"POL{i:05d}" for i in range(n)]

    # Age band (for fairness audit)
    age_band = [
        "18-30" if a < 30
        else "31-50" if a < 50
        else "51-70" if a < 70
        else "71+"
        for a in age
    ]

    # Income decile (for fairness audit)
    postcode_income_decile = rng.integers(1, 11, n).astype(str)

    df = pl.DataFrame({
        "policy_id": policy_ids,
        "age": age,
        "age_band": age_band,
        "ncd": ncd,
        "vehicle_age": vehicle_age,
        "region": region,
        "postcode_income_decile": postcode_income_decile,
        "expiring_premium": expiring,
        "renewal_premium": renewal,
        "enbp": enbp,
        "renewed": renewed_nullable,
        "start_date": start_dates,
        "end_date": end_dates,
        "tau_true": tau_true,
    })

    return df


@pytest.fixture(scope="session")
def raw_panel():
    """Full synthetic panel including censored policies."""
    return make_synthetic_panel(n=2000, seed=42)


@pytest.fixture(scope="session")
def built_panel(raw_panel):
    """Built panel from RetentionPanel.build()."""
    from insurance_uplift.data import RetentionPanel

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rp = RetentionPanel(
            raw_panel,
            renewal_indicator_col="renewed",
            enbp_col="enbp",
            censor_date=date(2024, 6, 30),
        )
        return rp.build()


@pytest.fixture(scope="session")
def clean_panel(built_panel):
    """Panel with censored policies removed."""
    return built_panel.filter(pl.col("censored_flag") == 0).with_columns(
        pl.col("renewed").cast(pl.Float64)
    )


@pytest.fixture(scope="session")
def fitted_model(clean_panel):
    """Fitted RetentionUpliftModel — session-scoped so it only trains once.

    Skips automatically if econml is not available.
    """
    econml = pytest.importorskip("econml", reason="econml not available")
    from insurance_uplift.fit import RetentionUpliftModel
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = RetentionUpliftModel(
            estimator="causal_forest",
            n_estimators=100,  # small for test speed
            inference=True,
            random_state=42,
        )
        model.fit(
            clean_panel,
            confounders=["age", "ncd", "vehicle_age"],
        )
    return model
