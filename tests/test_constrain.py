"""Tests for constrain.py — ENBPConstraint, FairnessAudit, ROIReport."""

from __future__ import annotations

import warnings

import numpy as np
import polars as pl
import pytest

from insurance_uplift.constrain import ENBPConstraint, FairnessAudit, ROIReport


def make_constrain_panel(n: int = 200, seed: int = 42) -> pl.DataFrame:
    """Synthetic panel with enbp column for constraint tests."""
    rng = np.random.default_rng(seed)
    expiring = rng.uniform(300, 1200, n)
    # Some recommended renewals exceed ENBP
    renewal = expiring * (1 + rng.uniform(-0.05, 0.12, n))
    enbp = expiring * (1 + rng.uniform(-0.02, 0.06, n))

    return pl.DataFrame({
        "policy_id": [f"P{i:04d}" for i in range(n)],
        "expiring_premium": expiring,
        "renewal_premium": renewal,
        "enbp": enbp,
    })


class TestENBPConstraintInit:
    def test_default_columns(self):
        c = ENBPConstraint()
        assert c.enbp_col == "enbp"
        assert c.expiring_premium_col == "expiring_premium"

    def test_custom_columns(self):
        c = ENBPConstraint(enbp_col="nbp", expiring_premium_col="last_premium")
        assert c.enbp_col == "nbp"
        assert c.expiring_premium_col == "last_premium"


class TestENBPConstraintApply:
    def test_returns_polars_series(self):
        df = make_constrain_panel()
        c = ENBPConstraint()
        rec = pl.Series(np.zeros(len(df)))
        result = c.apply(df, rec)
        assert isinstance(result, pl.Series)

    def test_length_matches_input(self):
        df = make_constrain_panel()
        c = ENBPConstraint()
        rec = pl.Series(np.zeros(len(df)))
        result = c.apply(df, rec)
        assert len(result) == len(df)

    def test_clipped_rate_never_exceeds_enbp_ratio(self):
        df = make_constrain_panel()
        c = ENBPConstraint()
        # Recommend large increases that will all be clipped
        rec = pl.Series(np.full(len(df), 0.20))  # +20% recommendation
        result = c.apply(df, rec)
        expiring = df["expiring_premium"].to_numpy()
        enbp = df["enbp"].to_numpy()
        clipped_renewal = expiring * (1 + result.to_numpy())
        # All renewals must be <= enbp (within floating point)
        assert np.all(clipped_renewal <= enbp + 1e-6)

    def test_below_enbp_not_clipped(self):
        df = make_constrain_panel(n=100)
        c = ENBPConstraint()
        # Recommend a -10% decrease (well below any ENBP)
        rec = pl.Series(np.full(len(df), -0.10))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = c.apply(df, rec)
        # None should be clipped (all are decreases)
        np.testing.assert_allclose(result.to_numpy(), rec.to_numpy(), atol=1e-6)

    def test_warning_when_clipping_occurs(self):
        df = make_constrain_panel()
        c = ENBPConstraint()
        rec = pl.Series(np.full(len(df), 0.20))
        with pytest.warns(UserWarning, match="ENBP constraint clipped"):
            c.apply(df, rec)

    def test_mismatched_length_raises(self):
        df = make_constrain_panel(n=100)
        c = ENBPConstraint()
        rec = pl.Series(np.zeros(50))
        with pytest.raises(ValueError, match="rows"):
            c.apply(df, rec)

    def test_missing_enbp_col_raises(self):
        df = make_constrain_panel().drop("enbp")
        c = ENBPConstraint()
        with pytest.raises(ValueError, match="missing"):
            c.apply(df, pl.Series(np.zeros(len(df))))

    def test_series_name_is_clipped(self):
        df = make_constrain_panel()
        c = ENBPConstraint()
        rec = pl.Series(np.zeros(len(df)))
        result = c.apply(df, rec)
        assert result.name == "clipped_rate_change"


class TestENBPConstraintAuditReport:
    def test_returns_dataframe(self):
        df = make_constrain_panel()
        c = ENBPConstraint()
        rec = pl.Series(np.full(len(df), 0.05))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = c.audit_report(df, rec)
        assert isinstance(result, pl.DataFrame)

    def test_audit_has_required_columns(self):
        df = make_constrain_panel()
        c = ENBPConstraint()
        rec = pl.Series(np.zeros(len(df)))
        result = c.audit_report(df, rec)
        for col in ["expiring_premium", "enbp", "recommended_renewal",
                    "clipped_renewal", "was_clipped", "clip_amount_pct"]:
            assert col in result.columns

    def test_policy_id_included_when_present(self):
        df = make_constrain_panel()
        c = ENBPConstraint()
        rec = pl.Series(np.zeros(len(df)))
        result = c.audit_report(df, rec)
        assert "policy_id" in result.columns

    def test_was_clipped_is_boolean_like(self):
        df = make_constrain_panel()
        c = ENBPConstraint()
        rec = pl.Series(np.full(len(df), 0.20))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = c.audit_report(df, rec)
        # Some should be clipped
        assert int(result["was_clipped"].sum()) > 0

    def test_clip_amount_zero_for_unclipped(self):
        df = make_constrain_panel(n=50)
        c = ENBPConstraint()
        # Zero rate change: no clipping expected for below-enbp policies
        rec = pl.Series(np.full(len(df), -0.10))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = c.audit_report(df, rec)
        # All clip amounts should be zero
        np.testing.assert_allclose(
            result.filter(~pl.col("was_clipped"))["clip_amount_pct"].to_numpy(),
            0.0,
            atol=1e-6,
        )


class TestFairnessAuditInit:
    def test_empty_proxies_raise(self):
        with pytest.raises(ValueError, match="protected_proxies"):
            FairnessAudit(protected_proxies=[])

    def test_stores_proxies(self):
        fa = FairnessAudit(protected_proxies=["age_band"])
        assert fa.protected_proxies == ["age_band"]

    def test_default_threshold_age(self):
        fa = FairnessAudit(protected_proxies=["age_band"])
        assert fa.vulnerability_threshold_age == 70


class TestFairnessAuditFit:
    def test_fit_returns_self(self):
        X = pl.DataFrame({"age_band": ["18-30", "31-50", "51-70", "71+"] * 25})
        tau = pl.Series(np.random.default_rng(0).normal(0, 0.3, 100))
        fa = FairnessAudit(protected_proxies=["age_band"])
        result = fa.fit(X, tau)
        assert result is fa

    def test_is_fitted_after_fit(self):
        X = pl.DataFrame({"age_band": ["18-30", "71+"] * 50})
        tau = pl.Series(np.random.default_rng(0).normal(0, 0.3, 100))
        fa = FairnessAudit(protected_proxies=["age_band"])
        fa.fit(X, tau)
        assert fa._is_fitted

    def test_missing_proxy_col_raises(self):
        X = pl.DataFrame({"other_col": ["A"] * 10})
        tau = pl.Series(np.zeros(10))
        fa = FairnessAudit(protected_proxies=["age_band"])
        with pytest.raises(ValueError, match="missing"):
            fa.fit(X, tau)


class TestFairnessAuditAudit:
    def test_returns_dataframe(self):
        X = pl.DataFrame({"age_band": ["18-30", "31-50", "51-70", "71+"] * 50})
        rng = np.random.default_rng(0)
        # Older age bands have higher tau (inelastic)
        age_tau = {
            "18-30": rng.normal(-0.5, 0.2, 50),
            "31-50": rng.normal(-0.1, 0.2, 50),
            "51-70": rng.normal(0.1, 0.2, 50),
            "71+": rng.normal(0.4, 0.2, 50),
        }
        tau_vals = np.concatenate([
            age_tau["18-30"], age_tau["31-50"],
            age_tau["51-70"], age_tau["71+"]
        ])
        X_ordered = pl.DataFrame({
            "age_band": ["18-30"] * 50 + ["31-50"] * 50 + ["51-70"] * 50 + ["71+"] * 50
        })
        fa = FairnessAudit(protected_proxies=["age_band"])
        fa.fit(X_ordered, pl.Series(tau_vals))
        result = fa.audit()
        assert isinstance(result, pl.DataFrame)

    def test_audit_has_required_columns(self):
        X = pl.DataFrame({"age_band": ["18-30", "71+"] * 50})
        tau = pl.Series(np.random.default_rng(0).normal(0, 0.5, 100))
        fa = FairnessAudit(protected_proxies=["age_band"])
        fa.fit(X, tau)
        result = fa.audit()
        for col in ["proxy_variable", "group", "n", "avg_tau",
                    "flagged_as_vulnerable", "regulatory_note"]:
            assert col in result.columns

    def test_elderly_inelastic_is_flagged(self):
        """71+ group with positive tau must be flagged as vulnerable."""
        rng = np.random.default_rng(0)
        X = pl.DataFrame({
            "age_band": ["71+"] * 100
        })
        # All elderly, all inelastic (positive tau)
        tau = pl.Series(rng.uniform(0.3, 0.8, 100))
        fa = FairnessAudit(protected_proxies=["age_band"], vulnerability_threshold_age=70)
        fa.fit(X, tau)
        result = fa.audit()
        flagged_rows = result.filter(pl.col("flagged_as_vulnerable"))
        assert len(flagged_rows) > 0

    def test_young_inelastic_not_flagged(self):
        """18-30 group with positive tau is inelastic but not a vulnerability proxy."""
        rng = np.random.default_rng(0)
        X = pl.DataFrame({"age_band": ["18-30"] * 100})
        tau = pl.Series(rng.uniform(0.3, 0.8, 100))
        fa = FairnessAudit(protected_proxies=["age_band"])
        fa.fit(X, tau)
        result = fa.audit()
        flagged_rows = result.filter(pl.col("flagged_as_vulnerable"))
        assert len(flagged_rows) == 0

    def test_income_decile_low_inelastic_flagged(self):
        """Low income decile (1-3) with inelastic tau should be flagged."""
        rng = np.random.default_rng(0)
        X = pl.DataFrame({"postcode_income_decile": ["2"] * 100})
        tau = pl.Series(rng.uniform(0.3, 0.8, 100))
        fa = FairnessAudit(protected_proxies=["postcode_income_decile"])
        fa.fit(X, tau)
        result = fa.audit()
        assert len(result.filter(pl.col("flagged_as_vulnerable"))) > 0

    def test_audit_before_fit_raises(self):
        fa = FairnessAudit(protected_proxies=["age_band"])
        with pytest.raises(RuntimeError, match="fit"):
            fa.audit()

    def test_multiple_proxies(self):
        n = 200
        rng = np.random.default_rng(0)
        X = pl.DataFrame({
            "age_band": rng.choice(["18-30", "31-50", "51-70", "71+"], n).tolist(),
            "postcode_income_decile": rng.choice(
                ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"], n
            ).tolist(),
        })
        tau = pl.Series(rng.normal(0, 0.3, n))
        fa = FairnessAudit(protected_proxies=["age_band", "postcode_income_decile"])
        fa.fit(X, tau)
        result = fa.audit()
        assert set(result["proxy_variable"].unique().to_list()) == {
            "age_band", "postcode_income_decile"
        }


class TestROIReportInit:
    def test_default_values(self):
        roi = ROIReport()
        assert roi.policy_premium_avg == 600.0

    def test_invalid_premium_raises(self):
        with pytest.raises(ValueError, match="premium"):
            ROIReport(policy_premium_avg=0.0)


class TestROIReportCompute:
    def test_returns_dict(self):
        n = 100
        roi = ROIReport(policy_premium_avg=600.0)
        df = make_constrain_panel(n=n)
        tau = pl.Series(np.full(n, -0.3))
        rec = pl.Series(np.ones(n, dtype=np.int32))
        result = roi.compute(df, tau, rec, discount_size=0.05)
        assert isinstance(result, dict)

    def test_required_keys_present(self):
        n = 100
        roi = ROIReport(policy_premium_avg=600.0)
        df = make_constrain_panel(n=n)
        tau = pl.Series(np.full(n, -0.3))
        rec = pl.Series(np.ones(n, dtype=np.int32))
        result = roi.compute(df, tau, rec, discount_size=0.05)
        required_keys = [
            "n_treated", "expected_additional_renewals", "expected_discount_cost",
            "expected_total_cost", "net_roi", "roi_pct",
            "break_even_retention_rate", "uplift_per_pound_spent",
        ]
        for k in required_keys:
            assert k in result

    def test_no_treated_returns_zeros(self):
        n = 100
        roi = ROIReport(policy_premium_avg=600.0)
        df = make_constrain_panel(n=n)
        tau = pl.Series(np.full(n, -0.3))
        rec = pl.Series(np.zeros(n, dtype=np.int32))
        with pytest.warns(UserWarning, match="No policies"):
            result = roi.compute(df, tau, rec, discount_size=0.05)
        assert result["n_treated"] == 0
        assert result["net_roi"] == 0.0

    def test_negative_discount_raises(self):
        n = 100
        roi = ROIReport()
        df = make_constrain_panel(n=n)
        tau = pl.Series(np.ones(n) * -0.3)
        rec = pl.Series(np.ones(n, dtype=np.int32))
        with pytest.raises(ValueError, match="discount_size"):
            roi.compute(df, tau, rec, discount_size=-0.05)

    def test_n_treated_counts_ones(self):
        n = 200
        roi = ROIReport(policy_premium_avg=600.0)
        df = make_constrain_panel(n=n)
        tau = pl.Series(np.full(n, -0.3))
        rec = pl.Series(np.array([1] * 80 + [0] * 120, dtype=np.int32))
        result = roi.compute(df, tau, rec, discount_size=0.05)
        assert result["n_treated"] == 80

    def test_positive_tau_customers_no_additional_renewals(self):
        """Customers with positive tau don't benefit from discounts."""
        n = 100
        roi = ROIReport(policy_premium_avg=600.0)
        df = make_constrain_panel(n=n)
        # Positive tau: price increase improves renewal, so discount has no benefit
        tau = pl.Series(np.full(n, 0.5))
        rec = pl.Series(np.ones(n, dtype=np.int32))
        result = roi.compute(df, tau, rec, discount_size=0.05)
        assert result["expected_additional_renewals"] == 0.0

    def test_discount_cost_proportional_to_n_treated(self):
        n = 100
        roi = ROIReport(policy_premium_avg=600.0)
        df = make_constrain_panel(n=n)
        tau = pl.Series(np.full(n, -0.3))
        rec50 = pl.Series(np.array([1] * 50 + [0] * 50, dtype=np.int32))
        rec100 = pl.Series(np.ones(n, dtype=np.int32))
        result50 = roi.compute(df, tau, rec50, discount_size=0.05)
        result100 = roi.compute(df, tau, rec100, discount_size=0.05)
        assert abs(result100["expected_discount_cost"] / result50["expected_discount_cost"] - 2.0) < 0.01
