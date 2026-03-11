"""Tests for data.py — RetentionPanel."""

from __future__ import annotations

import warnings
from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from insurance_uplift.data import RetentionPanel


def make_minimal_df(n: int = 100, seed: int = 0) -> pl.DataFrame:
    """Minimal policy DataFrame for data module tests."""
    rng = np.random.default_rng(seed)
    expiring = rng.uniform(300, 800, n)
    renewal = expiring * (1 + rng.uniform(-0.1, 0.15, n))
    enbp = expiring * (1 + rng.uniform(-0.05, 0.03, n))
    start = [date(2023, 1, 1) + timedelta(days=int(d)) for d in rng.uniform(0, 180, n)]
    end = [s + timedelta(days=365) for s in start]
    renewed = rng.binomial(1, 0.75, n).tolist()

    return pl.DataFrame({
        "policy_id": [f"P{i:04d}" for i in range(n)],
        "expiring_premium": expiring,
        "renewal_premium": renewal,
        "enbp": enbp,
        "renewed": renewed,
        "start_date": start,
        "end_date": end,
    })


class TestRetentionPanelBuild:
    def test_returns_polars_dataframe(self):
        df = make_minimal_df()
        rp = RetentionPanel(df, censor_date=date(2024, 6, 30))
        result = rp.build()
        assert isinstance(result, pl.DataFrame)

    def test_treatment_column_added(self):
        df = make_minimal_df()
        rp = RetentionPanel(df, censor_date=date(2024, 6, 30))
        result = rp.build()
        assert "treatment" in result.columns

    def test_treatment_is_log_price_ratio(self):
        df = make_minimal_df(n=10)
        rp = RetentionPanel(df, censor_date=date(2024, 6, 30))
        result = rp.build()
        expected = np.log(
            df["renewal_premium"].to_numpy() / df["expiring_premium"].to_numpy()
        )
        np.testing.assert_allclose(
            result["treatment"].to_numpy(), expected, rtol=1e-6
        )

    def test_censored_flag_column_added(self):
        df = make_minimal_df()
        rp = RetentionPanel(df, censor_date=date(2024, 6, 30))
        result = rp.build()
        assert "censored_flag" in result.columns

    def test_no_policies_censored_for_early_censor_date(self):
        df = make_minimal_df()
        rp = RetentionPanel(df, censor_date=date(2020, 1, 1))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = rp.build()
        # All end_dates are ~2024, so censor_date 2020 means all are censored
        assert int(result["censored_flag"].sum()) == len(df)

    def test_no_policies_censored_for_late_censor_date(self):
        df = make_minimal_df()
        rp = RetentionPanel(df, censor_date=date(2030, 1, 1))
        result = rp.build()
        assert int(result["censored_flag"].sum()) == 0

    def test_censored_warning_issued(self):
        df = make_minimal_df()
        # Set some end_dates past the censor date
        end_dates = df["end_date"].to_list()
        end_dates[0] = date(2025, 1, 1)
        df = df.with_columns(pl.Series("end_date", end_dates))
        rp = RetentionPanel(df, censor_date=date(2024, 6, 30))
        with pytest.warns(UserWarning, match="censored"):
            rp.build()

    def test_earned_exposure_column_added(self):
        df = make_minimal_df()
        rp = RetentionPanel(df, censor_date=date(2024, 6, 30))
        result = rp.build()
        assert "earned_exposure" in result.columns

    def test_earned_exposure_between_0_and_1(self):
        df = make_minimal_df()
        rp = RetentionPanel(df, censor_date=date(2024, 6, 30))
        result = rp.build()
        assert (result["earned_exposure"] >= 0).all()
        assert (result["earned_exposure"] <= 1.001).all()

    def test_policy_weight_column_is_ones(self):
        df = make_minimal_df()
        rp = RetentionPanel(df, censor_date=date(2024, 6, 30))
        result = rp.build()
        assert "policy_weight" in result.columns
        np.testing.assert_array_equal(result["policy_weight"].to_numpy(), np.ones(len(df)))

    def test_treatment_variation_flag_column_added(self):
        df = make_minimal_df()
        rp = RetentionPanel(df, censor_date=date(2024, 6, 30))
        result = rp.build()
        assert "treatment_variation_flag" in result.columns

    def test_original_columns_preserved(self):
        df = make_minimal_df()
        rp = RetentionPanel(df, censor_date=date(2024, 6, 30))
        result = rp.build()
        for col in df.columns:
            assert col in result.columns

    def test_missing_expiring_premium_raises(self):
        df = make_minimal_df().drop("expiring_premium")
        rp = RetentionPanel(df, censor_date=date(2024, 6, 30))
        with pytest.raises(ValueError, match="missing required columns"):
            rp.build()

    def test_zero_expiring_premium_raises(self):
        df = make_minimal_df()
        df = df.with_columns(pl.lit(0.0).alias("expiring_premium"))
        rp = RetentionPanel(df, censor_date=date(2024, 6, 30))
        with pytest.raises(ValueError):
            rp.build()

    def test_censor_date_inferred_if_not_provided(self):
        df = make_minimal_df()
        rp = RetentionPanel(df)
        with pytest.warns(UserWarning):
            result = rp.build()
        assert "censored_flag" in result.columns

    def test_without_start_date_col(self):
        df = make_minimal_df().drop("start_date")
        rp = RetentionPanel(df, start_date_col="start_date", censor_date=date(2024, 6, 30))
        result = rp.build()
        np.testing.assert_array_equal(result["earned_exposure"].to_numpy(), np.ones(len(df)))

    def test_enbp_col_none(self):
        """Panel builds without ENBP column."""
        df = make_minimal_df().drop("enbp")
        rp = RetentionPanel(df, enbp_col=None, censor_date=date(2024, 6, 30))
        result = rp.build()
        assert "enbp" not in result.columns

    def test_accepts_pandas_dataframe(self):
        """Polars conversion from pandas works."""
        try:
            import pandas as pd
        except ImportError:
            pytest.skip("pandas not installed")
        df_pl = make_minimal_df()
        df_pd = df_pl.to_pandas()
        rp = RetentionPanel(df_pd, censor_date=date(2024, 6, 30))
        result = rp.build()
        assert isinstance(result, pl.DataFrame)

    def test_invalid_type_raises(self):
        with pytest.raises(TypeError):
            RetentionPanel({"a": [1, 2]}, censor_date=date(2024, 6, 30))


class TestTreatmentVariationReport:
    def test_returns_dataframe(self, built_panel):
        from insurance_uplift.data import RetentionPanel

        # Need a RetentionPanel with _built set
        rp = RetentionPanel.__new__(RetentionPanel)
        rp._built = built_panel.filter(pl.col("censored_flag") == 0)
        result = rp.treatment_variation_report()
        assert isinstance(result, pl.DataFrame)

    def test_raises_if_not_built(self):
        df = make_minimal_df()
        rp = RetentionPanel(df)
        with pytest.raises(RuntimeError, match="build"):
            rp.treatment_variation_report()

    def test_with_confounder_cols(self, built_panel):
        from insurance_uplift.data import RetentionPanel

        rp = RetentionPanel.__new__(RetentionPanel)
        rp._built = built_panel.filter(pl.col("censored_flag") == 0)
        result = rp.treatment_variation_report(confounder_cols=["region"])
        assert "region" in result.columns
        assert "mean_treatment" in result.columns

    def test_low_variation_flag_present(self, built_panel):
        from insurance_uplift.data import RetentionPanel

        rp = RetentionPanel.__new__(RetentionPanel)
        rp._built = built_panel.filter(pl.col("censored_flag") == 0)
        result = rp.treatment_variation_report(confounder_cols=["region"])
        assert "low_variation_flag" in result.columns
