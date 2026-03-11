"""Tests for fit.py — RetentionUpliftModel."""

from __future__ import annotations

import warnings

import numpy as np
import polars as pl
import pytest

from insurance_uplift.fit import RetentionUpliftModel


class TestRetentionUpliftModelInit:
    def test_default_estimator(self):
        model = RetentionUpliftModel()
        assert model.estimator == "causal_forest"

    def test_invalid_estimator_raises(self):
        with pytest.raises(ValueError, match="estimator"):
            RetentionUpliftModel(estimator="random_forest")

    def test_invalid_outcome_raises(self):
        with pytest.raises(ValueError, match="outcome"):
            RetentionUpliftModel(outcome="time_series")

    def test_dr_learner_estimator(self):
        model = RetentionUpliftModel(estimator="dr_learner")
        assert model.estimator == "dr_learner"

    def test_x_learner_estimator(self):
        model = RetentionUpliftModel(estimator="x_learner")
        assert model.estimator == "x_learner"


class TestRetentionUpliftModelFit:
    def test_fit_returns_self(self, clean_panel):
        model = RetentionUpliftModel(n_estimators=50, inference=False, random_state=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = model.fit(clean_panel, confounders=["age", "ncd"])
        assert result is model

    def test_is_fitted_after_fit(self, clean_panel):
        model = RetentionUpliftModel(n_estimators=50, inference=False, random_state=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(clean_panel, confounders=["age", "ncd"])
        assert model._is_fitted

    def test_confounders_required(self, clean_panel):
        model = RetentionUpliftModel(n_estimators=50)
        with pytest.raises(ValueError, match="confounders"):
            model.fit(clean_panel, confounders=None)

    def test_missing_confounder_col_raises(self, clean_panel):
        model = RetentionUpliftModel(n_estimators=50)
        with pytest.raises(ValueError, match="missing"):
            model.fit(clean_panel, confounders=["age", "nonexistent_col"])

    def test_censored_policies_raise(self, built_panel):
        model = RetentionUpliftModel(n_estimators=50)
        # built_panel has censored_flag > 0 for some rows
        with pytest.raises(ValueError, match="censored"):
            model.fit(built_panel, confounders=["age"])

    def test_fit_with_dr_learner(self, clean_panel):
        model = RetentionUpliftModel(
            estimator="dr_learner", n_estimators=50, inference=False, random_state=0
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(clean_panel, confounders=["age", "ncd"])
        assert model._is_fitted

    def test_fit_with_x_learner(self, clean_panel):
        model = RetentionUpliftModel(estimator="x_learner", random_state=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(clean_panel, confounders=["age", "ncd"])
        assert model._is_fitted

    def test_nan_in_features_raises(self, clean_panel):
        panel_nan = clean_panel.with_columns(
            pl.when(pl.col("policy_id") == "POL00001")
            .then(None)
            .otherwise(pl.col("age"))
            .alias("age")
        )
        model = RetentionUpliftModel(n_estimators=50)
        with pytest.raises(ValueError, match="NaN"):
            model.fit(panel_nan, confounders=["age"])


class TestRetentionUpliftModelCate:
    def test_cate_returns_series(self, fitted_model, clean_panel):
        tau = fitted_model.cate(clean_panel)
        assert isinstance(tau, pl.Series)

    def test_cate_length_matches_input(self, fitted_model, clean_panel):
        tau = fitted_model.cate(clean_panel)
        assert len(tau) == len(clean_panel)

    def test_cate_name_is_tau_hat(self, fitted_model, clean_panel):
        tau = fitted_model.cate(clean_panel)
        assert tau.name == "tau_hat"

    def test_cate_is_numeric(self, fitted_model, clean_panel):
        tau = fitted_model.cate(clean_panel)
        assert tau.dtype in (pl.Float64, pl.Float32)

    def test_cate_has_variation(self, fitted_model, clean_panel):
        """CATE should not be constant — there should be heterogeneity."""
        tau = fitted_model.cate(clean_panel)
        assert float(tau.std()) > 0.01

    def test_cate_before_fit_raises(self, clean_panel):
        model = RetentionUpliftModel()
        with pytest.raises(RuntimeError, match="fit"):
            model.cate(clean_panel)

    def test_sign_direction_sensible(self, fitted_model, clean_panel):
        """Young customers (age<40) should have more negative tau than old (age>60)."""
        tau = fitted_model.cate(clean_panel)
        young = clean_panel.filter(pl.col("age") < 40)
        old = clean_panel.filter(pl.col("age") > 60)
        tau_young = fitted_model.cate(young)
        tau_old = fitted_model.cate(old)
        # On average, young should be more price-sensitive (more negative tau)
        assert float(tau_young.mean()) < float(tau_old.mean())


class TestRetentionUpliftModelCateInference:
    def test_returns_three_series(self, fitted_model, clean_panel):
        result = fitted_model.cate_inference(clean_panel)
        assert len(result) == 3

    def test_ci_covers_point_estimate(self, fitted_model, clean_panel):
        tau, lo, hi = fitted_model.cate_inference(clean_panel)
        assert (lo.to_numpy() <= tau.to_numpy() + 1e-6).all()
        assert (hi.to_numpy() >= tau.to_numpy() - 1e-6).all()

    def test_series_names(self, fitted_model, clean_panel):
        tau, lo, hi = fitted_model.cate_inference(clean_panel)
        assert tau.name == "tau_hat"
        assert lo.name == "lower_95"
        assert hi.name == "upper_95"

    def test_raises_without_inference(self, clean_panel):
        model = RetentionUpliftModel(n_estimators=50, inference=False, random_state=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(clean_panel, confounders=["age", "ncd"])
        with pytest.raises(RuntimeError, match="inference=True"):
            model.cate_inference(clean_panel)


class TestRetentionUpliftModelAte:
    def test_returns_three_floats(self, fitted_model):
        result = fitted_model.ate()
        assert len(result) == 3
        assert all(isinstance(v, float) for v in result)

    def test_ci_ordering(self, fitted_model):
        ate, lo, hi = fitted_model.ate()
        assert lo <= ate + 1e-6
        assert hi >= ate - 1e-6

    def test_ate_is_finite(self, fitted_model):
        ate, lo, hi = fitted_model.ate()
        assert np.isfinite(ate)
        assert np.isfinite(lo)
        assert np.isfinite(hi)

    def test_ate_before_fit_raises(self):
        model = RetentionUpliftModel()
        with pytest.raises(RuntimeError):
            model.ate()


class TestRetentionUpliftModelGate:
    def test_returns_dataframe(self, fitted_model, clean_panel):
        result = fitted_model.gate(clean_panel, by="region")
        assert isinstance(result, pl.DataFrame)

    def test_has_required_columns(self, fitted_model, clean_panel):
        result = fitted_model.gate(clean_panel, by="region")
        assert "group" in result.columns
        assert "gate" in result.columns
        assert "n" in result.columns

    def test_groups_match_unique_values(self, fitted_model, clean_panel):
        result = fitted_model.gate(clean_panel, by="region")
        n_groups = len(clean_panel["region"].unique())
        assert len(result) == n_groups

    def test_missing_by_col_raises(self, fitted_model, clean_panel):
        with pytest.raises(ValueError, match="not found"):
            fitted_model.gate(clean_panel, by="nonexistent")
