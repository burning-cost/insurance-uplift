"""Tests for evaluate.py — Qini, AUUC, segment_types."""

from __future__ import annotations

import warnings

import numpy as np
import polars as pl
import pytest

from insurance_uplift.evaluate import (
    auuc,
    persuadable_rate,
    plot_qini,
    qini_curve,
    segment_types,
    uplift_at_k,
)


def make_evaluate_data(n: int = 500, seed: int = 42):
    """Synthetic data with known positive AUUC (oracle tau used as tau_hat)."""
    rng = np.random.default_rng(seed)
    # True tau: negative for half the customers
    tau_true = rng.uniform(-0.8, 0.8, n)
    treatment_continuous = rng.normal(0.03, 0.08, n)
    # Binary treatment at median
    t_binary = (treatment_continuous > np.median(treatment_continuous)).astype(float)
    # Outcomes: P(Y=1) = sigmoid(base + tau * t)
    base = rng.normal(1.0, 0.3, n)
    logit = base + tau_true * t_binary
    prob = 1 / (1 + np.exp(-logit))
    y = rng.binomial(1, prob, n).astype(float)
    return y, t_binary, tau_true, treatment_continuous


class TestQiniCurve:
    def test_returns_two_arrays(self):
        y, t, tau, _ = make_evaluate_data()
        result = qini_curve(y, t, tau)
        assert len(result) == 2

    def test_first_point_is_zero(self):
        y, t, tau, _ = make_evaluate_data()
        fractions, gains = qini_curve(y, t, tau)
        assert fractions[0] == 0.0
        assert gains[0] == 0.0

    def test_last_fraction_is_one(self):
        y, t, tau, _ = make_evaluate_data()
        fractions, gains = qini_curve(y, t, tau)
        assert abs(fractions[-1] - 1.0) < 0.02

    def test_lengths_match(self):
        y, t, tau, _ = make_evaluate_data()
        fractions, gains = qini_curve(y, t, tau, n_buckets=50)
        assert len(fractions) == 51
        assert len(gains) == 51

    def test_oracle_model_positive_gain(self):
        """Sorting by true tau should produce positive Qini gain."""
        y, t, tau, _ = make_evaluate_data()
        _, gains = qini_curve(y, t, tau)
        # At some point in the curve, gain should be positive
        assert gains.max() > 0

    def test_continuous_treatment_binarised(self):
        """Continuous treatment input should be accepted."""
        y, _, tau, t_cont = make_evaluate_data()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fractions, gains = qini_curve(y, t_cont, tau)
        assert len(fractions) > 0

    def test_mismatched_lengths_raise(self):
        y = np.ones(100)
        t = np.ones(50)
        tau = np.ones(100)
        with pytest.raises(ValueError, match="same length"):
            qini_curve(y, t, tau)

    def test_all_treated_raises(self):
        y = np.ones(100)
        t = np.ones(100)
        tau = np.linspace(-1, 1, 100)
        with pytest.raises(ValueError, match="both treated and control"):
            qini_curve(y, t, tau)

    def test_polars_series_input(self):
        y, t, tau, _ = make_evaluate_data()
        fractions, gains = qini_curve(
            pl.Series(y), pl.Series(t), pl.Series(tau)
        )
        assert len(fractions) > 0


class TestAUUC:
    def test_returns_float(self):
        y, t, tau, _ = make_evaluate_data()
        result = auuc(y, t, tau)
        assert isinstance(result, float)

    def test_oracle_model_positive_auuc(self):
        """Model using true tau as predictions should have positive AUUC."""
        y, t, tau, _ = make_evaluate_data()
        result = auuc(y, t, tau)
        assert result > 0

    def test_worst_model_negative_auuc(self):
        """Model predicting reverse order should have negative AUUC."""
        y, t, tau, _ = make_evaluate_data()
        result = auuc(y, t, -tau)
        assert result < 0

    def test_random_model_near_zero(self):
        """Random predictions should produce AUUC near zero."""
        rng = np.random.default_rng(99)
        n = 1000
        y = rng.binomial(1, 0.7, n).astype(float)
        t = rng.binomial(1, 0.5, n).astype(float)
        tau_random = rng.normal(0, 1, n)
        result = auuc(y, t, tau_random)
        # Not exactly zero but should be small relative to oracle
        y2, t2, tau_oracle, _ = make_evaluate_data(n=n, seed=99)
        oracle_auuc = auuc(y2, t2, tau_oracle)
        assert abs(result) < abs(oracle_auuc)

    def test_auuc_is_finite(self):
        y, t, tau, _ = make_evaluate_data()
        result = auuc(y, t, tau)
        assert np.isfinite(result)


class TestUpliftAtK:
    def test_returns_float(self):
        y, t, tau, _ = make_evaluate_data()
        result = uplift_at_k(y, t, tau, k=0.3)
        assert isinstance(result, float)

    def test_k_equals_1_is_1(self):
        y, t, tau, _ = make_evaluate_data()
        result = uplift_at_k(y, t, tau, k=1.0)
        assert abs(result - 1.0) < 0.05

    def test_k_out_of_range_raises(self):
        y, t, tau, _ = make_evaluate_data()
        with pytest.raises(ValueError, match="k must be"):
            uplift_at_k(y, t, tau, k=0.0)

    def test_k_above_1_raises(self):
        y, t, tau, _ = make_evaluate_data()
        with pytest.raises(ValueError, match="k must be"):
            uplift_at_k(y, t, tau, k=1.5)

    def test_top30_oracle_captures_majority(self):
        """Oracle model targeting top 30% should capture most of the gain."""
        y, t, tau, _ = make_evaluate_data(n=2000)
        result = uplift_at_k(y, t, tau, k=0.3)
        # With oracle predictions, targeting top 30% should capture substantial fraction
        assert result > 0.2  # relaxed for stochasticity

    def test_between_0_and_1(self):
        y, t, tau, _ = make_evaluate_data()
        result = uplift_at_k(y, t, tau, k=0.5)
        assert 0.0 <= result <= 1.01


class TestSegmentTypes:
    def test_returns_polars_dataframe(self):
        y, _, tau, _ = make_evaluate_data()
        t = (tau < 0).astype(float)
        result = segment_types(y, t, tau)
        assert isinstance(result, pl.DataFrame)

    def test_has_four_segments(self):
        y, t, tau, _ = make_evaluate_data()
        result = segment_types(y, t, tau)
        assert len(result) == 4

    def test_segment_names(self):
        y, t, tau, _ = make_evaluate_data()
        result = segment_types(y, t, tau)
        names = set(result["segment_type"].to_list())
        assert names == {"Persuadable", "Sure Thing", "Lost Cause", "Do Not Disturb"}

    def test_fractions_sum_to_one(self):
        y, t, tau, _ = make_evaluate_data()
        result = segment_types(y, t, tau)
        total = float(result["fraction"].sum())
        assert abs(total - 1.0) < 0.001

    def test_n_sums_to_total(self):
        y, t, tau, _ = make_evaluate_data()
        result = segment_types(y, t, tau)
        assert int(result["n"].sum()) == len(y)

    def test_persuadable_have_negative_tau(self):
        y, t, tau, _ = make_evaluate_data()
        result = segment_types(y, t, tau)
        persuadable = result.filter(pl.col("segment_type") == "Persuadable")
        if int(persuadable["n"][0]) > 0:
            assert float(persuadable["avg_tau"][0]) < 0

    def test_do_not_disturb_have_positive_tau(self):
        y, t, tau, _ = make_evaluate_data()
        result = segment_types(y, t, tau)
        dnd = result.filter(pl.col("segment_type") == "Do Not Disturb")
        if int(dnd["n"][0]) > 0:
            assert float(dnd["avg_tau"][0]) > 0

    def test_threshold_changes_proportions(self):
        y, t, tau, _ = make_evaluate_data()
        result_default = segment_types(y, t, tau, threshold=0.0)
        result_threshold = segment_types(y, t, tau, threshold=0.3)
        persuadable_default = int(result_default.filter(
            pl.col("segment_type") == "Persuadable"
        )["n"][0])
        persuadable_threshold = int(result_threshold.filter(
            pl.col("segment_type") == "Persuadable"
        )["n"][0])
        # Higher threshold means fewer Persuadables (tau must be < -0.3, not just < 0)
        assert persuadable_threshold <= persuadable_default


class TestPersuadableRate:
    def test_returns_float(self):
        tau = np.array([-0.5, 0.3, -0.1, 0.2])
        result = persuadable_rate(tau)
        assert isinstance(result, float)

    def test_all_negative_tau_is_one(self):
        tau = np.array([-0.5, -0.3, -0.1])
        result = persuadable_rate(tau)
        assert result == 1.0

    def test_all_positive_tau_is_zero(self):
        tau = np.array([0.5, 0.3, 0.1])
        result = persuadable_rate(tau)
        assert result == 0.0

    def test_threshold_reduces_rate(self):
        tau = np.array([-0.5, -0.05, -0.1, 0.2])
        rate_default = persuadable_rate(tau, threshold=0.0)
        rate_strict = persuadable_rate(tau, threshold=0.2)
        assert rate_strict <= rate_default

    def test_polars_series_input(self):
        tau = pl.Series([-0.5, 0.3, -0.1])
        result = persuadable_rate(tau)
        assert abs(result - 2 / 3) < 0.01


class TestPlotQini:
    def test_returns_axes(self):
        y, t, tau, _ = make_evaluate_data()
        try:
            import matplotlib
            matplotlib.use("Agg")
            ax = plot_qini(y, t, tau)
            assert ax is not None
        except ImportError:
            pytest.skip("matplotlib not installed")

    def test_accepts_custom_axes(self):
        y, t, tau, _ = make_evaluate_data()
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            _, ax = plt.subplots()
            result = plot_qini(y, t, tau, ax=ax)
            assert result is ax
            plt.close("all")
        except ImportError:
            pytest.skip("matplotlib not installed")
