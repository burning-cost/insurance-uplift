"""Tests for _utils.py shared helpers."""

import numpy as np
import polars as pl
import pytest

from insurance_uplift._utils import (
    binarise_treatment,
    log_price_ratio,
    safe_divide,
    segment_label,
    to_numpy,
    to_numpy_2d,
    validate_min_samples,
    validate_panel_columns,
)


class TestToNumpy:
    def test_list_input(self):
        result = to_numpy([1.0, 2.0, 3.0])
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float64
        np.testing.assert_array_equal(result, [1.0, 2.0, 3.0])

    def test_polars_series_input(self):
        s = pl.Series("x", [1.0, 2.0, 3.0])
        result = to_numpy(s)
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, [1.0, 2.0, 3.0])

    def test_numpy_array_passthrough(self):
        arr = np.array([1.0, 2.0])
        result = to_numpy(arr)
        assert isinstance(result, np.ndarray)

    def test_integer_input_cast_to_float64(self):
        result = to_numpy([1, 2, 3])
        assert result.dtype == np.float64


class TestToNumpy2d:
    def test_polars_dataframe(self):
        df = pl.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
        result = to_numpy_2d(df)
        assert result.shape == (2, 2)
        assert result.dtype == np.float64

    def test_1d_numpy_becomes_2d(self):
        arr = np.array([1.0, 2.0, 3.0])
        result = to_numpy_2d(arr)
        assert result.shape == (3, 1)

    def test_2d_numpy_passthrough(self):
        arr = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = to_numpy_2d(arr)
        assert result.shape == (2, 2)


class TestLogPriceRatio:
    def test_equal_premiums(self):
        result = log_price_ratio([100.0], [100.0])
        np.testing.assert_allclose(result, [0.0], atol=1e-10)

    def test_price_increase(self):
        result = log_price_ratio([110.0], [100.0])
        assert float(result[0]) > 0

    def test_price_decrease(self):
        result = log_price_ratio([90.0], [100.0])
        assert float(result[0]) < 0

    def test_10pct_increase_approx_log(self):
        result = log_price_ratio([110.0], [100.0])
        np.testing.assert_allclose(result, [np.log(1.1)], rtol=1e-6)

    def test_zero_expiring_raises(self):
        with pytest.raises(ValueError, match="expiring_premium"):
            log_price_ratio([100.0], [0.0])

    def test_negative_expiring_raises(self):
        with pytest.raises(ValueError):
            log_price_ratio([100.0], [-50.0])

    def test_zero_renewal_raises(self):
        with pytest.raises(ValueError, match="renewal_premium"):
            log_price_ratio([0.0], [100.0])

    def test_polars_series_input(self):
        r = pl.Series([110.0, 90.0])
        e = pl.Series([100.0, 100.0])
        result = log_price_ratio(r, e)
        assert len(result) == 2
        assert result[0] > 0
        assert result[1] < 0

    def test_multiple_policies(self):
        renewals = [110.0, 95.0, 100.0]
        expiring = [100.0, 100.0, 100.0]
        result = log_price_ratio(renewals, expiring)
        assert len(result) == 3


class TestBinariseTreatment:
    def test_above_median_is_one(self):
        t = np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10])
        result = binarise_treatment(t)
        median = np.median(t)
        assert np.all(result[t > median] == 1)
        assert np.all(result[t <= median] == 0)

    def test_returns_binary(self):
        t = np.random.default_rng(0).normal(0, 0.05, 200)
        result = binarise_treatment(t)
        assert set(result).issubset({0, 1})

    def test_unsupported_method_raises(self):
        with pytest.raises(ValueError, match="median"):
            binarise_treatment(np.array([1.0, 2.0]), method="mean")

    def test_small_n_warns(self):
        t = np.concatenate([np.zeros(200), np.ones(5)])
        with pytest.warns(UserWarning, match="noisy"):
            binarise_treatment(t)


class TestValidatePanelColumns:
    def test_all_present_no_error(self):
        df = pl.DataFrame({"a": [1], "b": [2]})
        validate_panel_columns(df, ["a", "b"])  # no exception

    def test_missing_column_raises(self):
        df = pl.DataFrame({"a": [1]})
        with pytest.raises(ValueError, match="missing required columns"):
            validate_panel_columns(df, ["a", "b"])

    def test_error_message_includes_missing(self):
        df = pl.DataFrame({"a": [1]})
        with pytest.raises(ValueError, match="b"):
            validate_panel_columns(df, ["a", "b"])


class TestValidateMinSamples:
    def test_below_minimum_warns(self):
        with pytest.warns(UserWarning, match="observations"):
            validate_min_samples(50, min_n=200, context="test: ")

    def test_above_minimum_no_warning(self):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            validate_min_samples(500, min_n=200)


class TestSegmentLabel:
    def test_negative_tau_is_persuadable(self):
        assert segment_label(-0.5) == "Persuadable"

    def test_positive_tau_is_do_not_disturb(self):
        assert segment_label(0.5) == "Do Not Disturb"

    def test_zero_tau_is_near_zero(self):
        assert segment_label(0.0) == "Near Zero"

    def test_threshold_affects_boundary(self):
        assert segment_label(-0.05, threshold=0.1) == "Near Zero"
        assert segment_label(-0.2, threshold=0.1) == "Persuadable"


class TestSafeDivide:
    def test_normal_division(self):
        assert safe_divide(10.0, 2.0) == 5.0

    def test_zero_denominator_returns_fill(self):
        assert safe_divide(10.0, 0.0, fill=99.0) == 99.0

    def test_default_fill_is_zero(self):
        assert safe_divide(10.0, 0.0) == 0.0
