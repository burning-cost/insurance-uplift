"""Tests for segment.py — PolicyTree and SegmentSummary."""

from __future__ import annotations

import warnings

import numpy as np
import polars as pl
import pytest

from insurance_uplift.segment import PolicyTree, SegmentSummary


class TestPolicyTreeInit:
    def test_default_backend_is_sklearn(self, fitted_model):
        tree = PolicyTree(fitted_model)
        assert tree.backend == "sklearn"

    def test_max_depth_stored(self, fitted_model):
        tree = PolicyTree(fitted_model, max_depth=3)
        assert tree.max_depth == 3


class TestPolicyTreeFit:
    def test_fit_returns_self(self, fitted_model, clean_panel):
        tree = PolicyTree(fitted_model, max_depth=2)
        result = tree.fit(clean_panel)
        assert result is tree

    def test_is_fitted_after_fit(self, fitted_model, clean_panel):
        tree = PolicyTree(fitted_model, max_depth=2)
        tree.fit(clean_panel)
        assert tree._is_fitted

    def test_policytree_r_falls_back_with_warning(self, fitted_model, clean_panel):
        tree = PolicyTree(fitted_model, max_depth=2, backend="policytree_r")
        with pytest.warns(UserWarning, match="v0.2"):
            tree.fit(clean_panel)
        assert tree.backend == "sklearn"

    def test_unfit_model_raises(self, clean_panel):
        from insurance_uplift.fit import RetentionUpliftModel
        model = RetentionUpliftModel()
        tree = PolicyTree(model)
        with pytest.raises(RuntimeError, match="uplift_model must be fitted"):
            tree.fit(clean_panel)


class TestPolicyTreeRecommend:
    def test_returns_binary_series(self, fitted_model, clean_panel):
        tree = PolicyTree(fitted_model, max_depth=2)
        tree.fit(clean_panel)
        rec = tree.recommend(clean_panel)
        assert isinstance(rec, pl.Series)
        assert set(rec.to_list()).issubset({0, 1})

    def test_length_matches_input(self, fitted_model, clean_panel):
        tree = PolicyTree(fitted_model, max_depth=2)
        tree.fit(clean_panel)
        rec = tree.recommend(clean_panel)
        assert len(rec) == len(clean_panel)

    def test_series_name_is_recommend(self, fitted_model, clean_panel):
        tree = PolicyTree(fitted_model, max_depth=2)
        tree.fit(clean_panel)
        rec = tree.recommend(clean_panel)
        assert rec.name == "recommend"

    def test_budget_constraint_limits_targets(self, fitted_model, clean_panel):
        budget = 0.2
        tree = PolicyTree(fitted_model, max_depth=2, budget_constraint=budget)
        tree.fit(clean_panel)
        rec = tree.recommend(clean_panel)
        n_targeted = int(rec.sum())
        max_allowed = int(budget * len(clean_panel)) + 1
        assert n_targeted <= max_allowed

    def test_recommend_before_fit_raises(self, fitted_model, clean_panel):
        tree = PolicyTree(fitted_model)
        with pytest.raises(RuntimeError, match="fit"):
            tree.recommend(clean_panel)


class TestPolicyTreeWelfareGain:
    def test_returns_float(self, fitted_model, clean_panel):
        tree = PolicyTree(fitted_model, max_depth=2)
        tree.fit(clean_panel)
        gain = tree.welfare_gain()
        assert isinstance(gain, float)

    def test_welfare_gain_non_negative(self, fitted_model, clean_panel):
        tree = PolicyTree(fitted_model, max_depth=2)
        tree.fit(clean_panel)
        gain = tree.welfare_gain()
        assert gain >= 0.0

    def test_welfare_before_fit_raises(self, fitted_model):
        tree = PolicyTree(fitted_model)
        with pytest.raises(RuntimeError, match="fit"):
            tree.welfare_gain()


class TestPolicyTreeExportRules:
    def test_returns_list_of_dicts(self, fitted_model, clean_panel):
        tree = PolicyTree(fitted_model, max_depth=2)
        tree.fit(clean_panel)
        rules = tree.export_rules()
        assert isinstance(rules, list)
        assert all(isinstance(r, dict) for r in rules)

    def test_rule_dict_has_required_keys(self, fitted_model, clean_panel):
        tree = PolicyTree(fitted_model, max_depth=2)
        tree.fit(clean_panel)
        rules = tree.export_rules()
        for rule in rules:
            assert "node_id" in rule
            assert "rule" in rule
            assert "avg_tau" in rule
            assert "action" in rule

    def test_action_is_target_for_negative_tau(self, fitted_model, clean_panel):
        tree = PolicyTree(fitted_model, max_depth=2)
        tree.fit(clean_panel)
        rules = tree.export_rules()
        for rule in rules:
            if rule["avg_tau"] < 0:
                assert rule["action"] == "target"
            else:
                assert rule["action"] == "do_not_target"

    def test_export_before_fit_raises(self, fitted_model):
        tree = PolicyTree(fitted_model)
        with pytest.raises(RuntimeError):
            tree.export_rules()

    def test_depth_2_at_most_4_leaves(self, fitted_model, clean_panel):
        tree = PolicyTree(fitted_model, max_depth=2)
        tree.fit(clean_panel)
        rules = tree.export_rules()
        assert len(rules) <= 4


class TestSegmentSummary:
    def test_fit_returns_self(self, fitted_model, clean_panel):
        summary = SegmentSummary(fitted_model, max_depth=2)
        result = summary.fit(clean_panel)
        assert result is summary

    def test_segment_table_returns_dataframe(self, fitted_model, clean_panel):
        summary = SegmentSummary(fitted_model, max_depth=2, min_samples_leaf=20)
        summary.fit(clean_panel)
        table = summary.segment_table()
        assert isinstance(table, pl.DataFrame)

    def test_segment_table_has_required_columns(self, fitted_model, clean_panel):
        summary = SegmentSummary(fitted_model, max_depth=2, min_samples_leaf=20)
        summary.fit(clean_panel)
        table = summary.segment_table()
        required = ["segment_id", "rule_description", "n", "avg_tau",
                    "recommended_action", "avg_treatment_effect_pp"]
        for col in required:
            assert col in table.columns

    def test_recommended_action_valid_values(self, fitted_model, clean_panel):
        summary = SegmentSummary(fitted_model, max_depth=2, min_samples_leaf=20)
        summary.fit(clean_panel)
        table = summary.segment_table()
        valid = {"offer_discount", "hold_rate"}
        assert set(table["recommended_action"].to_list()).issubset(valid)

    def test_segment_table_before_fit_raises(self, fitted_model):
        summary = SegmentSummary(fitted_model)
        with pytest.raises(RuntimeError, match="fit"):
            summary.segment_table()

    def test_effect_pp_non_negative_for_discount_segments(self, fitted_model, clean_panel):
        summary = SegmentSummary(fitted_model, max_depth=2, min_samples_leaf=20)
        summary.fit(clean_panel)
        table = summary.segment_table()
        discount_segs = table.filter(pl.col("recommended_action") == "offer_discount")
        assert (discount_segs["avg_treatment_effect_pp"] >= 0).all()

    def test_plot_tree_returns_axes(self, fitted_model, clean_panel):
        try:
            import matplotlib
            matplotlib.use("Agg")
            summary = SegmentSummary(fitted_model, max_depth=2, min_samples_leaf=20)
            summary.fit(clean_panel)
            ax = summary.plot_tree()
            assert ax is not None
            import matplotlib.pyplot as plt
            plt.close("all")
        except ImportError:
            pytest.skip("matplotlib not installed")
