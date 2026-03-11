"""Integration tests: full pipeline from raw panel to ROI report."""

from __future__ import annotations

import warnings
from datetime import date

import numpy as np
import polars as pl
import pytest


class TestFullPipeline:
    """Run the complete pipeline and verify the outputs chain correctly."""

    def test_panel_to_cate_to_qini_pipeline(self, built_panel, fitted_model):
        """Panel -> CATE -> Qini in a single chain."""
        from insurance_uplift.evaluate import auuc

        clean = built_panel.filter(pl.col("censored_flag") == 0).with_columns(
            pl.col("renewed").cast(pl.Float64)
        )
        tau = fitted_model.cate(clean)
        score = auuc(clean["renewed"], clean["treatment"], tau)
        assert isinstance(score, float)
        assert np.isfinite(score)

    def test_cate_to_segment_types_pipeline(self, fitted_model, clean_panel):
        """CATE -> segment_types -> check proportions sum to 1."""
        from insurance_uplift.evaluate import segment_types

        tau = fitted_model.cate(clean_panel)
        segs = segment_types(clean_panel["renewed"], clean_panel["treatment"], tau)
        total_frac = float(segs["fraction"].sum())
        assert abs(total_frac - 1.0) < 0.001

    def test_cate_to_policy_tree_pipeline(self, fitted_model, clean_panel):
        """CATE -> PolicyTree -> recommendations."""
        from insurance_uplift.segment import PolicyTree

        tree = PolicyTree(fitted_model, max_depth=2)
        tree.fit(clean_panel)
        rec = tree.recommend(clean_panel)
        assert len(rec) == len(clean_panel)
        assert set(rec.to_list()).issubset({0, 1})

    def test_enbp_constraint_pipeline(self, clean_panel, fitted_model):
        """CATE -> rate recommendation -> ENBP clipping."""
        from insurance_uplift.constrain import ENBPConstraint

        tau = fitted_model.cate(clean_panel)
        # Convert tau to rate change recommendation: discount proportional to tau
        discount_log = np.log(0.90)
        rec_rate_change = pl.Series(
            (tau.to_numpy() * discount_log * 0.5).clip(-0.15, 0.0)
        )
        constraint = ENBPConstraint()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clipped = constraint.apply(clean_panel, rec_rate_change)
        assert len(clipped) == len(clean_panel)
        # Clipped values must be <= recommended (we can only reduce, not increase)
        assert (clipped.to_numpy() <= rec_rate_change.to_numpy() + 1e-6).all()

    def test_fairness_audit_pipeline(self, fitted_model, clean_panel):
        """CATE -> FairnessAudit -> flagged groups present."""
        from insurance_uplift.constrain import FairnessAudit

        tau = fitted_model.cate(clean_panel)
        audit = FairnessAudit(
            protected_proxies=["age_band"],
            vulnerability_threshold_age=70,
        )
        audit.fit(clean_panel.select(["age_band"]), tau)
        result = audit.audit()
        assert isinstance(result, pl.DataFrame)
        assert "flagged_as_vulnerable" in result.columns

    def test_roi_report_pipeline(self, fitted_model, clean_panel):
        """CATE -> PolicyTree -> ROIReport."""
        from insurance_uplift.constrain import ROIReport
        from insurance_uplift.segment import PolicyTree

        tree = PolicyTree(fitted_model, max_depth=2)
        tree.fit(clean_panel)
        rec = tree.recommend(clean_panel)
        tau = fitted_model.cate(clean_panel)

        roi = ROIReport(policy_premium_avg=650.0)
        result = roi.compute(clean_panel, tau, rec, discount_size=0.05)
        assert result["n_treated"] >= 0
        assert isinstance(result["net_roi"], float)
        assert np.isfinite(result["net_roi"])

    def test_gate_by_region_pipeline(self, fitted_model, clean_panel):
        """CATE -> GATE by region -> verify structure."""
        gate_result = fitted_model.gate(clean_panel, by="region")
        assert "group" in gate_result.columns
        assert "gate" in gate_result.columns
        n_regions = len(clean_panel["region"].unique())
        assert len(gate_result) == n_regions

    def test_segment_summary_pipeline(self, fitted_model, clean_panel):
        """CATE -> SegmentSummary -> segment table."""
        from insurance_uplift.segment import SegmentSummary

        summary = SegmentSummary(fitted_model, max_depth=2, min_samples_leaf=30)
        summary.fit(clean_panel)
        table = summary.segment_table()
        assert len(table) > 0
        assert "recommended_action" in table.columns

    def test_cate_inference_pipeline(self, fitted_model, clean_panel):
        """CATE with confidence intervals."""
        tau, lo, hi = fitted_model.cate_inference(clean_panel)
        assert len(tau) == len(lo) == len(hi)
        # CIs must bracket the point estimate
        assert (lo.to_numpy() <= tau.to_numpy() + 0.01).all()
        assert (hi.to_numpy() >= tau.to_numpy() - 0.01).all()

    def test_end_to_end_returns_serialisable_types(self, fitted_model, clean_panel):
        """All outputs should be Polars DataFrames, Series, or Python scalars."""
        from insurance_uplift.evaluate import auuc, segment_types
        from insurance_uplift.segment import PolicyTree
        from insurance_uplift.constrain import ROIReport

        tau = fitted_model.cate(clean_panel)
        assert isinstance(tau, pl.Series)

        score = auuc(clean_panel["renewed"], clean_panel["treatment"], tau)
        assert isinstance(score, float)

        segs = segment_types(clean_panel["renewed"], clean_panel["treatment"], tau)
        assert isinstance(segs, pl.DataFrame)

        tree = PolicyTree(fitted_model, max_depth=2)
        tree.fit(clean_panel)
        rec = tree.recommend(clean_panel)
        assert isinstance(rec, pl.Series)

        roi = ROIReport()
        result = roi.compute(clean_panel, tau, rec, discount_size=0.05)
        assert isinstance(result, dict)
