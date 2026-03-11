"""insurance-uplift: heterogeneous treatment effects for UK personal lines retention targeting.

This library addresses a specific problem in UK personal lines insurance renewal
pricing: given a renewal panel with observed outcomes and price changes, which
customers should be targeted with a retention discount, and what is the expected
ROI?

It wraps EconML's CausalForestDML behind an insurance-vocabulary API: renewal
panels, CATE estimates, Qini curves, the four-customer taxonomy from Guelman
et al., ENBP-constrained recommendations, and Consumer Duty fairness audits.

Modules
-------
data
    :class:`~insurance_uplift.data.RetentionPanel` — build a panel from a policy
    extract, compute log price ratio treatment, handle censored policies.
fit
    :class:`~insurance_uplift.fit.RetentionUpliftModel` — estimate CATE via
    CausalForestDML or DRLearner. Methods: ``.cate()``, ``.cate_inference()``,
    ``.ate()``, ``.gate()``.
evaluate
    :func:`~insurance_uplift.evaluate.qini_curve`,
    :func:`~insurance_uplift.evaluate.auuc`,
    :func:`~insurance_uplift.evaluate.uplift_at_k`,
    :func:`~insurance_uplift.evaluate.segment_types`,
    :func:`~insurance_uplift.evaluate.plot_qini` — targeting quality metrics.
segment
    :class:`~insurance_uplift.segment.PolicyTree`,
    :class:`~insurance_uplift.segment.SegmentSummary` — decision tree segmentation
    of τ̂(x) into actionable targeting rules.
constrain
    :class:`~insurance_uplift.constrain.ENBPConstraint`,
    :class:`~insurance_uplift.constrain.FairnessAudit`,
    :class:`~insurance_uplift.constrain.ROIReport` — regulatory compliance layer.

Quick start
-----------
::

    import polars as pl
    from insurance_uplift.data import RetentionPanel
    from insurance_uplift.fit import RetentionUpliftModel
    from insurance_uplift.evaluate import auuc, segment_types
    from insurance_uplift.constrain import ENBPConstraint, FairnessAudit

    # 1. Build panel
    panel_obj = RetentionPanel(df, enbp_col='enbp')
    panel = panel_obj.build()
    panel_clean = panel.filter(pl.col('censored_flag') == 0)

    # 2. Fit CATE model
    model = RetentionUpliftModel(estimator='causal_forest')
    model.fit(panel_clean, confounders=['age', 'ncd', 'vehicle_age', 'region'])

    # 3. Evaluate
    tau = model.cate(panel_clean)
    score = auuc(panel_clean['renewed'], panel_clean['treatment'], tau)
    segments = segment_types(panel_clean['renewed'], panel_clean['treatment'], tau)

    # 4. Constrain and audit
    constraint = ENBPConstraint()
    clipped = constraint.apply(panel_clean, recommended_rate_changes)
    audit = FairnessAudit(protected_proxies=['age_band'])
    audit.fit(panel_clean.select(['age_band']), tau)
    fairness_table = audit.audit()
"""

from .constrain import ENBPConstraint, FairnessAudit, ROIReport
from .data import RetentionPanel
from .evaluate import auuc, persuadable_rate, plot_qini, qini_curve, segment_types, uplift_at_k
from .fit import RetentionUpliftModel
from .segment import PolicyTree, SegmentSummary

__version__ = "0.1.0"
__all__ = [
    "RetentionPanel",
    "RetentionUpliftModel",
    "qini_curve",
    "auuc",
    "uplift_at_k",
    "segment_types",
    "persuadable_rate",
    "plot_qini",
    "PolicyTree",
    "SegmentSummary",
    "ENBPConstraint",
    "FairnessAudit",
    "ROIReport",
]
