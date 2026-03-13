# insurance-uplift

Heterogeneous treatment effects for UK personal lines insurance retention targeting.

## The problem

Every insurer running a renewal discount campaign faces the same question: which customers will lapse without a discount, and which would renew regardless? Getting this wrong in either direction is expensive. Discount a "Sure Thing" and you've given away margin for a renewal you'd have kept anyway. Miss a "Persuadable" and you lose a customer who would have stayed for a modest reduction.

Standard approaches — segment by NCD band, apply a blanket 5% retention discount, track overall renewal rate — don't answer this question. They measure average effects, not individual ones.

FCA PS21/5 (GIPP) makes this more urgent. Price walking is eliminated. You can't rely on inelastic customers overpaying indefinitely. You need to earn retention, and that requires knowing who responds to what.

The causal ML methods to answer this have existed since 2018 (causal forests, DRLearner). The insurance-specific pipeline — panel construction, continuous price treatment, ENBP compliance, Consumer Duty fairness audit — has not existed in Python until now.

## What this library does

1. **Build a renewal panel** from a raw policy extract. Handles censored policies (those still in-force at extract date). Computes the continuous treatment: `log(renewal_premium / expiring_premium)`.

2. **Estimate per-customer CATE** via CausalForestDML (EconML). τ̂(x) is the predicted marginal effect of a 1-unit log price increase on renewal probability. A customer with τ̂ = -0.8 will lose about 8 pp renewal probability from a +10% price increase.

3. **Evaluate targeting quality** with Qini curves and AUUC. These measure whether sorting customers by τ̂ actually identifies the ones whose behaviour changes — not whether τ̂ is a good point estimate.

4. **Classify customers** into the Guelman four-customer taxonomy: Persuadable, Sure Thing, Lost Cause, Do Not Disturb.

5. **Build a policy tree** that converts τ̂(x) into a human-readable decision rule: "target customers where age ≤ 45 AND NCD band ≤ 2."

6. **Apply ENBP constraint** (ICOBS 6B.2). Clips every recommendation to the equivalent new business price. Non-negotiable under FCA PS21/5.

7. **Run a Consumer Duty fairness audit**. Checks whether inelastic segments are dominated by vulnerability proxies (age >70, low-income postcode). Flags groups where targeting would raise Consumer Duty concerns.

8. **Compute campaign ROI**: expected incremental renewals, discount cost, net revenue impact.

## Installation

```bash
pip install insurance-uplift
```

Dependencies: polars, econml, catboost, scipy, matplotlib, numpy, scikit-learn.

## Quick start

```python
import numpy as np
import polars as pl
from datetime import date
from insurance_uplift.data import RetentionPanel
from insurance_uplift.fit import RetentionUpliftModel
from insurance_uplift.evaluate import auuc, segment_types
from insurance_uplift.segment import PolicyTree
from insurance_uplift.constrain import ENBPConstraint, FairnessAudit, ROIReport

rng = np.random.default_rng(42)
n = 1000

# Synthetic UK motor renewal panel
age = rng.integers(18, 75, n)
ncd = rng.integers(0, 5, n)
vehicle_age = rng.integers(0, 12, n)
region = rng.choice(["London", "SE", "NW", "Midlands", "Scotland"], n)
age_band = (age // 10 * 10).astype(str)
postcode_income_decile = rng.integers(1, 11, n)

expiring_premium = rng.uniform(300, 1200, n)
# True price elasticity: younger and higher-NCD customers more sensitive
true_tau = -0.4 - 0.005 * np.maximum(40 - age, 0) - 0.03 * ncd
# Rate change: mix of increases and flat renewals (post-GIPP, no walkbacks)
rate_change_log = rng.normal(0.03, 0.05, n)
renewal_premium = expiring_premium * np.exp(rate_change_log)
enbp = expiring_premium * rng.uniform(0.88, 0.98, n)  # ENBP slightly below expiring

# Renewal probability: logistic function of true_tau * rate_change
renew_prob = 1 / (1 + np.exp(-(0.8 + true_tau * rate_change_log)))
renewed = rng.binomial(1, renew_prob)

# Policy dates (all ending within the last 12 months)
import datetime
end_dates = [
    date(2024, 9, 30) - datetime.timedelta(days=int(d))
    for d in rng.integers(0, 365, n)
]
start_dates = [
    ed - datetime.timedelta(days=365)
    for ed in end_dates
]

df = pl.DataFrame({
    "policy_id": [f"POL{i:05d}" for i in range(n)],
    "age": age,
    "ncd": ncd,
    "vehicle_age": vehicle_age,
    "region": region,
    "age_band": age_band,
    "postcode_income_decile": postcode_income_decile,
    "expiring_premium": expiring_premium,
    "renewal_premium": renewal_premium,
    "enbp": enbp,
    "renewed": renewed,
    "start_date": start_dates,
    "end_date": end_dates,
})

# 1. Build the panel
panel_obj = RetentionPanel(
    df,
    renewal_premium_col="renewal_premium",
    expiring_premium_col="expiring_premium",
    renewal_indicator_col="renewed",
    enbp_col="enbp",
    censor_date=date(2024, 9, 30),
)
panel = panel_obj.build()
clean = panel.filter(pl.col("censored_flag") == 0)

# 2. Fit the CATE model
model = RetentionUpliftModel(estimator="causal_forest", n_estimators=2000)
model.fit(clean, confounders=["age", "ncd", "vehicle_age", "region"])

# 3. Get CATE estimates
tau = model.cate(clean)
ate, lo, hi = model.ate()
print(f"ATE: {ate:.4f} [{lo:.4f}, {hi:.4f}]")

# 4. Evaluate targeting
score = auuc(clean["renewed"], clean["treatment"], tau)
segments = segment_types(clean["renewed"], clean["treatment"], tau)

# 5. Build targeting rules
tree = PolicyTree(model, max_depth=2)
tree.fit(clean)
recommendations = tree.recommend(clean)
print(tree.export_rules())

# 6. Apply ENBP constraint
# rec_rate_changes: recommended log rate change per customer.
# Persuadable customers (tau < 0) get a discount; others get no change.
rec_rate_changes = pl.Series(
    "rec_rate_change",
    np.where(tau.to_numpy() < 0, -0.05, 0.0),  # -5% discount for persuadables
)
constraint = ENBPConstraint()
clipped = constraint.apply(clean, rec_rate_changes)
audit = constraint.audit_report(clean, rec_rate_changes)

# 7. Fairness audit
fairness = FairnessAudit(protected_proxies=["age_band", "postcode_income_decile"])
fairness.fit(clean.select(["age_band", "postcode_income_decile"]), tau)
print(fairness.audit().filter(pl.col("flagged_as_vulnerable")))

# 8. ROI
roi = ROIReport(policy_premium_avg=680.0)
result = roi.compute(clean, tau, recommendations, discount_size=0.05)
print(f"Net ROI: £{result['net_roi']:,.0f}")
```

## The treatment variable

Treatment is `log(renewal_premium / expiring_premium)`. Positive = price increase, negative = decrease. This is continuous by design.

Binarising the price into "discount / no discount" loses the dose-response information. A customer who received a 15% increase is much more stressed than one who received a 2% increase, and the causal forest can use that variation. The EconML DML estimator handles continuous treatment natively.

For Qini evaluation, the treatment is binarised at the median to produce a treated/control split. This is the standard adaptation when no explicit A/B assignment exists. The median split is explicit and documented, not hidden.

## The four-customer taxonomy

Taken from Guelman, Guillén & Pérez-Marín (2012). Every customer falls into one of four groups:

| Type | τ̂(x) | Outcome | What this means |
|---|---|---|---|
| Persuadable | < 0 | Lapsed | Would renew with a discount. The only group worth targeting. |
| Sure Thing | ≈ 0 | Renewed | Renews regardless. Discount is wasted margin. |
| Lost Cause | ≈ 0 | Lapsed | Lapses regardless. No intervention effect. |
| Do Not Disturb | > 0 | Any | Contact or rate change triggers comparison shopping — intervention makes things worse. |

The primary use case for this library is identifying Persuadable customers. Discounting Sure Things and Do Not Disturb customers is expensive and potentially counterproductive.

## ENBP compliance

The `ENBPConstraint` class clips all renewal recommendations to the ENBP floor. This is not optional configuration — it is required by FCA ICOBS 6B.2. Any campaign built with this library will automatically produce a ENBP audit report showing:

- How many recommendations were clipped
- By how much, on average
- Which customer segments had the most clipping

## Consumer Duty fairness audit

`FairnessAudit` checks whether inelastic segments (τ̂ > 0) are dominated by vulnerability proxies. Under Consumer Duty (PRIN 2A.4), being technically ENBP-compliant is insufficient if the practical effect is to charge more to systematically vulnerable customers.

The audit flags any group where:
1. Average τ̂ > threshold (inelastic: customers don't respond to price changes)
2. The group is defined by a recognised vulnerability proxy (age > 70, low income decile)

Flagged groups require a compliance review before any rate action. The audit report is designed to be dropped into a TCF/Consumer Duty governance pack.

## API reference

### `RetentionPanel`

```python
RetentionPanel(
    policy_df,
    policy_id_col='policy_id',
    renewal_premium_col='renewal_premium',
    expiring_premium_col='expiring_premium',
    renewal_indicator_col='renewed',
    start_date_col='start_date',
    end_date_col='end_date',
    enbp_col='enbp',
    censor_date=None,
)
.build() -> pl.DataFrame
.treatment_variation_report(confounder_cols=None) -> pl.DataFrame
```

### `RetentionUpliftModel`

```python
RetentionUpliftModel(
    estimator='causal_forest',  # or 'dr_learner', 'x_learner'
    outcome='binary',
    nuisance_model='catboost',
    n_estimators=2000,
    n_folds=5,
    min_samples_leaf=20,
    inference=True,
    random_state=42,
)
.fit(panel, treatment_col='treatment', outcome_col='renewed',
     confounders=[...], weight_col=None) -> self
.cate(X) -> pl.Series
.cate_inference(X) -> tuple[pl.Series, pl.Series, pl.Series]
.ate() -> tuple[float, float, float]
.gate(X, by='region') -> pl.DataFrame
```

### Evaluate

```python
qini_curve(y_true, treatment, tau_hat, n_buckets=100) -> (fractions, gains)
auuc(y_true, treatment, tau_hat) -> float
uplift_at_k(y_true, treatment, tau_hat, k=0.3) -> float
segment_types(y_true, treatment, tau_hat, threshold=0.0) -> pl.DataFrame
persuadable_rate(tau_hat, threshold=0.0) -> float
plot_qini(y_true, treatment, tau_hat, ax=None) -> Axes
```

### `PolicyTree`

```python
PolicyTree(uplift_model, max_depth=2, backend='sklearn', budget_constraint=None)
.fit(X) -> self
.recommend(X) -> pl.Series
.welfare_gain() -> float
.export_rules() -> list[dict]
```

### `SegmentSummary`

```python
SegmentSummary(uplift_model, max_depth=3, min_samples_leaf=50)
.fit(X) -> self
.segment_table() -> pl.DataFrame
.plot_tree(ax=None) -> Axes
```

### `ENBPConstraint`

```python
ENBPConstraint(enbp_col='enbp', expiring_premium_col='expiring_premium')
.apply(df, recommended_rate_change) -> pl.Series
.audit_report(df, recommended_rate_change) -> pl.DataFrame
```

### `FairnessAudit`

```python
FairnessAudit(
    protected_proxies=['age_band', 'postcode_income_decile'],
    vulnerability_threshold_age=70,
    inelasticity_threshold=0.0,
)
.fit(X, tau_hat) -> self
.audit() -> pl.DataFrame
.plot_tau_by_proxy(ax=None) -> Axes
```

### `ROIReport`

```python
ROIReport(discount_cost_per_unit=0.0, policy_premium_avg=600.0)
.compute(df, tau_hat, recommended_treatment, discount_size=0.05) -> dict
```

## Design choices

**Why continuous treatment?** Insurance renewal pricing is not a binary A/B test. Different customer segments receive different rate changes. The full log price ratio preserves this variation and lets the DML estimator learn the dose-response curve directly.

**Why CatBoost for nuisance models?** CatBoost handles categorical features natively (region, NCD band, vehicle type) without manual encoding. It also handles the moderate sample sizes typical of renewal panels (5,000–50,000 policies) well. The EconML default (random forest) underperforms on these datasets in our testing.

**Why binarise at median for Qini?** Qini curves require a treatment/control split. When treatment is continuous, the median split is transparent, reproducible, and avoids arbitrary threshold choice. The alternative — treating all policies as "treated" because they received any rate change — conflates treatment intensity with treatment assignment.

**Why not use scikit-uplift or CausalML directly?** Both require binary treatment. Neither handles the renewal panel construction, ENBP constraint, or Consumer Duty fairness checks. This library wraps EconML where it has the best continuous-treatment support and adds the insurance-specific pipeline on top.

## Relationship to insurance-elasticity

[insurance-elasticity](https://github.com/burning-cost/insurance-elasticity) answers: "how does price sensitivity vary by customer segment?" Its output is the CATE surface.

This library answers: "given the CATE estimates, which customers should I target, and what is the expected ROI?" Its output is the targeting decision.

The natural workflow is to use insurance-elasticity to understand the CATE surface, then use insurance-uplift to build the campaign. The two libraries share CausalForestDML internally but have distinct primary outputs.

## Limitations

- v0.1 is binary outcome only. The `outcome='survival'` parameter is reserved for v0.2, which will add causal survival forest via rpy2+grf for right-censored outcome modelling.
- The `policytree_r` backend (welfare-maximising policy tree via rpy2+policytree) is also deferred to v0.2. The current sklearn backend is a greedy heuristic that approximates welfare maximisation.
- CATE identification requires within-cell treatment variation. If a blanket rate change was applied uniformly (all customers received exactly +5%), the DML residualiser cannot separate price sensitivity from confounders. Use `treatment_variation_report()` to check.

## Regulatory notes

This library is a tool for pricing analysis. It does not constitute legal or regulatory advice. Compliance with FCA PS21/5, Consumer Duty, and related regulations is the responsibility of the firm using the library. The `ENBPConstraint` and `FairnessAudit` outputs are designed to support compliance workflows but do not substitute for legal review.

## References

- Guelman, Guillén & Pérez-Marín (2012): Random Forests for Uplift Modeling — the four-customer taxonomy
- Athey & Imbens (2016), Wager & Athey (2018): Causal forests
- Künzel et al. (2019): Meta-learners for HTE
- Rößler & Schoder (2022): Benchmarking uplift vs HTE methods
- FCA PS21/5 (2022): GIPP remedies, ENBP requirement
- FCA Consumer Duty FG22/5 (2022): fair value and vulnerable customer outcomes
- FCA EP25/2 (December 2025): GIPP evaluation confirming ENBP is effective and stable
