"""Regulatory compliance layer: ENBP constraint and Consumer Duty fairness audit.

Two classes cover the FCA compliance requirements for UK personal lines
renewal pricing:

:class:`ENBPConstraint`
    Clips renewal recommendations to the equivalent new business price per
    ICOBS 6B.2. ENBP-clipping is not optional — it is required by law for any
    insurer offering personal lines renewal products in the UK.

:class:`FairnessAudit`
    Checks whether inelastic customer segments (τ̂(x) ≈ 0 or τ̂(x) > 0) are
    dominated by vulnerability proxies. Under Consumer Duty (PRIN 2A), it is
    insufficient to be technically ENBP-compliant if the effect is to charge
    more to systematically vulnerable customers.

:class:`ROIReport`
    Computes campaign ROI: expected incremental renewals, discount cost, and
    net revenue impact given a targeting recommendation.

Usage
-----
::

    from insurance_uplift.constrain import ENBPConstraint, FairnessAudit, ROIReport

    # ENBP clipping
    constraint = ENBPConstraint(enbp_col='enbp', expiring_premium_col='expiring_premium')
    clipped = constraint.apply(panel, recommended_rate_change)
    report = constraint.audit_report(panel, recommended_rate_change)

    # Fairness audit
    audit = FairnessAudit(protected_proxies=['age_band', 'postcode_income_decile'])
    audit.fit(X, tau_hat)
    fairness_table = audit.audit()

    # ROI
    roi = ROIReport(discount_cost_per_unit=50.0, policy_premium_avg=650.0)
    results = roi.compute(panel, tau_hat, recommend, discount_size=0.10)
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import polars as pl

from ._utils import safe_divide, to_numpy, validate_panel_columns


class ENBPConstraint:
    """Clip renewal premium recommendations to the ENBP floor.

    Under FCA ICOBS 6B.2 (PS21/5), the renewal price offered to an existing
    customer must not exceed the price that would be quoted to a new customer
    for an equivalent policy. This constraint must be applied unconditionally
    before any renewal offer is communicated.

    Parameters
    ----------
    enbp_col:
        Column in the policy DataFrame containing the pre-computed equivalent
        new business price.
    expiring_premium_col:
        Column containing the expiring (prior year) premium.
    policy_id_col:
        Column containing the policy identifier (used in audit reports).
    """

    def __init__(
        self,
        enbp_col: str = "enbp",
        expiring_premium_col: str = "expiring_premium",
        policy_id_col: str = "policy_id",
    ) -> None:
        self.enbp_col = enbp_col
        self.expiring_premium_col = expiring_premium_col
        self.policy_id_col = policy_id_col

    def apply(
        self,
        df: pl.DataFrame,
        recommended_rate_change: pl.Series,
    ) -> pl.Series:
        """Clip recommended rate changes to the ENBP floor.

        Parameters
        ----------
        df:
            Policy DataFrame containing at least ``enbp_col`` and
            ``expiring_premium_col``.
        recommended_rate_change:
            Per-policy fractional rate change recommendation, e.g. ``-0.05``
            for a 5% discount, ``+0.03`` for a 3% increase. Length must match
            ``len(df)``.

        Returns
        -------
        pl.Series
            Clipped fractional rate changes. Any recommendation that would
            produce ``expiring × (1 + rate_change) > enbp`` is reduced to
            ``enbp / expiring - 1``.

        Raises
        ------
        ValueError
            If required columns are missing or lengths do not match.
        """
        validate_panel_columns(df, [self.enbp_col, self.expiring_premium_col])
        if len(df) != len(recommended_rate_change):
            raise ValueError(
                f"df has {len(df)} rows but recommended_rate_change has "
                f"{len(recommended_rate_change)} elements."
            )

        expiring = df[self.expiring_premium_col].to_numpy().astype(np.float64)
        enbp = df[self.enbp_col].to_numpy().astype(np.float64)
        rec = to_numpy(recommended_rate_change)

        # Recommended renewal premium
        rec_renewal = expiring * (1 + rec)

        # Maximum allowed rate change implied by ENBP
        max_rate_change = np.where(expiring > 0, enbp / expiring - 1, rec)

        # Clip: recommended renewal cannot exceed ENBP
        clipped = np.minimum(rec, max_rate_change)

        n_clipped = int(np.sum(clipped < rec))
        if n_clipped > 0:
            frac = n_clipped / len(rec)
            warnings.warn(
                f"ENBP constraint clipped {n_clipped} of {len(rec)} recommendations "
                f"({frac:.1%}). These policies had recommended renewal > ENBP.",
                UserWarning,
                stacklevel=2,
            )

        return pl.Series("clipped_rate_change", clipped)

    def audit_report(
        self,
        df: pl.DataFrame,
        recommended_rate_change: pl.Series,
        segment_type_col: Optional[str] = None,
    ) -> pl.DataFrame:
        """Produce a per-policy audit report of ENBP clipping.

        Parameters
        ----------
        df:
            Policy DataFrame.
        recommended_rate_change:
            Per-policy fractional rate change recommendation.
        segment_type_col:
            Optional column name in ``df`` containing customer segment labels
            (Persuadable / Sure Thing / Lost Cause / Do Not Disturb). If present,
            the report includes per-segment clipping statistics.

        Returns
        -------
        pl.DataFrame
            One row per policy with columns: ``policy_id``, ``expiring_premium``,
            ``enbp``, ``recommended_renewal``, ``clipped_renewal``,
            ``was_clipped``, ``clip_amount_pct``.
        """
        validate_panel_columns(df, [self.enbp_col, self.expiring_premium_col])

        expiring = df[self.expiring_premium_col].to_numpy().astype(np.float64)
        enbp = df[self.enbp_col].to_numpy().astype(np.float64)
        rec = to_numpy(recommended_rate_change)

        rec_renewal = expiring * (1 + rec)
        clipped = self.apply(df, recommended_rate_change).to_numpy()
        clipped_renewal = expiring * (1 + clipped)
        was_clipped = clipped < rec
        clip_amount = np.where(was_clipped, (rec_renewal - clipped_renewal) / expiring * 100, 0.0)

        report_data: dict = {
            "expiring_premium": expiring.tolist(),
            "enbp": enbp.tolist(),
            "recommended_renewal": rec_renewal.tolist(),
            "clipped_renewal": clipped_renewal.tolist(),
            "was_clipped": was_clipped.tolist(),
            "clip_amount_pct": clip_amount.tolist(),
        }

        if self.policy_id_col in df.columns:
            report_data = {self.policy_id_col: df[self.policy_id_col].to_list(), **report_data}

        if segment_type_col and segment_type_col in df.columns:
            report_data["segment_type"] = df[segment_type_col].to_list()

        return pl.DataFrame(report_data)


class FairnessAudit:
    """Audit CATE distribution across vulnerability proxies for Consumer Duty.

    Consumer Duty (PRIN 2A.4) requires firms to ensure their pricing does not
    produce systematically worse outcomes for customers with vulnerability
    characteristics. Price insensitivity (τ̂(x) ≈ 0 or τ̂(x) > 0) in a segment
    is commercially useful — it means you can hold or increase rates without
    losing those customers — but if that segment is dominated by older or
    lower-income customers, exploiting their inelasticity raises serious
    Consumer Duty concerns.

    This audit flags groups where:
    1. The average τ̂(x) is above a threshold (inelastic)
    2. The group is defined by a vulnerability proxy (age > 70, low-income postcode)

    Parameters
    ----------
    protected_proxies:
        List of column names representing vulnerability proxies. Common examples:
        ``'age_band'``, ``'postcode_income_decile'``.
    vulnerability_threshold_age:
        Age above which a customer is flagged as a potential vulnerable
        customer for Consumer Duty purposes. Default 70.
    inelasticity_threshold:
        τ̂(x) above this value is flagged as inelastic. Default 0.0.
    """

    def __init__(
        self,
        protected_proxies: list[str],
        vulnerability_threshold_age: int = 70,
        inelasticity_threshold: float = 0.0,
    ) -> None:
        if not protected_proxies:
            raise ValueError("protected_proxies must not be empty.")
        self.protected_proxies = protected_proxies
        self.vulnerability_threshold_age = vulnerability_threshold_age
        self.inelasticity_threshold = inelasticity_threshold
        self._X: Optional[pl.DataFrame] = None
        self._tau_hat: Optional[np.ndarray] = None
        self._is_fitted = False

    def fit(self, X: pl.DataFrame, tau_hat) -> "FairnessAudit":
        """Store features and CATE estimates for the audit.

        Parameters
        ----------
        X:
            Customer features DataFrame. Must contain all columns in
            ``protected_proxies``.
        tau_hat:
            Predicted CATE from the uplift model.

        Returns
        -------
        FairnessAudit
            Self, for method chaining.
        """
        validate_panel_columns(X, self.protected_proxies)
        self._X = X
        self._tau_hat = to_numpy(tau_hat)
        self._is_fitted = True
        return self

    def audit(self) -> pl.DataFrame:
        """Run the fairness audit and return flagged groups.

        Returns
        -------
        pl.DataFrame
            Columns: ``proxy_variable``, ``group``, ``n``, ``avg_tau``,
            ``min_tau``, ``max_tau``, ``frac_inelastic``,
            ``flagged_as_vulnerable``, ``regulatory_note``.

            A row is flagged when ``avg_tau > inelasticity_threshold`` and
            the group corresponds to a recognised vulnerability proxy
            (age > vulnerability_threshold_age, or postcode income decile
            in the lowest 3 deciles).
        """
        if not self._is_fitted:
            raise RuntimeError("Call .fit() before .audit().")

        rows = []
        for proxy in self.protected_proxies:
            groups = self._X[proxy]
            for g in groups.unique().sort():
                mask = (groups == g).to_numpy()
                tau_g = self._tau_hat[mask]
                n_g = len(tau_g)
                if n_g == 0:
                    continue

                avg_tau = float(np.mean(tau_g))
                frac_inelastic = float(np.mean(tau_g > self.inelasticity_threshold))

                # Determine vulnerability flag
                is_vulnerable = self._is_vulnerable_group(proxy, g)
                is_inelastic = avg_tau > self.inelasticity_threshold

                flagged = is_vulnerable and is_inelastic

                if flagged:
                    regulatory_note = (
                        f"Group '{g}' in '{proxy}' is both inelastic (avg τ={avg_tau:.3f}) "
                        f"and a recognised vulnerability proxy. "
                        f"Targeting with rate increases may breach Consumer Duty "
                        f"fair value outcome (PRIN 2A.4). Review with compliance."
                    )
                elif is_inelastic and not is_vulnerable:
                    regulatory_note = "Inelastic but not a vulnerability proxy. No immediate concern."
                elif is_vulnerable and not is_inelastic:
                    regulatory_note = "Vulnerability proxy but elastic. Monitor outcomes."
                else:
                    regulatory_note = ""

                rows.append(
                    {
                        "proxy_variable": proxy,
                        "group": str(g),
                        "n": n_g,
                        "avg_tau": avg_tau,
                        "min_tau": float(np.min(tau_g)),
                        "max_tau": float(np.max(tau_g)),
                        "frac_inelastic": frac_inelastic,
                        "flagged_as_vulnerable": flagged,
                        "regulatory_note": regulatory_note,
                    }
                )

        return pl.DataFrame(rows).sort(["proxy_variable", "group"])

    def _is_vulnerable_group(self, proxy: str, group_value) -> bool:
        """Determine if a group value represents a vulnerability proxy."""
        g_str = str(group_value).lower().strip()

        # Age-based proxy
        if "age" in proxy.lower():
            try:
                age_val = float(g_str.replace("+", "").split("-")[-1].strip())
                return age_val >= self.vulnerability_threshold_age
            except (ValueError, IndexError):
                # Try to extract numeric part
                import re
                nums = re.findall(r'\d+', g_str)
                if nums:
                    return max(int(n) for n in nums) >= self.vulnerability_threshold_age
                return False

        # Income / socioeconomic proxy: low deciles (1-3) are vulnerable
        if "income" in proxy.lower() or "decile" in proxy.lower() or "deprivation" in proxy.lower():
            import re
            nums = re.findall(r'\d+', g_str)
            if nums:
                return min(int(n) for n in nums) <= 3
            return False

        # Postcode-based proxies: flag if explicitly "low" income
        if "postcode" in proxy.lower():
            return any(kw in g_str for kw in ["low", "deprived", "1", "2", "3"])

        return False

    def plot_tau_by_proxy(self, ax=None):
        """Plot CATE distribution per group for each proxy variable.

        Parameters
        ----------
        ax:
            Matplotlib axes. If ``None``, creates one figure per proxy variable.

        Returns
        -------
        matplotlib.axes.Axes or list[matplotlib.axes.Axes]
        """
        if not self._is_fitted:
            raise RuntimeError("Call .fit() before .plot_tau_by_proxy().")

        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib is required for plot_tau_by_proxy.")

        audit_df = self.audit()

        if len(self.protected_proxies) == 1:
            proxy = self.protected_proxies[0]
            if ax is None:
                _, ax = plt.subplots(figsize=(10, 5))
            proxy_data = audit_df.filter(pl.col("proxy_variable") == proxy)
            groups = proxy_data["group"].to_list()
            avg_taus = proxy_data["avg_tau"].to_list()
            flagged = proxy_data["flagged_as_vulnerable"].to_list()
            colours = ["#d73027" if f else "#4575b4" for f in flagged]
            ax.bar(groups, avg_taus, color=colours)
            ax.axhline(y=self.inelasticity_threshold, linestyle="--", color="black", alpha=0.5)
            ax.set_xlabel(proxy)
            ax.set_ylabel("Average τ̂(x)")
            ax.set_title(f"CATE by {proxy} — Consumer Duty Fairness Audit")
            ax.tick_params(axis="x", rotation=45)
            return ax

        axes = []
        for proxy in self.protected_proxies:
            _, ax_i = plt.subplots(figsize=(10, 5))
            proxy_data = audit_df.filter(pl.col("proxy_variable") == proxy)
            groups = proxy_data["group"].to_list()
            avg_taus = proxy_data["avg_tau"].to_list()
            flagged = proxy_data["flagged_as_vulnerable"].to_list()
            colours = ["#d73027" if f else "#4575b4" for f in flagged]
            ax_i.bar(groups, avg_taus, color=colours)
            ax_i.axhline(y=self.inelasticity_threshold, linestyle="--", color="black", alpha=0.5)
            ax_i.set_xlabel(proxy)
            ax_i.set_ylabel("Average τ̂(x)")
            ax_i.set_title(f"CATE by {proxy} — Consumer Duty Fairness Audit")
            ax_i.tick_params(axis="x", rotation=45)
            axes.append(ax_i)
        return axes


class ROIReport:
    """Compute the expected ROI of a retention discount campaign.

    Connects CATE estimates to business outcomes: given a targeting
    recommendation and discount size, what are the expected incremental
    renewals, discount cost, and net revenue impact?

    Parameters
    ----------
    discount_cost_per_unit:
        The per-policy cost of administering the retention intervention
        (outbound call centre cost, gift, etc.) in addition to the premium
        discount. Set to 0 if the only cost is the premium reduction.
    policy_premium_avg:
        Average annual premium per policy (£). Used to compute expected
        incremental revenue from retained customers.
    """

    def __init__(
        self,
        discount_cost_per_unit: float = 0.0,
        policy_premium_avg: float = 600.0,
    ) -> None:
        if policy_premium_avg <= 0:
            raise ValueError("policy_premium_avg must be positive.")
        self.discount_cost_per_unit = discount_cost_per_unit
        self.policy_premium_avg = policy_premium_avg

    def compute(
        self,
        df: pl.DataFrame,
        tau_hat,
        recommended_treatment: pl.Series,
        discount_size: float = 0.05,
    ) -> dict:
        """Compute campaign ROI.

        Parameters
        ----------
        df:
            Policy DataFrame. Used for policy counts.
        tau_hat:
            Predicted CATE for each policy.
        recommended_treatment:
            Binary Series: 1 = this policy is targeted with a discount.
        discount_size:
            Fractional price discount offered to targeted policies (e.g. 0.05
            for a 5% discount). Must be positive.

        Returns
        -------
        dict
            Keys:

            - ``n_treated``: number of policies targeted
            - ``expected_additional_renewals``: E[incremental renewals from campaign]
            - ``expected_discount_cost``: total premium discount cost (£)
            - ``expected_admin_cost``: per-policy admin cost for targeted policies (£)
            - ``expected_total_cost``: sum of discount and admin costs (£)
            - ``expected_additional_premium_revenue``: incremental premium revenue from
              retained customers (£)
            - ``net_roi``: revenue - cost (£)
            - ``roi_pct``: net_roi / total_cost as percentage
            - ``break_even_retention_rate``: minimum additional retention rate needed
              for positive ROI
            - ``uplift_per_pound_spent``: incremental renewals per pound of campaign cost
        """
        if discount_size <= 0:
            raise ValueError("discount_size must be positive.")

        tau = to_numpy(tau_hat)
        rec = to_numpy(recommended_treatment).astype(bool)

        n_treated = int(rec.sum())
        n_total = len(rec)

        if n_treated == 0:
            warnings.warn(
                "No policies are recommended for treatment. ROI report will be zero.",
                UserWarning,
                stacklevel=2,
            )
            return {
                "n_treated": 0,
                "n_total": n_total,
                "expected_additional_renewals": 0.0,
                "expected_discount_cost": 0.0,
                "expected_admin_cost": 0.0,
                "expected_total_cost": 0.0,
                "expected_additional_premium_revenue": 0.0,
                "net_roi": 0.0,
                "roi_pct": 0.0,
                "break_even_retention_rate": 0.0,
                "uplift_per_pound_spent": 0.0,
            }

        # Expected incremental renewals from targeting Persuadable customers
        # For each treated customer: effect of discount = tau(x) * log(1 - discount_size)
        # This is the change in renewal probability from the price change
        discount_log = np.log(1 - discount_size)
        delta_renewal_prob = tau[rec] * discount_log  # positive for price-sensitive customers

        # Clip to valid probability range
        delta_renewal_prob = np.clip(delta_renewal_prob, -1.0, 1.0)
        expected_additional_renewals = float(np.sum(np.maximum(delta_renewal_prob, 0)))

        # Cost of discount: average premium * discount_size * n_treated
        avg_premium = self.policy_premium_avg
        expected_discount_cost = avg_premium * discount_size * n_treated
        expected_admin_cost = self.discount_cost_per_unit * n_treated
        expected_total_cost = expected_discount_cost + expected_admin_cost

        # Revenue from retained customers: they pay (1 - discount_size) * premium
        expected_additional_premium_revenue = (
            expected_additional_renewals * avg_premium * (1 - discount_size)
        )

        net_roi = expected_additional_premium_revenue - expected_total_cost
        roi_pct = safe_divide(net_roi, expected_total_cost, fill=0.0) * 100

        break_even_renewals = safe_divide(
            expected_total_cost,
            avg_premium * (1 - discount_size),
            fill=0.0,
        )
        break_even_rate = safe_divide(break_even_renewals, n_treated, fill=0.0)

        uplift_per_pound = safe_divide(
            expected_additional_renewals, expected_total_cost, fill=0.0
        )

        return {
            "n_treated": n_treated,
            "n_total": n_total,
            "expected_additional_renewals": expected_additional_renewals,
            "expected_discount_cost": expected_discount_cost,
            "expected_admin_cost": expected_admin_cost,
            "expected_total_cost": expected_total_cost,
            "expected_additional_premium_revenue": expected_additional_premium_revenue,
            "net_roi": net_roi,
            "roi_pct": roi_pct,
            "break_even_retention_rate": break_even_rate,
            "uplift_per_pound_spent": uplift_per_pound,
        }
