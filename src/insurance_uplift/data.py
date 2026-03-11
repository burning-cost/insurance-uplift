"""Renewal panel construction for uplift modelling.

The :class:`RetentionPanel` class takes a raw policy extract and returns a
panel DataFrame suitable for CATE estimation. Its primary jobs are:

1. Compute the continuous treatment: ``log(renewal_premium / expiring_premium)``.
2. Flag censored policies (end_date > censor_date) and handle them correctly.
3. Compute earned exposure weights.
4. Run a treatment variation check to warn if DML residualisation will struggle.

Usage
-----
::

    import polars as pl
    from insurance_uplift.data import RetentionPanel

    panel = RetentionPanel(
        policy_df=df,
        renewal_premium_col='renewal_premium',
        expiring_premium_col='expiring_premium',
        renewal_indicator_col='renewed',
        start_date_col='start_date',
        end_date_col='end_date',
        enbp_col='enbp',
    )
    built = panel.build()
    variation = panel.treatment_variation_report(confounder_cols=['age_band', 'ncd'])
"""

from __future__ import annotations

import warnings
from datetime import date
from typing import Optional

import numpy as np
import polars as pl

from ._utils import log_price_ratio, validate_panel_columns


class RetentionPanel:
    """Build a policy-level renewal panel for uplift modelling.

    Parameters
    ----------
    policy_df:
        Raw policy extract. Accepts Polars or Pandas DataFrames. Pandas frames
        are converted automatically.
    policy_id_col:
        Column containing a unique policy identifier.
    renewal_premium_col:
        Renewal offer premium (£).
    expiring_premium_col:
        Expiring (prior year) premium (£). Must be strictly positive.
    renewal_indicator_col:
        Binary outcome column: 1 = renewed, 0 = lapsed, None/null = censored.
        Censored rows are identified by null values in this column.
    start_date_col:
        Policy start date. Used to compute earned exposure.
    end_date_col:
        Policy end (renewal) date. Used to identify censored policies.
    enbp_col:
        Equivalent new business price. Required for :class:`~insurance_uplift.constrain.ENBPConstraint`.
        Pass ``None`` to omit.
    censor_date:
        Extract date. Policies with ``end_date > censor_date`` are censored.
        If ``None``, inferred as ``max(end_date)`` in the dataset.

    Notes
    -----
    Treatment is defined as ``log(renewal_premium / expiring_premium)``.
    Positive treatment = price increase; negative = price decrease. This
    continuous treatment maps directly to the log-linear form used in
    insurance price elasticity models.

    Censored policies are excluded from binary outcome modelling with a
    warning. They are retained (with ``censored_flag=1``) in the output
    so callers can inspect them.
    """

    def __init__(
        self,
        policy_df,
        policy_id_col: str = "policy_id",
        renewal_premium_col: str = "renewal_premium",
        expiring_premium_col: str = "expiring_premium",
        renewal_indicator_col: str = "renewed",
        start_date_col: str = "start_date",
        end_date_col: str = "end_date",
        enbp_col: Optional[str] = "enbp",
        censor_date: Optional[date] = None,
    ) -> None:
        # Convert pandas to polars if needed
        try:
            import pandas as pd  # noqa: F401

            if isinstance(policy_df, pd.DataFrame):
                policy_df = pl.from_pandas(policy_df)
        except ImportError:
            pass

        if not isinstance(policy_df, pl.DataFrame):
            raise TypeError(
                f"policy_df must be a Polars or Pandas DataFrame, got {type(policy_df)}."
            )

        self._df = policy_df
        self.policy_id_col = policy_id_col
        self.renewal_premium_col = renewal_premium_col
        self.expiring_premium_col = expiring_premium_col
        self.renewal_indicator_col = renewal_indicator_col
        self.start_date_col = start_date_col
        self.end_date_col = end_date_col
        self.enbp_col = enbp_col
        self.censor_date = censor_date
        self._built: Optional[pl.DataFrame] = None

    def _resolve_censor_date(self) -> date:
        """Return the censor date, inferring from data if not provided."""
        if self.censor_date is not None:
            return self.censor_date
        col = self._df[self.end_date_col]
        if col.dtype == pl.Utf8:
            col = col.str.to_date()
        max_date = col.max()
        if max_date is None:
            raise ValueError("Cannot infer censor_date: end_date column is empty.")
        warnings.warn(
            f"censor_date not provided. Inferred as max(end_date) = {max_date}. "
            "This may incorrectly classify policies near the extract date as censored. "
            "Provide censor_date explicitly for production use.",
            UserWarning,
            stacklevel=3,
        )
        return max_date

    def build(self) -> pl.DataFrame:
        """Construct the panel with treatment, outcome, and metadata columns.

        Returns
        -------
        pl.DataFrame
            One row per policy with columns:

            - All original columns from ``policy_df``
            - ``treatment``: log(renewal_premium / expiring_premium)
            - ``censored_flag``: 1 if policy end_date > censor_date, else 0
            - ``earned_exposure``: fraction of policy year earned at censor date
            - ``policy_weight``: 1.0 for all non-censored (reserved for future use)
            - ``treatment_variation_flag``: 1 if treatment is at the 5th or 95th percentile

        Raises
        ------
        ValueError
            If required columns are missing or premiums are non-positive.
        """
        required = [
            self.renewal_premium_col,
            self.expiring_premium_col,
            self.renewal_indicator_col,
            self.end_date_col,
        ]
        if self.start_date_col in self._df.columns:
            required.append(self.start_date_col)
        validate_panel_columns(self._df, required)

        df = self._df.clone()

        # Compute treatment
        r = df[self.renewal_premium_col].to_numpy().astype(np.float64)
        e = df[self.expiring_premium_col].to_numpy().astype(np.float64)
        treatment = log_price_ratio(r, e)
        df = df.with_columns(pl.Series("treatment", treatment))

        # Resolve censor date and flag censored policies
        censor_date = self._resolve_censor_date()
        end_col = df[self.end_date_col]
        if end_col.dtype == pl.Utf8:
            end_col = end_col.str.to_date()
            df = df.with_columns(end_col.alias(self.end_date_col))

        censored = (df[self.end_date_col] > censor_date).cast(pl.Int8)
        df = df.with_columns(censored.alias("censored_flag"))

        n_censored = int(censored.sum())
        n_total = len(df)
        if n_censored > 0:
            frac = n_censored / n_total
            warnings.warn(
                f"{n_censored} of {n_total} policies ({frac:.1%}) are censored "
                f"(end_date > {censor_date}). These are excluded from binary outcome "
                "modelling. Use outcome='survival' with rpy2+grf to retain them.",
                UserWarning,
                stacklevel=2,
            )

        # Earned exposure
        if self.start_date_col in df.columns:
            start_col = df[self.start_date_col]
            if start_col.dtype == pl.Utf8:
                start_col = start_col.str.to_date()
                df = df.with_columns(start_col.alias(self.start_date_col))

            from datetime import timedelta

            censor_dt = pl.lit(censor_date)
            start_dt = df[self.start_date_col]
            end_dt = df[self.end_date_col]

            # Exposure = min(end_date, censor_date) - start_date, as fraction of year
            # Clip end date to censor date for censored policies
            policy_duration = (end_dt - start_dt).dt.total_seconds() / (365.25 * 86400)
            elapsed = (
                pl.Series(
                    [
                        (min(e_date, censor_date) - s_date).days / 365.25
                        for e_date, s_date in zip(
                            df[self.end_date_col].to_list(),
                            df[self.start_date_col].to_list(),
                        )
                    ]
                ).clip(0.0, 1.0)
            )
            df = df.with_columns(elapsed.alias("earned_exposure"))
        else:
            df = df.with_columns(pl.lit(1.0).alias("earned_exposure"))

        # Policy weight (placeholder for future inverse-probability weighting)
        df = df.with_columns(pl.lit(1.0).alias("policy_weight"))

        # Treatment variation flag: policies at extreme percentiles
        t_low = float(np.percentile(treatment, 5))
        t_high = float(np.percentile(treatment, 95))
        variation_flag = ((df["treatment"] <= t_low) | (df["treatment"] >= t_high)).cast(pl.Int8)
        df = df.with_columns(variation_flag.alias("treatment_variation_flag"))

        self._built = df
        return df

    def treatment_variation_report(
        self, confounder_cols: Optional[list[str]] = None
    ) -> pl.DataFrame:
        """Check whether there is sufficient treatment variation for DML.

        DML residualisation requires within-cell treatment variation: the
        residual ``T - E[T|X]`` must have non-trivial variance. If all
        customers in a confounder cell received the same rate change (e.g.
        a blanket 10% renewal increase), the causal forest cannot distinguish
        price sensitivity from baseline renewal propensity.

        Parameters
        ----------
        confounder_cols:
            Columns to group by. If ``None``, reports overall treatment
            distribution statistics only.

        Returns
        -------
        pl.DataFrame
            Summary of treatment variation: mean, std, min, max, coefficient
            of variation per confounder cell. Cells with CV < 0.05 are
            flagged as potentially problematic.
        """
        if self._built is None:
            raise RuntimeError("Call .build() before .treatment_variation_report().")

        df = self._built.filter(pl.col("censored_flag") == 0)

        if confounder_cols is None:
            stats = df.select(
                pl.col("treatment").mean().alias("mean_treatment"),
                pl.col("treatment").std().alias("std_treatment"),
                pl.col("treatment").min().alias("min_treatment"),
                pl.col("treatment").max().alias("max_treatment"),
                pl.col("treatment").count().alias("n"),
            )
            stats = stats.with_columns(
                (pl.col("std_treatment") / pl.col("mean_treatment").abs()).alias(
                    "coeff_variation"
                )
            )
            return stats

        agg = df.group_by(confounder_cols).agg(
            pl.col("treatment").mean().alias("mean_treatment"),
            pl.col("treatment").std().alias("std_treatment"),
            pl.col("treatment").min().alias("min_treatment"),
            pl.col("treatment").max().alias("max_treatment"),
            pl.col("treatment").count().alias("n"),
        )
        agg = agg.with_columns(
            (pl.col("std_treatment") / (pl.col("mean_treatment").abs() + 1e-9)).alias(
                "coeff_variation"
            )
        )
        agg = agg.with_columns(
            (pl.col("coeff_variation") < 0.05).alias("low_variation_flag")
        )
        n_flagged = int(agg["low_variation_flag"].sum())
        if n_flagged > 0:
            warnings.warn(
                f"{n_flagged} confounder cell(s) have treatment coefficient of variation < 0.05. "
                "DML residualisation may be unreliable in these cells. "
                "Consider whether a blanket rate change was applied uniformly.",
                UserWarning,
                stacklevel=2,
            )
        return agg.sort(confounder_cols)
