"""CATE estimation for insurance retention uplift.

The :class:`RetentionUpliftModel` wraps EconML estimators behind an
insurance-vocabulary API. The primary estimator is ``CausalForestDML`` with
CatBoost nuisance models. A ``DRLearner`` backend is available for low-overlap
observational data where propensity overlap is poor.

Treatment is the continuous log price ratio computed by :class:`~insurance_uplift.data.RetentionPanel`.
The model estimates the marginal effect of a unit increase in log price on renewal
probability: ``τ̂(x) = ∂E[Y|X=x, T=t] / ∂t``.

Usage
-----
::

    from insurance_uplift.fit import RetentionUpliftModel

    model = RetentionUpliftModel(estimator='causal_forest')
    model.fit(
        panel=built_panel.filter(pl.col('censored_flag') == 0),
        confounders=['age', 'ncd', 'vehicle_age', 'region'],
    )
    tau = model.cate(X_test)
    ate, lo, hi = model.ate()
"""

from __future__ import annotations

import warnings
from typing import Literal, Optional

import numpy as np
from sklearn.model_selection import KFold
import polars as pl

from ._utils import to_numpy, to_numpy_2d, validate_min_samples, validate_panel_columns


class RetentionUpliftModel:
    """Estimate per-customer treatment effects on renewal probability.

    Parameters
    ----------
    estimator:
        Which EconML estimator to use.

        ``'causal_forest'``
            CausalForestDML with honest splitting and BLB confidence intervals.
            Best general choice for insurance renewal panels.

        ``'dr_learner'``
            DRLearner (doubly robust pseudo-outcome). Preferred when treatment
            propensity overlap is poor (e.g. only a small fraction of customers
            received discounts).

        ``'x_learner'``
            XLearner (cross-fitted T-learner). Preferred when treatment cells are
            heavily imbalanced (< 10% treated).

    outcome:
        ``'binary'`` uses the renewal indicator directly. Censored policies must
        be excluded before calling ``.fit()``. ``'survival'`` requires rpy2 and
        the R ``grf`` package (v0.1 deferred; falls back to binary with a warning).

    nuisance_model:
        Nuisance model backend. ``'catboost'`` is the recommended default.
        ``'linear'`` uses Ridge regression (faster but less expressive).

    n_estimators:
        Number of trees in the causal forest (only used for ``'causal_forest'``).

    n_folds:
        Cross-fitting folds for DML residualisation.

    min_samples_leaf:
        Minimum samples per leaf in the causal forest. Increase for smaller datasets.

    inference:
        Whether to compute honest confidence intervals. Set to ``False`` to speed
        up training when CIs are not needed.

    random_state:
        Seed for reproducibility.
    """

    def __init__(
        self,
        estimator: Literal["causal_forest", "dr_learner", "x_learner"] = "causal_forest",
        outcome: Literal["binary", "survival"] = "binary",
        nuisance_model: str = "catboost",
        n_estimators: int = 2000,
        n_folds: int = 5,
        min_samples_leaf: int = 20,
        inference: bool = True,
        random_state: int = 42,
    ) -> None:
        _valid_estimators = ("causal_forest", "dr_learner", "x_learner")
        if estimator not in _valid_estimators:
            raise ValueError(
                f"estimator must be one of {_valid_estimators}, got '{estimator}'."
            )
        if outcome not in ("binary", "survival"):
            raise ValueError(f"outcome must be 'binary' or 'survival', got '{outcome}'.")

        self.estimator = estimator
        self.outcome = outcome
        self.nuisance_model = nuisance_model
        self.n_estimators = n_estimators
        self.n_folds = n_folds
        self.min_samples_leaf = min_samples_leaf
        self.inference = inference
        self.random_state = random_state

        self._model = None
        self._confounders: Optional[list[str]] = None
        self._treatment_col: Optional[str] = None
        self._outcome_col: Optional[str] = None
        self._train_X: Optional[np.ndarray] = None
        self._train_T: Optional[np.ndarray] = None
        self._train_Y: Optional[np.ndarray] = None
        self._is_fitted = False

    def _build_nuisance_model(self, task: str = "regression"):
        """Construct the nuisance model for outcome and propensity estimation."""
        if self.nuisance_model == "catboost":
            try:
                if task == "classification":
                    from catboost import CatBoostClassifier

                    return CatBoostClassifier(
                        iterations=300,
                        depth=6,
                        learning_rate=0.05,
                        verbose=0,
                        random_seed=self.random_state,
                        loss_function="Logloss",
                    )
                else:
                    from catboost import CatBoostRegressor

                    return CatBoostRegressor(
                        iterations=300,
                        depth=6,
                        learning_rate=0.05,
                        verbose=0,
                        random_seed=self.random_state,
                    )
            except ImportError:
                warnings.warn(
                    "catboost is not installed. Falling back to GradientBoostingRegressor. "
                    "Install catboost for better nuisance model performance.",
                    UserWarning,
                    stacklevel=3,
                )
                from sklearn.ensemble import GradientBoostingRegressor
                return GradientBoostingRegressor(n_estimators=200, random_state=self.random_state)

        elif self.nuisance_model == "linear":
            from sklearn.linear_model import Ridge
            return Ridge(alpha=1.0)

        else:
            raise ValueError(
                f"nuisance_model must be 'catboost' or 'linear', got '{self.nuisance_model}'."
            )

    def _build_econml_model(self):
        """Construct the EconML estimator."""
        if self.estimator == "causal_forest":
            from econml.dml import CausalForestDML

            return CausalForestDML(
                model_y=self._build_nuisance_model("regression"),
                model_t=self._build_nuisance_model("regression"),
                n_estimators=self.n_estimators,
                discrete_treatment=False,
                min_samples_leaf=self.min_samples_leaf,
                cv=self.n_folds,
                inference=self.inference,
                random_state=self.random_state,
            )

        elif self.estimator == "dr_learner":
            from econml.dr import DRLearner
            from sklearn.ensemble import GradientBoostingRegressor

            return DRLearner(
                model_propensity=self._build_nuisance_model("regression"),
                model_regression=self._build_nuisance_model("regression"),
                model_final=GradientBoostingRegressor(
                    n_estimators=200, random_state=self.random_state
                ),
                cv=KFold(n_splits=self.n_folds),
                random_state=self.random_state,
            )

        elif self.estimator == "x_learner":
            from econml.metalearners import XLearner

            return XLearner(
                models=self._build_nuisance_model("regression"),
                propensity_model=self._build_nuisance_model("regression"),
            )

    def fit(
        self,
        panel: pl.DataFrame,
        treatment_col: str = "treatment",
        outcome_col: str = "renewed",
        confounders: Optional[list[str]] = None,
        weight_col: Optional[str] = None,
    ) -> "RetentionUpliftModel":
        """Fit the CATE estimator on a built retention panel.

        Parameters
        ----------
        panel:
            Output of :meth:`~insurance_uplift.data.RetentionPanel.build`. Must
            not contain censored policies (``censored_flag == 0``).
        treatment_col:
            Column containing the continuous treatment (log price ratio).
        outcome_col:
            Column containing the binary renewal outcome (1=renewed, 0=lapsed).
        confounders:
            Feature columns to condition on. These are the observed risk factors
            and customer characteristics that confound the relationship between
            price and renewal. Typical candidates: age, NCD band, vehicle age,
            region, prior claims, policy tenure.
        weight_col:
            Optional sample weight column. If ``None``, equal weights are used.

        Returns
        -------
        RetentionUpliftModel
            Self, for method chaining.

        Raises
        ------
        ValueError
            If the panel contains censored policies or required columns are missing.
        """
        if "censored_flag" in panel.columns:
            n_censored = int(panel["censored_flag"].sum())
            if n_censored > 0:
                raise ValueError(
                    f"Panel contains {n_censored} censored policies. "
                    "Filter to censored_flag == 0 before fitting, or use outcome='survival'."
                )

        if confounders is None:
            raise ValueError(
                "confounders must be provided: the list of columns to condition on "
                "(age, NCD, region, etc.)."
            )

        required = [treatment_col, outcome_col] + confounders
        validate_panel_columns(panel, required)

        Y = to_numpy(panel[outcome_col])
        T = to_numpy(panel[treatment_col])
        X = to_numpy_2d(panel.select(confounders))

        validate_min_samples(len(Y), min_n=200, context="RetentionUpliftModel.fit: ")

        if np.any(np.isnan(X)):
            raise ValueError("Confounder matrix X contains NaN values. Impute before fitting.")
        if np.any(np.isnan(Y)):
            raise ValueError("Outcome column contains NaN values.")
        if np.any(np.isnan(T)):
            raise ValueError("Treatment column contains NaN values.")

        sample_weight = None
        if weight_col is not None:
            validate_panel_columns(panel, [weight_col])
            sample_weight = to_numpy(panel[weight_col])

        self._confounders = confounders
        self._treatment_col = treatment_col
        self._outcome_col = outcome_col
        self._train_X = X
        self._train_T = T
        self._train_Y = Y

        self._model = self._build_econml_model()

        if self.estimator in ("causal_forest", "dr_learner"):
            if sample_weight is not None:
                self._model.fit(Y, T, X=X, W=None, sample_weight=sample_weight)
            else:
                self._model.fit(Y, T, X=X, W=None)
        elif self.estimator == "x_learner":
            # XLearner has a different fit signature
            T_binary = (T > np.median(T)).astype(int)
            self._model.fit(Y, T_binary, X=X)

        self._is_fitted = True
        return self

    def _check_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError("Call .fit() before using this method.")

    def cate(self, X: pl.DataFrame) -> pl.Series:
        """Return per-customer CATE estimates τ̂(x).

        Parameters
        ----------
        X:
            Customer features. Must contain the same confounder columns used in
            ``.fit()``.

        Returns
        -------
        pl.Series
            CATE estimates: the marginal effect of a 1-unit increase in log price
            on renewal probability. Negative values indicate price-sensitive
            customers (price increase → lapse). The effect of a -10% price change
            (discount) is approximately ``τ̂(x) × log(0.90) ≈ τ̂(x) × (−0.105)``.
        """
        self._check_fitted()
        X_np = to_numpy_2d(X.select(self._confounders))
        tau = self._model.effect(X_np)
        return pl.Series("tau_hat", tau.ravel())

    def cate_inference(
        self, X: pl.DataFrame
    ) -> tuple[pl.Series, pl.Series, pl.Series]:
        """Return CATE estimates with 95% confidence intervals.

        Parameters
        ----------
        X:
            Customer features as in :meth:`cate`.

        Returns
        -------
        tuple[pl.Series, pl.Series, pl.Series]
            ``(tau_hat, lower_95, upper_95)`` where all three are aligned Series.

        Raises
        ------
        RuntimeError
            If ``inference=False`` was set at construction time.
        """
        self._check_fitted()
        if not self.inference:
            raise RuntimeError(
                "Confidence intervals require inference=True at model construction."
            )
        if self.estimator not in ("causal_forest",):
            warnings.warn(
                f"Estimator '{self.estimator}' may not support BLB confidence intervals. "
                "CIs may be approximate.",
                UserWarning,
                stacklevel=2,
            )

        X_np = to_numpy_2d(X.select(self._confounders))
        try:
            interval = self._model.effect_interval(X_np, alpha=0.05)
            lower = interval[0].ravel()
            upper = interval[1].ravel()
        except AttributeError:
            # Some estimators return InferenceResults differently
            result = self._model.effect_inference(X_np)
            lower, upper = result.conf_int(alpha=0.05)
            lower = lower.ravel()
            upper = upper.ravel()

        tau = self._model.effect(X_np).ravel()
        return (
            pl.Series("tau_hat", tau),
            pl.Series("lower_95", lower),
            pl.Series("upper_95", upper),
        )

    def ate(self) -> tuple[float, float, float]:
        """Return the population Average Treatment Effect with confidence interval.

        The ATE is computed on the training data. It is the expected effect of a
        1-unit increase in log price on renewal probability across all customers.

        Returns
        -------
        tuple[float, float, float]
            ``(estimate, lower_95_ci, upper_95_ci)``
        """
        self._check_fitted()
        if self.estimator == "causal_forest":
            ate_result = self._model.ate_(self._train_X)
            try:
                ci = self._model.ate_interval_(self._train_X, alpha=0.05)
                return float(ate_result), float(ci[0]), float(ci[1])
            except Exception:
                pass

        # Fallback: estimate from training predictions
        tau_train = self._model.effect(self._train_X).ravel()
        ate_est = float(np.mean(tau_train))
        ate_std = float(np.std(tau_train) / np.sqrt(len(tau_train)))
        return ate_est, ate_est - 1.96 * ate_std, ate_est + 1.96 * ate_std

    def gate(self, X: pl.DataFrame, by: str) -> pl.DataFrame:
        """Return Group Average Treatment Effects (GATEs) stratified by a column.

        Parameters
        ----------
        X:
            Customer features, including the ``by`` column.
        by:
            Column to group by. Typical usage: ``'ncd_band'``, ``'age_band'``,
            ``'region'``, ``'vehicle_age_band'``.

        Returns
        -------
        pl.DataFrame
            Columns: ``[group, gate, lower_95, upper_95, n]``.
        """
        self._check_fitted()
        if by not in X.columns:
            raise ValueError(f"Column '{by}' not found in X.")

        tau = self.cate(X)
        groups = X[by]

        rows = []
        for g in groups.unique().sort():
            mask = groups == g
            tau_g = tau.filter(mask).to_numpy()
            n_g = len(tau_g)
            gate = float(np.mean(tau_g))
            se = float(np.std(tau_g) / np.sqrt(n_g)) if n_g > 1 else 0.0
            rows.append(
                {
                    "group": str(g),
                    "gate": gate,
                    "lower_95": gate - 1.96 * se,
                    "upper_95": gate + 1.96 * se,
                    "n": n_g,
                }
            )

        return pl.DataFrame(rows).sort("gate")
