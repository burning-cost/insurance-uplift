"""Policy tree and segment summary for uplift-based targeting decisions.

The :class:`PolicyTree` converts CATE estimates into an actionable, human-readable
decision rule: "target customers where age > 50 AND ncd_band <= 2."

v0.1 uses a DecisionTreeRegressor on τ̂(x) as the sklearn backend. This is an
approximation to the welfare-maximising policy tree from Athey & Wager (2021).
The approximation gap for depth=2 trees is typically <5% welfare loss in practice.
The rpy2/policytree backend (exact welfare maximisation) is deferred to v0.2.

The :class:`SegmentSummary` wraps PolicyTree to produce a formatted table of
targeting segments suitable for a pricing committee presentation.

Usage
-----
::

    from insurance_uplift.segment import PolicyTree, SegmentSummary

    tree = PolicyTree(uplift_model=model, max_depth=2)
    tree.fit(X_panel)
    recommendations = tree.recommend(X_panel)
    welfare = tree.welfare_gain()
    rules = tree.export_rules()

    summary = SegmentSummary(uplift_model=model, max_depth=3)
    summary.fit(X_panel)
    table = summary.segment_table()
"""

from __future__ import annotations

import warnings
from typing import Literal, Optional

import numpy as np
import polars as pl


class PolicyTree:
    """Fit a decision tree that maximises targeting welfare.

    Uses a DecisionTreeRegressor on τ̂(x) as the sklearn backend (v0.1). This
    approximates but does not solve welfare maximisation exactly. For
    governance presentations requiring the optimality claim, use the
    ``policytree_r`` backend once rpy2 support is added in v0.2.

    Parameters
    ----------
    uplift_model:
        A fitted :class:`~insurance_uplift.fit.RetentionUpliftModel` instance.
    max_depth:
        Maximum tree depth. Depth 2 gives at most 4 leaf segments. Depth 3 gives
        8 segments. Deeper trees are harder to operationalise in underwriting
        systems.
    backend:
        ``'sklearn'``: greedy tree on τ̂ (available now). ``'policytree_r'``:
        exhaustive welfare maximisation via rpy2 + R policytree (v0.2).
    budget_constraint:
        Maximum fraction of the portfolio to target with discounts. If set,
        the tree recommendation is constrained so that at most this fraction
        of customers receive ``recommend=1``.
    """

    def __init__(
        self,
        uplift_model,
        max_depth: int = 2,
        backend: Literal["sklearn", "policytree_r"] = "sklearn",
        budget_constraint: Optional[float] = None,
    ) -> None:
        self.uplift_model = uplift_model
        self.max_depth = max_depth
        self.backend = backend
        self.budget_constraint = budget_constraint

        self._tree = None
        self._feature_names: Optional[list[str]] = None
        self._tau_hat_train: Optional[np.ndarray] = None
        self._is_fitted = False

    def fit(
        self,
        X: pl.DataFrame,
        sample_weight: Optional[np.ndarray] = None,
    ) -> "PolicyTree":
        """Fit the policy tree on customer features.

        Parameters
        ----------
        X:
            Customer features. Must contain the confounders used to train
            the uplift model.
        sample_weight:
            Optional per-customer weights.

        Returns
        -------
        PolicyTree
            Self, for method chaining.
        """
        if not self.uplift_model._is_fitted:
            raise RuntimeError("uplift_model must be fitted before fitting PolicyTree.")

        if self.backend == "policytree_r":
            warnings.warn(
                "policytree_r backend is not yet implemented (deferred to v0.2). "
                "Falling back to sklearn backend.",
                UserWarning,
                stacklevel=2,
            )
            self.backend = "sklearn"

        tau_hat = self.uplift_model.cate(X).to_numpy()
        self._tau_hat_train = tau_hat
        self._feature_names = self.uplift_model._confounders

        X_np = X.select(self._feature_names).to_numpy().astype(np.float64)

        from sklearn.tree import DecisionTreeRegressor

        tree = DecisionTreeRegressor(
            max_depth=self.max_depth,
            min_samples_leaf=max(1, len(tau_hat) // 20),
            random_state=self.uplift_model.random_state,
        )
        tree.fit(X_np, tau_hat, sample_weight=sample_weight)
        self._tree = tree
        self._X_train = X_np
        self._is_fitted = True
        return self

    def recommend(self, X: pl.DataFrame) -> pl.Series:
        """Return binary targeting recommendations.

        Parameters
        ----------
        X:
            Customer features.

        Returns
        -------
        pl.Series
            1 = target with discount (customer is Persuadable); 0 = do not target.
            If ``budget_constraint`` is set, at most ``budget_constraint × len(X)``
            customers receive ``1``.
        """
        self._check_fitted()
        X_np = X.select(self._feature_names).to_numpy().astype(np.float64)
        tau_hat = self._tree.predict(X_np)

        if self.budget_constraint is not None:
            # Target the n_target customers with the most negative predicted tau
            # (highest price sensitivity / Persuadable) up to the budget limit.
            # Use argsort-based approach to guarantee the count constraint is met
            # regardless of ties.
            n_target = int(np.floor(self.budget_constraint * len(tau_hat)))
            recommend = np.zeros(len(tau_hat), dtype=np.int32)
            if n_target > 0:
                # argsort ascending: first n_target indices are the most negative
                top_negative_idx = np.argsort(tau_hat)[:n_target]
                recommend[top_negative_idx] = 1
        else:
            recommend = (tau_hat < 0).astype(np.int32)

        return pl.Series("recommend", recommend)

    def welfare_gain(self) -> float:
        """Estimated welfare gain vs uniform (no targeting) policy.

        Returns the expected improvement in portfolio retention rate from using
        the policy tree, compared to treating all customers uniformly.

        Returns
        -------
        float
            Welfare gain in percentage points of retention rate.
        """
        self._check_fitted()
        tau_hat_all = self._tau_hat_train
        recommend_mask = tau_hat_all < 0  # Persuadable customers

        if not recommend_mask.any():
            return 0.0

        # Expected gain from targeting Persuadable customers with a discount
        # = -1 × E[τ | τ < 0] × P(τ < 0) × effect_of_discount
        # Here we return the expected absolute improvement assuming a -10% discount
        discount_log = np.log(0.90)  # -10% price change
        avg_tau_persuadable = float(np.mean(tau_hat_all[recommend_mask]))
        frac_persuadable = float(recommend_mask.mean())

        # Welfare gain = |τ(x)| × |log(0.90)| for each Persuadable customer
        welfare = abs(avg_tau_persuadable * discount_log) * frac_persuadable * 100
        return welfare

    def export_rules(self) -> list[dict]:
        """Export the decision tree rules as machine-readable dicts.

        Suitable for import into underwriting or campaign management systems.

        Returns
        -------
        list[dict]
            One dict per leaf node with keys: ``node_id``, ``rule``,
            ``avg_tau``, ``n_samples``, ``action``.
        """
        self._check_fitted()
        from sklearn.tree import _tree

        tree = self._tree
        feature_names = self._feature_names

        def recurse(node, depth, parent_rule):
            rules = []
            if tree.tree_.feature[node] == _tree.TREE_UNDEFINED:
                # Leaf node
                avg_tau = float(tree.tree_.value[node][0][0])
                n_samples = int(tree.tree_.n_node_samples[node])
                action = "target" if avg_tau < 0 else "do_not_target"
                rules.append(
                    {
                        "node_id": int(node),
                        "rule": " AND ".join(parent_rule) if parent_rule else "ALL",
                        "avg_tau": avg_tau,
                        "n_samples": n_samples,
                        "action": action,
                    }
                )
            else:
                feat = feature_names[tree.tree_.feature[node]]
                threshold = tree.tree_.threshold[node]
                # Left child: feature <= threshold
                rules.extend(
                    recurse(
                        tree.tree_.children_left[node],
                        depth + 1,
                        parent_rule + [f"{feat} <= {threshold:.4f}"],
                    )
                )
                # Right child: feature > threshold
                rules.extend(
                    recurse(
                        tree.tree_.children_right[node],
                        depth + 1,
                        parent_rule + [f"{feat} > {threshold:.4f}"],
                    )
                )
            return rules

        return recurse(0, 0, [])

    def _check_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError("Call .fit() before using this method.")


class SegmentSummary:
    """Fit a policy tree and produce a formatted targeting segment table.

    Designed for pricing committee presentations. Each segment gets a
    human-readable rule description, CATE summary statistics, and a
    recommended action.

    Parameters
    ----------
    uplift_model:
        A fitted :class:`~insurance_uplift.fit.RetentionUpliftModel` instance.
    max_depth:
        Depth of the underlying policy tree.
    min_samples_leaf:
        Minimum number of policies required in each segment. Segments with
        fewer policies are merged upward.
    """

    def __init__(
        self,
        uplift_model,
        max_depth: int = 3,
        min_samples_leaf: int = 50,
    ) -> None:
        self.uplift_model = uplift_model
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self._policy_tree: Optional[PolicyTree] = None
        self._X_fit: Optional[pl.DataFrame] = None
        self._is_fitted = False

    def fit(self, X: pl.DataFrame) -> "SegmentSummary":
        """Fit the underlying policy tree on customer features.

        Parameters
        ----------
        X:
            Customer features.

        Returns
        -------
        SegmentSummary
            Self, for method chaining.
        """
        self._policy_tree = PolicyTree(
            uplift_model=self.uplift_model,
            max_depth=self.max_depth,
        )
        self._policy_tree._tree = None
        self._policy_tree._is_fitted = False

        tau_hat = self.uplift_model.cate(X).to_numpy()
        feature_names = self.uplift_model._confounders
        self._policy_tree._tau_hat_train = tau_hat
        self._policy_tree._feature_names = feature_names

        X_np = X.select(feature_names).to_numpy().astype(np.float64)

        from sklearn.tree import DecisionTreeRegressor

        tree = DecisionTreeRegressor(
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.uplift_model.random_state,
        )
        tree.fit(X_np, tau_hat)
        self._policy_tree._tree = tree
        self._policy_tree._X_train = X_np
        self._policy_tree._is_fitted = True
        self._X_fit = X
        self._is_fitted = True
        return self

    def segment_table(self) -> pl.DataFrame:
        """Return a formatted table of targeting segments.

        Returns
        -------
        pl.DataFrame
            Columns:

            - ``segment_id``: integer ID for the segment (leaf node)
            - ``rule_description``: human-readable decision rule
            - ``n``: number of policies in this segment
            - ``avg_tau``: average predicted CATE
            - ``lower_ci``: 2.5th percentile of τ̂ in segment (approximate CI)
            - ``upper_ci``: 97.5th percentile of τ̂ in segment
            - ``recommended_action``: 'offer_discount' or 'hold_rate'
            - ``avg_treatment_effect_pp``: expected effect of a -10% discount
              on renewal probability (percentage points)
        """
        if not self._is_fitted:
            raise RuntimeError("Call .fit() before .segment_table().")

        rules = self._policy_tree.export_rules()
        X_np = self._policy_tree._X_train
        tau_hat = self._policy_tree._tau_hat_train

        # Get leaf node assignments for training data
        leaf_ids = self._policy_tree._tree.apply(X_np)

        rows = []
        for rule_dict in rules:
            node_id = rule_dict["node_id"]
            mask = leaf_ids == node_id
            tau_seg = tau_hat[mask]
            n_seg = len(tau_seg)

            if n_seg == 0:
                continue

            avg_tau = float(np.mean(tau_seg))
            lower_ci = float(np.percentile(tau_seg, 2.5)) if n_seg > 1 else avg_tau
            upper_ci = float(np.percentile(tau_seg, 97.5)) if n_seg > 1 else avg_tau
            discount_log = np.log(0.90)
            effect_pp = abs(avg_tau * discount_log) * 100 if avg_tau < 0 else 0.0

            rows.append(
                {
                    "segment_id": node_id,
                    "rule_description": rule_dict["rule"],
                    "n": n_seg,
                    "avg_tau": avg_tau,
                    "lower_ci": lower_ci,
                    "upper_ci": upper_ci,
                    "recommended_action": (
                        "offer_discount" if avg_tau < 0 else "hold_rate"
                    ),
                    "avg_treatment_effect_pp": effect_pp,
                }
            )

        return pl.DataFrame(rows).sort("avg_tau")

    def plot_tree(self, ax=None):
        """Visualise the policy tree structure.

        Parameters
        ----------
        ax:
            Matplotlib axes. If ``None``, a new figure is created.

        Returns
        -------
        matplotlib.axes.Axes
        """
        if not self._is_fitted:
            raise RuntimeError("Call .fit() before .plot_tree().")

        try:
            import matplotlib.pyplot as plt
            from sklearn.tree import plot_tree as sklearn_plot_tree
        except ImportError:
            raise ImportError(
                "matplotlib and scikit-learn are required for plot_tree."
            )

        if ax is None:
            fig, ax = plt.subplots(figsize=(min(20, 4 * 2 ** self.max_depth), 8))

        sklearn_plot_tree(
            self._policy_tree._tree,
            feature_names=self._policy_tree._feature_names,
            filled=True,
            rounded=True,
            fontsize=10,
            ax=ax,
        )
        ax.set_title("Policy Tree — Targeting Segments by Predicted τ̂(x)")
        return ax
