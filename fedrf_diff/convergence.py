"""Section 4 · combined convergence criterion.

    Delta M_t = |M_t - M_(t-1)|                                       metric stability
    Delta P_t = (1 / |V|) sum_i ||P_t(i, .) - P_(t-1)(i, .)||_1 / 2     prediction stability
    Converged_t = (Delta M_t <= eps_M) AND (Delta P_t <= eps_P)

The success counter grows when both tests pass and resets otherwise; training stops once it
reaches H or after R rounds. Round 1 only initialises M_1 and P_1.
"""

from dataclasses import dataclass

import numpy as np


def prediction_change_sum(proba, prev_proba):
    """sum_i ||P_t(i, .) - P_(t-1)(i, .)||_1 / 2 over one client's validation patients."""
    return float(0.5 * np.abs(proba - prev_proba).sum())


@dataclass(frozen=True)
class RoundCheck:
    delta_metric: float | None
    delta_pred: float | None
    metric_stable: bool
    pred_stable: bool
    stable_rounds: int
    converged: bool


class CombinedConvergence:
    def __init__(self, eps_metric, eps_pred, patience):
        self.eps_metric, self.eps_pred, self.patience = eps_metric, eps_pred, patience
        self.prev_metric = None
        self.stable_rounds = 0

    def update(self, metric, delta_pred):
        """Record M_t and Delta P_t (None in the first round) and apply the combined test."""
        if self.prev_metric is None:
            self.prev_metric = metric
            return RoundCheck(None, None, False, False, 0, False)
        delta_metric = abs(metric - self.prev_metric)
        self.prev_metric = metric
        metric_stable = delta_metric <= self.eps_metric
        pred_stable = delta_pred <= self.eps_pred
        self.stable_rounds = self.stable_rounds + 1 if metric_stable and pred_stable else 0
        return RoundCheck(delta_metric, delta_pred, metric_stable, pred_stable, self.stable_rounds,
                          self.stable_rounds >= self.patience)
