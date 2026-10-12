"""Step 6 · decentralized three-part convergence test, evaluated by every client on its own
fixed validation patients D_k^val:

    1  loss stability          |L_k^(t) - L_k^(t-1)| / (|L_k^(t-1)| + eps)          <= tau_loss
    2  prediction stability    (1/|D_val|) sum_i 1[y_k^(t)(x_i) != y_k^(t-1)(x_i)]   <= tau_pred
    3  global/local agreement  (1/|D_val|) sum_i 1[y_Fk^(t)(x_i) != y_G^(t)(x_i)]    <= tau_agree

All three pass: q_k = q_k + 1, otherwise q_k = 0. The client sends only b_k^(t) = [q_k >= P].
L is the validation log-loss of the local model F_k; round 0 initialises L_k^(0) and y_k^(0)
from the initial forest F_k^(0).
"""

from dataclasses import dataclass

import numpy as np

PROBA_FLOOR = 1e-15


def log_loss(y, proba, classes):
    """Mean cross-entropy of the class probabilities (columns in `classes` order)."""
    idx = np.searchsorted(classes, y)
    return float(-np.mean(np.log(np.clip(proba[np.arange(len(y)), idx], PROBA_FLOOR, 1.0))))


@dataclass(frozen=True)
class LocalCheck:
    loss: float
    rel_loss_change: float
    pred_change: float
    disagreement: float
    loss_stable: bool
    pred_stable: bool
    agrees: bool
    patience: int
    flag: bool


class LocalConvergence:
    """One client's patience counter q_k and its three tests."""

    def __init__(self, tau_loss, tau_pred, tau_agree, patience, eps=1e-12):
        self.tau_loss, self.tau_pred, self.tau_agree = tau_loss, tau_pred, tau_agree
        self.patience, self.eps = patience, eps
        self.q = 0
        self.prev_loss = self.prev_pred = None

    def initialise(self, loss, pred):
        """Round 0: L_k^(0) and y_k^(0) of the initial local forest F_k^(0)."""
        self.prev_loss, self.prev_pred = loss, pred

    def update(self, loss, pred, global_pred):
        rel = abs(loss - self.prev_loss) / (abs(self.prev_loss) + self.eps)
        change = float(np.mean(pred != self.prev_pred))
        disagree = 1.0 if global_pred is None else float(np.mean(pred != global_pred))  # no G^(t): no agreement
        ok = (rel <= self.tau_loss, change <= self.tau_pred, disagree <= self.tau_agree)
        self.q = self.q + 1 if all(ok) else 0
        self.prev_loss, self.prev_pred = loss, pred
        return LocalCheck(loss, rel, change, disagree, *ok, self.q, self.q >= self.patience)
