"""Federated client (one hospital): local training, whole-tree export, DP, proof, evaluation."""

import hashlib
import json
from dataclasses import dataclass

import numpy as np

from . import zkp
from .alignment import check_local_data, encode_features, imputation_stats
from .convergence import prediction_change_sum
from .data import TARGET
from .metrics import confusion_counts
from .privacy import laplace_scale, privatize_leaves
from .rf_kernel import export_tree, train_local_forest, tree_to_message

REPORT_BYTES = 4 * 5 + 8  # confusion counts + validation size (int32) and the Delta P sum (float64)


@dataclass(frozen=True)
class EvalReport:
    """What a client sends back after scoring the global forest: counts only, no patients."""
    counts: np.ndarray  # [tn, fp, fn, tp]
    pred_change_sum: float | None  # sum_i ||P_t(i, .) - P_(t-1)(i, .)||_1 / 2, None in round 1
    n: int


class HospitalClient:
    def __init__(self, cid, hospital, train, val, schema, cfg, with_zkp=True):
        check_local_data(train)
        check_local_data(val)
        self.cid, self.hospital = cid, hospital
        self.name = f"Client {cid}({hospital})"
        self.client_id = f"client_{cid}"
        self.schema, self.cfg = schema, cfg
        self.train_raw, self.val_raw = train, val
        self.keypair = zkp.keygen(cfg["zkp_lambda"]) if with_zkp else None
        self._val_proba = {}  # tree_id -> class probabilities of that global tree on the validation patients
        self._prev_proba = None

    def imputation_stats(self):
        return imputation_stats(self.train_raw)

    def apply_preprocessing(self, fill_values):
        self.X_train = encode_features(self.train_raw, fill_values, self.schema)
        self.y_train = self.train_raw[TARGET].to_numpy()
        self.X_val = encode_features(self.val_raw, fill_values, self.schema)
        self.y_val = self.val_raw[TARGET].to_numpy()

    def _seed(self, rnd, stream):
        return np.random.SeedSequence([self.cfg["seed"], rnd, self.cid, stream])

    # Step 3 · local Random Forest with m new trees
    def train_round(self, rnd):
        seed = int(self._seed(rnd, 0).generate_state(1)[0])
        return train_local_forest(self.X_train, self.y_train, self.cfg["trees_per_client"], seed,
                                  self.cfg["rf_params"])

    # Step 4 · explicit tree export
    def export(self, forest, rnd):
        classes = list(self.schema.classes)
        return [export_tree(est, classes, f"{self.client_id}_r{rnd:03d}_t{j:03d}", self.client_id)
                for j, est in enumerate(forest.estimators_)]

    # Laplace DP on every leaf vector before anything leaves the hospital
    def privatize(self, trees, epsilon, rnd):
        if epsilon is None:
            return trees
        rng = np.random.default_rng(self._seed(rnd, 1))
        return [t.with_values(privatize_leaves(t.value, t.is_leaf, epsilon, self.cfg["dp_clip"], rng)) for t in trees]

    # Step 4-5 · serialization of the whole trees and their metadata (transport format, not an encoding)
    def serialize(self, trees, rnd, epsilon):
        privacy = None if epsilon is None else {
            "mechanism": "laplace", "epsilon": epsilon, "clip_l1": self.cfg["dp_clip"],
            "scale": laplace_scale(epsilon, self.cfg["dp_clip"])}
        update = {"schema_id": self.schema.schema_id, "schema_version": self.schema.version,
                  "client_id": self.client_id, "round": rnd, "privacy": privacy,
                  "trees": [tree_to_message(t, self.schema) for t in trees]}
        return json.dumps(update, separators=(",", ":"), allow_nan=False).encode()

    # Step 5 · Modified Pedersen proof bound to this exact update
    def prove(self, payload, rnd):
        context = zkp.update_context(self.schema.schema_id, rnd, self.client_id)
        return zkp.prove(self.keypair, hashlib.sha256(payload).digest(), context)

    # Step 8 · score the global forest G_t on the local validation patients
    def receive_global_update(self, added, removed):
        for tree_id in removed:
            del self._val_proba[tree_id]
        for tree in added:
            self._val_proba[tree.tree_id] = tree.predict_proba(self.X_val)

    def evaluate(self):
        proba = sum(self._val_proba.values()) / len(self._val_proba)
        pred = np.asarray(self.schema.classes)[proba.argmax(axis=1)]
        change = None if self._prev_proba is None else prediction_change_sum(proba, self._prev_proba)
        self._prev_proba = proba
        return EvalReport(confusion_counts(self.y_val, pred), change, len(self.y_val))
