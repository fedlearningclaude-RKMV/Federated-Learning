"""Federated client (one hospital).

    Step 1  train the initial forest F_k^(0) on the private training split
    Step 2  upload the complete forest F_k^(t-1) (explicit trees + schema metadata) with a ZKP
    Step 5  receive G^(t), train a supplementary forest H_k^(t) and build
            F_k^(t) = CapOrPrune(TreePool(G^(t), H_k^(t)), B_k)
    Step 6  run the three-part convergence test on the fixed validation split and return only
            the boolean flag b_k^(t)

CapOrPrune keeps the B_k pool trees with the highest balanced accuracy on the client's
training patients that each tree did not train on: its out-of-bag patients for a tree this
client grew, every training patient for a tree grown elsewhere. The validation split is never
used for pruning, so the convergence test stays an honest held-out check.
"""

import hashlib
import json
import time

import numpy as np

from . import zkp
from .alignment import check_local_data, encode_features, imputation_stats
from .convergence import LocalConvergence, log_loss
from .data import TARGET
from .metrics import confusion_counts
from .privacy import privatize_leaves
from .rf_kernel import export_tree, train_local_forest, tree_to_message, tune_forest_params

FLAG_BYTES = 1  # b_k^(t) is the only thing a client reports about its validation data
_TUNING_CACHE = {}  # tuning is a deterministic function of the training data and seed; the DP sweep reuses it


def balanced_accuracy(y, pred, classes):
    recalls = [np.mean(pred[y == c] == c) for c in classes if np.any(y == c)]
    return float(np.mean(recalls)) if recalls else 0.0


class HospitalClient:
    def __init__(self, cid, hospital, train, val, schema, cfg, with_zkp=True):
        check_local_data(train)
        check_local_data(val)
        self.cid, self.hospital = cid, hospital
        self.name = f"Client {cid}({hospital})"
        self.client_id = f"client_{cid}"
        self.schema, self.cfg = schema, cfg
        self.classes = np.asarray(schema.classes)
        self.train_raw, self.val_raw = train, val
        self.keypair = zkp.keygen(cfg["zkp_lambda"]) if with_zkp else None
        self.forest = []  # F_k^(t)
        self.global_forest = []  # G^(t) as last received
        self.convergence = LocalConvergence(cfg["tau_loss"], cfg["tau_pred"], cfg["tau_agree"], cfg["patience"])
        self._in_bag = {}  # tree_id -> training rows a tree grown here was fitted on
        self._val_proba = {}  # tree_id -> class probabilities on the validation patients (cache)
        self._train_score = {}  # tree_id -> CapOrPrune score (cache)
        self.rf_params = dict(cfg["rf_params"])  # replaced by the locally tuned setting every round
        self.tuning = None  # (best parameters, CV balanced accuracy, number of CV fits) of the last round
        self.tuning_seconds = 0.0

    def imputation_stats(self):
        return imputation_stats(self.train_raw)

    def apply_preprocessing(self, fill_values):
        self.X_train = encode_features(self.train_raw, fill_values, self.schema)
        self.y_train = self.train_raw[TARGET].to_numpy()
        self.X_val = encode_features(self.val_raw, fill_values, self.schema)
        self.y_val = self.val_raw[TARGET].to_numpy()

    def _seed(self, rnd, stream):
        return np.random.SeedSequence([self.cfg["seed"], rnd, self.cid, stream])

    # Steps 1 and 5 · a local forest of explicit trees, Laplace DP on every leaf vector
    def tune(self, rnd):
        """Local hyperparameter selection by stratified k-fold CV on the training patients only."""
        tuning = self.cfg["local_tuning"]
        if not tuning:
            return
        seed = int(self._seed(rnd, 2).generate_state(1)[0])
        key = (hashlib.sha256(self.X_train.tobytes() + self.y_train.tobytes()).hexdigest(), seed,
               repr(tuning), repr(self.cfg["rf_params"]))
        t0 = time.perf_counter()
        if key not in _TUNING_CACHE:
            _TUNING_CACHE[key] = (tune_forest_params(self.X_train, self.y_train, tuning["grid"], tuning["folds"],
                                                     tuning["cv_trees"], seed, self.cfg["rf_params"]),
                                  time.perf_counter() - t0)
        self.tuning, self.tuning_seconds = _TUNING_CACHE[key]
        self.rf_params = self.tuning[0]

    def grow(self, rnd, n_trees, epsilon):
        """F_k^(0) at round 0, the supplementary forest H_k^(t) afterwards (after local tuning)."""
        self.tune(rnd)
        rf = train_local_forest(self.X_train, self.y_train, n_trees, int(self._seed(rnd, 0).generate_state(1)[0]),
                                self.rf_params)
        rng = np.random.default_rng(self._seed(rnd, 1))
        trees = []
        for j, (est, rows) in enumerate(zip(rf.estimators_, rf.estimators_samples_)):
            tree = export_tree(est, list(self.schema.classes), f"{self.client_id}_r{rnd:03d}_t{j:03d}", self.client_id)
            if epsilon is not None:
                tree = tree.with_values(privatize_leaves(tree.value, tree.is_leaf, epsilon, self.cfg["dp_clip"], rng))
            self._in_bag[tree.tree_id] = np.unique(rows)
            trees.append(tree)
        return trees

    def initialise(self, epsilon):
        """Step 1 · F_k^(0) and the round-0 reference L_k^(0), y_k^(0)."""
        self.forest = self.grow(0, self.cfg["initial_trees"], epsilon)
        proba = self.proba(self.forest)
        self.convergence.initialise(log_loss(self.y_val, proba, self.classes), self.predict(proba))

    # Step 2 · the complete local forest with its schema metadata (transport format, not an encoding)
    def serialize(self, rnd, privacy=None):
        update = {"schema_id": self.schema.schema_id, "schema_version": self.schema.version,
                  "client_id": self.client_id, "round": rnd, "privacy": privacy,
                  "trees": [tree_to_message(t, self.schema) for t in self.forest]}
        return json.dumps(update, separators=(",", ":"), allow_nan=False).encode()

    def prove(self, payload, rnd):
        context = zkp.update_context(self.schema.schema_id, rnd, self.client_id)
        return zkp.prove(self.keypair, hashlib.sha256(payload).digest(), context)

    # Step 5 · local refinement
    def _score(self, tree):
        if tree.tree_id not in self._train_score:
            rows = np.ones(len(self.y_train), dtype=bool)
            if tree.tree_id in self._in_bag:
                rows[self._in_bag[tree.tree_id]] = False
            pred = self.predict(tree.predict_proba(self.X_train[rows])) if rows.any() else np.array([])
            self._train_score[tree.tree_id] = balanced_accuracy(self.y_train[rows], pred, self.classes)
        return self._train_score[tree.tree_id]

    def cap_or_prune(self, pool):
        budget = self.cfg["local_budget"]
        ranked = sorted(pool, key=lambda t: (self._score(t), t.tree_id), reverse=True)
        return ranked[:budget]

    def refine(self, global_forest, rnd, epsilon):
        """F_k^(t) = CapOrPrune(TreePool(G^(t), H_k^(t)), B_k); returns H_k^(t)."""
        self.global_forest = list(global_forest)
        supplementary = self.grow(rnd, self.cfg["supplementary_trees"], epsilon)
        pool = {t.tree_id: t for t in self.global_forest + supplementary}
        self.forest = self.cap_or_prune(list(pool.values()))
        return supplementary

    # Step 6 · three-part convergence test on the fixed validation split
    def proba(self, trees):
        for t in trees:
            if t.tree_id not in self._val_proba:
                self._val_proba[t.tree_id] = t.predict_proba(self.X_val)
        return sum(self._val_proba[t.tree_id] for t in trees) / len(trees)

    def predict(self, proba):
        return self.classes[proba.argmax(axis=1)]

    def convergence_check(self):
        local_proba = self.proba(self.forest)
        local_pred = self.predict(local_proba)
        global_pred = self.predict(self.proba(self.global_forest)) if self.global_forest else None
        check = self.convergence.update(log_loss(self.y_val, local_proba, self.classes), local_pred, global_pred)
        global_counts = None if global_pred is None else confusion_counts(self.y_val, global_pred)
        return check, confusion_counts(self.y_val, local_pred), global_counts
