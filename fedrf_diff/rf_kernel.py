"""Local Random Forest training, explicit whole-tree export and forest inference.

No feature or latent encoding is involved: a tree is kept and transferred as its explicit
node structure

    feature[i], threshold[i]   split of internal node i (x[feature] <= threshold goes left)
    left[i], right[i]          child node ids (LEAF = -1 at a leaf)
    value[i]                   class-probability vector of leaf i (unused at internal nodes)

and the global forest predicts by averaging the class probabilities of all its trees.
"""

from dataclasses import dataclass, field, replace

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold

LEAF = -1


@dataclass(frozen=True)
class Tree:
    tree_id: str
    client_id: str
    feature: np.ndarray
    threshold: np.ndarray
    left: np.ndarray
    right: np.ndarray
    value: np.ndarray  # (n_nodes, n_classes)
    is_leaf: np.ndarray = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        object.__setattr__(self, "is_leaf", self.left == LEAF)

    @property
    def n_nodes(self):
        return len(self.left)

    @property
    def n_leaves(self):
        return int(self.is_leaf.sum())

    def apply(self, X):
        """Leaf id reached by every row of X."""
        node = np.zeros(len(X), dtype=np.int64)
        rows = np.arange(len(X))
        active = ~self.is_leaf[node]
        while active.any():
            n = node[active]
            go_left = X[rows[active], self.feature[n]] <= self.threshold[n]
            node[active] = np.where(go_left, self.left[n], self.right[n])
            active = ~self.is_leaf[node]
        return node

    def predict_proba(self, X):
        return self.value[self.apply(X)]

    def with_values(self, value):
        return replace(self, value=value)


def train_local_forest(X, y, n_trees, seed, rf_params):
    """Step 3 · a fresh local Random Forest of n_trees trees with balanced class weights."""
    rf = RandomForestClassifier(n_estimators=n_trees, random_state=seed, class_weight="balanced", **rf_params)
    return rf.fit(X, y)


def tune_forest_params(X, y, grid, folds, n_trees, seed, rf_params):
    """Local model selection: stratified k-fold grid search on the client's training patients.

    The cross-validation forests stay inside the hospital (they are never exported, so they
    consume no DP budget); only the winning hyperparameters shape the trees that are shared.
    Returns (best parameters, mean balanced accuracy of the best setting, number of fits).
    """
    base = dict(rf_params)
    for key in grid:
        base.pop(key, None)
    search = GridSearchCV(RandomForestClassifier(n_estimators=n_trees, random_state=seed, class_weight="balanced",
                                                 **base),
                          grid, scoring="balanced_accuracy",
                          cv=StratifiedKFold(folds, shuffle=True, random_state=seed), n_jobs=1, refit=False)
    search.fit(X, y)
    return {**base, **search.best_params_}, float(search.best_score_), len(search.cv_results_["params"]) * folds


def export_tree(estimator, classes, tree_id, client_id):
    """Step 4 · the explicit node structure of one fitted scikit-learn tree."""
    t = estimator.tree_
    left = t.children_left.astype(np.int64)
    leaf = left == LEAF
    # Leaf class distributions in the agreed class order
    counts = np.zeros((t.node_count, len(classes)))
    for j, c in enumerate(estimator.classes_):
        counts[:, classes.index(int(c))] = t.value[:, 0, j]
    totals = counts.sum(axis=1, keepdims=True)
    value = np.divide(counts, totals, out=np.full_like(counts, 1 / len(classes)), where=totals > 0)
    return Tree(tree_id=tree_id, client_id=client_id,
                feature=np.where(leaf, LEAF, t.feature).astype(np.int64),
                threshold=np.where(leaf, 0.0, t.threshold).astype(np.float64),
                left=left, right=t.children_right.astype(np.int64), value=value)


def tree_to_message(tree, schema):
    """TreeMessage: the tree as explicit nodes plus the metadata the server checks."""
    nodes = []
    for i in range(tree.n_nodes):
        if tree.is_leaf[i]:
            nodes.append({"node_id": i, "leaf": True, "class_distribution": tree.value[i].tolist()})
        else:
            nodes.append({"node_id": i, "feature": int(tree.feature[i]), "threshold": float(tree.threshold[i]),
                          "left_child": int(tree.left[i]), "right_child": int(tree.right[i])})
    return {"schema_version": schema.version, "task": "classification", "classes": list(schema.classes),
            "client_id": tree.client_id, "tree_id": tree.tree_id, "nodes": nodes}


def forest_proba(trees, X):
    """P_G(y = c | x) = (1 / |T_G|) sum over trees of P_h(y = c | x)."""
    return sum(t.predict_proba(X) for t in trees) / len(trees)
