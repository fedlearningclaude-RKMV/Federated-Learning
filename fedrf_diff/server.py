"""Federated server: authentication, Step 6 tree validation and the global forest (Steps 6-9).

The clients send their whole trees as explicit node structures, so there is nothing to decode:
the server only parses the JSON transport format and checks every received tree before it
joins the global forest (schema id and version, class order, sender and round, payload size,
unique dense node ids, valid feature indices and finite thresholds, child references that form
one rooted tree, leaf outputs that are valid class distributions, node count and depth limits,
replayed tree ids). A malformed tree is rejected on its own; a malformed update as a whole.
"""

import hashlib
import json
import math
from collections import deque
from dataclasses import dataclass

import numpy as np

from . import zkp
from .metrics import metrics_from_counts
from .rf_kernel import LEAF, Tree, tree_to_message

INTERNAL_KEYS = {"node_id", "feature", "threshold", "left_child", "right_child"}
LEAF_KEYS = {"node_id", "leaf", "class_distribution"}
TREE_KEYS = {"schema_version", "task", "classes", "client_id", "tree_id", "nodes"}
UPDATE_KEYS = {"schema_id", "schema_version", "client_id", "round", "privacy", "trees"}
DISTRIBUTION_TOL = 1e-6


@dataclass(frozen=True)
class Limits:
    max_payload_bytes: int
    max_trees: int
    max_nodes: int
    max_depth: int

    @classmethod
    def from_config(cls, cfg):
        return cls(cfg["max_payload_bytes"], cfg["max_trees_per_update"], cfg["max_nodes_per_tree"],
                   cfg["max_tree_depth"])


class InvalidUpdate(ValueError):
    pass


class InvalidTree(ValueError):
    pass


def _is_int(v):
    return isinstance(v, int) and not isinstance(v, bool)


def _is_real(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _reject_constant(token):
    raise InvalidUpdate(f"non-finite number {token} in payload")


def check_update(payload, schema, limits, client_id, rnd):
    """Check one received update; returns (valid trees, [(tree_id, reason), ...] for rejected trees)."""
    if len(payload) > limits.max_payload_bytes:
        raise InvalidUpdate(f"payload of {len(payload)} bytes exceeds {limits.max_payload_bytes}")
    try:
        update = json.loads(payload, parse_constant=_reject_constant)
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise InvalidUpdate(f"unparseable payload: {e}") from None
    if not isinstance(update, dict) or set(update) != UPDATE_KEYS:
        raise InvalidUpdate("update fields do not match the protocol")
    if update["schema_id"] != schema.schema_id or update["schema_version"] != schema.version:
        raise InvalidUpdate("schema mismatch")
    if update["client_id"] != client_id or update["round"] != rnd:
        raise InvalidUpdate("sender or round mismatch")
    if not isinstance(update["trees"], list) or len(update["trees"]) > limits.max_trees:
        raise InvalidUpdate(f"trees must be a list of at most {limits.max_trees}")
    trees, rejected = [], []
    for msg in update["trees"]:
        try:
            trees.append(validate_tree(msg, schema, limits, client_id))
        except InvalidTree as e:
            tree_id = msg.get("tree_id") if isinstance(msg, dict) else None
            rejected.append((str(tree_id), str(e)))
    return trees, rejected


def validate_tree(msg, schema, limits, client_id):
    """Check one TreeMessage and rebuild its node arrays."""
    if not isinstance(msg, dict) or set(msg) != TREE_KEYS:
        raise InvalidTree("tree fields do not match the protocol")
    if msg["schema_version"] != schema.version or msg["task"] != "classification":
        raise InvalidTree("schema version or task mismatch")
    if msg["classes"] != list(schema.classes):
        raise InvalidTree("class order mismatch")
    if msg["client_id"] != client_id:
        raise InvalidTree("tree claims another client")
    tree_id = msg["tree_id"]
    if not isinstance(tree_id, str) or not tree_id.startswith(f"{client_id}_"):
        raise InvalidTree("tree id must start with the sender's client id")
    nodes = msg["nodes"]
    if not isinstance(nodes, list) or not 1 <= len(nodes) <= limits.max_nodes:
        raise InvalidTree(f"a tree needs 1 to {limits.max_nodes} nodes")

    n, k = len(nodes), schema.n_classes
    feature = np.full(n, LEAF, dtype=np.int64)
    threshold = np.zeros(n)
    left = np.full(n, LEAF, dtype=np.int64)
    right = np.full(n, LEAF, dtype=np.int64)
    value = np.full((n, k), 1 / k)
    seen = np.zeros(n, dtype=bool)
    for node in nodes:
        if not isinstance(node, dict) or not _is_int(node.get("node_id")) or not 0 <= node["node_id"] < n:
            raise InvalidTree("node ids must be the integers 0 .. n-1")
        i = node["node_id"]
        if seen[i]:
            raise InvalidTree(f"duplicate node id {i}")
        seen[i] = True
        if set(node) == LEAF_KEYS:
            dist = node["class_distribution"]
            if node["leaf"] is not True or not isinstance(dist, list) or len(dist) != k:
                raise InvalidTree(f"leaf {i} needs a class distribution of length {k}")
            if not all(_is_real(p) and p >= 0 for p in dist) or abs(sum(dist) - 1) > DISTRIBUTION_TOL:
                raise InvalidTree(f"leaf {i} output is not a probability distribution")
            value[i] = dist
        elif set(node) == INTERNAL_KEYS:
            f, t, lc, rc = node["feature"], node["threshold"], node["left_child"], node["right_child"]
            if not _is_int(f) or not 0 <= f < schema.n_features:
                raise InvalidTree(f"node {i} splits on an unknown feature")
            if not _is_real(t):
                raise InvalidTree(f"node {i} threshold is not a finite number")
            if not (_is_int(lc) and _is_int(rc) and 0 < lc < n and 0 < rc < n and lc != rc):
                raise InvalidTree(f"node {i} has invalid child references")
            feature[i], threshold[i], left[i], right[i] = f, t, lc, rc
        else:
            raise InvalidTree(f"node {i} is neither a valid leaf nor a valid split")

    # One rooted tree: every node except the root has exactly one parent and is reachable from 0
    depth = np.full(n, -1, dtype=np.int64)
    depth[0] = 0
    queue = deque([0])
    while queue:
        i = queue.popleft()
        if left[i] == LEAF:
            continue
        for child in (left[i], right[i]):
            if depth[child] != -1:
                raise InvalidTree(f"node {child} has more than one parent or closes a cycle")
            depth[child] = depth[i] + 1
            queue.append(child)
    if (depth == -1).any():
        raise InvalidTree("unreachable nodes")
    if depth.max() > limits.max_depth:
        raise InvalidTree(f"depth {depth.max()} exceeds {limits.max_depth}")
    return Tree(tree_id=tree_id, client_id=client_id, feature=feature, threshold=threshold,
                left=left, right=right, value=value)


def load_forest(path, schema, limits):
    """Reload a saved global forest, validating every tree again."""
    saved = json.loads(path.read_text())
    if saved["schema_id"] != schema.schema_id:
        raise InvalidUpdate("saved forest was built under another schema")
    return [validate_tree(msg, schema, limits, msg["client_id"]) for msg in saved["trees"]]



class FederatedServer:
    def __init__(self, schema, limits, zkp_lambda, max_trees_per_client=None):
        self.schema, self.limits = schema, limits
        self.zkp_lambda = zkp_lambda
        self.max_trees_per_client = max_trees_per_client
        self.public_keys = {}
        self.forest = {}  # tree_id -> Tree, in acceptance order
        self.tree_size = {}  # tree_id -> bytes of its TreeMessage (what a client downloads)

    def register(self, client_id, commitment):
        if not zkp.in_group(commitment):
            raise ValueError(f"{client_id}: commitment is not an element of the {zkp.GROUP} subgroup")
        self.public_keys[client_id] = commitment

    def verify(self, client_id, payload, proof, rnd):
        if client_id not in self.public_keys:
            return False
        context = zkp.update_context(self.schema.schema_id, rnd, client_id)
        return zkp.verify(self.public_keys[client_id], hashlib.sha256(payload).digest(), context, proof,
                          self.zkp_lambda)

    def receive(self, client_id, payload, rnd):
        """Step 6 · valid trees of one authenticated update and the rejected ones with reasons."""
        try:
            trees, rejected = check_update(payload, self.schema, self.limits, client_id, rnd)
        except InvalidUpdate as e:
            return [], [("<update>", str(e))]
        valid, seen = [], set()
        for t in trees:
            if t.tree_id in self.forest or t.tree_id in seen:
                rejected.append((t.tree_id, "replayed tree id"))
            else:
                seen.add(t.tree_id)
                valid.append(t)
        return valid, rejected

    def aggregate(self, trees):
        """Step 7 · G_t = G_(t-1) plus the valid trees of this round (no parameter averaging).

        With a per-client cap only the newest trees of every client are kept. Returns the ids
        added and removed, which is what the clients need to update their copy of G_t.
        """
        previous = set(self.forest)
        for t in trees:
            self.forest[t.tree_id] = t
        if self.max_trees_per_client is not None:
            by_client = {}
            for tree_id, t in self.forest.items():
                by_client.setdefault(t.client_id, []).append(tree_id)
            for ids in by_client.values():
                for tree_id in ids[:-self.max_trees_per_client]:
                    del self.forest[tree_id]
        added = [t for t in trees if t.tree_id in self.forest]
        removed = [tree_id for tree_id in previous if tree_id not in self.forest]
        for tree_id in removed:
            del self.tree_size[tree_id]
        for t in added:
            self.tree_size[t.tree_id] = len(json.dumps(tree_to_message(t, self.schema), separators=(",", ":")))
        return added, removed

    @property
    def trees(self):
        return list(self.forest.values())

    @staticmethod
    def global_metrics(reports):
        """Global Server row: metrics of the summed confusion counts of all validation patients."""
        return metrics_from_counts(np.sum([r.counts for r in reports], axis=0))

    @staticmethod
    def prediction_change(reports):
        """Delta P_t over V = all clients' validation patients, from the clients' partial sums."""
        if any(r.pred_change_sum is None for r in reports):
            return None
        return sum(r.pred_change_sum for r in reports) / sum(r.n for r in reports)

    def save_forest(self, path, rnd):
        saved = {"schema_id": self.schema.schema_id, "schema_version": self.schema.version,
                 "features": list(self.schema.features), "classes": list(self.schema.classes), "round": rnd,
                 "trees": [tree_to_message(t, self.schema) for t in self.trees]}
        path.write_text(json.dumps(saved, separators=(",", ":")))
