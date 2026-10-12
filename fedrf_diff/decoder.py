"""Step 6 · server-side decoding and validation of whole-tree updates.

The server parses an authenticated update and checks the schema id and version, the class
order, the sender and round, the payload size and every tree: unique dense node ids, valid
feature indices and finite thresholds, child references that form one rooted tree (no cycles,
no shared or orphaned nodes), leaf outputs that are valid class distributions, and the
allowed node count and depth. A malformed tree is rejected on its own; a malformed update
is rejected as a whole.
"""

import json
import math
from collections import deque
from dataclasses import dataclass

import numpy as np

from .rf_kernel import LEAF, Tree

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


def decode_update(payload, schema, limits, client_id, rnd):
    """Parse one update; returns (valid trees, [(tree_id, reason), ...] for rejected trees)."""
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
