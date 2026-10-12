"""Federated server: authentication, structural validation, tree-level aggregation, stopping.

Step 3 · every client uploads its complete forest F_k^(t-1) as explicit node structures, so
there is nothing to decode: the server parses the JSON transport format and checks every tree
(schema id and version, class order, sender and round, payload size, unique dense node ids,
valid feature indices and finite thresholds, child references that form one rooted tree, leaf
outputs that are valid class distributions, node count and depth limits). A tree keeps the id
and the creator it was born with; a copy of a known tree must be identical to the original and
only the creator may introduce a new tree id. A malformed tree is rejected on its own, a
malformed upload as a whole.

The valid trees form the candidate pool U^(t) = union of T_k^(t-1), and the aggregation policy
A selects at most B of them: G^(t) = SelectTrees(U^(t), A, B). Tree parameters are never
averaged. Step 7 · the server never sees validation metrics: it only tallies the clients'
boolean convergence flags.
"""

import hashlib
import json
import math
from collections import deque
from dataclasses import dataclass

import numpy as np

from . import zkp
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


def check_update(payload, schema, limits, client_id, rnd, creators):
    """Check one uploaded forest; returns ([(tree, digest), ...] valid, [(tree_id, reason), ...] rejected)."""
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
            trees.append((validate_tree(msg, schema, limits, creators), tree_digest(msg)))
        except InvalidTree as e:
            tree_id = msg.get("tree_id") if isinstance(msg, dict) else None
            rejected.append((str(tree_id), str(e)))
    return trees, rejected


def tree_digest(msg):
    return hashlib.sha256(json.dumps(msg, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def validate_tree(msg, schema, limits, creators):
    """Check one TreeMessage (created by one of `creators`) and rebuild its node arrays."""
    if not isinstance(msg, dict) or set(msg) != TREE_KEYS:
        raise InvalidTree("tree fields do not match the protocol")
    if msg["schema_version"] != schema.version or msg["task"] != "classification":
        raise InvalidTree("schema version or task mismatch")
    if msg["classes"] != list(schema.classes):
        raise InvalidTree("class order mismatch")
    client_id = msg["client_id"]
    if client_id not in creators:
        raise InvalidTree("tree created by an unknown client")
    tree_id = msg["tree_id"]
    if not isinstance(tree_id, str) or not tree_id.startswith(f"{client_id}_"):
        raise InvalidTree("tree id must start with its creator's client id")
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


def save_forest(path, trees, schema, rnd):
    saved = {"schema_id": schema.schema_id, "schema_version": schema.version, "features": list(schema.features),
             "classes": list(schema.classes), "round": rnd, "trees": [tree_to_message(t, schema) for t in trees]}
    path.write_text(json.dumps(saved, separators=(",", ":")))


def load_forest(path, schema, limits):
    """Reload a saved forest, validating every tree again."""
    saved = json.loads(path.read_text())
    if saved["schema_id"] != schema.schema_id:
        raise InvalidUpdate("saved forest was built under another schema")
    return [validate_tree(msg, schema, limits, {msg["client_id"]}) for msg in saved["trees"]]


def select_trees(pool, policy, budget, votes=None):
    """SelectTrees(U, A, B) for the predeclared policy A.

    consensus        round-robin over the creating clients; inside each client's queue the trees
                     kept by the most client forests come first (a tree every client retained
                     after its own CapOrPrune is a tree that serves every site), then the newest
    client-balanced  round-robin over the creating clients, newest trees first
    newest           the B newest trees regardless of creator
    Round-robin over creators keeps any one client (however large) from filling the budget.
    """
    newest_first = sorted(pool, key=lambda t: t.tree_id, reverse=True)  # ids carry zero-padded round / index
    if policy == "newest":
        return newest_first[:budget]
    if policy == "consensus":
        votes = votes or {}
        newest_first = sorted(newest_first, key=lambda t: votes.get(t.tree_id, 0), reverse=True)  # stable sort
    elif policy != "client-balanced":
        raise ValueError(f"unknown aggregation policy {policy!r}")
    queues = {}
    for t in newest_first:
        queues.setdefault(t.client_id, []).append(t)
    selected, order = [], sorted(queues)
    while len(selected) < budget and any(queues.values()):
        for creator in order:
            if queues[creator] and len(selected) < budget:
                selected.append(queues[creator].pop(0))
    return selected



class FederatedServer:
    def __init__(self, schema, limits, zkp_lambda, policy, budget):
        self.schema, self.limits = schema, limits
        self.zkp_lambda = zkp_lambda
        self.policy, self.budget = policy, budget
        self.public_keys = {}
        self.known = {}  # tree_id -> digest of the first accepted copy
        self.global_forest = []  # G^(t)
        self.tree_size = {}  # tree_id -> bytes of its TreeMessage

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

    def receive(self, client_id, payload, rnd, creators):
        """Step 3 · structural validation of one uploaded forest F_k^(t-1)."""
        try:
            checked, rejected = check_update(payload, self.schema, self.limits, client_id, rnd, creators)
        except InvalidUpdate as e:
            return [], [("<update>", str(e))]
        valid, seen = [], set()
        for tree, digest in checked:
            known = self.known.get(tree.tree_id)
            if tree.tree_id in seen:
                rejected.append((tree.tree_id, "duplicate tree id in one forest"))
            elif known is None and tree.client_id != client_id:
                rejected.append((tree.tree_id, "new tree id claimed for another client"))
            elif known is not None and known != digest:
                rejected.append((tree.tree_id, "altered copy of a known tree"))
            else:
                seen.add(tree.tree_id)
                self.known.setdefault(tree.tree_id, digest)
                valid.append(tree)
        return valid, rejected

    def aggregate(self, valid_forests):
        """Step 3 · G^(t) = SelectTrees(U^(t), A, B) with U^(t) the union of the valid client trees."""
        pool, votes = {}, {}
        for trees in valid_forests:
            for t in trees:
                pool.setdefault(t.tree_id, t)
                votes[t.tree_id] = votes.get(t.tree_id, 0) + 1  # in how many client forests the tree sits
        self.global_forest = select_trees(list(pool.values()), self.policy, self.budget, votes)
        for t in self.global_forest:
            if t.tree_id not in self.tree_size:
                self.tree_size[t.tree_id] = len(json.dumps(tree_to_message(t, self.schema), separators=(",", ":")))
        return len(pool)

    @property
    def global_bytes(self):
        """Step 4 · size of the broadcast G^(t)."""
        return sum(self.tree_size[t.tree_id] for t in self.global_forest)

    @staticmethod
    def stopping_decision(flags, rnd, max_rounds):
        """Step 7 · the server sees only the boolean flags b_k^(t)."""
        if all(flags):
            return "all-client convergence"
        if rnd >= max_rounds:
            return "maximum rounds reached"
        return None
