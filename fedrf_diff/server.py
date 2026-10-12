"""Federated server: authentication, tree validation and the global forest (Steps 6-9)."""

import hashlib
import json

import numpy as np

from . import zkp
from .decoder import InvalidUpdate, decode_update
from .metrics import metrics_from_counts
from .rf_kernel import tree_to_message


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

    def decode(self, client_id, payload, rnd):
        """Step 6 · valid trees of one authenticated update and the rejected ones with reasons."""
        try:
            trees, rejected = decode_update(payload, self.schema, self.limits, client_id, rnd)
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
