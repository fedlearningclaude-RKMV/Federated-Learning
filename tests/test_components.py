import copy
import hashlib
import json

import numpy as np
import pytest
from sklearn.metrics import (accuracy_score, cohen_kappa_score, f1_score, matthews_corrcoef, precision_score,
                             recall_score)

from fedrf_diff import zkp
from fedrf_diff.alignment import check_local_data
from fedrf_diff.config import CONFIG
from fedrf_diff.convergence import CombinedConvergence, prediction_change_sum
from fedrf_diff.data import load_hospitals
from fedrf_diff.metrics import MCC, confusion_counts, metrics_from_counts
from fedrf_diff.privacy import PrivacyAccountant, epsilon_schedule, laplace_scale, privatize_leaves
from fedrf_diff.rf_kernel import forest_proba
from fedrf_diff.server import InvalidUpdate, Limits, check_update, load_forest


# ── Data and schema ─────────────────────────────────────────────────────────
def test_hospitals_are_horizontal_partitions_without_duplicates():
    df = load_hospitals(CONFIG["raw_csv"])
    sizes = df.hospital.value_counts().to_dict()
    assert sizes == {"Hungary": 293, "Switzerland": 123, "VA Long Beach": 199, "Cleveland": 303}
    assert not df.drop(columns="hospital").duplicated().any()


def test_clients_share_one_schema(federation):
    schema, clients, _ = federation
    assert [c.name for c in clients] == ["Client 1(Hungary)", "Client 2(Switzerland)",
                                         "Client 3(VA Long Beach)", "Client 4(Cleveland)"]
    for c in clients:
        assert c.X_train.shape[1] == c.X_val.shape[1] == schema.n_features
        assert not np.isnan(c.X_train).any()


def test_unknown_label_is_refused(federation):
    _, clients, _ = federation
    bad = clients[0].train_raw.copy()
    bad.iloc[0, bad.columns.get_loc("target")] = 2
    with pytest.raises(ValueError):
        check_local_data(bad)


# ── Modified Pedersen ZKP ───────────────────────────────────────────────────
def test_group_parameters_are_rederivable():
    p, q, i, j = zkp.derive_group(q_start=zkp.Q_COUNTER, p_start=zkp.P_COUNTER)
    assert (p, q, i, j) == (zkp.P, zkp.Q, zkp.Q_COUNTER, zkp.P_COUNTER)
    assert (zkp.P - 1) % zkp.Q == 0 and zkp.P.bit_length() == 2048 and zkp.Q.bit_length() == 256
    assert all(zkp.in_group(g) for g in zkp.generators(3))


@pytest.mark.parametrize("lam", [1, 2, 3])
def test_proof_accepts_only_the_bound_update(lam):
    key = zkp.keygen(lam)
    digest = hashlib.sha256(b"update").digest()
    context = zkp.update_context("schema", 3, "client_1")
    proof = zkp.prove(key, digest, context)
    assert zkp.verify(key.commitment, digest, context, proof, lam)
    assert not zkp.verify(key.commitment, hashlib.sha256(b"updatf").digest(), context, proof, lam)
    assert not zkp.verify(key.commitment, digest, zkp.update_context("schema", 4, "client_1"), proof, lam)
    assert not zkp.verify(key.commitment, digest, zkp.update_context("schema", 3, "client_2"), proof, lam)
    assert not zkp.verify(zkp.keygen(lam).commitment, digest, context, proof, lam)
    forged = zkp.Proof(proof.announcement, ((proof.responses[0] + 1) % zkp.Q, *proof.responses[1:]))
    assert not zkp.verify(key.commitment, digest, context, forged, lam)


def test_server_rejects_tampered_update(federation):
    schema, clients, server = federation
    c = clients[0]
    payload = c.serialize(c.export(c.train_round(1), 1), 1, None)
    proof = c.prove(payload, 1)
    assert server.verify(c.client_id, payload, proof, 1)
    tampered = payload.replace(b'"threshold":', b'"threshold":1', 1)
    assert not server.verify(c.client_id, tampered, proof, 1)
    assert not server.verify(clients[1].client_id, payload, proof, 1)


# ── Whole-tree export and server validation ─────────────────────────────────
def test_explicit_trees_reproduce_the_local_forest(federation):
    schema, clients, server = federation
    c = clients[3]
    forest = c.train_round(1)
    trees = c.export(forest, 1)
    np.testing.assert_allclose(forest_proba(trees, c.X_val), forest.predict_proba(c.X_val), atol=1e-12)
    received, rejected = server.receive(c.client_id, c.serialize(trees, 1, None), 1)
    assert rejected == []
    np.testing.assert_array_equal(forest_proba(received, c.X_val), forest_proba(trees, c.X_val))


def _valid_update(federation):
    schema, clients, server = federation
    c = clients[0]
    return json.loads(c.serialize(c.export(c.train_round(1), 1), 1, None)), c.client_id, schema


def _set_node(update, i, **fields):
    nodes = update["trees"][0]["nodes"]
    node = next(n for n in nodes if n["node_id"] == i)
    node.update(fields)


@pytest.mark.parametrize("corrupt, reason", [
    (lambda u: u["trees"][0].update(schema_version="features-v0"), "schema version"),
    (lambda u: u["trees"][0].update(classes=[1, 0]), "class order"),
    (lambda u: u["trees"][0].update(tree_id="client_9_r001_t000"), "tree id"),
    (lambda u: _set_node(u, 0, feature=99), "unknown feature"),
    (lambda u: _set_node(u, 0, feature=True), "unknown feature"),
    (lambda u: _set_node(u, 0, threshold="1.5"), "threshold"),
    (lambda u: _set_node(u, 0, left_child=0), "child references"),
    (lambda u: _set_node(u, 0, right_child=10 ** 6), "child references"),
    (lambda u: u["trees"][0]["nodes"][1].update(node_id=0), "duplicate node id"),
    (lambda u: u["trees"][0]["nodes"].append({"node_id": len(u["trees"][0]["nodes"]), "leaf": True,
                                              "class_distribution": [0.5, 0.5]}), "unreachable"),
    (lambda u: _set_node(u, 0, right_child=u["trees"][0]["nodes"][0]["left_child"]), "child references"),
])
def test_malformed_tree_is_rejected_alone(federation, limits, corrupt, reason):
    update, client_id, schema = _valid_update(federation)
    n_trees = len(update["trees"])
    corrupt(update)
    trees, rejected = check_update(json.dumps(update).encode(), schema, limits, client_id, 1)
    assert len(trees) == n_trees - 1
    assert len(rejected) == 1 and reason in rejected[0][1]


def test_invalid_leaf_and_shared_child_are_rejected(federation, limits):
    update, client_id, schema = _valid_update(federation)
    nodes = update["trees"][0]["nodes"]
    leaf = next(n for n in nodes if n.get("leaf"))
    leaf["class_distribution"] = [0.9, 0.3]
    # Second tree: an inner node points at a node that already has a parent
    nodes2 = update["trees"][1]["nodes"]
    inner = [n for n in nodes2 if not n.get("leaf") and n["node_id"] != 0]
    inner[0]["left_child"] = nodes2[0]["left_child"] if nodes2[0]["left_child"] != inner[0]["node_id"] else 0
    trees, rejected = check_update(json.dumps(update).encode(), schema, limits, client_id, 1)
    assert len(rejected) == 2
    assert "probability distribution" in rejected[0][1]


def test_tree_too_deep_is_rejected(federation):
    update, client_id, schema = _valid_update(federation)
    shallow = Limits(10 ** 8, 500, 20_000, max_depth=2)
    trees, rejected = check_update(json.dumps(update).encode(), schema, shallow, client_id, 1)
    assert rejected and all("depth" in r for _, r in rejected)


def test_malformed_update_is_rejected_whole(federation, limits):
    update, client_id, schema = _valid_update(federation)
    raw = json.dumps(update).encode()
    with_nan = copy.deepcopy(update)
    next(n for n in with_nan["trees"][0]["nodes"] if "threshold" in n)["threshold"] = float("nan")
    for payload, rnd, cid in [(raw, 2, client_id), (raw, 1, "client_2"), (b"{not json", 1, client_id),
                              (json.dumps(with_nan).encode(), 1, client_id)]:
        with pytest.raises(InvalidUpdate):
            check_update(payload, schema, limits, cid, rnd)
    wrong = copy.deepcopy(update)
    wrong["schema_id"] = "0" * 16
    with pytest.raises(InvalidUpdate):
        check_update(json.dumps(wrong).encode(), schema, limits, client_id, 1)
    with pytest.raises(InvalidUpdate):
        check_update(raw, schema, Limits(100, 500, 20_000, 64), client_id, 1)


def test_replayed_trees_and_per_client_cap(federation):
    schema, clients, server = federation
    c = clients[0]
    payload = c.serialize(c.export(c.train_round(1), 1), 1, None)
    trees, _ = server.receive(c.client_id, payload, 1)
    added, removed = server.aggregate(trees)
    assert len(added) == len(trees) and removed == []
    again, rejected = server.receive(c.client_id, payload, 1)
    assert again == [] and all(r == "replayed tree id" for _, r in rejected)
    server.max_trees_per_client = 3
    newer, _ = server.receive(c.client_id, c.serialize(c.export(c.train_round(2), 2), 2, None), 2)
    added, removed = server.aggregate(newer)
    assert len(server.forest) == 3 and set(removed) == {t.tree_id for t in trees}
    assert [t.tree_id for t in server.trees] == [t.tree_id for t in newer[-3:]]


def test_saved_forest_reloads(federation, tmp_path, limits):
    schema, clients, server = federation
    c = clients[2]
    trees, _ = server.receive(c.client_id, c.serialize(c.export(c.train_round(1), 1), 1, None), 1)
    server.aggregate(trees)
    server.save_forest(tmp_path / "forest.json", 1)
    loaded = load_forest(tmp_path / "forest.json", schema, limits)
    np.testing.assert_array_equal(forest_proba(loaded, c.X_val), forest_proba(trees, c.X_val))


# ── Laplace DP ──────────────────────────────────────────────────────────────
def test_dp_leaves_are_valid_distributions_and_structure_is_kept(federation):
    _, clients, _ = federation
    c = clients[1]
    trees = c.export(c.train_round(1), 1)
    noisy = c.privatize(trees, 2.0, 1)
    for t, n in zip(trees, noisy):
        np.testing.assert_array_equal(t.threshold, n.threshold)
        np.testing.assert_array_equal(t.left, n.left)
        leaves = n.value[n.is_leaf]
        assert (leaves >= 0).all() and np.allclose(leaves.sum(axis=1), 1)
        assert not np.allclose(leaves, t.value[t.is_leaf])
    assert c.privatize(trees, None, 1) is trees


def test_laplace_mechanism_uses_b_equal_2c_over_epsilon():
    eps, clip = 2.5, 1.0
    assert laplace_scale(eps, clip) == pytest.approx(0.8)
    value = np.random.default_rng(0).dirichlet([1, 1], 1000)
    is_leaf = np.arange(1000) % 3 != 0
    out = privatize_leaves(value, is_leaf, eps, clip, np.random.default_rng(7))
    noisy = np.clip(value[is_leaf] + np.random.default_rng(7).laplace(0, 0.8, (is_leaf.sum(), 2)), 0, None)
    totals = noisy.sum(axis=1, keepdims=True)
    expected = np.where(totals > 0, noisy / np.where(totals > 0, totals, 1), 0.5)
    np.testing.assert_allclose(out[is_leaf], expected)
    np.testing.assert_array_equal(out[~is_leaf], value[~is_leaf])


def test_epsilon_schedule_and_accountant():
    sched = epsilon_schedule((2.0, 3.0), 10, 42)
    assert all(2 <= e <= 3 for e in sched) and sched == epsilon_schedule((2.0, 3.0), 10, 42)
    assert epsilon_schedule((2.0, 3.0), 4, 42) == sched[:4]
    assert epsilon_schedule(3, 2, 42) == [3.0, 3.0] and epsilon_schedule(None, 2, 42) == [None, None]
    acc = PrivacyAccountant(trees_per_round=25)
    assert acc.spend(2.0) == 50.0 and acc.spend(None) == 0.0 and acc.total == 50.0


# ── Metrics and convergence ─────────────────────────────────────────────────
def test_metrics_match_scikit_learn():
    rng = np.random.default_rng(1)
    y, p = rng.integers(0, 2, 300), rng.integers(0, 2, 300)
    m = metrics_from_counts(confusion_counts(y, p))
    assert m["Accuracy"] == pytest.approx(accuracy_score(y, p))
    assert m["Precision"] == pytest.approx(precision_score(y, p))
    assert m["Recall"] == pytest.approx(recall_score(y, p))
    assert m["F1 Score"] == pytest.approx(f1_score(y, p))
    assert m[MCC] == pytest.approx(matthews_corrcoef(y, p))
    assert m["Cohen's Kappa"] == pytest.approx(cohen_kappa_score(y, p))
    # Global Server = summed counts, same as pooling every validation patient
    y2, p2 = rng.integers(0, 2, 50), rng.integers(0, 2, 50)
    pooled = metrics_from_counts(confusion_counts(y, p) + confusion_counts(y2, p2))
    assert pooled[MCC] == pytest.approx(matthews_corrcoef(np.r_[y, y2], np.r_[p, p2]))


def test_prediction_change_is_normalised():
    a = np.array([[1.0, 0.0], [0.5, 0.5]])
    b = np.array([[0.0, 1.0], [0.5, 0.5]])
    assert prediction_change_sum(a, b) == pytest.approx(1.0)  # one patient flipped fully, one unchanged
    assert prediction_change_sum(a, a) == 0.0


def test_combined_convergence_needs_both_tests_for_h_rounds():
    conv = CombinedConvergence(eps_metric=0.001, eps_pred=0.005, patience=3)
    first = conv.update(0.70, None)
    assert first.delta_metric is None and not first.converged
    seq = [(0.70, 0.001), (0.70, 0.001), (0.75, 0.001), (0.75, 0.001), (0.75, 0.01), (0.75, 0.001),
           (0.7505, 0.002), (0.7505, 0.0)]
    counters = [conv.update(m, dp).stable_rounds for m, dp in seq]
    assert counters == [1, 2, 0, 1, 0, 1, 2, 3]
    assert conv.stable_rounds >= conv.patience
