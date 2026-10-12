import json

import numpy as np
import pandas as pd
import pytest

from fedrf_diff.metrics import MCC
from fedrf_diff.pipeline import CONVERGED, MAX_ROUNDS, main, run_dp_sweep, run_federation
from fedrf_diff.report import SERVER
from fedrf_diff.rf_kernel import forest_proba


def test_federation_rounds(small_cfg, capsys):
    res = run_federation(small_cfg, small_cfg["dp_epsilon"], small_cfg["max_rounds"])
    out = capsys.readouterr().out
    assert "Modified Pedersen ZKP authentication: pedersen2048 group, lambda = 2" in out
    assert "Authentication Successful (proof valid, update accepted)" in out
    assert "Round 0 (initialisation)" in out and "Local three-part convergence test" in out
    assert res.rounds_run == 3 and res.status in (CONVERGED, MAX_ROUNDS)
    assert (res.zkp.Proof == "Valid").all() and (res.rounds["Trees Rejected"] == 0).all()
    # Round 1 uploads F_k^(0); later rounds upload the capped F_k^(t-1)
    assert res.zkp[res.zkp.Round == 1]["Trees"].tolist() == [4] * 4
    assert (res.zkp[res.zkp.Round > 1]["Trees"] <= small_cfg["local_budget"]).all()
    assert (res.rounds["Global Trees |G|"] <= small_cfg["global_budget"]).all()
    assert all(len(c.forest) <= small_cfg["local_budget"] for c in res.clients)
    assert set(res.metrics.Client) == {c.name for c in res.clients} | {SERVER}
    # Global Server row = G^(t) scored on all validation patients pooled
    X = np.concatenate([c.X_val for c in res.clients])
    y = np.concatenate([c.y_val for c in res.clients])
    acc = np.mean(forest_proba(res.server.global_forest, X).argmax(axis=1) == y)
    last = res.metrics[(res.metrics.Round == 3) & (res.metrics.Client == SERVER)]
    assert last["Accuracy"].item() == pytest.approx(acc)
    # Per patient: F_k^(0) plus one supplementary forest per round
    assert res.epsilon_per_patient == pytest.approx(4 * sum(res.schedule))


def test_uploads_carry_trees_and_schema_only(small_cfg):
    res = run_federation(small_cfg, None, 1, with_zkp=False, verbose=False)
    upload = json.loads(res.clients[0].serialize(2))
    assert set(upload) == {"schema_id", "schema_version", "client_id", "round", "privacy", "trees"}
    assert all(set(t) == {"schema_version", "task", "classes", "client_id", "tree_id", "nodes"}
               for t in upload["trees"])


def test_dp_sweep_reproduces_the_main_run(small_cfg):
    main_run = run_federation(small_cfg, (2.0, 3.0), 3, verbose=False, stop_early=False)
    names = [c.name for c in main_run.clients]
    avg, last = run_dp_sweep(small_cfg, 3, names)
    row = avg[avg["Epsilon per round"] == "U[2, 3]"].iloc[0]
    expected = main_run.metrics.groupby("Client")["Accuracy"].mean()
    for p in names + [SERVER]:
        assert row[p] == pytest.approx(expected[p], rel=1e-12)
    assert avg.loc[avg["Epsilon per round"] == "No DP", "Epsilon spent per patient"].item() == 0.0


def test_all_client_convergence_stops_the_run(small_cfg):
    cfg = dict(small_cfg, tau_loss=10.0, tau_pred=1.0, tau_agree=1.0, patience=2, max_rounds=10)
    res = run_federation(cfg, None, 10, with_zkp=False, verbose=False)
    assert res.status == CONVERGED and res.rounds_run == 2
    assert res.local.pivot(index="Round", columns="Client", values="q_k").values.tolist() == [[1] * 4, [2] * 4]
    strict = dict(small_cfg, tau_loss=0.0, tau_pred=0.0, tau_agree=0.0)
    res = run_federation(strict, None, 3, with_zkp=False, verbose=False)
    assert res.status == MAX_ROUNDS and res.rounds_run == 3


def test_main_writes_outputs(small_cfg, capsys):
    cfg = dict(small_cfg, max_rounds=2, dp_sweep=[2.0, None])
    main(cfg)
    out = capsys.readouterr().out
    assert "=== Global model G^(t): metrics averaged over all" in out and "Accuracy of G^(t) vs DP budget" in out
    for name in ["validation_metrics", "local_convergence", "zkp_times", "tree_transfer_times",
                 "communication_computation_cost", "rounds", "dp_budget_accuracy"]:
        assert not pd.read_csv(cfg["out_dir"] / f"{name}.csv").empty
    forests = cfg["out_dir"] / "forests"
    assert (forests / "global_forest.json").exists() and len(list(forests.glob("client_*_forest.json"))) == 4
    assert np.isfinite(pd.read_csv(cfg["out_dir"] / "validation_metrics.csv")[MCC]).all()
