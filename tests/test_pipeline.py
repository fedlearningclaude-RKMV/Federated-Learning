import numpy as np
import pandas as pd
import pytest

from fedrf_diff.metrics import MCC
from fedrf_diff.pipeline import main, run_dp_sweep, run_federation
from fedrf_diff.report import SERVER
from fedrf_diff.rf_kernel import forest_proba


def test_federation_rounds(small_cfg, capsys):
    res = run_federation(small_cfg, small_cfg["dp_epsilon"], small_cfg["max_rounds"])
    out = capsys.readouterr().out
    assert "Modified Pedersen ZKP authentication: pedersen2048 group, lambda = 2" in out
    assert "Authentication Successful (proof valid, update accepted)" in out
    assert res.rounds_run == 3
    n_trees = 4 * small_cfg["trees_per_client"]
    assert res.convergence["Global Trees"].tolist() == [n_trees, 2 * n_trees, 3 * n_trees]
    assert (res.zkp.Proof == "Valid").all() and (res.convergence["Trees Rejected"] == 0).all()
    assert res.convergence["Delta M"].isna().tolist() == [True, False, False]
    assert set(res.metrics.Client) == {c.name for c in res.clients} | {SERVER}
    # The Delta P aggregated from the clients' partial sums equals Delta P of the pooled global probabilities
    X_val = np.concatenate([c.X_val for c in res.clients])
    trees = res.server.trees
    now, before = forest_proba(trees, X_val), forest_proba(trees[:-n_trees], X_val)
    assert res.convergence["Delta P"].iloc[-1] == pytest.approx(0.5 * np.abs(now - before).sum(axis=1).mean())
    assert res.epsilon_per_patient == sum(4 * e for e in res.schedule)


def test_dp_sweep_reproduces_the_main_run(small_cfg):
    main_run = run_federation(small_cfg, (2.0, 3.0), 3, verbose=False)
    names = [c.name for c in main_run.clients]
    avg, last = run_dp_sweep(small_cfg, 3, names)
    row = avg[avg["Epsilon per round"] == "U[2, 3]"].iloc[0]
    expected = main_run.metrics.groupby("Client")["Accuracy"].mean()
    for p in names + [SERVER]:
        assert row[p] == pytest.approx(expected[p], rel=1e-12)
    assert avg.loc[avg["Epsilon per round"] == "No DP", "Epsilon spent per patient"].item() == 0.0


def test_combined_criterion_stops_the_run(small_cfg):
    cfg = dict(small_cfg, eps_metric=1.0, eps_pred=1.0, patience=2, max_rounds=10)
    res = run_federation(cfg, None, 10, with_zkp=False, verbose=False)
    assert res.converged and res.rounds_run == 3  # round 1 initialises, rounds 2 and 3 pass
    assert res.convergence["Stable Rounds"].tolist() == [0, 1, 2]


def test_main_writes_outputs(small_cfg, capsys):
    cfg = dict(small_cfg, max_rounds=2, dp_sweep=[2.0, None])
    main(cfg)
    out = capsys.readouterr().out
    assert "=== Metrics averaged over all 2 rounds ===" in out and "Accuracy vs DP budget" in out
    for name in ["validation_metrics", "zkp_times", "tree_transfer_times", "communication_computation_cost",
                 "convergence", "dp_budget_accuracy"]:
        assert not pd.read_csv(cfg["out_dir"] / f"{name}.csv").empty
    assert (cfg["out_dir"] / "global_forest.json").exists()
    metrics = pd.read_csv(cfg["out_dir"] / "validation_metrics.csv")
    assert np.isfinite(metrics[MCC]).all()
