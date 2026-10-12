"""Every hyperparameter of the experiment in one place."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

CONFIG = {
    "raw_csv": ROOT / "data/raw/heart_statlog_cleveland_hungary_final.csv",
    "clients_dir": ROOT / "data/clients",
    "out_dir": ROOT / "outputs/run",
    "seed": 42,
    "val_size": 0.20,  # validation share of every hospital (stratified)
    # Steps 1 and 5 · local Random Forests
    "initial_trees": 25,  # trees in F_k^(0)
    "supplementary_trees": 25,  # trees in H_k^(t), grown every round
    "rf_params": {"max_features": "sqrt", "min_samples_leaf": 2},  # + balanced class weights
    "local_budget": 50,  # B_k, CapOrPrune keeps at most this many trees in F_k^(t)
    # Steps 1 and 5 · local model selection before growing F_k^(0) / H_k^(t): stratified k-fold grid search
    # on the training patients (never shared, no DP cost). This is the bulk of the round time; grow or
    # shrink the grid, folds or cv_trees to change it, or set "local_tuning": None to skip tuning.
    "local_tuning": {
        "grid": {"max_features": ["sqrt", 0.5, None], "min_samples_leaf": [1, 2, 4, 8]},
        "folds": 5,
        "cv_trees": 60,
    },
    # Step 3 · structural validation limits and tree-level aggregation
    "max_payload_bytes": 16 * 1024 * 1024,
    "max_trees_per_update": 500,
    "max_nodes_per_tree": 20_000,
    "max_tree_depth": 64,
    "aggregation_policy": "consensus",  # A: "consensus", "client-balanced" or "newest" (server.select_trees)
    "global_budget": 100,  # B, trees in G^(t)
    # Step 6 · decentralized three-part convergence test (each client, own validation split)
    "max_rounds": 50,  # R_max
    "tau_loss": 0.05,  # relative change of the validation log-loss
    "tau_pred": 0.05,  # share of validation predictions that changed since the last round
    "tau_agree": 0.15,  # share of validation predictions where F_k^(t) and G^(t) disagree
    "patience": 3,  # P consecutive passing rounds
    # Laplace DP on the shared leaf vectors (privacy.py)
    "dp_epsilon": (2.0, 3.0),  # epsilon per round: a number, a (low, high) uniform range, or None for no DP
    "dp_clip": 1.0,  # C, L1 clip norm of every leaf vector
    # Modified Pedersen ZKP (zkp.py): number of generators / secret witness components
    "zkp_lambda": 2,
    # Accuracy vs DP budget experiment (no ZKP, same seed, as many rounds as the main run)
    "run_dp_sweep": True,
    "dp_sweep": [1.0, 2.0, 3.0, 4.0, 5.0, (2.0, 3.0), None],
    "make_figures": True,
}
