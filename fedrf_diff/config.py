"""Every hyperparameter of the experiment in one place."""

from pathlib import Path

from .metrics import MCC

ROOT = Path(__file__).resolve().parent.parent

CONFIG = {
    "raw_csv": ROOT / "data/raw/heart_statlog_cleveland_hungary_final.csv",
    "clients_dir": ROOT / "data/clients",
    "out_dir": ROOT / "outputs/run",
    "seed": 42,
    "val_size": 0.20,  # validation share of every hospital (stratified)
    # Step 3 · local Random Forest: every client trains m new trees per round
    "trees_per_client": 25,  # m
    "rf_params": {"max_features": "sqrt", "min_samples_leaf": 2},  # + balanced class weights
    # Step 6 · server-side validation limits for received trees
    "max_payload_bytes": 16 * 1024 * 1024,
    "max_trees_per_update": 500,
    "max_nodes_per_tree": 20_000,
    "max_tree_depth": 64,
    # Step 7 · global forest policy: keep every valid tree (None) or only the newest N trees per client
    "max_trees_per_client": None,
    # Section 4 · combined convergence criterion
    "max_rounds": 50,  # R
    "convergence_metric": MCC,  # M_t, fixed before training
    "eps_metric": 0.001,  # eps_M
    "eps_pred": 0.005,  # eps_P
    "patience": 3,  # H consecutive rounds
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
