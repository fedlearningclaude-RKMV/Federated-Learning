# FedRF-Diff: Federated Random Forest with Tree-Level Aggregation

A horizontal federated Random Forest for heart disease prediction across 4 hospitals. Clients exchange
**complete Random Forests as explicit decision trees** (feature, threshold, children and leaf class
distribution for every node). The server **never averages parameters** and **never computes the stopping
condition**: every client runs a strict three-part convergence test on its own validation patients and sends
only a boolean flag.

```bash
pip install -r requirements.txt
python -m fedrf_diff        # full run, CSVs, forests and journal figures in outputs/run/
python -m pytest            # unit and pipeline tests
```

## Algorithm

1. **Initialisation (t = 0)**: each client trains `F_k^(0)` (25 trees) on its private data. `G^(0)` is empty and `q_k = 0`.
2. **Client → server**: at the start of round t each client uploads its complete forest `F_k^(t-1)` with its
   schema metadata, signed with a Modified Pedersen ZKP. It sends no raw data and no latent encodings.
3. **Tree-level aggregation**: the server validates every tree (schema, classes, node references, splits, leaves,
   size and depth, provenance: a copy of a known tree must be identical, and only its creator may introduce a tree id).
   It pools `U^(t) = ∪ T_k^(t-1)` and selects `G^(t) = SelectTrees(U^(t), A, B = 100)`. Policy `A = consensus`
   is a round-robin over creating clients that prefers the trees kept by the most client forests, newest first.
   The `client-balanced` and `newest` policies are also available.
4. **Broadcast**: `G^(t)`, round t and the verified schema go to every client.
5. **Local refinement**: each client first selects its Random Forest hyperparameters with a stratified 5-fold
   grid search on its own training patients (`max_features` × `min_samples_leaf`, 12 settings, balanced accuracy).
   The cross-validation forests are never shared, so they cost no DP budget. This local model selection is the bulk of
   the round time, about 30 s per round on a 4-core machine; adjust `local_tuning` in `config.py` to change it.
   The client then grows a supplementary forest `H_k^(t)` (25 trees) and builds
   `F_k^(t) = CapOrPrune(TreePool(G^(t), H_k^(t)), B_k = 50)`. This keeps the trees with the highest balanced
   accuracy on the client's training patients that each tree did not train on: out-of-bag patients for its own
   trees, all training patients for foreign trees. Validation data is never used for pruning.
6. **Three-part convergence test** (each client, on its fixed validation split):
   relative log-loss change `≤ τ_loss = 0.05`, share of changed predictions `≤ τ_pred = 0.05`, and share of
   predictions where `F_k^(t)` and `G^(t)` disagree `≤ τ_agree = 0.15`. All pass → `q_k += 1`, otherwise `q_k = 0`.
   The client sends only `b_k = [q_k ≥ P = 3]`.
7. **Stopping**: the server tallies the flags. If every client reports True it stops with "all-client convergence";
   at `t = R_max = 50` it stops with "maximum rounds reached". It returns `G^(t)` and every `F_k^(t)`.

Each newly grown tree gets Laplace DP on its leaf vectors (`b = 2C/ε`, `C = 1`, `ε ~ U[2, 3]` per round).
All parameters are in `fedrf_diff/config.py`.

The per-round validation tables in the console are written by the **evaluation harness** for the paper.
In the protocol itself, no validation metric reaches the server.

## Data

`data/raw/heart_statlog_cleveland_hungary_final.csv` (1190 rows) stacks five UCI sources in a fixed order:
Hungary 0–293, Switzerland 294–416, VA Long Beach 417–616, Statlog 617–886, Cleveland 887–1189.
Statlog is Cleveland Clinic data, so it joins the Cleveland client, and its copies drop out with the 272 exact duplicates.

| Client | Patients | Train / validation | Heart disease | Cholesterol missing |
|---|---|---|---|---|
| Client 1(Hungary) | 293 | 234 / 59 | 36.2% | 0% |
| Client 2(Switzerland) | 123 | 98 / 25 | 93.5% | 100% |
| Client 3(VA Long Beach) | 199 | 159 / 40 | 74.4% | 24.6% |
| Client 4(Cleveland) | 303 | 242 / 61 | 45.9% | 0% |

## Outputs (`outputs/run/`)

`validation_metrics.csv` (G^(t) per client and Global Server), `local_tuning.csv` (selected hyperparameters, CV score and tuning time), `local_convergence.csv` (the three tests, q_k, b_k
and F_k^(t) metrics), `rounds.csv` (|U|, |G|, flags, decision, DP budget), `zkp_times.csv`, `tree_transfer_times.csv`,
`communication_computation_cost.csv`, `dp_budget_accuracy.csv`, `console_output.txt`, and `forests/`
(the returned G and every F_k as explicit trees). The figures are `convergence`, `zkp_times`, `computation_cost`,
`communication_cost` and `accuracy_vs_dp_budget`.

## Privacy scope

Each released leaf vector is ε-DP (pure DP). Leaves partition a tree's training sample, so a patient costs ε per
tree, `m·ε` for the m trees grown in a round, and these add up over rounds (re-sent trees are free post-processing) (the per-patient bound is printed as "per patient").
Split features and thresholds are learned on the raw records and are **not** privatised, so whole-tree sharing
is not an end-to-end DP release. The noise comes from a seeded numpy generator for reproducibility, not from a
hardened DP sampler. The ZKP authenticates updates; it does not hide the trees from the server.
