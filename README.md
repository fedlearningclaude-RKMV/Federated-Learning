# FedRF-Diff: Federated Random Forest with Whole-Tree Sharing

A horizontal federated Random Forest for heart disease prediction across 4 hospitals. Each
hospital trains a local Random Forest and sends the server its **complete, explicit decision trees**:
feature, threshold, child references and leaf class distribution for every node. It shares no raw
records, embeddings, latent codes or split histograms. The server validates the trees and concatenates
the valid ones into the global forest. Training stops when a **combined convergence criterion** holds.

```bash
pip install -r requirements.txt
python -m fedrf_diff        # full run, CSVs and journal figures in outputs/run/
python -m pytest            # unit and pipeline tests
```

## Package

| Module | Pipeline step |
|---|---|
| `fedrf_diff/data.py` | Hospital partition of the raw CSV, duplicates, missing values, stratified 80/20 split |
| `fedrf_diff/alignment.py` | Step 1 schema agreement (feature order, classes, missing-value and split rules, schema id); Step 2 federated imputation and one-hot encoding |
| `fedrf_diff/rf_kernel.py` | Step 3 local Random Forest; Step 4 explicit tree export (TreeMessage) and forest inference |
| `fedrf_diff/privacy.py` | Laplace DP on every leaf vector, epsilon schedule, privacy accountant |
| `fedrf_diff/zkp.py` | Modified Pedersen ZKP (pedersen2048 group, lambda = 2) binding each update to its sender, round and schema |
| `fedrf_diff/client.py` | Hospital client: train, export, privatize, serialize, prove, score the global forest |
| `fedrf_diff/server.py` | Authentication, Step 6 validation of every received tree, global forest update (concatenation, optional per-client cap), global metrics |
| `fedrf_diff/convergence.py` | Section 4 combined convergence criterion |
| `fedrf_diff/metrics.py` | Accuracy, precision, recall, specificity, F1, MCC, Cohen's kappa from confusion counts |
| `fedrf_diff/pipeline.py` | The round loop and the accuracy-vs-DP-budget experiment |
| `fedrf_diff/report.py`, `plotting.py` | Console tables; 600-dpi PNG + vector PDF figures |
| `fedrf_diff/config.py` | Every hyperparameter |

## Algorithm (one round t)

1. **Local training**: each client trains `m = 25` new Random Forest trees on its own patients.
2. **Explicit export + DP**: each tree becomes explicit nodes. Every leaf class-distribution vector `v`
   (L1-clipped to `C = 1`) is released as `v + Lap(2C/ε)`, clamped and renormalised, with `ε ~ U[2, 3]` per round.
3. **Transfer**: the update (trees + schema id, class order, client id, tree ids) is serialized and signed with
   a Modified Pedersen proof. This proves knowledge of the opening of the client's registered commitment
   `Y = g1^x · g2^r`, with a Fiat–Shamir challenge bound to SHA-256(update), the schema, the round and the client id.
4. **Server validation**: the server verifies the proof, then checks the schema, class order, sender and round,
   payload size, node ids, feature indices, finite thresholds, one rooted tree (no cycles or orphans), valid leaf
   distributions, depth/size limits and replayed tree ids. Invalid trees are rejected individually.
5. **Global forest**: `G_t = G_(t-1) ∪ {valid trees}`, with no parameter averaging. Prediction averages the class
   probabilities of all trees and takes the argmax.
6. **Evaluation**: every client scores `G_t` on its own validation patients and returns only its confusion counts
   and its share of `ΔP_t`. The Global Server row uses the summed counts.
7. **Combined convergence**: `ΔM_t = |M_t − M_(t−1)|` (M = MCC) and
   `ΔP_t = (1/|V|) Σ_i ||P_t(i,·) − P_(t−1)(i,·)||₁ / 2`. A round passes when `ΔM_t ≤ 0.001` **and**
   `ΔP_t ≤ 0.005`. The run stops after `H = 3` consecutive passing rounds or at `R = 50`.

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

`validation_metrics.csv`, `zkp_times.csv`, `tree_transfer_times.csv` (serialization and server tree-validation times, accepted/rejected trees),
`communication_computation_cost.csv`, `convergence.csv` (ΔM, ΔP, stable rounds, DP budget per round),
`dp_budget_accuracy.csv`, and `global_forest.json` (the final global forest as explicit trees).
The figures are `convergence`, `zkp_times`, `computation_cost`, `communication_cost` and `accuracy_vs_dp_budget`.

## Privacy scope

Each released leaf vector is ε-DP (pure DP). Leaves partition a tree's training sample, so a patient costs ε per
tree, `m·ε` per round, and these add up over rounds (the per-patient bound is printed as "per patient").
Split features and thresholds are learned on the raw records and are **not** privatised, so whole-tree sharing
is not an end-to-end DP release. The noise comes from a seeded numpy generator for reproducibility, not from a
hardened DP sampler. The ZKP authenticates updates; it does not hide the trees from the server.
