# FedRF — Federated Random Forest for Heart Disease

A federated Random Forest trained across 4 hospitals that never share patient records.
Every client grows a local Random Forest, signs each update with a Schnorr zero-knowledge
proof, and the global server merges the verified trees into one global Random Forest.

```bash
pip install -r requirements.txt
python3 federated_rf.py
```

| File | Content |
|---|---|
| `federated_rf.py` | The full pipeline, including the hospital split, Schnorr ZKP and convergence rule |
| `fl_plots.py` | Publication-quality figures (600-dpi PNG + vector PDF) |
| `outputs/run/` | Per-round CSVs (validation metrics, ZKP, encoding/decoding, communication/computation cost, convergence) and the saved global Random Forest |
| `outputs/run/figures/` | Convergence, ZKP times, computation cost, uplink, downlink and validation metrics |

## Algorithm

1. **Dataset preparation**: assign each row to its hospital, drop duplicates, mark missing values
   (cholesterol = 0, resting BP = 0), one-hot encode chest pain type, resting ECG and ST slope.
2. **Split**: stratified 80 / 20 train / validation inside every hospital.
3. **Distribute**: one non-IID client per hospital; missing values are imputed with federated
   means (clients share only per-feature sums and counts).
4. **Local training**: every round each client adds `T_c = 10` trees to its Random Forest.
5. **Communication**: the new trees are serialized (encoding) and signed with a non-interactive
   Schnorr proof (RFC 3526 2048-bit MODP group) bound to the update's SHA-256 digest, the round and the client id.
6. **Server aggregation**: the server verifies each proof, rejects invalid updates, decodes the accepted
   trees and merges them into one scikit-learn Random Forest `F_global = F_1 ∪ … ∪ F_K`.
7. **Validation**: every client scores the global Random Forest on its own validation patients;
   the Global Server row scores all validation patients together.
8. **Combined convergence**: stop at round `r` when all three hold:
   `r ≥ R_min` (5), no validation-MCC improvement `> δ` (0.001) for `P` (3) rounds, and the share of validation
   predictions that flipped since the previous round `D_j ≤ τ` (2 %) for the latest `P` rounds.
   Final safeguard: if the best MCC is below `M_min` (0.60) the run is reported as stuck, not converged.
   The best-MCC global Random Forest is saved.
9. **Reporting**: each round prints the Schnorr ZKP table, authentication status, encoding/decoding times,
   communication and computation cost, and the seven validation metrics for every client and the Global Server.
   The run ends with averages over all rounds and min / max / mean analyses.

## Data

`data/raw/heart_statlog_cleveland_hungary_final.csv` (1190 rows) is five UCI sources stacked in a fixed order.
Statlog is Cleveland Clinic data, so it joins the Cleveland client. After removing 272 exact duplicates:

| Client | Rows | Heart disease | Cholesterol missing |
|---|---|---|---|
| Client 1 (Hungary) | 293 | 36.2% | 0% |
| Client 2 (Switzerland) | 123 | 93.5% | 100% |
| Client 3 (VA Long Beach) | 199 | 74.4% | 24.6% |
| Client 4 (Cleveland) | 303 | 45.9% | 0% |

`data/clients/*.csv` hold the per-hospital data written by the pipeline.
