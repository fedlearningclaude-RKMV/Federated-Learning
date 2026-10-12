# Federated Learning — Heart Disease

A federated Random Forest trained across 4 hospitals that never share patient records.
Clients train local forests and send only tree structures. The server merges them into
a global forest and stops training when the validation MCC stops improving.

```bash
pip install -r requirements.txt
python3 federated_rf.py
```

| File | Content |
|---|---|
| `federated_rf.py` | The full pipeline, one section per algorithm step (1–10), including the hospital split |
| `fl_plots.py` | Publication-quality figures (300-dpi PNG + vector PDF) |
| `results/` | Per-round metrics and costs, averages, test metrics, saved global model |
| `results/figures/` | Convergence, test metrics, uplink, downlink and computation cost figures |

## Algorithm

1. **Dataset preparation**: assign each row to its hospital, drop duplicates, mark missing values
   (cholesterol = 0, resting BP = 0), one-hot encode chest pain type, resting ECG and ST slope.
2. **Split**: stratified 70 / 10 / 20 train / validation / test inside every hospital.
3. **Distribute**: one non-IID client per hospital; missing values are imputed with federated
   means (clients share only per-feature sums and counts).
4. **Local training**: each client trains a Random Forest with `T_c = 50` trees.
5. **Communication**: clients upload serialized trees.
6. **Aggregation**: `F_global = F_1 ∪ … ∪ F_K`. Optional tree selection keeps the `K × T_c` trees with the
   highest federated validation MCC from the new trees plus the previous global forest.
7. **Validation**: clients evaluate the global forest locally and return only confusion counts; the
   server computes the MCC.
8. **Convergence**: stop when MCC improves by less than δ = 0.001 for 3 rounds, or at round 30;
   keep the best round's model.
9. **Evaluation**: every round prints Accuracy, Precision, Recall, Specificity, F1, MCC and Cohen's Kappa
   for each hospital, the client average and the global server (pooled test set), together with that round's
   uplink / downlink bytes and local training, validation and aggregation time. The test set is only reported
   per round; model selection uses the validation MCC. At the end the best model's test metrics, the averages
   over all rounds and the total costs are printed.
10. **Deployment**: the global model is saved with joblib and used to score an example patient.

## Data

`data/raw/heart_statlog_cleveland_hungary_final.csv` (1190 rows) is five UCI sources stacked in a fixed order.
Statlog is Cleveland Clinic data, so it joins the Cleveland client. After removing 272 exact duplicates:

| Client | Rows | Heart disease | Cholesterol missing |
|---|---|---|---|
| Hungary | 293 | 36.2% | 0% |
| Switzerland | 123 | 93.5% | 100% |
| VA Long Beach | 199 | 74.4% | 24.6% |
| Cleveland | 303 | 45.9% | 0% |

`data/clients/*.csv` hold the per-hospital data written by the pipeline.
