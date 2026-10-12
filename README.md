# Federated Learning — Heart Disease

Federated heart disease prediction across 4 hospitals.

## Data

`data/raw/heart_statlog_cleveland_hungary_final.csv` (1190 rows, 11 features + binary `target`)
is made of five UCI sources stacked in a fixed order. `scripts/partition_by_hospital.py`
assigns each row to its hospital by position, merges Statlog into Cleveland (it is
Cleveland Clinic data), drops the 272 exact duplicates, and writes one CSV per client:

| Client | Rows | Heart disease | Cholesterol = 0 (missing) |
|---|---|---|---|
| `data/clients/hungary.csv` | 293 | 36.2% | 0% |
| `data/clients/switzerland.csv` | 123 | 93.5% | 100% |
| `data/clients/va_long_beach.csv` | 199 | 74.4% | 24.6% |
| `data/clients/cleveland.csv` | 303 | 45.9% | 0% |

The clients are strongly non-IID: their sizes, positive rates and missing values differ.

Regenerate with:

```bash
python3 scripts/partition_by_hospital.py
```
