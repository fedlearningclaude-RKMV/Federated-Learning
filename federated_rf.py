"""
Federated Random Forest for heart disease prediction across 4 hospitals.

The pipeline follows the 10-step algorithm:

    1  Dataset preparation      (hospital partitioning + preprocessing)
    2  Train / validation / test split
    3  Distribute training data to clients (non-IID, one client per hospital)
    4  Client-side local training (one Random Forest per client)
    5  Communication to server  (serialized trees are sent, never raw data)
    6  Server-side aggregation  (union of client trees + optional tree selection)
    7  Validation during training (federated MCC on the validation split)
    8  Iterative federated training with convergence criteria
    9  Final model evaluation on the unseen test set
   10  Final model and deployment

Run:   python3 federated_rf.py
"""

import json
import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

import fl_plots

ROOT = Path(__file__).resolve().parent
console = Console()

# ════════════════════════════════════════════════════════════════════════════
#  Configuration
# ════════════════════════════════════════════════════════════════════════════
CONFIG = {
    "raw_csv": ROOT / "data/raw/heart_statlog_cleveland_hungary_final.csv",
    "clients_dir": ROOT / "data/clients",
    "results_dir": ROOT / "results",
    "seed": 42,
    # Step 2: fractions of every hospital's data
    "test_size": 0.20,  # alpha: kept unseen during training
    "val_size": 0.10,  # used by the server for MCC-based convergence
    # Step 4: local Random Forest
    "trees_per_client": 50,  # T_c
    "rf_params": {
        "max_features": "sqrt",
        "min_samples_leaf": 2,
        "class_weight": "balanced",
        "n_jobs": 1,
    },
    # Step 6: optional tree selection
    "tree_selection": True,
    # Step 8: convergence criteria
    "max_rounds": 30,  # R_max
    "patience": 3,  # P
    "min_delta": 0.001,  # delta (minimum MCC improvement)
}

TARGET = "target"
NUMERIC = ["age", "resting bp s", "cholesterol", "max heart rate", "oldpeak"]
BINARY = ["sex", "fasting blood sugar", "exercise angina"]
CATEGORICAL = {"chest pain type": [1, 2, 3, 4], "resting ecg": [0, 1, 2], "ST slope": [1, 2, 3]}

# ════════════════════════════════════════════════════════════════════════════
#  Client division: the raw CSV is five UCI sources stacked in a fixed order.
#  Statlog Heart is a re-release of the Cleveland Clinic data, so it is merged
#  into the Cleveland client, giving 4 hospitals.
# ════════════════════════════════════════════════════════════════════════════
SOURCE_BLOCKS = [  # (hospital, first row, end row) 0-based, end exclusive
    ("Hungary", 0, 294),
    ("Switzerland", 294, 417),
    ("VA Long Beach", 417, 617),
    ("Cleveland", 617, 920),
    ("Cleveland", 920, 1190),  # Statlog Heart (Cleveland Clinic)
]
HOSPITALS = ["Hungary", "Switzerland", "VA Long Beach", "Cleveland"]
GLOBAL = "Global (server)"


def divide_into_hospitals(path):
    """Label every row with its hospital and drop exact duplicates."""
    df = pd.read_csv(path)
    if len(df) != SOURCE_BLOCKS[-1][2]:
        raise ValueError(f"expected {SOURCE_BLOCKS[-1][2]} rows, got {len(df)}")
    df["hospital"] = ""
    for hospital, start, end in SOURCE_BLOCKS:
        df.loc[start : end - 1, "hospital"] = hospital
    n_raw = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    return df, n_raw - len(df)


def save_client_csvs(df, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    for hospital, part in df.groupby("hospital", sort=False):
        name = hospital.lower().replace(" ", "_")
        part.drop(columns="hospital").to_csv(out_dir / f"{name}.csv", index=False)


# ════════════════════════════════════════════════════════════════════════════
#  Metrics
# ════════════════════════════════════════════════════════════════════════════
METRIC_NAMES = ["Accuracy", "Precision", "Recall", "Specificity", "F1 Score", "MCC", "Kappa"]


def compute_metrics(y_true, y_pred, y_prob=None):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    m = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "Specificity": tn / (tn + fp) if tn + fp else 0.0,
        "F1 Score": f1_score(y_true, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
        "Kappa": cohen_kappa_score(y_true, y_pred),
        "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn),
    }
    if y_prob is not None and len(np.unique(y_true)) == 2:
        m["AUC"] = roc_auc_score(y_true, y_prob)
    return m


def mcc_from_counts(tp, tn, fp, fn):
    """MCC from (aggregated) confusion counts, so clients never share labels."""
    denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return np.where(denom > 0, (tp * tn - fp * fn) / np.where(denom > 0, denom, 1), 0.0)


def confusion_counts(y_true, y_pred):
    """Return [tp, tn, fp, fn] along the last axis; y_pred may be (n_trees, n)."""
    y_true = np.asarray(y_true, dtype=bool)
    y_pred = np.asarray(y_pred, dtype=bool)
    return np.stack(
        [(y_pred & y_true).sum(-1), (~y_pred & ~y_true).sum(-1),
         (y_pred & ~y_true).sum(-1), (~y_pred & y_true).sum(-1)], axis=-1
    ).astype(np.int32)


def nbytes(obj):
    return len(pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL))


# ════════════════════════════════════════════════════════════════════════════
#  Global model: an ensemble of decision trees F_global = F_1 ∪ … ∪ F_K
# ════════════════════════════════════════════════════════════════════════════
class GlobalForest:
    def __init__(self, trees, feature_names):
        self.trees = list(trees)
        self.feature_names = list(feature_names)

    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float32)
        return np.mean([t.predict_proba(X)[:, 1] for t in self.trees], axis=0)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)

    @property
    def n_nodes(self):
        return sum(t.tree_.node_count for t in self.trees)


# ════════════════════════════════════════════════════════════════════════════
#  Cost bookkeeping
# ════════════════════════════════════════════════════════════════════════════
@dataclass
class CostTracker:
    rows: list = field(default_factory=list)

    def log(self, rnd, party, direction, what, n_bytes):
        self.rows.append({"round": rnd, "party": party, "direction": direction,
                          "payload": what, "bytes": int(n_bytes)})

    def frame(self):
        return pd.DataFrame(self.rows)


@dataclass
class TimeTracker:
    rows: list = field(default_factory=list)

    def log(self, rnd, party, task, seconds):
        self.rows.append({"round": rnd, "party": party, "task": task, "seconds": seconds})

    def frame(self):
        return pd.DataFrame(self.rows)


COMM = CostTracker()
COMP = TimeTracker()


# ════════════════════════════════════════════════════════════════════════════
#  Federated client (hospital)
# ════════════════════════════════════════════════════════════════════════════
class HospitalClient:
    def __init__(self, cid, name, train, val, test):
        self.cid, self.name = cid, name
        self.train_raw, self.val_raw, self.test_raw = train, val, test

    # Step 3 helper: share only sums and counts for federated imputation
    def missing_value_stats(self):
        x = self.train_raw[NUMERIC]
        return {"sum": x.sum(skipna=True).to_dict(), "count": x.notna().sum().to_dict()}

    def apply_preprocessing(self, fill_values):
        def prep(df):
            X = encode_features(df.drop(columns=[TARGET]), fill_values)
            return X.to_numpy(np.float32), df[TARGET].to_numpy(int)
        self.X_train, self.y_train = prep(self.train_raw)
        self.X_val, self.y_val = prep(self.val_raw)
        self.X_test, self.y_test = prep(self.test_raw)
        self.feature_names = list(encode_features(self.train_raw.head(1).drop(columns=[TARGET]), fill_values).columns)

    # Step 4: local training
    def train_local_forest(self, n_trees, seed):
        rf = RandomForestClassifier(n_estimators=n_trees, random_state=seed, **CONFIG["rf_params"])
        rf.fit(self.X_train, self.y_train)
        return rf.estimators_

    # Step 7 helpers: the server sends trees, clients return confusion counts only
    def per_tree_counts(self, trees):
        preds = np.stack([t.predict(self.X_val) for t in trees])
        return confusion_counts(self.y_val, preds)

    def ensemble_counts(self, model):
        return confusion_counts(self.y_val, model.predict(self.X_val))


# ════════════════════════════════════════════════════════════════════════════
#  Federated server
# ════════════════════════════════════════════════════════════════════════════
class FederatedServer:
    def __init__(self, clients, feature_names):
        self.clients = clients
        self.feature_names = feature_names
        self.global_trees = []  # trees of the current global forest
        self.history = []

    def aggregate(self, local_forests):
        """F_global = F_1 ∪ … ∪ F_K, plus the previous global forest as candidates."""
        new_trees = [t for forest in local_forests.values() for t in forest]
        return self.global_trees + new_trees

    def select_trees(self, pool, tree_mcc, keep):
        """Keep the `keep` trees with the highest federated validation MCC."""
        order = np.argsort(-tree_mcc, kind="stable")[:keep]
        return [pool[i] for i in sorted(order)]


# ════════════════════════════════════════════════════════════════════════════
#  Preprocessing (Step 1)
# ════════════════════════════════════════════════════════════════════════════
def mark_missing(df):
    """Zeros in cholesterol / resting BP and ST slope 0 are missing values."""
    df = df.copy()
    df.loc[df["cholesterol"] == 0, "cholesterol"] = np.nan
    df.loc[df["resting bp s"] == 0, "resting bp s"] = np.nan
    df.loc[df["ST slope"] == 0, "ST slope"] = np.nan
    return df


def encode_features(df, fill_values):
    """Impute numeric features, one-hot encode categorical ones.

    Feature scaling is not needed: tree splits are invariant to monotone
    rescaling. Outliers are kept for the same reason.
    """
    out = df[NUMERIC].fillna(fill_values).copy()
    for col in BINARY:
        out[col] = df[col].astype(int)
    for col, cats in CATEGORICAL.items():
        for c in cats:
            out[f"{col}={c}"] = (df[col] == c).astype(int)
    return out


# ════════════════════════════════════════════════════════════════════════════
#  Pretty printing helpers
# ════════════════════════════════════════════════════════════════════════════
def step(n, title):
    console.print()
    console.print(Rule(f"[bold white] STEP {n} · {title} ", style="bright_blue"))


def fmt_bytes(b):
    for unit in ["B", "KB", "MB", "GB"]:
        if b < 1024 or unit == "GB":
            return f"{b:,.1f} {unit}" if unit != "B" else f"{b:,.0f} B"
        b /= 1024


AVERAGE = "Average (clients)"
PARTY_STYLE = {AVERAGE: "bold yellow", GLOBAL: "bold green"}
COST_COLS = ["Uplink", "Downlink", "Train (s)", "Val (s)", "Agg (s)", "Total (s)"]


def metrics_table(title, results):
    t = Table(title=title, box=box.ROUNDED, header_style="bold cyan", title_style="bold")
    t.add_column("Party", style="bold")
    for m in METRIC_NAMES:
        t.add_column(m, justify="right")
    for name, r in results.items():
        t.add_row(name, *[f"{r[m]:.4f}" for m in METRIC_NAMES], style=PARTY_STYLE.get(name))
    return t


def party_table(title, rows):
    """Metrics + communication + computation cost, one row per party."""
    t = Table(title=title, title_justify="left", box=box.ROUNDED, header_style="bold cyan", title_style="bold")
    t.add_column("Party", style="bold")
    for m in METRIC_NAMES:
        t.add_column(m.replace(" Score", ""), justify="right")
    for col in COST_COLS:
        t.add_column(col, justify="right", style="magenta" if col in ("Uplink", "Downlink") else "blue")
    for name, r in rows.items():
        t.add_row(name, *[f"{r[m]:.4f}" for m in METRIC_NAMES], fmt_bytes(r["Uplink"]), fmt_bytes(r["Downlink"]),
                  *[f"{r[c]:.4f}" if r[c] else "–" for c in COST_COLS[2:]], style=PARTY_STYLE.get(name))
    return t


def evaluate_parties(model, clients):
    """Test metrics per client, their average, and on the pooled test set (global server)."""
    results, preds = {}, {}
    for c in clients:
        prob = model.predict_proba(c.X_test)
        preds[c.name] = (c.y_test, prob)
        results[c.name] = compute_metrics(c.y_test, (prob >= 0.5).astype(int), prob)
    results[AVERAGE] = {m: float(np.mean([results[c.name][m] for c in clients])) for m in METRIC_NAMES}
    y = np.concatenate([preds[c.name][0] for c in clients])
    p = np.concatenate([preds[c.name][1] for c in clients])
    results[GLOBAL] = compute_metrics(y, (p >= 0.5).astype(int), p)
    return results, preds


def cost_rows(rnd, clients, comm, comp):
    """Communication (bytes) and computation (seconds) of one round per client, their average and the server."""
    comm, comp = comm[comm["round"] == rnd], comp[comp["round"] == rnd]

    def secs(party, task):
        return comp[(comp.party == party) & (comp.task == task)].seconds.sum()

    rows = {}
    for c in clients:
        cc = comm[comm.party == c.name]
        r = {"Uplink": int(cc[cc.direction == "uplink"].bytes.sum()),
             "Downlink": int(cc[cc.direction == "downlink"].bytes.sum()),
             "Train (s)": secs(c.name, "local training"), "Val (s)": secs(c.name, "validation"), "Agg (s)": 0.0}
        r["Total (s)"] = r["Train (s)"] + r["Val (s)"]
        rows[c.name] = r
    rows[AVERAGE] = {k: float(np.mean([rows[c.name][k] for c in clients])) for k in COST_COLS}
    agg = secs("Server", "aggregation")
    rows[GLOBAL] = {"Uplink": int(comm[comm.direction == "uplink"].bytes.sum()),
                    "Downlink": int(comm[comm.direction == "downlink"].bytes.sum()),
                    "Train (s)": 0.0, "Val (s)": 0.0, "Agg (s)": agg, "Total (s)": agg}
    return rows


# ════════════════════════════════════════════════════════════════════════════
#  Main pipeline
# ════════════════════════════════════════════════════════════════════════════
def main():
    cfg = CONFIG
    rng_seed = cfg["seed"]
    out_dir = cfg["results_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.perf_counter()

    console.print(Panel.fit(
        "[bold]Federated Random Forest[/bold] · Heart Disease Prediction\n"
        "[dim]4 hospitals · non-IID clients · tree-union aggregation · MCC-based early stopping[/dim]",
        border_style="bright_blue"))

    # ── STEP 1 ─────────────────────────────────────────────────────────────
    step(1, "Dataset Preparation")
    df, n_dup = divide_into_hospitals(cfg["raw_csv"])
    save_client_csvs(df, cfg["clients_dir"])
    df = mark_missing(df)
    console.print(f"Loaded [bold]{len(df) + n_dup}[/bold] rows, dropped [bold]{n_dup}[/bold] exact duplicates "
                  f"→ [bold]{len(df)}[/bold] patients")
    console.print("Preprocessing: missing values marked (cholesterol=0, resting BP=0, ST slope=0), "
                  "one-hot encoding of chest pain / resting ECG / ST slope, target already binary (0/1). "
                  "[dim]Scaling and outlier removal skipped: tree models are invariant to them.[/dim]")

    # ── STEP 2 ─────────────────────────────────────────────────────────────
    step(2, "Train / Validation / Test Split")
    splits = {}
    for h in HOSPITALS:
        part = df[df.hospital == h].drop(columns="hospital")
        rest, test = train_test_split(part, test_size=cfg["test_size"], stratify=part[TARGET], random_state=rng_seed)
        rel_val = cfg["val_size"] / (1 - cfg["test_size"])
        train, val = train_test_split(rest, test_size=rel_val, stratify=rest[TARGET], random_state=rng_seed)
        splits[h] = (train, val, test)
    n_tr = sum(len(s[0]) for s in splits.values())
    n_va = sum(len(s[1]) for s in splits.values())
    n_te = sum(len(s[2]) for s in splits.values())
    console.print(f"Stratified split inside every hospital → train [bold]{n_tr}[/bold] · "
                  f"validation [bold]{n_va}[/bold] · test [bold]{n_te}[/bold] (α = {cfg['test_size']}, kept unseen)")

    # ── STEP 3 ─────────────────────────────────────────────────────────────
    step(3, "Distribute Training Data to Clients (non-IID)")
    clients = [HospitalClient(i + 1, h, *splits[h]) for i, h in enumerate(HOSPITALS)]

    # Federated imputation: clients send per-feature sums and counts only
    stats = {}
    for c in clients:
        s = c.missing_value_stats()
        COMM.log(0, c.name, "uplink", "imputation stats", nbytes(s))
        stats[c.name] = s
    fill_values = {f: sum(s["sum"][f] for s in stats.values()) / sum(s["count"][f] for s in stats.values())
                   for f in NUMERIC}
    for c in clients:
        COMM.log(0, c.name, "downlink", "imputation values", nbytes(fill_values))
        c.apply_preprocessing(fill_values)
    feature_names = clients[0].feature_names

    t = Table(box=box.ROUNDED, header_style="bold cyan")
    for col in ["Client", "Hospital", "Train", "Val", "Test", "Total", "Heart disease %", "Cholesterol missing %"]:
        t.add_column(col, justify="left" if col in ("Client", "Hospital") else "right")
    for c in clients:
        raw = pd.concat([c.train_raw, c.val_raw, c.test_raw])
        t.add_row(f"D{c.cid}", c.name, str(len(c.y_train)), str(len(c.y_val)), str(len(c.y_test)), str(len(raw)),
                  f"{100 * raw[TARGET].mean():.1f}", f"{100 * raw['cholesterol'].isna().mean():.1f}")
    console.print(t)
    console.print(f"Federated imputation values: " +
                  ", ".join(f"{k} = {v:.1f}" for k, v in fill_values.items()) +
                  f"  ·  {len(feature_names)} features after encoding")

    # ── STEPS 4–8 ──────────────────────────────────────────────────────────
    step("4–8", "Iterative Federated Training")
    console.print(f"T_c = {cfg['trees_per_client']} trees per client · K = {len(clients)} clients · "
                  f"R_max = {cfg['max_rounds']} · patience = {cfg['patience']} · δ = {cfg['min_delta']} · "
                  f"tree selection = {'on' if cfg['tree_selection'] else 'off'}")
    console.print("[dim]Each round: metrics of the round's global model on every client's test set (reporting only; "
                  "model selection uses the validation MCC), plus that round's communication and computation cost.[/dim]")
    server = FederatedServer(clients, feature_names)
    keep = cfg["trees_per_client"] * len(clients)  # K × T_c
    best = {"mcc": -np.inf, "round": 0, "trees": None}
    no_improve = 0
    round_rows = []  # one row per (round, party): metrics + costs

    for rnd in range(1, cfg["max_rounds"] + 1):
        # STEP 4 · local training in parallel (simulated sequentially)
        local_forests = {}
        for c in clients:
            t0 = time.perf_counter()
            local_forests[c.name] = c.train_local_forest(cfg["trees_per_client"], seed=rng_seed + 1000 * rnd + c.cid)
            COMP.log(rnd, c.name, "local training", time.perf_counter() - t0)

        # STEP 5 · clients upload their trees (model parameters / tree structures)
        tree_bytes = {}
        for c in clients:
            tree_bytes[c.name] = nbytes(local_forests[c.name])
            COMM.log(rnd, c.name, "uplink", "local trees", tree_bytes[c.name])

        # STEP 6 · server aggregation: union of client trees (+ previous global forest)
        t0 = time.perf_counter()
        pool = server.aggregate(local_forests)
        agg_time = time.perf_counter() - t0

        # The server forwards every client the trees it does not have yet
        for c in clients:
            others = sum(b for n, b in tree_bytes.items() if n != c.name)
            COMM.log(rnd, c.name, "downlink", "other clients' trees", others)

        # STEP 6 (optional) · tree selection on federated per-tree validation MCC
        if cfg["tree_selection"] and len(pool) > keep:
            counts = np.zeros((len(pool), 4), dtype=np.int64)
            for c in clients:
                t0 = time.perf_counter()
                cnt = c.per_tree_counts(pool)
                COMP.log(rnd, c.name, "validation", time.perf_counter() - t0)
                COMM.log(rnd, c.name, "uplink", "per-tree val counts", cnt.nbytes)
                counts += cnt
            t0 = time.perf_counter()
            tree_mcc = mcc_from_counts(*counts.T.astype(float))
            selected = server.select_trees(pool, tree_mcc, keep)
            agg_time += time.perf_counter() - t0
            for c in clients:
                COMM.log(rnd, c.name, "downlink", "selected tree ids", 4 * len(selected))
        else:
            selected = pool[-keep:] if not cfg["tree_selection"] else pool
        COMP.log(rnd, "Server", "aggregation", agg_time)
        server.global_trees = selected
        model = GlobalForest(selected, feature_names)

        # STEP 7 · validation: clients evaluate F_global locally, send confusion counts
        agg = np.zeros(4, dtype=np.int64)
        for c in clients:
            t0 = time.perf_counter()
            cnt = c.ensemble_counts(model)
            COMP.log(rnd, c.name, "validation", time.perf_counter() - t0)
            COMM.log(rnd, c.name, "uplink", "ensemble val counts", cnt.nbytes)
            agg += cnt
        tp, tn, fp, fn = agg.astype(float)
        val_mcc = float(mcc_from_counts(tp, tn, fp, fn))
        val_acc = (tp + tn) / agg.sum()

        # STEP 8 · convergence check
        delta = val_mcc - best["mcc"] if np.isfinite(best["mcc"]) else val_mcc
        if val_mcc > best["mcc"] + cfg["min_delta"]:
            best = {"mcc": val_mcc, "round": rnd, "trees": list(selected)}
            no_improve, status = 0, "[green]★ new best[/green]"
        else:
            no_improve += 1
            status = f"[yellow]no improvement {no_improve}/{cfg['patience']}[/yellow]"

        # Round report: metrics + communication + computation for every party
        results_r, _ = evaluate_parties(model, clients)
        rows = cost_rows(rnd, clients, COMM.frame(), COMP.frame())
        for party, row in rows.items():
            row.update({m: results_r[party][m] for m in METRIC_NAMES})
            round_rows.append({"round": rnd, "party": party, **row})
        parallel_time = max(rows[c.name]["Total (s)"] for c in clients) + agg_time
        server.history.append({"round": rnd, "val_mcc": val_mcc, "val_acc": val_acc, "pool": len(pool),
                               "kept": len(selected), "uplink_bytes": rows[GLOBAL]["Uplink"],
                               "downlink_bytes": rows[GLOBAL]["Downlink"],
                               "round_time_parallel_s": parallel_time, "best": rnd == best["round"]})
        title = (f"Round {rnd}  ·  trees {len(pool)} → {len(selected)}  ·  val Acc {val_acc:.4f}  ·  "
                 f"val MCC {val_mcc:.4f} ({delta:+.4f})  ·  parallel round time {parallel_time:.3f} s  ·  {status}")
        console.print()
        console.print(party_table(title, {**{p: rows[p] for p in rows}}))

        if no_improve >= cfg["patience"]:
            stop_reason = f"no MCC improvement > {cfg['min_delta']} for {cfg['patience']} consecutive rounds"
            break
    else:
        stop_reason = f"reached R_max = {cfg['max_rounds']}"

    console.print("[dim]Uplink/Downlink of Global (server) = total bytes received/sent by the server. "
                  "Total (s) = training + validation + aggregation. Parallel round time = slowest client + "
                  "server aggregation.[/dim]")
    console.print(Panel.fit(
        f"Stopped after round [bold]{rnd}[/bold]: {stop_reason}.\n"
        f"Best global model: round [bold]{best['round']}[/bold] · validation MCC [bold]{best['mcc']:.4f}[/bold] · "
        f"{len(best['trees'])} trees", title="Convergence", border_style="green"))
    final_model = GlobalForest(best["trees"], feature_names)
    rounds_run = len(server.history)
    per_round = pd.DataFrame(round_rows)

    # ── STEP 9 ─────────────────────────────────────────────────────────────
    step(9, "Final Model Evaluation on the Test Set")
    results, _ = evaluate_parties(final_model, clients)
    console.print(metrics_table(f"Best global model (round {best['round']}) · test-set metrics", results))

    # Averages over all rounds
    step("9b", "Averages over all rounds")
    avg = per_round.groupby("party", sort=False)[METRIC_NAMES + COST_COLS].mean()
    console.print(party_table(f"Average per round over {rounds_run} rounds · metrics, communication and computation",
                              {p: avg.loc[p].to_dict() for p in avg.index}))

    tot = per_round.groupby("party", sort=False)[COST_COLS].sum()
    tt = Table(title=f"Total cost over all {rounds_run} rounds", box=box.ROUNDED, header_style="bold cyan",
               title_style="bold")
    tt.add_column("Party", style="bold")
    for col in COST_COLS:
        tt.add_column(col, justify="right")
    for p in tot.index:
        r = tot.loc[p]
        tt.add_row(p, fmt_bytes(r["Uplink"]), fmt_bytes(r["Downlink"]),
                   *[f"{r[c]:.4f}" if r[c] else "–" for c in COST_COLS[2:]], style=PARTY_STYLE.get(p))
    console.print(tt)

    comm, comp = COMM.frame(), COMP.frame()
    par_total = sum(h["round_time_parallel_s"] for h in server.history)
    raw_bytes = sum(nbytes(np.vstack([c.X_train, c.X_val])) + nbytes(np.concatenate([c.y_train, c.y_val]))
                    for c in clients)
    console.print(f"Simulated parallel training time: [bold]{par_total:.3f} s[/bold] "
                  f"(avg {par_total / rounds_run:.3f} s per round) vs. {comp.seconds.sum():.3f} s sequential.\n"
                  f"One-off setup (federated imputation): "
                  f"{fmt_bytes(comm[comm['round'] == 0].bytes.sum())}, not included in the per-round tables.\n"
                  f"Final global model: {len(final_model.trees)} trees · {final_model.n_nodes:,} nodes · "
                  f"{fmt_bytes(nbytes(final_model.trees))} serialized. Uploading the raw training data once would be "
                  f"{fmt_bytes(raw_bytes)}, and would expose patient records.")

    # ── STEP 10 ────────────────────────────────────────────────────────────
    step(10, "Final Model and Deployment")
    model_path = out_dir / "global_model.joblib"
    joblib.dump({"model": final_model, "fill_values": fill_values, "config": {k: str(v) for k, v in cfg.items()}},
                model_path)
    new_patient = pd.DataFrame([{
        "age": 58, "sex": 1, "chest pain type": 4, "resting bp s": 140, "cholesterol": 0, "fasting blood sugar": 0,
        "resting ecg": 1, "max heart rate": 118, "exercise angina": 1, "oldpeak": 1.8, "ST slope": 2}])
    x_new = encode_features(mark_missing(new_patient), fill_values).to_numpy(np.float32)
    p_new = final_model.predict_proba(x_new)[0]
    console.print(f"Model saved to [bold]{model_path.relative_to(ROOT)}[/bold]")
    console.print(f"Example new patient (58 y, male, asymptomatic chest pain, exercise angina, flat ST slope) → "
                  f"[bold]{'Heart disease' if p_new >= 0.5 else 'No heart disease'}[/bold] (P = {p_new:.3f})")

    # ── Save results & figures ───────────────────────────────────────────
    hist = pd.DataFrame(server.history)
    hist.to_csv(out_dir / "training_history.csv", index=False)
    per_round.to_csv(out_dir / "round_metrics_and_costs.csv", index=False)
    avg.to_csv(out_dir / "averages_over_rounds.csv")
    comm.to_csv(out_dir / "communication_cost.csv", index=False)
    comp.to_csv(out_dir / "computation_cost.csv", index=False)
    pd.DataFrame(results).T.to_csv(out_dir / "test_metrics.csv")
    with open(out_dir / "summary.json", "w") as f:
        json.dump({"best_round": best["round"], "best_val_mcc": best["mcc"], "rounds_run": rounds_run,
                   "stop_reason": stop_reason, "test_metrics": results}, f, indent=2, default=float)

    step("★", "Figures")
    files = fl_plots.make_all(out_dir / "figures", history=hist, best_round=best["round"], cfg=cfg, results=results,
                              per_round=per_round, clients=HOSPITALS, global_name=GLOBAL)
    for fpath in files:
        console.print(f"  • {fpath.relative_to(ROOT)}")
    console.print(f"\n[bold green]Done[/bold green] in {time.perf_counter() - t_start:.1f} s.")


if __name__ == "__main__":
    main()
