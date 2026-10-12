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
    "min_delta": 0.002,  # delta (minimum MCC improvement)
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


def metrics_table(title, results, names):
    t = Table(title=title, box=box.ROUNDED, header_style="bold cyan", title_style="bold")
    t.add_column("Client", style="bold")
    for m in METRIC_NAMES:
        t.add_column(m, justify="right")
    t.add_column("n (test)", justify="right", style="dim")
    for name in names:
        r = results[name]
        style = "bold green" if name == GLOBAL else None
        n = r["TP"] + r["TN"] + r["FP"] + r["FN"]
        t.add_row(name, *[f"{r[m]:.4f}" for m in METRIC_NAMES], str(n), style=style)
    return t


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
                  f"tree selection = {'on' if cfg['tree_selection'] else 'off'}\n")
    server = FederatedServer(clients, feature_names)
    keep = cfg["trees_per_client"] * len(clients)  # K × T_c
    best = {"mcc": -np.inf, "round": 0, "trees": None}
    no_improve = 0

    rt = Table(box=box.SIMPLE_HEAVY, header_style="bold cyan")
    for col in ["Round", "Trees (pool → kept)", "Val Acc", "Val MCC", "Δ vs best", "Uplink", "Downlink",
                "Round time*", "Status"]:
        rt.add_column(col, justify="right")

    for rnd in range(1, cfg["max_rounds"] + 1):
        # STEP 4 · local training in parallel (simulated sequentially)
        local_forests, client_times = {}, {}
        for c in clients:
            t0 = time.perf_counter()
            local_forests[c.name] = c.train_local_forest(cfg["trees_per_client"], seed=rng_seed + 1000 * rnd + c.cid)
            client_times[c.name] = time.perf_counter() - t0
            COMP.log(rnd, c.name, "local training", client_times[c.name])

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
            status = f"[yellow]no gain {no_improve}/{cfg['patience']}[/yellow]"

        comm = COMM.frame()
        comm_r = comm[comm["round"] == rnd]
        up = comm_r[comm_r.direction == "uplink"].bytes.sum()
        down = comm_r[comm_r.direction == "downlink"].bytes.sum()
        comp = COMP.frame()
        comp_r = comp[comp["round"] == rnd]
        parallel_time = comp_r[comp_r.party != "Server"].groupby("party").seconds.sum().max() + agg_time
        server.history.append({"round": rnd, "val_mcc": val_mcc, "val_acc": val_acc, "pool": len(pool),
                               "kept": len(selected), "uplink_bytes": int(up), "downlink_bytes": int(down),
                               "round_time_parallel_s": parallel_time, "best": rnd == best["round"]})
        rt.add_row(str(rnd), f"{len(pool)} → {len(selected)}", f"{val_acc:.4f}", f"{val_mcc:.4f}",
                   f"{delta:+.4f}", fmt_bytes(up), fmt_bytes(down), f"{parallel_time:.2f} s", status)

        if no_improve >= cfg["patience"]:
            stop_reason = f"no MCC improvement > {cfg['min_delta']} for {cfg['patience']} consecutive rounds"
            break
    else:
        stop_reason = f"reached R_max = {cfg['max_rounds']}"

    console.print(rt)
    console.print("[dim]* simulated parallel round time = slowest client (training + validation) + server aggregation[/dim]")
    console.print(Panel.fit(
        f"Stopped after round [bold]{rnd}[/bold]: {stop_reason}.\n"
        f"Best global model: round [bold]{best['round']}[/bold] · validation MCC [bold]{best['mcc']:.4f}[/bold] · "
        f"{len(best['trees'])} trees", title="Convergence", border_style="green"))
    final_model = GlobalForest(best["trees"], feature_names)

    # ── STEP 9 ─────────────────────────────────────────────────────────────
    step(9, "Final Model Evaluation on the Test Set")
    results, preds = {}, {}
    for c in clients:
        prob = final_model.predict_proba(c.X_test)
        y_pred = (prob >= 0.5).astype(int)
        results[c.name] = compute_metrics(c.y_test, y_pred, prob)
        preds[c.name] = (c.y_test, y_pred, prob)
    y_all = np.concatenate([preds[c.name][0] for c in clients])
    p_all = np.concatenate([preds[c.name][2] for c in clients])
    yhat_all = (p_all >= 0.5).astype(int)
    results[GLOBAL] = compute_metrics(y_all, yhat_all, p_all)
    preds[GLOBAL] = (y_all, yhat_all, p_all)
    names = HOSPITALS + [GLOBAL]
    console.print(metrics_table("Federated global model · test-set metrics", results, names))

    cm = Table(title="Confusion matrices (test set)", box=box.ROUNDED, header_style="bold cyan", title_style="bold")
    for col in ["Client", "TP", "FN", "FP", "TN", "AUC"]:
        cm.add_column(col, justify="right" if col != "Client" else "left")
    for n in names:
        r = results[n]
        cm.add_row(n, str(r["TP"]), str(r["FN"]), str(r["FP"]), str(r["TN"]),
                   f"{r['AUC']:.4f}" if "AUC" in r else "–", style="bold green" if n == GLOBAL else None)
    console.print(cm)

    # Baselines: isolated local training and (privacy-violating) centralized training
    n_trees_total = keep
    baselines = {"Local only": {}, "Centralized": {}}
    for c in clients:
        rf = RandomForestClassifier(n_estimators=n_trees_total, random_state=rng_seed, **cfg["rf_params"])
        rf.fit(c.X_train, c.y_train)
        baselines["Local only"][c.name] = compute_metrics(c.y_test, rf.predict(c.X_test))
    X_pool = np.vstack([np.vstack([c.X_train, c.X_val]) for c in clients])
    y_pool = np.concatenate([np.concatenate([c.y_train, c.y_val]) for c in clients])
    central = RandomForestClassifier(n_estimators=n_trees_total, random_state=rng_seed, **cfg["rf_params"])
    central.fit(X_pool, y_pool)
    for c in clients:
        baselines["Centralized"][c.name] = compute_metrics(c.y_test, central.predict(c.X_test))
    X_te = np.vstack([c.X_test for c in clients])
    baselines["Centralized"][GLOBAL] = compute_metrics(y_all, central.predict(X_te))
    baselines["Federated"] = results

    bt = Table(title="Federated vs. baselines (test MCC / Accuracy)", box=box.ROUNDED,
               header_style="bold cyan", title_style="bold")
    bt.add_column("Client", style="bold")
    for meth in ["Local only", "Federated", "Centralized"]:
        bt.add_column(f"{meth}\nMCC", justify="right")
        bt.add_column(f"{meth}\nAcc", justify="right")
    for n in names:
        row = [n]
        for meth in ["Local only", "Federated", "Centralized"]:
            r = baselines[meth].get(n)
            row += [f"{r['MCC']:.4f}", f"{r['Accuracy']:.4f}"] if r else ["–", "–"]
        bt.add_row(*row, style="bold green" if n == GLOBAL else None)
    console.print(bt)
    console.print("[dim]Local only: each hospital trains alone on its own data. Centralized: all data pooled "
                  "in one place (upper bound, not privacy-preserving).[/dim]")

    # Communication & computation cost
    step("9b", "Communication and Computation Cost")
    comm, comp = COMM.frame(), COMP.frame()
    ct = Table(title="Communication cost per client (all rounds)", box=box.ROUNDED,
               header_style="bold cyan", title_style="bold")
    for col in ["Client", "Uplink", "Downlink", "Total", "Messages", "Avg / round"]:
        ct.add_column(col, justify="right" if col != "Client" else "left")
    rounds_run = len(server.history)
    for c in clients:
        cc = comm[comm.party == c.name]
        up, down = cc[cc.direction == "uplink"].bytes.sum(), cc[cc.direction == "downlink"].bytes.sum()
        ct.add_row(c.name, fmt_bytes(up), fmt_bytes(down), fmt_bytes(up + down), str(len(cc)),
                   fmt_bytes((up + down) / rounds_run))
    up, down = comm[comm.direction == "uplink"].bytes.sum(), comm[comm.direction == "downlink"].bytes.sum()
    ct.add_row("All clients", fmt_bytes(up), fmt_bytes(down), fmt_bytes(up + down), str(len(comm)),
               fmt_bytes((up + down) / rounds_run), style="bold green")
    console.print(ct)

    pt = Table(title="Communication cost by payload", box=box.ROUNDED, header_style="bold cyan", title_style="bold")
    for col in ["Payload", "Direction", "Total", "Share"]:
        pt.add_column(col, justify="right" if col in ("Total", "Share") else "left")
    by_payload = comm.groupby(["payload", "direction"]).bytes.sum().sort_values(ascending=False)
    for (payload, direction), b in by_payload.items():
        pt.add_row(payload, direction, fmt_bytes(b), f"{100 * b / (up + down):.2f} %")
    console.print(pt)

    raw_bytes = sum(nbytes(np.vstack([c.X_train, c.X_val])) + nbytes(np.concatenate([c.y_train, c.y_val]))
                    for c in clients)
    console.print(f"Final global model: {len(final_model.trees)} trees · {final_model.n_nodes:,} nodes · "
                  f"{fmt_bytes(nbytes(final_model.trees))} serialized.  For reference, uploading the raw training "
                  f"data once would be {fmt_bytes(raw_bytes)} — and would expose patient records.")

    tt = Table(title="Computation cost (wall-clock seconds)", box=box.ROUNDED, header_style="bold cyan",
               title_style="bold")
    for col in ["Party", "Local training", "Validation", "Aggregation", "Total", "Avg / round"]:
        tt.add_column(col, justify="right" if col != "Party" else "left")
    for party in [c.name for c in clients] + ["Server"]:
        pc = comp[comp.party == party]
        tr = pc[pc.task == "local training"].seconds.sum()
        va = pc[pc.task == "validation"].seconds.sum()
        ag = pc[pc.task == "aggregation"].seconds.sum()
        tt.add_row(party, f"{tr:.3f}", f"{va:.3f}", f"{ag:.4f}", f"{tr + va + ag:.3f}",
                   f"{(tr + va + ag) / rounds_run:.3f}")
    seq_total = comp.seconds.sum()
    par_total = sum(h["round_time_parallel_s"] for h in server.history)
    tt.add_row("All (sequential)", "", "", "", f"{seq_total:.3f}", f"{seq_total / rounds_run:.3f}",
               style="bold green")
    console.print(tt)
    console.print(f"Simulated parallel training time: [bold]{par_total:.2f} s[/bold] over {rounds_run} rounds "
                  f"(vs. {seq_total:.2f} s sequential)")

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
    comm.to_csv(out_dir / "communication_cost.csv", index=False)
    comp.to_csv(out_dir / "computation_cost.csv", index=False)
    pd.DataFrame(results).T.to_csv(out_dir / "test_metrics.csv")
    with open(out_dir / "summary.json", "w") as f:
        json.dump({"best_round": best["round"], "best_val_mcc": best["mcc"], "rounds_run": rounds_run,
                   "stop_reason": stop_reason, "test_metrics": results,
                   "baselines": {k: v for k, v in baselines.items() if k != "Federated"}}, f, indent=2, default=float)

    step("★", "Figures")
    data_summary = {c.name: {"train": len(c.y_train), "val": len(c.y_val), "test": len(c.y_test),
                             "pos": int(np.concatenate([c.y_train, c.y_val, c.y_test]).sum()),
                             "neg": int((1 - np.concatenate([c.y_train, c.y_val, c.y_test])).sum())}
                     for c in clients}
    files = fl_plots.make_all(
        out_dir / "figures", data=data_summary, history=hist, best_round=best["round"], cfg=cfg,
        results=results, preds=preds, baselines=baselines, comm=comm, comp=comp, clients=HOSPITALS, global_name=GLOBAL)
    for fpath in files:
        console.print(f"  • {fpath.relative_to(ROOT)}")
    console.print(f"\n[bold green]Done[/bold green] in {time.perf_counter() - t_start:.1f} s.")


if __name__ == "__main__":
    main()
