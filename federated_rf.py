"""
FedRF: federated Random Forest for heart disease prediction across 4 hospitals.

    1  Dataset preparation      hospital partitioning, missing values, one-hot encoding
    2  Split                    stratified train / validation split inside every hospital
    3  Distribute               one non-IID client per hospital, federated imputation
    4  Local training           every client grows its Random Forest by T_c trees per round
    5  Communication            trees are serialized (encoding) and signed with a Schnorr ZKP
    6  Server aggregation       proofs are verified, accepted trees decoded and merged into
                                one global Random Forest  F_global = F_1 ∪ … ∪ F_K
    7  Validation               clients score F_global on their validation patients
    8  Combined convergence     stop at round r when
                                    r >= R_min
                                    no MCC improvement > delta for P rounds
                                    D_j <= tau for the latest P rounds
                                and check the final safeguard MCC >= M_min
    9  Reporting                per-round and averaged metrics, ZKP, encoding/decoding,
                                communication and computation cost; figures and CSVs
   10  Deployment               the best global Random Forest is saved with joblib

Run:   python3 federated_rf.py
"""

import hashlib
import pickle
import secrets
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

import fl_plots

ROOT = Path(__file__).resolve().parent

# ════════════════════════════════════════════════════════════════════════════
#  Configuration
# ════════════════════════════════════════════════════════════════════════════
CONFIG = {
    "raw_csv": ROOT / "data/raw/heart_statlog_cleveland_hungary_final.csv",
    "clients_dir": ROOT / "data/clients",
    "out_dir": ROOT / "outputs/run",
    "seed": 42,
    "val_size": 0.20,  # validation share of every hospital
    # Local Random Forest: every client adds T_c new trees per round
    "trees_per_client": 10,
    "rf_params": {"max_features": "sqrt", "min_samples_leaf": 2, "n_jobs": 1},  # + balanced class weights
    # Combined convergence criterion
    "max_rounds": 30,  # R_max
    "min_rounds": 5,  # R_min
    "patience": 3,  # P
    "min_delta": 0.001,  # delta: smallest MCC gain that counts as improvement
    "max_disagreement": 0.02,  # tau: largest share of validation predictions allowed to flip
    "min_mcc": 0.60,  # M_min: final safeguard
}

TARGET = "target"
NUMERIC = ["age", "resting bp s", "cholesterol", "max heart rate", "oldpeak"]
BINARY = ["sex", "fasting blood sugar", "exercise angina"]
CATEGORICAL = {"chest pain type": [1, 2, 3, 4], "resting ecg": [0, 1, 2], "ST slope": [1, 2, 3]}
MCC = "Matthews Correlation Coefficient (MCC)"
METRICS = ["Accuracy", "Precision", "Recall", "Specificity", "F1 Score", MCC, "Cohen's Kappa"]
SERVER = "Global Server"

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


def divide_into_hospitals(path):
    """Label every row with its hospital and drop exact duplicates."""
    df = pd.read_csv(path)
    if len(df) != SOURCE_BLOCKS[-1][2]:
        raise ValueError(f"expected {SOURCE_BLOCKS[-1][2]} rows, got {len(df)}")
    df["hospital"] = ""
    for hospital, start, end in SOURCE_BLOCKS:
        df.loc[start : end - 1, "hospital"] = hospital
    return df.drop_duplicates().reset_index(drop=True)


def save_client_csvs(df, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    for hospital, part in df.groupby("hospital", sort=False):
        part.drop(columns="hospital").to_csv(out_dir / f"{hospital.lower().replace(' ', '_')}.csv", index=False)


def mark_missing(df):
    """Zeros in cholesterol / resting BP and ST slope 0 are missing values."""
    df = df.copy()
    df.loc[df["cholesterol"] == 0, "cholesterol"] = np.nan
    df.loc[df["resting bp s"] == 0, "resting bp s"] = np.nan
    df.loc[df["ST slope"] == 0, "ST slope"] = np.nan
    return df


def encode_features(df, fill_values):
    """Impute numeric features and one-hot encode categorical ones (trees need no scaling)."""
    out = df[NUMERIC].fillna(fill_values).copy()
    for col in BINARY:
        out[col] = df[col].astype(int)
    for col, cats in CATEGORICAL.items():
        for c in cats:
            out[f"{col}={c}"] = (df[col] == c).astype(int)
    return out


# ════════════════════════════════════════════════════════════════════════════
#  Schnorr zero-knowledge proof (RFC 3526 MODP group 14, 2048-bit, g = 2)
#  A client proves knowledge of its secret key x (y = g^x mod p) and binds the
#  proof to the SHA-256 digest of its model update, the round and its id
#  (non-interactive via Fiat-Shamir). The server rejects updates whose proof fails.
# ════════════════════════════════════════════════════════════════════════════
MODP2048_P = int(
    "FFFFFFFFFFFFFFFFC90FDAA22168C234C4C6628B80DC1CD129024E088A67CC74020BBEA63B139B22514A08798E3404DD"
    "EF9519B3CD3A431B302B0A6DF25F14374FE1356D6D51C245E485B576625E7EC6F44C42E9A637ED6B0BFF5CB6F406B7ED"
    "EE386BFB5A899FA5AE9F24117C4B1FE649286651ECE45B3DC2007CB8A163BF0598DA48361C55D39A69163FA8FD24CF5F"
    "83655D23DCA3AD961C62F356208552BB9ED529077096966D670C354E4ABC9804F1746C08CA18217C32905E462E36CE3B"
    "E39E772C180E86039B2783A2EC07A28FB5C55DF06F4C52C9DE2BCBF6955817183995497CEA956AE515D2261898FA0510"
    "15728E5A8AACAA68FFFFFFFFFFFFFFFF", 16)
MODP2048_G = 2
MODP2048_Q = (MODP2048_P - 1) // 2  # prime order of the subgroup generated by g
PROOF_BYTES = 2 * 256  # (t, s), each a 2048-bit integer
COUNT_BYTES = 5 * 4  # confusion counts + flip count sent back after validation (int32)


def _challenge(y, t, digest, context):
    data = b"|".join([y.to_bytes(256, "big"), t.to_bytes(256, "big"), digest, context.encode()])
    return int.from_bytes(hashlib.sha256(data).digest(), "big") % MODP2048_Q


def schnorr_keygen():
    x = secrets.randbelow(MODP2048_Q - 1) + 1
    return x, pow(MODP2048_G, x, MODP2048_P)


def schnorr_prove(x, y, digest, context):
    k = secrets.randbelow(MODP2048_Q - 1) + 1
    t = pow(MODP2048_G, k, MODP2048_P)
    s = (k + _challenge(y, t, digest, context) * x) % MODP2048_Q
    return t, s


def schnorr_verify(y, digest, context, proof):
    t, s = proof
    c = _challenge(y, t, digest, context)
    return pow(MODP2048_G, s, MODP2048_P) == (t * pow(y, c, MODP2048_P)) % MODP2048_P


# ════════════════════════════════════════════════════════════════════════════
#  Metrics
# ════════════════════════════════════════════════════════════════════════════
def compute_metrics(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "Specificity": tn / (tn + fp) if tn + fp else 0.0,
        "F1 Score": f1_score(y_true, y_pred, zero_division=0),
        MCC: matthews_corrcoef(y_true, y_pred),
        "Cohen's Kappa": cohen_kappa_score(y_true, y_pred),
    }


def ms(seconds):
    return 1000 * seconds


# ════════════════════════════════════════════════════════════════════════════
#  Federated client (hospital)
# ════════════════════════════════════════════════════════════════════════════
class HospitalClient:
    def __init__(self, cid, hospital, train, val, seed):
        self.cid, self.hospital = cid, hospital
        self.name = f"Client {cid}({hospital})"
        self.train_raw, self.val_raw = train, val
        self.seed = seed + cid
        self.secret_key, self.public_key = schnorr_keygen()
        self.prev_pred = None

    def missing_value_stats(self):
        """Federated imputation: only per-feature sums and counts leave the hospital."""
        x = self.train_raw[NUMERIC]
        return {"sum": x.sum(skipna=True).to_dict(), "count": x.notna().sum().to_dict()}

    def apply_preprocessing(self, fill_values):
        def prep(df):
            return encode_features(df.drop(columns=[TARGET]), fill_values).to_numpy(np.float32), df[TARGET].to_numpy()
        self.X_train, self.y_train = prep(self.train_raw)
        self.X_val, self.y_val = prep(self.val_raw)
        # Balanced class weights computed once, so every warm-started round uses the same weights
        weights = compute_class_weight("balanced", classes=np.array([0, 1]), y=self.y_train)
        self.forest = RandomForestClassifier(n_estimators=0, warm_start=True, random_state=self.seed,
                                             class_weight={0: weights[0], 1: weights[1]}, **CONFIG["rf_params"])

    # Step 4 · grow the local forest by T_c trees (warm start keeps earlier trees)
    def train_round(self, n_new):
        self.forest.n_estimators += n_new
        self.forest.fit(self.X_train, self.y_train)
        return self.forest.estimators_[-n_new:]

    # Step 5 · encoding + Schnorr proof bound to the update
    @staticmethod
    def encode(trees):
        return pickle.dumps(trees, protocol=pickle.HIGHEST_PROTOCOL)

    def prove(self, payload, rnd):
        digest = hashlib.sha256(payload).digest()
        return schnorr_prove(self.secret_key, self.public_key, digest, f"round={rnd}|client={self.cid}")

    # Step 7 · local validation of the global Random Forest
    def validate(self, global_rf):
        pred = global_rf.predict(self.X_val)
        flips = None if self.prev_pred is None else int((pred != self.prev_pred).sum())
        self.prev_pred = pred
        return pred, flips


# ════════════════════════════════════════════════════════════════════════════
#  Federated server
# ════════════════════════════════════════════════════════════════════════════
class FederatedServer:
    def __init__(self, n_features):
        self.n_features = n_features
        self.public_keys = {}
        self.trees = []

    def register(self, client):
        self.public_keys[client.cid] = client.public_key

    def verify(self, cid, payload, proof, rnd):
        digest = hashlib.sha256(payload).digest()
        return schnorr_verify(self.public_keys[cid], digest, f"round={rnd}|client={cid}", proof)

    @staticmethod
    def decode(payload):
        return pickle.loads(payload)

    def aggregate(self, accepted_trees):
        """F_global = F_1 ∪ … ∪ F_K as one scikit-learn Random Forest."""
        for trees in accepted_trees:
            self.trees.extend(trees)
        return build_random_forest(self.trees, self.n_features)


def build_random_forest(trees, n_features):
    rf = RandomForestClassifier(n_estimators=len(trees))
    rf.estimators_ = list(trees)
    rf.classes_ = np.array([0, 1])
    rf.n_classes_ = 2
    rf.n_outputs_ = 1
    rf.n_features_in_ = n_features
    return rf


# ════════════════════════════════════════════════════════════════════════════
#  Terminal output helpers
# ════════════════════════════════════════════════════════════════════════════
def show(df):
    print(df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))


def stats_table(df, column, unit, parties):
    g = df.groupby("Client", sort=False)[column]
    show(pd.DataFrame({"Client": parties,
                       f"Min ({unit})": [g.min()[p] for p in parties],
                       f"Max ({unit})": [g.max()[p] for p in parties],
                       f"Mean ({unit})": [g.mean()[p] for p in parties]}))


def fmt(value, spec):
    return "n/a" if value is None else format(value, spec)


# ════════════════════════════════════════════════════════════════════════════
#  Main pipeline
# ════════════════════════════════════════════════════════════════════════════
def main():
    cfg = CONFIG
    seed = cfg["seed"]
    out_dir = cfg["out_dir"]
    fig_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    # STEPS 1-3 · data, split, clients, federated imputation
    df = divide_into_hospitals(cfg["raw_csv"])
    save_client_csvs(df, cfg["clients_dir"])
    df = mark_missing(df)
    clients = []
    for cid, h in enumerate(HOSPITALS, start=1):
        part = df[df.hospital == h].drop(columns="hospital")
        train, val = train_test_split(part, test_size=cfg["val_size"], stratify=part[TARGET], random_state=seed)
        clients.append(HospitalClient(cid, h, train, val, seed))
    stats = [c.missing_value_stats() for c in clients]
    fill_values = {f: sum(s["sum"][f] for s in stats) / sum(s["count"][f] for s in stats) for f in NUMERIC}
    for c in clients:
        c.apply_preprocessing(fill_values)
    server = FederatedServer(clients[0].X_train.shape[1])
    for c in clients:
        server.register(c)
    names = [c.name for c in clients]
    parties = names + [SERVER]

    P, R_min, delta, tau, M_min = (cfg["patience"], cfg["min_rounds"], cfg["min_delta"],
                                   cfg["max_disagreement"], cfg["min_mcc"])
    print(f"FedRF | seed {seed} | up to {cfg['max_rounds']} rounds | Schnorr ZKP: modp2048 group")
    print("Clients (train/validation patients): " +
          ", ".join(f"{c.name} {len(c.y_train)}/{len(c.y_val)}" for c in clients))
    print(f"Random Forest on every client and on the global server: each client adds {cfg['trees_per_client']} "
          f"trees per round, the server merges all client trees into one global Random Forest")
    print(f"Stop when r >= {R_min}, no MCC improvement > {delta} for {P} rounds and D_j <= {tau} for the latest "
          f"{P} rounds; final safeguard M_min = {M_min}")
    print(f"{SERVER} = the global model scored on all clients' validation patients together.")

    zkp_rows, codec_rows, cost_rows, metric_rows, conv_rows = [], [], [], [], []
    best = {"mcc": -np.inf, "round": 0, "trees": None}
    stale = stable = 0
    converged = False

    for rnd in range(1, cfg["max_rounds"] + 1):
        t_round = time.perf_counter()
        print(f"\n{'=' * 30} Round {rnd} {'=' * 30}")

        # STEP 4 · local training ── STEP 5 · encoding + Schnorr proof
        updates = {}
        for c in clients:
            t0 = time.perf_counter()
            trees = c.train_round(cfg["trees_per_client"])
            t_train = time.perf_counter() - t0
            t0 = time.perf_counter()
            payload = c.encode(trees)
            t_enc = time.perf_counter() - t0
            t0 = time.perf_counter()
            proof = c.prove(payload, rnd)
            t_gen = time.perf_counter() - t0
            updates[c.cid] = {"payload": payload, "proof": proof, "n_trees": len(trees),
                              "train": t_train, "enc": t_enc, "gen": t_gen}

        # STEP 6 · proof verification, decoding, aggregation into the global Random Forest
        accepted = []
        for c in clients:
            u = updates[c.cid]
            t0 = time.perf_counter()
            u["valid"] = server.verify(c.cid, u["payload"], u["proof"], rnd)
            u["ver"] = time.perf_counter() - t0
            t0 = time.perf_counter()
            if u["valid"]:
                accepted.append(server.decode(u["payload"]))
            u["dec"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        global_rf = server.aggregate(accepted)
        t_agg = time.perf_counter() - t0

        # STEP 7 · every client validates the global Random Forest locally
        preds, flips = {}, {}
        for c in clients:
            t0 = time.perf_counter()
            preds[c.name], flips[c.name] = c.validate(global_rf)
            updates[c.cid]["val"] = time.perf_counter() - t0

        # Bookkeeping: ZKP, encoding/decoding, communication and computation
        for c in clients:
            u = updates[c.cid]
            others = sum(len(updates[o.cid]["payload"]) for o in clients
                         if o.cid != c.cid and updates[o.cid]["valid"])
            zkp_rows.append({"Round": rnd, "Client": c.name, "Proof": "Valid" if u["valid"] else "Invalid",
                             "Trees": u["n_trees"], "ZKP Authentication Time (ms)": ms(u["gen"] + u["ver"]),
                             "ZKP Proof Generation Time (ms)": ms(u["gen"]),
                             "ZKP Proof Verification Time (ms)": ms(u["ver"])})
            codec_rows.append({"Round": rnd, "Client": c.name, "Encoding Time (ms)": ms(u["enc"]),
                               "Decoding Time (ms)": ms(u["dec"])})
            cost_rows.append({"Round": rnd, "Client": c.name,
                              "Uplink (KB)": (len(u["payload"]) + PROOF_BYTES + COUNT_BYTES) / 1024,
                              "Downlink (KB)": others / 1024, "Local Training Time (ms)": ms(u["train"]),
                              "Validation Time (ms)": ms(u["val"]), "Aggregation Time (ms)": np.nan})
        round_costs = pd.DataFrame(cost_rows[-len(clients):])
        cost_rows.append({"Round": rnd, "Client": SERVER, "Uplink (KB)": round_costs["Uplink (KB)"].sum(),
                          "Downlink (KB)": round_costs["Downlink (KB)"].sum(), "Local Training Time (ms)": np.nan,
                          "Validation Time (ms)": np.nan, "Aggregation Time (ms)": ms(t_agg)})

        # Metrics: each client on its own validation patients, the server on all of them
        round_metrics = {c.name: compute_metrics(c.y_val, preds[c.name]) for c in clients}
        y_all = np.concatenate([c.y_val for c in clients])
        p_all = np.concatenate([preds[c.name] for c in clients])
        round_metrics[SERVER] = compute_metrics(y_all, p_all)
        for party, m in round_metrics.items():
            metric_rows.append({"Round": rnd, "Client": party, **m})

        # STEP 8 · combined convergence criterion
        mcc = round_metrics[SERVER][MCC]
        d_mcc = None if rnd == 1 else mcc - best["mcc"]
        if rnd == 1 or mcc > best["mcc"] + delta:
            best = {"mcc": mcc, "round": rnd, "trees": list(server.trees)}
            stale = 0
        else:
            stale += 1
        d_j = None if rnd == 1 else sum(flips.values()) / len(y_all)
        stable = stable + 1 if d_j is not None and d_j <= tau else 0
        round_time = time.perf_counter() - t_round
        conv_rows.append({"Round": rnd, "Trees": len(server.trees), "Validation MCC": mcc,
                          "Delta MCC": d_mcc, "D_j": d_j, "MCC patience": stale, "D_j patience": stable,
                          "Best": best["round"] == rnd, "Round Time (s)": round_time})

        # ── Round report ──
        rz = pd.DataFrame(zkp_rows[-len(clients):]).drop(columns="Round")
        print("Schnorr ZKP (per client):")
        show(rz)
        print("\nZKP authentication status:")
        for _, r in rz.iterrows():
            status = ("Successful (proof valid, update accepted)" if r["Proof"] == "Valid"
                      else "Failed (proof invalid, update rejected)")
            print(f"  {r['Client']:<24}: Authentication {status}")
        print("\nEncoding and decoding times (per client):")
        show(pd.DataFrame(codec_rows[-len(clients):]).drop(columns="Round"))
        print("\nCommunication and computation cost (per client and global server):")
        show(pd.DataFrame(cost_rows[-len(parties):]).drop(columns="Round").astype(object).fillna("-"))
        print(f"\nValidation metrics of the global model after round {rnd}:")
        show(pd.DataFrame([{"Client": p, **m} for p, m in round_metrics.items()]))
        print(f"Server: Global RF {len(server.trees)} trees | MCC = {mcc:.4f} | Delta MCC = {fmt(d_mcc, '+.4f')} "
              f"(delta {delta}) | D_j = {fmt(d_j, '.4f')} (tau {tau}) | MCC patience {min(stale, P)}/{P} | "
              f"D_j patience {min(stable, P)}/{P} | round time {round_time:.1f} s")

        if rnd >= R_min and stale >= P and stable >= P:
            converged = True
            print(f"Converged at round {rnd}.")
            break

    rounds_run = rnd
    if not converged:
        print(f"Reached R_max = {cfg['max_rounds']} without meeting the combined convergence criterion.")
    if best["mcc"] < M_min:
        verdict = "below M_min: the model did not converge successfully (stuck at a poor solution)."
    elif converged:
        verdict = ">= M_min: successful convergence."
    else:
        verdict = ">= M_min, but the convergence criterion was not met within R_max."
    print(f"Final safeguard: best validation MCC {best['mcc']:.4f} (round {best['round']}, "
          f"{len(best['trees'])} trees) {verdict}")

    # ── Summaries over all rounds ──
    metrics_df, zkp_df = pd.DataFrame(metric_rows), pd.DataFrame(zkp_rows)
    codec_df, cost_df, conv_df = pd.DataFrame(codec_rows), pd.DataFrame(cost_rows), pd.DataFrame(conv_rows)

    print(f"\n=== Metrics averaged over all {rounds_run} rounds ===")
    show(metrics_df.groupby("Client", sort=False)[METRICS].mean().reindex(parties).reset_index())

    print(f"\n=== ZKP time analysis over {rounds_run} rounds (min, max and mean per client) ===")
    for col in ["ZKP Authentication Time (ms)", "ZKP Proof Generation Time (ms)", "ZKP Proof Verification Time (ms)"]:
        print(f"{col}:")
        stats_table(zkp_df, col, "ms", names)

    print("\n=== ZKP authentication summary ===")
    valid = (zkp_df.Proof == "Valid").groupby(zkp_df.Client, sort=False).sum()
    summary = pd.DataFrame({"Client": names, "Rounds": rounds_run,
                            "Successful": [int(valid[n]) for n in names]})
    summary["Failed"] = summary.Rounds - summary.Successful
    summary["ZKP Authentication"] = np.where(summary.Failed == 0, "Successful in every round",
                                             "Failed in " + summary.Failed.astype(str) + " round(s)")
    show(summary)
    n_valid = int(valid.sum())
    print(f"Schnorr proofs: {n_valid}/{len(zkp_df)} valid (updates accepted), "
          f"{len(zkp_df) - n_valid} invalid (updates rejected)")

    print(f"\n=== Encoding and decoding time analysis over {rounds_run} rounds (min, max and mean per client) ===")
    for col in ["Encoding Time (ms)", "Decoding Time (ms)"]:
        print(f"{col}:")
        stats_table(codec_df, col, "ms", names)

    print(f"\n=== Communication and computation cost analysis over {rounds_run} rounds "
          f"(min, max and mean per client and global server) ===")
    for col, unit, who in [("Uplink (KB)", "KB", parties), ("Downlink (KB)", "KB", parties),
                           ("Local Training Time (ms)", "ms", names), ("Validation Time (ms)", "ms", names),
                           ("Aggregation Time (ms)", "ms", [SERVER])]:
        print(f"{col}:")
        stats_table(cost_df, col, unit, who)
    server_costs = cost_df[cost_df.Client == SERVER]
    print(f"Total communication: uplink {server_costs['Uplink (KB)'].sum() / 1024:.4f} MB, downlink "
          f"{server_costs['Downlink (KB)'].sum() / 1024:.4f} MB | mean round time "
          f"{conv_df['Round Time (s)'].mean():.4f} s")

    print(f"\n{'Converged' if converged else 'Stopped'} after {rounds_run} rounds (seed {seed}).")

    # STEP 10 · deployment: the best global Random Forest, plus CSVs and figures
    best_rf = build_random_forest(best["trees"], server.n_features)
    joblib.dump({"model": best_rf, "fill_values": fill_values, "best_round": best["round"]},
                out_dir / "global_random_forest.joblib")
    metrics_df.to_csv(out_dir / "validation_metrics.csv", index=False)
    zkp_df.to_csv(out_dir / "zkp_times.csv", index=False)
    codec_df.to_csv(out_dir / "encoding_decoding_times.csv", index=False)
    cost_df.to_csv(out_dir / "communication_computation_cost.csv", index=False)
    conv_df.to_csv(out_dir / "convergence.csv", index=False)

    print("\nFigures (600-dpi PNG + vector PDF):")
    for path in fl_plots.make_all(fig_dir, conv_df, metrics_df, zkp_df, codec_df, cost_df, names, SERVER, cfg,
                                  best["round"]):
        print(f"  saved {path} (+ .pdf)")
    print(f"\nCSV files and the global Random Forest in {out_dir} and figures in {fig_dir}")


if __name__ == "__main__":
    main()
