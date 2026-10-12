"""Console report: per-round tables and the end-of-run journal tables."""

import numpy as np
import pandas as pd

from .metrics import MCC, METRICS

SERVER = "Global Server"
ZKP_TIMES = ["ZKP Authentication Time (ms)", "ZKP Proof Generation Time (ms)", "ZKP Proof Verification Time (ms)"]
CODEC_TIMES = ["Encoding Time (ms)", "Decoding Time (ms)"]


def show(df):
    print(df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))


def fmt(value, spec):
    return "n/a" if value is None else format(value, spec)


def short_metric(name):
    return "MCC" if name == MCC else name


def stats_table(df, column, unit, parties):
    g = df.groupby("Client", sort=False)[column]
    show(pd.DataFrame({"Client": parties,
                       f"Min ({unit})": [g.min()[p] for p in parties],
                       f"Max ({unit})": [g.max()[p] for p in parties],
                       f"Mean ({unit})": [g.mean()[p] for p in parties]}))


def header(cfg, clients, schema, dp_text, zkp_group):
    print(f"FedRF-Diff | seed {cfg['seed']} | up to {cfg['max_rounds']} rounds | Modified Pedersen ZKP "
          f"authentication: {zkp_group} group, lambda = {cfg['zkp_lambda']}")
    print("Clients (train/validation patients): "
          + ", ".join(f"{c.name} {len(c.y_train)}/{len(c.y_val)}" for c in clients)
          + f"; schema {schema.version} ({schema.n_features} features, id {schema.schema_id})")
    print(f"Whole-tree sharing: every client trains {cfg['trees_per_client']} new trees per round and sends them "
          f"as explicit nodes (feature, threshold, children, leaf class distribution; no feature/latent "
          f"encoding); the server validates them and concatenates the valid trees into the global forest. "
          f"Vectors = leaf class-distribution vectors in an update.")
    print(f"{dp_text}; stop when Delta M <= {cfg['eps_metric']:g} ({short_metric(cfg['convergence_metric'])}) "
          f"and Delta P <= {cfg['eps_pred']:g} for {cfg['patience']} consecutive rounds")
    print(f"{SERVER} = the global model scored on all clients' validation patients together.")


def round_report(rnd, zkp_rows, codec_rows, round_metrics, server_line):
    print(f"\n{'=' * 30} Round {rnd} {'=' * 30}")
    if zkp_rows:
        rz = pd.DataFrame(zkp_rows).drop(columns="Round")
        print("ZKP client authentication (per client):")
        show(rz)
        print("\nZKP authentication status:")
        for _, r in rz.iterrows():
            status = ("Successful (proof valid, update accepted)" if r["Proof"] == "Valid"
                      else "Failed (proof invalid, update rejected)")
            print(f"  {r['Client']:<24}: Authentication {status}")
        print()
    print("Encoding and decoding times (per client):")
    show(pd.DataFrame(codec_rows)[["Client", *CODEC_TIMES]])
    print(f"\nValidation metrics of the global model after round {rnd}:")
    show(pd.DataFrame([{"Client": p, **m} for p, m in round_metrics.items()]))
    print(server_line)


def run_summary(res, names):
    parties = names + [SERVER]
    n = res.rounds_run
    print(f"\n=== Metrics averaged over all {n} rounds ===")
    show(res.metrics.groupby("Client", sort=False)[METRICS].mean().reindex(parties).reset_index())

    print(f"\n=== ZKP time analysis over {n} rounds (min, max and mean per client) ===")
    for col in ZKP_TIMES:
        print(f"{col}:")
        stats_table(res.zkp, col, "ms", names)

    print("\n=== ZKP authentication summary ===")
    valid = (res.zkp.Proof == "Valid").groupby(res.zkp.Client, sort=False).sum()
    summary = pd.DataFrame({"Client": names, "Rounds": n, "Successful": [int(valid[c]) for c in names]})
    summary["Failed"] = summary.Rounds - summary.Successful
    summary["ZKP Authentication"] = np.where(summary.Failed == 0, "Successful in every round",
                                             "Failed in " + summary.Failed.astype(str) + " round(s)")
    show(summary)
    ok = int(valid.sum())
    print(f"ZKP authentications: {ok}/{len(res.zkp)} successful (updates accepted), "
          f"{len(res.zkp) - ok} failed (updates rejected)")

    print("\n=== Server tree validation summary ===")
    g = res.codec.groupby("Client", sort=False)
    trees = pd.DataFrame({"Client": names, "Trees Sent": [int(g["Trees Sent"].sum()[c]) for c in names],
                          "Trees Accepted": [int(g["Trees Accepted"].sum()[c]) for c in names]})
    trees["Trees Rejected"] = trees["Trees Sent"] - trees["Trees Accepted"]
    show(trees)
    print(f"Tree validation: {trees['Trees Accepted'].sum()}/{trees['Trees Sent'].sum()} trees accepted, "
          f"{trees['Trees Rejected'].sum()} rejected (schema, node references, split fields, leaf outputs, "
          f"size and depth limits)")

    print(f"\n=== Encoding and decoding time analysis over {n} rounds (min, max and mean per client) ===")
    for col in CODEC_TIMES:
        print(f"{col}:")
        stats_table(res.codec, col, "ms", names)


def dp_sweep_report(avg, last, rounds):
    print(f"Accuracy averaged over the {rounds} rounds:")
    show(avg)
    print(f"Accuracy after round {rounds}:")
    show(last)
