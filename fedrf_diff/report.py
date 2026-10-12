"""Console report: per-round tables and the end-of-run journal tables."""

import numpy as np
import pandas as pd

from .metrics import METRICS

SERVER = "Global Server"
ZKP_TIMES = ["ZKP Authentication Time (ms)", "ZKP Proof Generation Time (ms)", "ZKP Proof Verification Time (ms)"]
TRANSFER_TIMES = ["Serialization Time (ms)", "Tree Validation Time (ms)"]
LOCAL_COLUMNS = ["Client", "Trees in F_k", "Kept from G", "Loss L_k", "Rel. Loss Change", "Prediction Change",
                 "G/F Disagreement", "q_k", "b_k"]


def show(df):
    print(df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))


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
    print(f"Tree-level aggregation: F_k^(0) = {cfg['initial_trees']} trees; every round each client uploads its "
          f"complete forest F_k^(t-1) as explicit trees (no feature/latent encoding), the server validates them and "
          f"selects G^(t) = SelectTrees(U^(t), {cfg['aggregation_policy']}, B = {cfg['global_budget']}) without "
          f"averaging, and each client builds F_k^(t) = CapOrPrune(G^(t) + H_k^(t), B_k = {cfg['local_budget']}) "
          f"with {cfg['supplementary_trees']} supplementary trees. Vectors = leaf class-distribution vectors in an upload.")
    print(f"{dp_text}")
    print(f"Decentralized stopping: each client tests on its own validation patients relative loss change <= "
          f"{cfg['tau_loss']:g}, prediction change <= {cfg['tau_pred']:g} and G/F disagreement <= "
          f"{cfg['tau_agree']:g}; q_k counts consecutive passes and only b_k = [q_k >= {cfg['patience']}] is sent; "
          f"the server stops when all b_k are True or at t = {cfg['max_rounds']}")
    print(f"{SERVER} = the global model G^(t) scored on all clients' validation patients together (evaluation "
          f"harness only; the server receives no metrics).")


def round_report(rnd, zkp_rows, transfer_rows, round_metrics, local_rows, server_line):
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
    print("Whole-tree transfer times (per client):")
    show(pd.DataFrame(transfer_rows)[["Client", *TRANSFER_TIMES]])
    if round_metrics:
        print(f"\nValidation metrics of the global model G^({rnd}) after round {rnd}:")
        show(pd.DataFrame([{"Client": p, **m} for p, m in round_metrics.items()]))
    print(f"\nLocal three-part convergence test of F_k^({rnd}) (on each client's own validation patients; "
          f"only b_k is sent):")
    show(pd.DataFrame(local_rows)[LOCAL_COLUMNS])
    print(server_line)


def run_summary(res, names):
    parties = names + [SERVER]
    n = res.rounds_run
    print(f"\n=== Global model G^(t): metrics averaged over all {n} rounds ===")
    show(res.metrics.groupby("Client", sort=False)[METRICS].mean().reindex(parties).reset_index())
    print(f"\n=== Local models F_k^(t): metrics averaged over all {n} rounds ===")
    show(res.local.groupby("Client", sort=False)[METRICS].mean().reindex(names).reset_index())
    print(f"\n=== Final models after round {n} ===")
    last_g = res.metrics[res.metrics.Round == n].set_index("Client").reindex(parties)
    last_f = res.local[res.local.Round == n].set_index("Client").reindex(names)
    show(pd.concat([last_g[METRICS].assign(Model="G").reset_index(),
                    last_f[METRICS].assign(Model="F_k").reset_index()])[["Model", "Client", *METRICS]])

    print(f"\n=== Decentralized convergence summary over {n} rounds ===")
    g = res.local.groupby("Client", sort=False)
    show(pd.DataFrame({"Client": names,
                       "Loss Stable (rounds)": [int(g["Loss Stable"].sum()[c]) for c in names],
                       "Prediction Stable (rounds)": [int(g["Prediction Stable"].sum()[c]) for c in names],
                       "Agreement (rounds)": [int(g["Agreement"].sum()[c]) for c in names],
                       "Max q_k": [int(g["q_k"].max()[c]) for c in names],
                       "Flag True (rounds)": [int(g["b_k"].sum()[c]) for c in names],
                       "Final b_k": [bool(g["b_k"].last()[c]) for c in names]}))

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
    g = res.transfer.groupby("Client", sort=False)
    trees = pd.DataFrame({"Client": names, "Trees Sent": [int(g["Trees Sent"].sum()[c]) for c in names],
                          "Trees Accepted": [int(g["Trees Accepted"].sum()[c]) for c in names]})
    trees["Trees Rejected"] = trees["Trees Sent"] - trees["Trees Accepted"]
    show(trees)
    print(f"Tree validation: {trees['Trees Accepted'].sum()}/{trees['Trees Sent'].sum()} uploaded trees accepted, "
          f"{trees['Trees Rejected'].sum()} rejected (schema, node references, split fields, leaf outputs, "
          f"size and depth limits, tree provenance)")

    print(f"\n=== Whole-tree transfer time analysis over {n} rounds (min, max and mean per client) ===")
    for col in TRANSFER_TIMES:
        print(f"{col}:")
        stats_table(res.transfer, col, "ms", names)


def dp_sweep_report(avg, last, rounds):
    print(f"Accuracy of G^(t) averaged over the {rounds} rounds:")
    show(avg)
    print(f"Accuracy of G^({rounds}) after round {rounds}:")
    show(last)
