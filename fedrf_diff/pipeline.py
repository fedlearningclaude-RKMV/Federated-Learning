"""End-to-end pipeline (Steps 1-10) and the accuracy-vs-DP-budget experiment."""

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import report, zkp
from .alignment import agree_schema, federated_fill_values
from .client import REPORT_BYTES, HospitalClient
from .config import CONFIG
from .convergence import CombinedConvergence
from .data import load_hospitals, mark_missing, save_client_csvs, split_hospitals
from .decoder import Limits
from .metrics import metrics_from_counts
from .privacy import PrivacyAccountant, describe, epsilon_schedule, laplace_scale
from .report import SERVER, fmt, short_metric
from .server import FederatedServer


@dataclass
class RunResult:
    rounds_run: int
    converged: bool
    metrics: pd.DataFrame
    zkp: pd.DataFrame
    codec: pd.DataFrame
    cost: pd.DataFrame
    convergence: pd.DataFrame
    schedule: list
    epsilon_per_patient: float
    server: FederatedServer
    clients: list


def ms(seconds):
    return 1000 * seconds


def build_federation(cfg, with_zkp=True):
    """Steps 1-2 · schema agreement, hospital clients, federated imputation, key registration."""
    schema = agree_schema()
    df = mark_missing(load_hospitals(cfg["raw_csv"]))
    clients = [HospitalClient(cid, hospital, train, val, schema, cfg, with_zkp)
               for cid, (hospital, train, val) in enumerate(split_hospitals(df, cfg["val_size"], cfg["seed"]), 1)]
    fill_values = federated_fill_values([c.imputation_stats() for c in clients])
    for c in clients:
        c.apply_preprocessing(fill_values)
    server = FederatedServer(schema, Limits.from_config(cfg), cfg["zkp_lambda"], cfg["max_trees_per_client"])
    if with_zkp:
        for c in clients:
            server.register(c.client_id, c.keypair.commitment)
    return schema, clients, server


def dp_description(spec, cfg):
    if spec is None:
        return "No DP: leaf vectors are shared without noise"
    if isinstance(spec, (tuple, list)):
        draw = f"epsilon drawn uniformly from [{spec[0]:g}, {spec[1]:g}] for each round (seed {cfg['seed']}"
    else:
        draw = f"epsilon = {spec:g} for each round ("
    return (f"Laplace DP on every leaf vector: {draw}; pure DP, delta = 0), L1 clip norm C = {cfg['dp_clip']:g}, "
            f"Laplace scale b = 2C / epsilon (privacy.py)")


def run_federation(cfg, dp_spec, rounds, with_zkp=True, verbose=True, stop_on_convergence=True):
    """Steps 3-9 for up to `rounds` rounds."""
    schema, clients, server = build_federation(cfg, with_zkp)
    if verbose:
        report.header(cfg, clients, schema, dp_description(dp_spec, cfg), zkp.GROUP)
    schedule = epsilon_schedule(dp_spec, rounds, cfg["seed"])
    conv = CombinedConvergence(cfg["eps_metric"], cfg["eps_pred"], cfg["patience"])
    accountant = PrivacyAccountant(cfg["trees_per_client"])
    metric_name = cfg["convergence_metric"]
    zkp_rows, codec_rows, cost_rows, metric_rows, conv_rows = [], [], [], [], []
    converged, rnd = False, 0

    for rnd in range(1, rounds + 1):
        t_round = time.perf_counter()
        epsilon = schedule[rnd - 1]

        # Steps 3-5 · local training, explicit export, Laplace DP, serialization, proof
        updates = {}
        for c in clients:
            u = updates[c.cid] = {}
            t0 = time.perf_counter()
            forest = c.train_round(rnd)
            u["train"] = time.perf_counter() - t0
            t0 = time.perf_counter()
            trees = c.export(forest, rnd)
            t_export = time.perf_counter() - t0
            t0 = time.perf_counter()
            trees = c.privatize(trees, epsilon, rnd)
            u["dp"] = time.perf_counter() - t0
            t0 = time.perf_counter()
            u["payload"] = c.serialize(trees, rnd, epsilon)
            u["enc"] = t_export + time.perf_counter() - t0
            u["n_trees"], u["vectors"] = len(trees), sum(t.n_leaves for t in trees)
            t0 = time.perf_counter()
            u["proof"] = c.prove(u["payload"], rnd) if with_zkp else None
            u["gen"] = time.perf_counter() - t0

        # Step 6 · authentication, decoding and validation of every tree
        accepted = []
        for c in clients:
            u = updates[c.cid]
            t0 = time.perf_counter()
            u["valid"] = server.verify(c.client_id, u["payload"], u["proof"], rnd) if with_zkp else True
            u["ver"] = time.perf_counter() - t0
            t0 = time.perf_counter()
            trees, rejected = (server.decode(c.client_id, u["payload"], rnd) if u["valid"]
                               else ([], [("<update>", "authentication failed")]))
            u["dec"] = time.perf_counter() - t0
            u["accepted"], u["rejected"] = len(trees), rejected
            accepted += trees

        # Step 7 · global forest update (concatenation of valid trees)
        t0 = time.perf_counter()
        added, removed = server.aggregate(accepted)
        t_agg = time.perf_counter() - t0
        if not server.forest:
            print(f"Round {rnd}: no valid tree yet, the global forest is empty.")
            continue

        # Step 8 · every client scores G_t on its validation patients and returns counts only
        reports = {}
        for c in clients:
            t0 = time.perf_counter()
            c.receive_global_update(added, removed)
            reports[c.cid] = c.evaluate()
            updates[c.cid]["val"] = time.perf_counter() - t0
        round_metrics = {c.name: metrics_from_counts(reports[c.cid].counts) for c in clients}
        round_metrics[SERVER] = server.global_metrics(list(reports.values()))

        # Step 9 · combined convergence test
        m_t = round_metrics[SERVER][metric_name]
        check = conv.update(m_t, server.prediction_change(list(reports.values())))
        eps_patient = accountant.spend(epsilon)
        forest_kb = sum(server.tree_size.values()) / 1024
        round_time = time.perf_counter() - t_round

        # Bookkeeping
        added_bytes = {c.client_id: 0 for c in clients}
        for t in added:
            added_bytes[t.client_id] += server.tree_size[t.tree_id]
        for c in clients:
            u = updates[c.cid]
            if with_zkp:
                zkp_rows.append({"Round": rnd, "Client": c.name, "Proof": "Valid" if u["valid"] else "Invalid",
                                 "Vectors": u["vectors"], "ZKP Authentication Time (ms)": ms(u["gen"] + u["ver"]),
                                 "ZKP Proof Generation Time (ms)": ms(u["gen"]),
                                 "ZKP Proof Verification Time (ms)": ms(u["ver"])})
            codec_rows.append({"Round": rnd, "Client": c.name, "Encoding Time (ms)": ms(u["enc"]),
                               "Decoding Time (ms)": ms(u["dec"]), "Trees Sent": u["n_trees"],
                               "Trees Accepted": u["accepted"],
                               "Rejections": "; ".join(f"{i}: {r}" for i, r in u["rejected"])})
            proof_bytes = u["proof"].size_bytes if with_zkp else 0
            cost_rows.append({"Round": rnd, "Client": c.name,
                              "Uplink (KB)": (len(u["payload"]) + proof_bytes + REPORT_BYTES) / 1024,
                              "Downlink (KB)": sum(b for cid, b in added_bytes.items() if cid != c.client_id) / 1024,
                              "Local Training Time (ms)": ms(u["train"]), "DP Time (ms)": ms(u["dp"]),
                              "Validation Time (ms)": ms(u["val"]), "Aggregation Time (ms)": np.nan,
                              "Global Forest (KB)": np.nan})
        round_costs = pd.DataFrame(cost_rows[-len(clients):])
        cost_rows.append({"Round": rnd, "Client": SERVER, "Uplink (KB)": round_costs["Uplink (KB)"].sum(),
                          "Downlink (KB)": round_costs["Downlink (KB)"].sum(), "Local Training Time (ms)": np.nan,
                          "DP Time (ms)": np.nan, "Validation Time (ms)": np.nan,
                          "Aggregation Time (ms)": ms(t_agg), "Global Forest (KB)": forest_kb})
        for party, m in round_metrics.items():
            metric_rows.append({"Round": rnd, "Client": party, **m})
        n_accepted = sum(u["accepted"] for u in updates.values())
        n_rejected = sum(u["n_trees"] for u in updates.values()) - n_accepted
        conv_rows.append({"Round": rnd, "Global Trees": len(server.forest), "Trees Accepted": n_accepted,
                          "Trees Rejected": n_rejected, "M_t": m_t, "Delta M": check.delta_metric,
                          "Delta P": check.delta_pred, "Metric Stable": check.metric_stable,
                          "Prediction Stable": check.pred_stable, "Stable Rounds": check.stable_rounds,
                          "Converged": check.converged, "Epsilon per Leaf Vector": epsilon,
                          "Laplace Scale b": None if epsilon is None else laplace_scale(epsilon, cfg["dp_clip"]),
                          "Epsilon per Patient (round)": eps_patient,
                          "Epsilon per Patient (total)": accountant.total,
                          "Global Forest (KB)": forest_kb, "Round Time (s)": round_time})

        if verbose:
            dp = ("DP off" if epsilon is None else
                  f"DP budget: epsilon = {epsilon:.3f} per leaf vector this round (pure DP); per patient "
                  f"{eps_patient:.2f} this round, {accountant.total:.2f} in total")
            line = (f"Server: global forest {len(server.forest)} trees (+{n_accepted} valid, {n_rejected} rejected) | "
                    f"{short_metric(metric_name)} = {m_t:.4f} | Delta M = {fmt(check.delta_metric, '.4f')} "
                    f"(eps_M {cfg['eps_metric']:g}) | Delta P = {fmt(check.delta_pred, '.5f')} "
                    f"(eps_P {cfg['eps_pred']:g}) | patience {check.stable_rounds}/{cfg['patience']} | "
                    f"round time {round_time:.1f} s | {dp}")
            report.round_report(rnd, zkp_rows[-len(clients):] if with_zkp else [], codec_rows[-len(clients):],
                                round_metrics, line)

        if check.converged and stop_on_convergence:
            converged = True
            break

    return RunResult(rnd, converged, pd.DataFrame(metric_rows), pd.DataFrame(zkp_rows), pd.DataFrame(codec_rows),
                     pd.DataFrame(cost_rows), pd.DataFrame(conv_rows), schedule[:rnd], accountant.total,
                     server, clients)


def run_dp_sweep(cfg, rounds, names):
    """Accuracy of the same federation (same seed, no ZKP) under every DP budget for `rounds` rounds."""
    avg_rows, last_rows = [], []
    for spec in cfg["dp_sweep"]:
        res = run_federation(cfg, spec, rounds, with_zkp=False, verbose=False, stop_on_convergence=False)
        eps = [e for e in res.schedule if e is not None]
        info = {"Epsilon per round": describe(spec),
                "Mean epsilon per round": float(np.mean(eps)) if eps else 0.0,
                "Mean Laplace scale b": float(np.mean([laplace_scale(e, cfg["dp_clip"]) for e in eps])) if eps else 0.0,
                "Epsilon spent per patient": res.epsilon_per_patient}
        acc = res.metrics.pivot(index="Round", columns="Client", values="Accuracy")
        avg_rows.append({**info, **{p: acc[p].mean() for p in names + [SERVER]}})
        last_rows.append({**info, **{p: acc[p].iloc[-1] for p in names + [SERVER]}})
    return pd.DataFrame(avg_rows), pd.DataFrame(last_rows)


def main(cfg=CONFIG):
    out_dir = cfg["out_dir"]
    fig_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    save_client_csvs(load_hospitals(cfg["raw_csv"]), cfg["clients_dir"])

    res = run_federation(cfg, cfg["dp_epsilon"], cfg["max_rounds"])
    names = [c.name for c in res.clients]
    report.run_summary(res, names)

    n = res.rounds_run
    if res.converged:
        print(f"\nConverged at round {n}: Delta M <= {cfg['eps_metric']:g} and Delta P <= {cfg['eps_pred']:g} for "
              f"{cfg['patience']} consecutive rounds (seed {cfg['seed']}).")
    else:
        print(f"\nStopped at max_rounds without converging after {n} rounds (seed {cfg['seed']}).")
    forest_path = out_dir / "global_forest.json"
    res.server.save_forest(forest_path, n)
    print(f"Global forest G_{n}: {len(res.server.forest)} trees saved as explicit tree structures to {forest_path} "
          f"({forest_path.stat().st_size / 1024 ** 2:.2f} MB)")
    if cfg["dp_epsilon"] is not None:
        print(f"Privacy: every leaf vector epsilon-DP with epsilon = {min(res.schedule):.3f} to {max(res.schedule):.3f}; "
              f"per-patient bound {res.epsilon_per_patient:.2f} over {n} rounds x {cfg['trees_per_client']} trees "
              f"(basic composition; split thresholds are not privatised)")

    res.metrics.to_csv(out_dir / "validation_metrics.csv", index=False)
    res.zkp.to_csv(out_dir / "zkp_times.csv", index=False)
    res.codec.to_csv(out_dir / "encoding_decoding_times.csv", index=False)
    res.cost.to_csv(out_dir / "communication_computation_cost.csv", index=False)
    res.convergence.to_csv(out_dir / "convergence.csv", index=False)

    if cfg["make_figures"]:
        from . import plotting  # matplotlib is only needed for figures

        print("\nFigures (600-dpi PNG + vector PDF):")
        for path in plotting.make_all(fig_dir, res, names, cfg):
            print(f"  saved {path} (+ .pdf)")

    sweep = None
    if cfg["run_dp_sweep"]:
        n_budgets = sum(spec is not None for spec in cfg["dp_sweep"])
        print(f"\nAccuracy vs DP budget: {n_budgets} budgets + No DP, {n} rounds each, seed {cfg['seed']} "
              f"(no ZKP; takes a few minutes) ...")
        sweep = run_dp_sweep(cfg, n, names)
        report.dp_sweep_report(*sweep, n)
        pd.concat([sweep[0].assign(Accuracy=f"mean over {n} rounds"),
                   sweep[1].assign(Accuracy=f"after round {n}")]).to_csv(out_dir / "dp_budget_accuracy.csv",
                                                                       index=False)
        if cfg["make_figures"]:
            print(f"  saved {plotting.fig_dp_budget(fig_dir, *sweep, names, n)} (+ .pdf)")
    print(f"\nCSV files in {out_dir} and figures in {fig_dir}")
    return res, sweep
