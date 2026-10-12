"""Publication-quality figures for the federated Random Forest experiment.

Every figure is saved as a 300-dpi PNG and as a vector PDF.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import roc_curve

# ── Palette (validated categorical order) and chart ink ─────────────────────
CLIENT_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # Hungary, Switzerland, VA, Cleveland
GLOBAL_COLOR = "#0b0b0b"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#ffffff"
BLUES = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SEQ = LinearSegmentedColormap.from_list("seq_blue", ["#f4f8fd"] + BLUES)
METHOD_COLORS = {"Local only": "#c3c2b7", "Federated": "#2a78d6", "Centralized": "#52514e"}
METRICS = ["Accuracy", "Precision", "Recall", "Specificity", "F1 Score", "MCC", "Kappa"]

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.titleweight": "bold",
    "axes.labelsize": 9,
    "axes.labelcolor": INK2,
    "axes.edgecolor": AXIS,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelcolor": INK2,
    "ytick.labelcolor": INK2,
    "legend.frameon": False,
    "legend.fontsize": 8,
    "figure.dpi": 110,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,
})


def _save(fig, out_dir, name):
    paths = []
    for ext in ("png", "pdf"):
        p = out_dir / f"{name}.{ext}"
        fig.savefig(p, facecolor=SURFACE)
        paths.append(p)
    plt.close(fig)
    return paths[0]


def _panel(ax, letter):
    ax.text(-0.02, 1.04, letter, transform=ax.transAxes, fontsize=11, fontweight="bold", color=INK,
            ha="right", va="bottom")


def _short(name):
    return {"Switzerland": "Switzer-\nland", "VA Long Beach": "VA Long\nBeach", "Global (server)": "Global\n(server)"}.get(name, name)


# ── Figure 1 · client data distribution ─────────────────────────────────────
def fig_data(out_dir, data, clients):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))
    x = np.arange(len(clients))
    neg = np.array([data[c]["neg"] for c in clients])
    pos = np.array([data[c]["pos"] for c in clients])

    ax = axes[0]
    ax.bar(x, neg, 0.6, color="#86b6ef", edgecolor=SURFACE, linewidth=1, label="No heart disease")
    ax.bar(x, pos, 0.6, bottom=neg, color="#184f95", edgecolor=SURFACE, linewidth=1, label="Heart disease")
    for i, (n, p) in enumerate(zip(neg, pos)):
        ax.text(i, n + p + 6, f"{100 * p / (n + p):.0f}% pos.", ha="center", fontsize=7.5, color=INK2)
    ax.set_xticks(x, [_short(c) for c in clients])
    ax.set_ylabel("Patients")
    ax.set_title("Class balance per hospital", loc="left")
    ax.grid(axis="x", visible=False)
    ax.set_ylim(0, (neg + pos).max() * 1.3)
    ax.legend(loc="upper center", ncols=2)
    _panel(ax, "a")

    ax = axes[1]
    bottom = np.zeros(len(clients))
    for key, label, col in [("train", "Train", "#256abf"), ("val", "Validation", "#6da7ec"), ("test", "Test", "#b7d3f6")]:
        vals = np.array([data[c][key] for c in clients])
        ax.bar(x, vals, 0.6, bottom=bottom, color=col, edgecolor=SURFACE, linewidth=1, label=label)
        bottom += vals
    ax.set_xticks(x, [_short(c) for c in clients])
    ax.set_title("Train / validation / test split", loc="left")
    ax.grid(axis="x", visible=False)
    ax.set_ylim(0, bottom.max() * 1.3)
    ax.legend(loc="upper left", ncols=3)
    _panel(ax, "b")
    fig.tight_layout()
    return _save(fig, out_dir, "fig1_client_data_distribution")


# ── Figure 2 · convergence ──────────────────────────────────────────────────
def fig_convergence(out_dir, history, best_round, cfg):
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    r = history["round"].to_numpy()
    ax.plot(r, history["val_mcc"], color=CLIENT_COLORS[0], lw=2, marker="o", ms=5, label="Validation MCC")
    ax.plot(r, history["val_acc"], color=CLIENT_COLORS[1], lw=2, marker="s", ms=4.5, label="Validation accuracy")
    b = history.loc[history["round"] == best_round].iloc[0]
    ax.scatter([best_round], [b.val_mcc], s=160, marker="*", color=INK, zorder=5, edgecolor=SURFACE, linewidth=1.5,
               label=f"Best global model (round {best_round})")
    if r[-1] > best_round:
        ax.axvspan(best_round, r[-1], color=GRID, alpha=0.5, lw=0)
        ax.text((best_round + r[-1]) / 2, 0.97, f"no improvement > δ (patience = {cfg['patience']})", ha="center",
                va="top", fontsize=7.5, color=INK2, transform=ax.get_xaxis_transform())
    ax.set_xticks(r)
    ax.set_xlim(r[0] - 0.3, r[-1] + 0.3)
    ax.set_xlabel("Communication round r")
    ax.set_ylabel("Score")
    ax.set_title("Federated training convergence", loc="left")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncols=3, fontsize=7.5)
    lo = min(history.val_mcc.min(), history.val_acc.min())
    hi = max(history.val_mcc.max(), history.val_acc.max())
    pad = (hi - lo) * 0.25 + 0.01
    ax.set_ylim(lo - pad, hi + 2 * pad)
    fig.tight_layout()
    return _save(fig, out_dir, "fig2_convergence")


# ── Figure 3 · metrics heatmap and grouped bars ──────────────────────────────
def fig_metrics(out_dir, results, names):
    M = np.array([[results[n][m] for m in METRICS] for n in names])
    fig, ax = plt.subplots(figsize=(7.2, 2.9))
    im = ax.imshow(M, cmap=SEQ, vmin=min(0, M.min()), vmax=1, aspect="auto")
    ax.set_xticks(range(len(METRICS)), METRICS)
    ax.set_yticks(range(len(names)), names)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=8,
                    color="white" if v > 0.6 else INK, fontweight="bold" if names[i].startswith("Global") else None)
    ax.axhline(len(names) - 1.5, color=SURFACE, lw=3)
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=7, color=MUTED)
    ax.set_title("Test-set performance of the federated global model", loc="left")
    fig.tight_layout()
    p1 = _save(fig, out_dir, "fig3_metrics_heatmap")

    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    w = 0.8 / len(names)
    x = np.arange(len(METRICS))
    colors = CLIENT_COLORS + [GLOBAL_COLOR]
    for i, n in enumerate(names):
        ax.bar(x + (i - (len(names) - 1) / 2) * w, M[i], w * 0.88, color=colors[i], label=n.replace("\n", " "))
    ax.set_xticks(x, METRICS)
    ax.set_ylabel("Score")
    ax.set_ylim(min(0, M.min() - 0.05), 1.05)
    ax.axhline(0, color=AXIS, lw=0.8)
    ax.grid(axis="x", visible=False)
    ax.legend(ncols=5, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    ax.set_title("Per-client and global test metrics", loc="left")
    fig.tight_layout()
    _save(fig, out_dir, "fig4_metrics_bars")
    return p1


# ── Figure 5 · confusion matrices ───────────────────────────────────────────
def fig_confusion(out_dir, results, names):
    fig, axes = plt.subplots(1, len(names), figsize=(9.0, 2.2))
    for ax, n in zip(axes, names):
        r = results[n]
        cm = np.array([[r["TP"], r["FN"]], [r["FP"], r["TN"]]])
        norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
        ax.imshow(norm, cmap=SEQ, vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]}\n{100 * norm[i, j]:.0f}%", ha="center", va="center", fontsize=8,
                        color="white" if norm[i, j] > 0.6 else INK)
        ax.set_xticks([0, 1], ["Pos.", "Neg."])
        ax.set_yticks([0, 1], ["Pos.", "Neg."])
        ax.grid(False)
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(n, fontsize=9)
        ax.set_xlabel("Predicted", fontsize=8)
    axes[0].set_ylabel("Actual", fontsize=8)
    fig.suptitle("Confusion matrices on the test set (row-normalized)", x=0.01, ha="left", fontweight="bold",
                 fontsize=10)
    fig.tight_layout()
    return _save(fig, out_dir, "fig5_confusion_matrices")


# ── Figure 6 · ROC curves ────────────────────────────────────────────────────
def fig_roc(out_dir, preds, results, names):
    fig, ax = plt.subplots(figsize=(3.8, 3.6))
    colors = CLIENT_COLORS + [GLOBAL_COLOR]
    for n, col in zip(names, colors):
        y, _, p = preds[n]
        if len(np.unique(y)) < 2:
            continue
        fpr, tpr, _ = roc_curve(y, p)
        ls = "--" if n.startswith("Global") else "-"
        ax.plot(fpr, tpr, color=col, lw=2, ls=ls, label=f"{n}  (AUC {results[n]['AUC']:.3f})")
    ax.plot([0, 1], [0, 1], color=AXIS, lw=1, ls=":")
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    ax.set_aspect("equal")
    ax.set_xlabel("False positive rate (1 − specificity)")
    ax.set_ylabel("True positive rate (sensitivity)")
    ax.set_title("ROC curves on the test set", loc="left")
    ax.legend(loc="lower right", fontsize=7)
    fig.tight_layout()
    return _save(fig, out_dir, "fig6_roc_curves")


# ── Figure 7 · federated vs. baselines ──────────────────────────────────────
def fig_baselines(out_dir, baselines, names):
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.0), sharey=False)
    methods = ["Local only", "Federated", "Centralized"]
    x = np.arange(len(names))
    w = 0.26
    for ax, metric, letter in zip(axes, ["MCC", "Accuracy"], "ab"):
        for i, m in enumerate(methods):
            vals = [baselines[m].get(n, {}).get(metric, np.nan) for n in names]
            ax.bar(x + (i - 1) * w, vals, w * 0.9, color=METHOD_COLORS[m], label=m)
        ax.set_xticks(x, [_short(n) for n in names], fontsize=8)
        ax.set_title(f"Test {metric}", loc="left")
        ax.grid(axis="x", visible=False)
        ax.axhline(0, color=AXIS, lw=0.8)
        ax.set_ylim(min(0, np.nanmin([baselines[m].get(n, {}).get(metric, 0) for m in methods for n in names]) - 0.05),
                    1.0)
        _panel(ax, letter)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.legend(handles, labels, loc="lower center", ncols=3)
    return _save(fig, out_dir, "fig7_federated_vs_baselines")


# ── Figure 8 · communication cost ───────────────────────────────────────────
def fig_communication(out_dir, comm, clients):
    comm = comm[comm["round"] > 0].copy()
    comm["KB"] = comm.bytes / 1024
    rounds = sorted(comm["round"].unique())
    fig, axes = plt.subplots(1, 3, figsize=(10.0, 3.0))

    ax = axes[0]
    bottom = np.zeros(len(rounds))
    for c, col in zip(clients, CLIENT_COLORS):
        v = comm[(comm.party == c) & (comm.direction == "uplink")].groupby("round").KB.sum().reindex(rounds,
                                                                                                    fill_value=0)
        ax.bar(rounds, v.values, 0.7, bottom=bottom, color=col, edgecolor=SURFACE, linewidth=1, label=c)
        bottom += v.values
    ax.set_xlabel("Round")
    ax.set_ylabel("Uplink (KB)")
    ax.set_title("Uplink per round", loc="left")
    ax.set_xticks(rounds)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncols=2, fontsize=7)
    _panel(ax, "a")

    ax = axes[1]
    for d, col, mk in [("uplink", CLIENT_COLORS[0], "o"), ("downlink", CLIENT_COLORS[1], "s")]:
        v = comm[comm.direction == d].groupby("round").KB.sum().reindex(rounds, fill_value=0).cumsum() / 1024
        ax.plot(rounds, v.values, color=col, lw=2, marker=mk, ms=4.5, label=d.capitalize())
        ax.annotate(f"{v.values[-1]:.2f} MB", (rounds[-1], v.values[-1]), xytext=(4, 0), textcoords="offset points",
                    fontsize=7.5, color=INK2, va="center")
    ax.set_xlabel("Round")
    ax.set_ylabel("Cumulative (MB)")
    ax.set_title("Cumulative communication", loc="left")
    ax.set_xticks(rounds)
    ax.legend(loc="upper left")
    _panel(ax, "b")

    ax = axes[2]
    x = np.arange(len(clients))
    up = [comm[(comm.party == c) & (comm.direction == "uplink")].KB.sum() / 1024 for c in clients]
    down = [comm[(comm.party == c) & (comm.direction == "downlink")].KB.sum() / 1024 for c in clients]
    ax.bar(x - 0.18, up, 0.34, color=CLIENT_COLORS[0], label="Uplink")
    ax.bar(x + 0.18, down, 0.34, color=CLIENT_COLORS[1], label="Downlink")
    ax.set_ylim(0, max(down + up) * 1.3)
    ax.set_xticks(x, [_short(c) for c in clients], fontsize=7.5)
    ax.set_ylabel("Total (MB)")
    ax.set_title("Total traffic per client", loc="left")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper center", ncols=2)
    _panel(ax, "c")
    fig.tight_layout()
    return _save(fig, out_dir, "fig8_communication_cost")


# ── Figure 9 · computation cost ─────────────────────────────────────────────
def fig_computation(out_dir, comp, clients):
    comp = comp[comp["round"] > 0]
    rounds = sorted(comp["round"].unique())
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))

    ax = axes[0]
    for c, col, mk in zip(clients, CLIENT_COLORS, "osD^"):
        v = comp[(comp.party == c) & (comp.task == "local training")].groupby("round").seconds.sum()
        ax.plot(v.index, v.values, color=col, lw=2, marker=mk, ms=4.5, label=c)
    ax.set_xlabel("Round")
    ax.set_ylabel("Seconds")
    ax.set_xticks(rounds)
    ax.set_title("Local training time per round", loc="left")
    ax.set_ylim(bottom=0)
    ax.legend(loc="lower left", ncols=2, fontsize=7)
    _panel(ax, "a")

    ax = axes[1]
    parties = clients + ["Server"]
    tasks = [("local training", "Local training", "#256abf"), ("validation", "Validation", "#86b6ef"),
             ("aggregation", "Aggregation", "#52514e")]
    y = np.arange(len(parties))
    left = np.zeros(len(parties))
    for key, label, col in tasks:
        v = np.array([comp[(comp.party == p) & (comp.task == key)].seconds.sum() for p in parties])
        ax.barh(y, v, 0.6, left=left, color=col, edgecolor=SURFACE, linewidth=1, label=label)
        left += v
    for i, t in enumerate(left):
        ax.text(t, i, f"  {t:.2f} s" if t >= 0.01 else f"  {1000 * t:.1f} ms", va="center", fontsize=7.5, color=INK2)
    ax.set_yticks(y, [_short(p).replace("\n", " ") for p in parties])
    ax.invert_yaxis()
    ax.set_xlabel("Total seconds (all rounds)")
    ax.set_xlim(0, left.max() * 1.25)
    ax.grid(axis="y", visible=False)
    ax.set_title("Computation breakdown", loc="left")
    ax.legend(loc="lower right", fontsize=7)
    _panel(ax, "b")
    fig.tight_layout()
    return _save(fig, out_dir, "fig9_computation_cost")


def make_all(out_dir, data, history, best_round, cfg, results, preds, baselines, comm, comp, clients, global_name):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    names = clients + [global_name]
    return [
        fig_data(out_dir, data, clients),
        fig_convergence(out_dir, history, best_round, cfg),
        fig_metrics(out_dir, results, names),
        out_dir / "fig4_metrics_bars.png",
        fig_confusion(out_dir, results, names),
        fig_roc(out_dir, preds, results, names),
        fig_baselines(out_dir, baselines, names),
        fig_communication(out_dir, comm, clients),
        fig_computation(out_dir, comp, clients),
    ]
