"""Publication-quality figures for the federated Random Forest experiment.

Every figure is saved as a 300-dpi PNG and as a vector PDF.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── Palette (validated categorical order) and chart ink ─────────────────────
CLIENT_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # Hungary, Switzerland, VA, Cleveland
AVERAGE_COLOR = "#898781"
GLOBAL_COLOR = "#0b0b0b"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#ffffff"
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
    return {"Switzerland": "Switzer-\nland", "VA Long Beach": "VA Long\nBeach"}.get(name, name)


# ── Figure 1 · convergence ──────────────────────────────────────────────────
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
    return _save(fig, out_dir, "fig1_convergence")


# ── Figure 2 · final test metrics per client and global ─────────────────────
def fig_metrics(out_dir, results, names):
    M = np.array([[results[n][m] for m in METRICS] for n in names])
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    w = 0.8 / len(names)
    x = np.arange(len(METRICS))
    colors = CLIENT_COLORS + [AVERAGE_COLOR, GLOBAL_COLOR]
    for i, n in enumerate(names):
        ax.bar(x + (i - (len(names) - 1) / 2) * w, M[i], w * 0.88, color=colors[i], label=n.replace("\n", " "))
    ax.set_xticks(x, METRICS)
    ax.set_ylabel("Score")
    ax.set_ylim(min(0, M.min() - 0.05), 1.05)
    ax.axhline(0, color=AXIS, lw=0.8)
    ax.grid(axis="x", visible=False)
    ax.legend(ncols=6, loc="upper center", bbox_to_anchor=(0.5, -0.12), fontsize=7.5)
    ax.set_title("Test metrics of the best global model", loc="left")
    fig.tight_layout()
    return _save(fig, out_dir, "fig2_test_metrics")


# ── Figures 3 & 4 · uplink and downlink communication ───────────────────────
def fig_traffic(out_dir, per_round, clients, direction, name):
    col = direction.capitalize()
    rounds = sorted(per_round["round"].unique())
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.0), gridspec_kw={"width_ratios": [1.5, 1]})

    ax = axes[0]
    bottom = np.zeros(len(rounds))
    for c, color in zip(clients, CLIENT_COLORS):
        v = per_round[per_round.party == c].set_index("round")[col].reindex(rounds).to_numpy() / 1024
        ax.bar(rounds, v, 0.7, bottom=bottom, color=color, edgecolor=SURFACE, linewidth=1, label=c)
        bottom += v
    ax.set_xlabel("Communication round r")
    ax.set_ylabel(f"{col} (KB)")
    ax.set_xticks(rounds)
    ax.set_ylim(0, bottom.max() * 1.3)
    ax.grid(axis="x", visible=False)
    ax.set_title(f"{col} per round", loc="left")
    ax.legend(loc="upper center", ncols=4, fontsize=7)
    _panel(ax, "a")

    ax = axes[1]
    x = np.arange(len(clients))
    totals = [per_round[per_round.party == c][col].sum() / 1024 ** 2 for c in clients]
    ax.bar(x, totals, 0.6, color=CLIENT_COLORS)
    for i, t in enumerate(totals):
        ax.text(i, t, f"{t:.2f}", ha="center", va="bottom", fontsize=7.5, color=INK2)
    ax.set_xticks(x, [_short(c) for c in clients], fontsize=7.5)
    ax.set_ylabel(f"Total {direction} (MB)")
    ax.set_ylim(0, max(totals) * 1.2)
    ax.grid(axis="x", visible=False)
    ax.set_title(f"Total {direction} per client", loc="left")
    _panel(ax, "b")
    fig.tight_layout()
    return _save(fig, out_dir, name)


# ── Figure 5 · computation cost ─────────────────────────────────────────────
def fig_computation(out_dir, per_round, clients, global_name):
    rounds = sorted(per_round["round"].unique())
    fig, axes = plt.subplots(1, 3, figsize=(10.0, 3.0))
    tasks = [("Train (s)", "Local training time"), ("Val (s)", "Validation time")]
    for ax, (col, title), letter in zip(axes[:2], tasks, "ab"):
        for c, color, mk in zip(clients, CLIENT_COLORS, "osD^"):
            v = per_round[per_round.party == c].set_index("round")[col].reindex(rounds)
            ax.plot(rounds, v.values, color=color, lw=2, marker=mk, ms=4.5, label=c)
        ax.set_xlabel("Communication round r")
        ax.set_ylabel("Seconds")
        ax.set_xticks(rounds)
        ax.set_ylim(bottom=0)
        ax.set_title(f"{title} per round", loc="left")
        _panel(ax, letter)
    axes[0].legend(loc="lower left", ncols=2, fontsize=7)

    ax = axes[2]
    v = per_round[per_round.party == global_name].set_index("round")["Agg (s)"].reindex(rounds) * 1000
    ax.plot(rounds, v.values, color=GLOBAL_COLOR, lw=2, marker="o", ms=4.5)
    ax.set_xlabel("Communication round r")
    ax.set_ylabel("Milliseconds")
    ax.set_xticks(rounds)
    ax.set_ylim(bottom=0)
    ax.set_title("Server aggregation time", loc="left")
    _panel(ax, "c")
    fig.tight_layout()
    return _save(fig, out_dir, "fig5_computation_cost")


def make_all(out_dir, history, best_round, cfg, results, per_round, clients, global_name):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("fig*"):
        old.unlink()
    return [
        fig_convergence(out_dir, history, best_round, cfg),
        fig_metrics(out_dir, results, clients + ["Average (clients)", global_name]),
        fig_traffic(out_dir, per_round, clients, "uplink", "fig3_uplink_communication"),
        fig_traffic(out_dir, per_round, clients, "downlink", "fig4_downlink_communication"),
        fig_computation(out_dir, per_round, clients, global_name),
    ]
