"""Publication-quality figures for the federated Random Forest experiment.

Every figure is saved as a 600-dpi PNG and as a vector PDF.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── Palette (validated categorical order) and chart ink ─────────────────────
CLIENT_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # Hungary, Switzerland, VA, Cleveland
GLOBAL_COLOR = "#0b0b0b"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#ffffff"
MARKERS = "osD^"
MCC = "Matthews Correlation Coefficient (MCC)"
METRICS = ["Accuracy", "Precision", "Recall", "Specificity", "F1 Score", MCC, "Cohen's Kappa"]
METRIC_LABELS = ["Accuracy", "Precision", "Recall", "Specificity", "F1 Score", "MCC", "Kappa"]

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
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,
})


def _save(fig, out_dir, name):
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"{name}.{ext}", facecolor=SURFACE)
    plt.close(fig)
    return out_dir / f"{name}.png"


def _panel(ax, letter):
    ax.text(-0.02, 1.04, letter, transform=ax.transAxes, fontsize=11, fontweight="bold", color=INK,
            ha="right", va="bottom")


def _short(name):
    """'Client 3(VA Long Beach)' -> 'C3\nVA Long\nBeach' style tick labels."""
    cid, hospital = name.split("(")
    hospital = hospital.rstrip(")").replace("VA Long Beach", "VA Long\nBeach").replace("Switzerland", "Switzer-\nland")
    return f"C{cid.split()[-1]}\n{hospital}"


def _rounds_axis(ax, rounds):
    step = 1 if len(rounds) <= 15 else 2
    ax.set_xticks(rounds[::step])
    ax.set_xlim(rounds[0] - 0.4, rounds[-1] + 0.4)
    ax.set_xlabel("Communication round r")


def _per_client_lines(ax, df, column, clients, scale=1.0):
    rounds = sorted(df["Round"].unique())
    for c, color, mk in zip(clients, CLIENT_COLORS, MARKERS):
        v = df[df.Client == c].set_index("Round")[column].reindex(rounds) * scale
        ax.plot(rounds, v.values, color=color, lw=2, marker=mk, ms=4, label=c)
    _rounds_axis(ax, rounds)
    ax.set_ylim(bottom=0)


def _client_legend(fig, axes):
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncols=len(labels), bbox_to_anchor=(0.5, -0.01))


# ── Convergence: validation MCC and prediction disagreement D_j ─────────────
def fig_convergence(out_dir, conv, cfg, best_round):
    rounds = conv["Round"].to_numpy()
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.0))

    ax = axes[0]
    ax.plot(rounds, conv["Validation MCC"], color=CLIENT_COLORS[0], lw=2, marker="o", ms=4.5,
            label="Global Server validation MCC")
    b = conv.loc[conv["Round"] == best_round, "Validation MCC"].iloc[0]
    ax.scatter([best_round], [b], s=170, marker="*", color=INK, zorder=5, edgecolor=SURFACE, linewidth=1.5,
               label=f"Best global model (round {best_round})")
    ax.axhline(cfg["min_mcc"], color=MUTED, lw=1, ls="--", label=f"M_min = {cfg['min_mcc']}")
    ax.axvspan(rounds[0] - 0.4, cfg["min_rounds"] - 0.5, color=GRID, alpha=0.6, lw=0, label=f"r < R_min = {cfg['min_rounds']}")
    _rounds_axis(ax, rounds)
    ax.set_ylabel("MCC")
    lo, hi = min(conv["Validation MCC"].min(), cfg["min_mcc"]), conv["Validation MCC"].max()
    ax.set_ylim(lo - 0.05, hi + 0.08)
    ax.set_title("Performance plateau", loc="left")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncols=2, fontsize=7)
    _panel(ax, "a")

    ax = axes[1]
    d = conv.dropna(subset=["D_j"])
    ax.plot(d["Round"], d["D_j"] * 100, color=CLIENT_COLORS[1], lw=2, marker="s", ms=4.5,
            label="Disagreement D_j")
    ax.axhline(cfg["max_disagreement"] * 100, color=MUTED, lw=1, ls="--", label=f"τ = {cfg['max_disagreement']:.0%}")
    ax.axvspan(rounds[0] - 0.4, cfg["min_rounds"] - 0.5, color=GRID, alpha=0.6, lw=0)
    _rounds_axis(ax, rounds)
    ax.set_ylabel("Predictions flipped (%)")
    ax.set_ylim(0, max(d["D_j"].max(), cfg["max_disagreement"]) * 100 * 1.3)
    ax.set_title("Prediction stability", loc="left")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncols=2, fontsize=7)
    _panel(ax, "b")
    fig.tight_layout()
    return _save(fig, out_dir, "convergence")


# ── Validation metrics of the best global model ─────────────────────────────
def fig_metrics(out_dir, metrics, names, server, best_round):
    parties = names + [server]
    m = metrics[metrics["Round"] == best_round].set_index("Client").reindex(parties)
    M = m[METRICS].to_numpy()
    fig, ax = plt.subplots(figsize=(7.6, 3.0))
    w = 0.8 / len(parties)
    x = np.arange(len(METRICS))
    for i, (p, color) in enumerate(zip(parties, CLIENT_COLORS + [GLOBAL_COLOR])):
        ax.bar(x + (i - (len(parties) - 1) / 2) * w, M[i], w * 0.88, color=color, label=p)
    ax.set_xticks(x, METRIC_LABELS)
    ax.set_ylabel("Score")
    ax.set_ylim(min(0, np.nanmin(M) - 0.05), 1.05)
    ax.axhline(0, color=AXIS, lw=0.8)
    ax.grid(axis="x", visible=False)
    ax.legend(ncols=5, loc="upper center", bbox_to_anchor=(0.5, -0.12), fontsize=7)
    ax.set_title(f"Validation metrics of the best global model (round {best_round})", loc="left")
    fig.tight_layout()
    return _save(fig, out_dir, "validation_metrics")


# ── Schnorr ZKP times ───────────────────────────────────────────────────────
def fig_zkp(out_dir, zkp, names):
    cols = [("ZKP Proof Generation Time (ms)", "Proof generation"),
            ("ZKP Proof Verification Time (ms)", "Proof verification"),
            ("ZKP Authentication Time (ms)", "Authentication (total)")]
    fig, axes = plt.subplots(1, 3, figsize=(10.0, 3.1))
    for ax, (col, title), letter in zip(axes, cols, "abc"):
        _per_client_lines(ax, zkp, col, names)
        ax.set_ylabel("Milliseconds")
        ax.set_title(title, loc="left")
        _panel(ax, letter)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    _client_legend(fig, axes)
    return _save(fig, out_dir, "zkp_times")


# ── Computation cost ────────────────────────────────────────────────────────
def fig_computation(out_dir, cost, codec, names, server):
    fig, axes = plt.subplots(2, 3, figsize=(10.0, 5.6))
    panels = [(cost, "Local Training Time (ms)", "Local training"), (cost, "Validation Time (ms)", "Validation"),
              (codec, "Encoding Time (ms)", "Encoding (client)"), (codec, "Decoding Time (ms)", "Decoding (server)")]
    for ax, (df, col, title), letter in zip(axes.flat, panels, "abcd"):
        _per_client_lines(ax, df, col, names)
        ax.set_ylabel("Milliseconds")
        ax.set_title(title, loc="left")
        _panel(ax, letter)

    ax = axes.flat[4]
    s = cost[cost.Client == server]
    ax.plot(s["Round"], s["Aggregation Time (ms)"], color=GLOBAL_COLOR, lw=2, marker="o", ms=4)
    _rounds_axis(ax, sorted(s["Round"]))
    ax.set_ylim(bottom=0)
    ax.set_ylabel("Milliseconds")
    ax.set_title("Global Server aggregation", loc="left")
    _panel(ax, "e")

    ax = axes.flat[5]
    totals = {"Training": cost.groupby("Client")["Local Training Time (ms)"].mean(),
              "Validation": cost.groupby("Client")["Validation Time (ms)"].mean(),
              "Encoding": codec.groupby("Client")["Encoding Time (ms)"].mean()}
    y = np.arange(len(names))
    left = np.zeros(len(names))
    for (label, v), color in zip(totals.items(), ["#256abf", "#86b6ef", "#c3c2b7"]):
        vals = v.reindex(names).to_numpy()
        ax.barh(y, vals, 0.6, left=left, color=color, edgecolor=SURFACE, linewidth=1, label=label)
        left += vals
    ax.set_yticks(y, [_short(n).replace("\n", " ", 1) for n in names], fontsize=7)
    ax.invert_yaxis()
    ax.set_xlim(0, left.max() * 1.6)
    ax.set_xlabel("Mean milliseconds per round")
    ax.grid(axis="y", visible=False)
    ax.set_title("Mean client cost per round", loc="left")
    ax.legend(loc="lower right", fontsize=7)
    _panel(ax, "f")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    _client_legend(fig, axes)
    return _save(fig, out_dir, "computation_cost")


# ── Uplink and downlink communication (separate figures) ────────────────────
def fig_traffic(out_dir, cost, names, column, name):
    direction = column.split(" ")[0]
    rounds = sorted(cost["Round"].unique())
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.0), gridspec_kw={"width_ratios": [1.6, 1]})

    ax = axes[0]
    bottom = np.zeros(len(rounds))
    for c, color in zip(names, CLIENT_COLORS):
        v = cost[cost.Client == c].set_index("Round")[column].reindex(rounds).to_numpy()
        ax.bar(rounds, v, 0.7, bottom=bottom, color=color, edgecolor=SURFACE, linewidth=0.8, label=c)
        bottom += v
    _rounds_axis(ax, rounds)
    ax.set_ylabel(f"{direction} (KB)")
    ax.set_ylim(0, bottom.max() * 1.35)
    ax.grid(axis="x", visible=False)
    ax.set_title(f"{direction} per round", loc="left")
    ax.legend(loc="upper center", ncols=2, fontsize=7)
    _panel(ax, "a")

    ax = axes[1]
    x = np.arange(len(names))
    totals = [cost[cost.Client == c][column].sum() / 1024 for c in names]
    ax.bar(x, totals, 0.6, color=CLIENT_COLORS)
    for i, t in enumerate(totals):
        ax.text(i, t, f"{t:.2f}", ha="center", va="bottom", fontsize=7.5, color=INK2)
    ax.set_xticks(x, [_short(n) for n in names], fontsize=7)
    ax.set_ylabel(f"Total {direction.lower()} (MB)")
    ax.set_ylim(0, max(totals) * 1.2)
    ax.grid(axis="x", visible=False)
    ax.set_title(f"Total {direction.lower()} per client", loc="left")
    _panel(ax, "b")
    fig.tight_layout()
    return _save(fig, out_dir, name)


def make_all(out_dir, conv, metrics, zkp, codec, cost, names, server, cfg, best_round):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("*.p*"):
        old.unlink()
    return [
        fig_convergence(out_dir, conv, cfg, best_round),
        fig_zkp(out_dir, zkp, names),
        fig_computation(out_dir, cost, codec, names, server),
        fig_traffic(out_dir, cost, names, "Uplink (KB)", "uplink_communication"),
        fig_traffic(out_dir, cost, names, "Downlink (KB)", "downlink_communication"),
        fig_metrics(out_dir, metrics, names, server, best_round),
    ]
