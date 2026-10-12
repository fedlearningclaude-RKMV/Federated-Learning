"""Journal figures (600-dpi PNG + vector PDF) for the FedRF-Diff run."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .metrics import MCC
from .report import SERVER

# Validated categorical order (one hue per hospital); the Global Server is the ink-coloured aggregate
CLIENT_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # Hungary, Switzerland, VA, Cleveland
GLOBAL_COLOR = "#0b0b0b"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#ffffff"
MARKERS = "osD^"

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
    """'Client 3(VA Long Beach)' -> 'C3 VA Long Beach'."""
    cid, hospital = name.split("(")
    return f"C{cid.split()[-1]} {hospital.rstrip(')')}"


def _rounds_axis(ax, rounds):
    rounds = list(rounds)
    step = max(1, int(np.ceil(len(rounds) / 15)))
    ax.set_xticks(rounds[::step])
    ax.set_xlim(rounds[0] - 0.5, rounds[-1] + 0.5)
    ax.set_xlabel("Communication round t")


def _per_client_lines(ax, df, column, clients):
    rounds = sorted(df["Round"].unique())
    for c, color, mk in zip(clients, CLIENT_COLORS, MARKERS):
        v = df[df.Client == c].set_index("Round")[column].reindex(rounds)
        ax.plot(rounds, v.values, color=color, lw=2, marker=mk, ms=4, label=c)
    _rounds_axis(ax, rounds)
    ax.set_ylim(bottom=0)


def _client_legend(fig, axes):
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncols=len(labels), bbox_to_anchor=(0.5, -0.01))


# ── Decentralized convergence: G^(t) performance, the three local tests, q_k and the flags ──
def _per_client_test(ax, local, column, names, tau):
    rounds = sorted(local["Round"].unique())
    for c, color, mk in zip(names, CLIENT_COLORS, MARKERS):
        v = local[local.Client == c].set_index("Round")[column].reindex(rounds)
        ax.plot(rounds, v.values, color=color, lw=1.6, marker=mk, ms=3.5, label=c)
    ax.axhline(tau, color=INK2, lw=1, ls="--")
    _rounds_axis(ax, rounds)
    ax.set_ylim(bottom=0)


def fig_convergence(out_dir, res, cfg, names):
    local, rounds_df = res.local, res.rounds
    rounds = rounds_df["Round"].to_numpy()
    server = res.metrics[res.metrics.Client == SERVER].set_index("Round")
    fig, axes = plt.subplots(2, 3, figsize=(10.4, 6.2))

    ax = axes[0, 0]
    ax.plot(server.index, server["Accuracy"], color=GLOBAL_COLOR, lw=2, marker="s", ms=4, label="Accuracy")
    ax.plot(server.index, server[MCC], color=CLIENT_COLORS[0], lw=2, marker="o", ms=4, label="MCC")
    if res.converged:
        ax.axvline(rounds[-1], color=MUTED, lw=1, ls="--", label=f"Stop (round {rounds[-1]})")
    _rounds_axis(ax, rounds)
    ax.set_ylabel("Global Server score")
    lo = min(server["Accuracy"].min(), server[MCC].min())
    ax.set_ylim(lo - 0.12, max(server["Accuracy"].max(), server[MCC].max()) + 0.03)
    ax.set_title("Global model G^(t)", loc="left")
    ax.legend(loc="lower right", ncols=3, fontsize=7)
    _panel(ax, "a")

    for ax, col, tau, title, letter in [
            (axes[0, 1], "Rel. Loss Change", cfg["tau_loss"], "Loss stability", "b"),
            (axes[0, 2], "Prediction Change", cfg["tau_pred"], "Prediction stability", "c"),
            (axes[1, 0], "G/F Disagreement", cfg["tau_agree"], "Global/local agreement", "d")]:
        _per_client_test(ax, local, col, names, tau)
        ax.set_ylabel(col)
        ax.set_title(f"{title} (τ = {tau:g})", loc="left")
        _panel(ax, letter)

    ax = axes[1, 1]
    for c, color, mk in zip(names, CLIENT_COLORS, MARKERS):
        v = local[local.Client == c].set_index("Round")["q_k"].reindex(rounds)
        ax.step(rounds, v.values, where="mid", color=color, lw=1.8, label=c)
    ax.axhline(cfg["patience"], color=INK2, lw=1, ls="--")
    _rounds_axis(ax, rounds)
    ax.set_ylim(bottom=0)
    ax.yaxis.get_major_locator().set_params(integer=True)
    ax.set_ylabel("Patience counter q_k")
    ax.set_title(f"Consecutive passes (P = {cfg['patience']})", loc="left")
    _panel(ax, "e")

    ax = axes[1, 2]
    ax.bar(rounds, rounds_df["Flags True"], 0.7, color=CLIENT_COLORS[0])
    ax.axhline(len(names), color=INK2, lw=1, ls="--")
    _rounds_axis(ax, rounds)
    ax.set_ylim(0, len(names) + 0.6)
    ax.set_yticks(range(len(names) + 1))
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("Clients with b_k = True")
    ax.set_title("Server tally of convergence flags", loc="left")
    _panel(ax, "f")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    _client_legend(fig, axes[:, 1:])
    return _save(fig, out_dir, "convergence")


# ── Modified Pedersen ZKP times ─────────────────────────────────────────────
def fig_zkp(out_dir, zkp, names):
    cols = [("ZKP Proof Generation Time (ms)", "Proof generation (client)"),
            ("ZKP Proof Verification Time (ms)", "Proof verification (server)"),
            ("ZKP Authentication Time (ms)", "Authentication (total)")]
    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.1))
    for ax, (col, title), letter in zip(axes, cols, "abc"):
        _per_client_lines(ax, zkp, col, names)
        ax.set_ylabel("Milliseconds")
        ax.set_title(title, loc="left")
        _panel(ax, letter)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    _client_legend(fig, axes)
    return _save(fig, out_dir, "zkp_times")


# ── Computation cost ────────────────────────────────────────────────────────
def fig_computation(out_dir, res, names):
    cost, transfer, zkp = res.cost, res.transfer, res.zkp
    fig, axes = plt.subplots(2, 3, figsize=(10.4, 5.8))
    panels = [(cost, "Local Refinement Time (ms)", "Local refinement H_k + CapOrPrune (client)"),
              (transfer, "Serialization Time (ms)", "Forest serialization (client)"),
              (transfer, "Tree Validation Time (ms)", "Structural validation (server)"),
              (cost, "Convergence Check Time (ms)", "Three-part convergence test (client)")]
    for ax, (df, col, title), letter in zip(axes.flat, panels, "abcd"):
        _per_client_lines(ax, df, col, names)
        ax.set_ylabel("Milliseconds")
        ax.set_title(title, loc="left", fontsize=9)
        _panel(ax, letter)

    ax = axes.flat[4]
    s = cost[cost.Client == SERVER]
    ax.plot(s["Round"], s["Aggregation Time (ms)"], color=GLOBAL_COLOR, lw=2, marker="o", ms=4)
    _rounds_axis(ax, sorted(s["Round"]))
    ax.set_ylim(bottom=0)
    ax.set_ylabel("Milliseconds")
    ax.set_title("SelectTrees aggregation (server)", loc="left", fontsize=9)
    _panel(ax, "e")

    ax = axes.flat[5]
    parts = {"Refinement": cost.groupby("Client")["Local Refinement Time (ms)"].mean(),
             "Serialization": transfer.groupby("Client")["Serialization Time (ms)"].mean(),
             "ZKP proof": zkp.groupby("Client")["ZKP Proof Generation Time (ms)"].mean(),
             "Convergence test": cost.groupby("Client")["Convergence Check Time (ms)"].mean()}
    y = np.arange(len(names))
    left = np.zeros(len(names))
    for (label, v), color in zip(parts.items(), ["#184f95", "#2a78d6", "#86b6ef", "#c3c2b7"]):
        vals = v.reindex(names).to_numpy()
        ax.barh(y, vals, 0.6, left=left, color=color, edgecolor=SURFACE, linewidth=1, label=label)
        left += vals
    ax.set_yticks(y, [_short(n) for n in names], fontsize=7)
    ax.invert_yaxis()
    ax.set_xlim(0, left.max() * 1.15)
    ax.set_xlabel("Mean milliseconds per round")
    ax.grid(axis="y", visible=False)
    ax.set_title("Mean client cost per round", loc="left", fontsize=9)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncols=2, fontsize=7)
    _panel(ax, "f")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    _client_legend(fig, axes[:, :2])
    return _save(fig, out_dir, "computation_cost")


# ── Communication cost and forest sizes ─────────────────────────────────────
def fig_communication(out_dir, res, names):
    cost = res.cost
    rounds = sorted(cost["Round"].unique())
    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.2))
    for ax, col, title, letter in [(axes[0], "Uplink (KB)", "Uplink: forests F_k^(t-1)", "a"),
                                   (axes[1], "Downlink (KB)", "Downlink: broadcast of G^(t)", "b")]:
        bottom = np.zeros(len(rounds))
        for c, color in zip(names, CLIENT_COLORS):
            v = cost[cost.Client == c].set_index("Round")[col].reindex(rounds).to_numpy() / 1024
            ax.bar(rounds, v, 0.75, bottom=bottom, color=color, edgecolor=SURFACE, linewidth=0.8, label=c)
            bottom += v
        _rounds_axis(ax, rounds)
        ax.set_ylabel(col.replace("(KB)", "(MB)"))
        ax.set_ylim(0, bottom.max() * 1.15)
        ax.grid(axis="x", visible=False)
        ax.set_title(title, loc="left")
        _panel(ax, letter)

    ax = axes[2]
    r = res.rounds
    ax.plot(r["Round"], r["Candidate Pool |U|"], color=MUTED, lw=2, marker="o", ms=4, label="Candidate pool |U^(t)|")
    ax.plot(r["Round"], r["Global Trees |G|"], color=GLOBAL_COLOR, lw=2, marker="s", ms=4, label="Global forest |G^(t)|")
    _rounds_axis(ax, rounds)
    ax.set_ylim(0, r["Candidate Pool |U|"].max() * 1.2)
    ax.set_ylabel("Trees")
    ax.set_title("Tree-level aggregation", loc="left")
    ax.legend(loc="lower right", fontsize=7)
    _panel(ax, "c")
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    _client_legend(fig, axes[:2])
    return _save(fig, out_dir, "communication_cost")


# ── Accuracy vs DP budget ───────────────────────────────────────────────────
def fig_dp_budget(out_dir, avg, last, names, rounds):
    """Accuracy of the global model G^(t) under every DP budget."""
    out_dir = Path(out_dir)
    parties = names + [SERVER]
    labels = [f"ε = {b}" if b[0].isdigit() else b for b in avg["Epsilon per round"]]
    x = np.arange(len(labels))
    w = 0.8 / len(parties)
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.4), sharey=True)
    for ax, df, title, letter in [(axes[0], avg, f"Mean accuracy over {rounds} rounds", "a"),
                                  (axes[1], last, f"Accuracy after round {rounds}", "b")]:
        for i, (p, color) in enumerate(zip(parties, CLIENT_COLORS + [GLOBAL_COLOR])):
            ax.bar(x + (i - (len(parties) - 1) / 2) * w, df[p], w * 0.86, color=color, label=p)
        ax.set_xticks(x, labels, fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.grid(axis="x", visible=False)
        ax.set_xlabel("Laplace DP budget per leaf vector and round")
        ax.set_title(title, loc="left")
        _panel(ax, letter)
    axes[0].set_ylabel("Validation accuracy")
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    _client_legend(fig, axes)
    return _save(fig, out_dir, "accuracy_vs_dp_budget")


def make_all(out_dir, res, names, cfg):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in [*out_dir.glob("*.png"), *out_dir.glob("*.pdf")]:
        old.unlink()
    return [
        fig_convergence(out_dir, res, cfg, names),
        fig_zkp(out_dir, res.zkp, names),
        fig_computation(out_dir, res, names),
        fig_communication(out_dir, res, names),
    ]
