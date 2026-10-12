"""Journal figures (600-dpi PNG + vector PDF) for the FedRF-Diff run."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .report import SERVER, short_metric

# Validated categorical order (one hue per hospital); the Global Server is the ink-coloured aggregate
CLIENT_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # Hungary, Switzerland, VA, Cleveland
GLOBAL_COLOR = "#0b0b0b"
PASS_COLOR, FAIL_COLOR = "#2a78d6", "#eb6834"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE, BAND = "#e1e0d9", "#c3c2b7", "#ffffff", "#eeede8"
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


def _stable_band(axes, conv, patience):
    """Shade the final run of consecutive stable rounds that met the stopping rule."""
    last = conv.iloc[-1]
    if last["Converged"]:
        for ax in axes:
            ax.axvspan(last["Round"] - patience + 0.5, last["Round"] + 0.5, color=BAND, lw=0, zorder=0)


# ── Combined convergence: M_t, Delta M_t and Delta P_t ─────────────────────
def fig_convergence(out_dir, res, cfg):
    conv = res.convergence
    rounds = conv["Round"].to_numpy()
    server = res.metrics[res.metrics.Client == SERVER].set_index("Round")
    metric = short_metric(cfg["convergence_metric"])
    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.2))
    _stable_band(axes, conv, cfg["patience"])

    ax = axes[0]
    ax.plot(rounds, server.loc[rounds, cfg["convergence_metric"]], color=CLIENT_COLORS[0], lw=2, marker="o",
            ms=4, label=f"{metric} (M_t)")
    if cfg["convergence_metric"] != "Accuracy":
        ax.plot(rounds, server.loc[rounds, "Accuracy"], color=GLOBAL_COLOR, lw=2, marker="s", ms=4,
                label="Accuracy")
    if res.converged:
        ax.axvline(rounds[-1], color=MUTED, lw=1, ls="--", label=f"Converged (round {rounds[-1]})")
    _rounds_axis(ax, rounds)
    ax.set_ylabel("Global Server score")
    ax.set_title("Global validation performance", loc="left")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncols=2, fontsize=7)
    _panel(ax, "a")

    for ax, col, eps, name, letter in [(axes[1], "Delta M", cfg["eps_metric"], "Metric stability", "b"),
                                       (axes[2], "Delta P", cfg["eps_pred"], "Prediction stability", "c")]:
        d = conv.dropna(subset=[col])
        ok = d[col] <= eps
        ax.plot(d["Round"], d[col], color=AXIS, lw=1.2, zorder=1)
        ax.scatter(d.loc[ok, "Round"], d.loc[ok, col], s=30, color=PASS_COLOR, edgecolor=SURFACE, lw=1, zorder=3,
                   label=f"pass ({col} <= {eps:g})")
        ax.scatter(d.loc[~ok, "Round"], d.loc[~ok, col], s=30, marker="s", color=FAIL_COLOR, edgecolor=SURFACE,
                   lw=1, zorder=3, label="fail")
        ax.axhline(eps, color=INK2, lw=1, ls="--")
        if col == "Delta M":
            ax.set_yscale("symlog", linthresh=eps)
            ax.set_ylim(0, max(d[col].max(), eps) * 2)
        else:
            ax.set_yscale("log")
        _rounds_axis(ax, rounds)
        ax.set_ylabel(f"{col}_t")
        ax.set_title(name, loc="left")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncols=2, fontsize=7)
        _panel(ax, letter)
    fig.tight_layout()
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
    cost, codec, zkp = res.cost, res.codec, res.zkp
    fig, axes = plt.subplots(2, 3, figsize=(10.4, 5.8))
    panels = [(cost, "Local Training Time (ms)", "Local training (client)"),
              (codec, "Encoding Time (ms)", "Tree export + serialization (client)"),
              (codec, "Decoding Time (ms)", "Decoding + tree validation (server)"),
              (cost, "Validation Time (ms)", "Scoring the global forest (client)")]
    for ax, (df, col, title), letter in zip(axes.flat, panels, "abcd"):
        _per_client_lines(ax, df, col, names)
        ax.set_ylabel("Milliseconds")
        ax.set_title(title, loc="left")
        _panel(ax, letter)

    ax = axes.flat[4]
    s = cost[cost.Client == SERVER]
    ax.plot(s["Round"], s["Aggregation Time (ms)"], color=GLOBAL_COLOR, lw=2, marker="o", ms=4)
    _rounds_axis(ax, sorted(s["Round"]))
    ax.set_ylim(bottom=0)
    ax.set_ylabel("Milliseconds")
    ax.set_title("Global forest update (server)", loc="left")
    _panel(ax, "e")

    ax = axes.flat[5]
    parts = {"Training": cost.groupby("Client")["Local Training Time (ms)"].mean(),
             "Laplace DP": cost.groupby("Client")["DP Time (ms)"].mean(),
             "Encoding": codec.groupby("Client")["Encoding Time (ms)"].mean(),
             "ZKP proof": zkp.groupby("Client")["ZKP Proof Generation Time (ms)"].mean(),
             "Scoring": cost.groupby("Client")["Validation Time (ms)"].mean()}
    y = np.arange(len(names))
    left = np.zeros(len(names))
    for (label, v), color in zip(parts.items(), ["#184f95", "#2a78d6", "#86b6ef", "#c3c2b7", "#e1e0d9"]):
        vals = v.reindex(names).to_numpy()
        ax.barh(y, vals, 0.6, left=left, color=color, edgecolor=SURFACE, linewidth=1, label=label)
        left += vals
    ax.set_yticks(y, [_short(n) for n in names], fontsize=7)
    ax.invert_yaxis()
    ax.set_xlim(0, left.max() * 1.15)
    ax.set_xlabel("Mean milliseconds per round")
    ax.grid(axis="y", visible=False)
    ax.set_title("Mean client cost per round", loc="left")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncols=3, fontsize=7)
    _panel(ax, "f")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    _client_legend(fig, axes[:, :2])
    return _save(fig, out_dir, "computation_cost")


# ── Communication cost and global forest size ───────────────────────────────
def fig_communication(out_dir, res, names):
    cost = res.cost
    rounds = sorted(cost["Round"].unique())
    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.2))
    for ax, col, title, letter in [(axes[0], "Uplink (KB)", "Uplink per round", "a"),
                                   (axes[1], "Downlink (KB)", "Downlink per round", "b")]:
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
    s = res.convergence
    ax.plot(s["Round"], s["Global Forest (KB)"] / 1024, color=GLOBAL_COLOR, lw=2, marker="o", ms=4)
    last = s.iloc[-1]
    ax.annotate(f"{int(last['Global Trees'])} trees", (last["Round"], last["Global Forest (KB)"] / 1024),
                xytext=(-8, 0), textcoords="offset points", ha="right", va="center", fontsize=8, color=INK2)
    _rounds_axis(ax, rounds)
    ax.set_ylim(0, s["Global Forest (KB)"].max() / 1024 * 1.12)
    ax.set_ylabel("Global forest size (MB)")
    ax.set_title("Global forest G_t (explicit trees)", loc="left")
    _panel(ax, "c")
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    _client_legend(fig, axes[:2])
    return _save(fig, out_dir, "communication_cost")


# ── Accuracy vs DP budget ───────────────────────────────────────────────────
def fig_dp_budget(out_dir, avg, last, names, rounds):
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
        fig_convergence(out_dir, res, cfg),
        fig_zkp(out_dir, res.zkp, names),
        fig_computation(out_dir, res, names),
        fig_communication(out_dir, res, names),
    ]
