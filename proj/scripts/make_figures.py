from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

PROJ_ROOT = Path(__file__).resolve().parent.parent
THESIS_ROOT = PROJ_ROOT.parent
FIGURES_DIR = THESIS_ROOT / "paper" / "assets" / "figures"

BLUE = "#2a78d6"
ORANGE = "#eb6834"
INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
SURFACE = "#fcfcfb"
SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#5598e7", "#2a78d6", "#184f95"]

MODELS = [
    ("Mean pooling", PROJ_ROOT / "artifacts" / "silence_experiment_default_mean.tsv", BLUE),
    ("Max pooling", PROJ_ROOT / "artifacts" / "silence_experiment_default_max.tsv", ORANGE),
]

N_BINS = 9


def apply_style():
    plt.rcParams.update(
        {
            "font.family": "Palatino",
            # "font.size": 9.5,
            "font.size": 12,
            # "text.color": INK,
            "axes.edgecolor": BASELINE,
            # "axes.labelcolor": INK,
            # "axes.titlecolor": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "axes.axisbelow": True,
            # "figure.facecolor": SURFACE,
            # "axes.facecolor": SURFACE,
            # "savefig.facecolor": SURFACE,
        }
    )


def binned_stats(df: pd.DataFrame, x_col: str, y_col: str, bins: np.ndarray):
    cut = pd.cut(df[x_col], bins=bins, include_lowest=True)
    grouped = df.groupby(cut, observed=True)[y_col]
    mean = grouped.mean()
    n = grouped.count()
    if y_col == "correct":
        se = np.sqrt(mean * (1 - mean) / n)
    else:
        se = grouped.std() / np.sqrt(n)
    centers = np.array([interval.mid for interval in mean.index])
    return centers, mean.to_numpy(), se.to_numpy()


def plot_series(ax, x, y, se, color, label):
    lower = np.clip(y - 1.96 * se, 0, 1)
    upper = np.clip(y + 1.96 * se, 0, 1)
    ax.fill_between(x, lower, upper, color=color, alpha=0.12, linewidth=0, zorder=2)
    ax.plot(
        x,
        y,
        color=color,
        linewidth=2,
        marker="o",
        markersize=5,
        markerfacecolor=color,
        markeredgecolor=SURFACE,
        markeredgewidth=1.2,
        label=label,
        zorder=3,
    )


def savefig(fig, name: str):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / f"{name}.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(FIGURES_DIR / f"{name}.png", bbox_inches="tight", dpi=150)
    plt.close(fig)


def fig_duration_position():
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))

    dur_bins = np.linspace(0, 2.0, N_BINS + 1)
    pos_bins = np.linspace(0, 1.0, N_BINS + 1)

    for label, path, color in MODELS:
        df = pd.read_csv(path, sep="\t")

        x, y, se = binned_stats(df, "duration_s", "correct", dur_bins)
        plot_series(axes[0], x, y, se, color, label)

        bona = df[df.true_label == "bonafide"]
        x, y, se = binned_stats(bona, "duration_s", "prob_bonafide", dur_bins)
        plot_series(axes[1], x, y, se, color, label)

        x, y, se = binned_stats(df, "position", "correct", pos_bins)
        plot_series(axes[2], x, y, se, color, label)

    axes[0].set_title("Accuracy vs. silence duration", fontsize=10)
    axes[0].set_xlabel("Masked duration (s)")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_ylim(0.85, 1.0)

    axes[1].set_title("Bonafide confidence vs. duration", fontsize=10)
    axes[1].set_xlabel("Masked duration (s)")
    axes[1].set_ylabel("Mean P(bonafide)\n(true-bonafide utterances)")
    axes[1].set_ylim(0.0, 1.05)

    axes[2].set_title("Accuracy vs. silence position", fontsize=10)
    axes[2].set_xlabel("Position (fraction of utterance)")
    axes[2].set_ylabel("Accuracy")
    axes[2].set_ylim(0.85, 1.0)

    for ax in axes:
        ax.grid(axis="y", visible=True)
        ax.grid(axis="x", visible=False)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.1))

    fig.tight_layout(rect=(0, 0, 1, 0.92))
    savefig(fig, "silence_duration_effect")


def fig_heatmap():
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    cmap = LinearSegmentedColormap.from_list("seqblue", SEQ_BLUE)

    dur_bins = np.linspace(0, 2.0, 6)
    pos_bins = np.linspace(0, 1.0, 6)

    im = None
    for ax, (label, path, _color) in zip(axes, MODELS):
        df = pd.read_csv(path, sep="\t")
        dur_cut = pd.cut(df.duration_s, bins=dur_bins, include_lowest=True)
        pos_cut = pd.cut(df.position, bins=pos_bins, include_lowest=True)
        pivot = df.groupby([dur_cut, pos_cut], observed=True).correct.mean().unstack()

        im = ax.imshow(pivot.to_numpy(), cmap=cmap, vmin=0.85, vmax=1.0, aspect="auto", origin="lower")
        ax.set_title(label, fontsize=10)
        ax.set_xlabel("Position (fraction)")
        if ax is axes[0]:
            ax.set_ylabel("Masked duration (s)")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([f"{c.mid:.2f}" for c in pivot.columns], fontsize=8)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([f"{r.mid:.2f}" for r in pivot.index], fontsize=8)
        for spine in ax.spines.values():
            spine.set_visible(False)

    cbar = fig.colorbar(im, ax=axes, fraction=0.046, pad=0.03)
    cbar.set_label("Accuracy")

    savefig(fig, "silence_duration_position_heatmap")


if __name__ == "__main__":
    apply_style()
    fig_duration_position()
    fig_heatmap()
    print(f"Figures written to {FIGURES_DIR}")
