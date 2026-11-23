"""Plotting utilities for intra-sentence UID profiles."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PALETTE = {
    "ar": "#ff7f0e",
    "en": "#1f77b4",
    "fr": "#d62728",
    "fi": "#9467bd",
    "zh": "#2ca02c",
    "tr": "#8c564b",
    "hi": "#e377c2",
    "ja": "#111111",
}


def plot_uid_panels(
    df_g: pd.DataFrame,
    df_mu: pd.DataFrame,
    title: str = "Intra-sentence information profile (UID)",
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Plot a 3-panel summary: G(x), G(x) - x, and center of mass mu with 95% CIs."""
    langs = list(dict.fromkeys(df_mu["lang"].tolist()))
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.2), dpi=200)
    ax_g, ax_d, ax_m = axes

    ax_g.plot([0, 1], [0, 1], linestyle="--", color="black", lw=1.0, alpha=0.6, label="Uniform (y=x)")
    ax_d.axhline(0.0, linestyle="--", color="black", lw=1.0, alpha=0.6)
    ax_m.axvline(0.50, linestyle="--", color="black", lw=1.0, alpha=0.7)

    for idx, lang in enumerate(langs):
        col = PALETTE.get(lang, f"C{idx}")
        sub_g = df_g[df_g["lang"] == lang].sort_values("x")
        ax_g.plot(sub_g["x"], sub_g["G_est"], color=col, lw=2.0, label=lang.upper())
        ax_g.fill_between(sub_g["x"], sub_g["G_lo"], sub_g["G_hi"], color=col, alpha=0.15)

        ax_d.plot(sub_g["x"], sub_g["diff_est"], color=col, lw=2.0, label=lang.upper())
        ax_d.fill_between(sub_g["x"], sub_g["diff_lo"], sub_g["diff_hi"], color=col, alpha=0.15)

        row = df_mu[df_mu["lang"] == lang].iloc[0]
        xerr = [[row["mu_est"] - row["mu_lo"]], [row["mu_hi"] - row["mu_est"]]]
        ax_m.errorbar(
            row["mu_est"], idx, xerr=xerr, fmt="o", color=col,
            ecolor=col, elinewidth=2.0, capsize=4, markersize=6.5,
        )
        ax_m.text(row["mu_hi"] + 0.004, idx, f"{row['mu_est']:.3f}", va="center", fontsize=8.5, color=col)

    ax_g.set_xlim(0, 1)
    ax_g.set_ylim(0, 1)
    ax_g.set_xlabel("Normalized sentence position x")
    ax_g.set_ylabel("Cumulative information G(x)")
    ax_g.set_title("(a) Cumulative curve G(x)")
    ax_g.grid(alpha=0.25)
    ax_g.legend(loc="lower right", fontsize=8.5)

    ax_d.set_xlim(0, 1)
    ax_d.set_xlabel("Normalized sentence position x")
    ax_d.set_ylabel("G(x) - x")
    ax_d.set_title("(b) Deviation from uniformity")
    ax_d.grid(alpha=0.25)

    ax_m.set_yticks(np.arange(len(langs)))
    ax_m.set_yticklabels([l.upper() for l in langs], fontweight="bold")
    ax_m.set_xlabel("Center of mass mu")
    ax_m.set_title("(c) Information center of mass (95% CI)")
    x_min = min(0.45, float(df_mu["mu_lo"].min()) - 0.02)
    x_max = max(0.55, float(df_mu["mu_hi"].max()) + 0.03)
    ax_m.set_xlim(x_min, x_max)
    ax_m.grid(axis="x", alpha=0.25)

    fig.suptitle(title, fontsize=12, fontweight="bold")
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig
