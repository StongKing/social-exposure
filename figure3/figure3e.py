# -*- coding: utf-8 -*-
"""
Fig.3e: Distribution of origin-level reassigned visits across budgets

For each active origin i and budget k:

    A_i(k) = 0.5 * sum_j |H_ij(k) - F_ij|

The figure plots the empirical CDF:

    x-axis: reassigned visits per origin
    y-axis: cumulative share of active origins

Required public input:
    figure3e_data.csv
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os


# ============================================================
# Basic settings
# ============================================================

plot_ks = [
    0.01,
    0.05,
    0.20
]


data_path = (
    "figure3e_data.csv"
)



fig_path = (
    "figure3e.pdf"
)


dpi = 300


# ============================================================
# Load preprocessed data
# ============================================================

figure3e_data = pd.read_csv(
    data_path,
    dtype={
        "GEOID": str
    }
)


if "GEOID" not in figure3e_data.columns:

    raise ValueError(
        f"`GEOID` column not found in "
        f"{data_path}"
    )


figure3e_data = (
    figure3e_data
    .set_index("GEOID")
)


# ============================================================
# Plot Fig.3e
# ============================================================

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10.5,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 8.5
})


line_styles = [
    "-",
    "--",
    ":"
]


colors = [
    "#3498db",
    "#7c5bb8",
    "#e74c3c"
]


fig, ax = plt.subplots(
    figsize=(5, 5),
    dpi=dpi
)


summary_rows = []


for i, k in enumerate(
    plot_ks
):

    colname = (
        f"reassigned_k_{k:.2f}"
    )


    if colname not in figure3e_data.columns:

        print(
            f"[WARN] skip k={k}, "
            f"data not found"
        )

        continue


    # The values below are exactly:
    #
    # A_i(k) = 0.5 * sum_j |H_ij(k) - F_ij|
    #
    # computed by pre_figure3e.py.

    per_origin = (
        figure3e_data[
            colname
        ]
        .sort_values(
            ascending=False
        )
    )


    vals = (
        per_origin.values
    )


    sorted_vals = (
        np.sort(vals)
        .ravel()
    )


    n = len(
        sorted_vals
    )


    y = (
        np.arange(
            1,
            n + 1
        )
        / n
    )


    n_zero = int(
        (
            sorted_vals == 0
        ).sum()
    )


    n_changed = (
        n - n_zero
    )


    budget_label = (
        f"{int(round(k * 100))}% budget"
    )


    ax.step(
        sorted_vals,
        y,
        where="post",
        linestyle=line_styles[
            i % len(line_styles)
        ],
        color=colors[
            i % len(colors)
        ],
        linewidth=2.4,
        label=(
            f"{budget_label}, "
            f"unchanged {n_zero}/{n}"
        )
    )


    summary_rows.append({

        "k":
            k,

        "budget_label":
            budget_label,

        "n_origins":
            n,

        "n_changed":
            n_changed,

        "n_unchanged":
            n_zero,

        "unchanged_share":
            (
                n_zero / n
                if n > 0
                else np.nan
            ),

        "total_reassigned_visits":
            vals.sum(),

        "median_reassigned_visits":
            np.median(vals),

        "p90_reassigned_visits":
            np.percentile(
                vals,
                90
            ),

        "p99_reassigned_visits":
            np.percentile(
                vals,
                99
            ),

        "max_reassigned_visits":
            np.max(vals)

    })


# ============================================================
# Save summary table
# ============================================================

summary_df = pd.DataFrame(
    summary_rows
)


summary_path = "figure3e_reassigned_visits_cdf_summary.csv"


summary_df.to_csv(
    summary_path,
    index=False
)


print(
    "\nFig.3e summary:"
)


print(
    summary_df
)


print(
    f"\nSaved summary to: "
    f"{summary_path}"
)


# ============================================================
# Figure styling
# ============================================================

ax.set_xlabel(
    "Reassigned visits per origin"
)


ax.set_ylabel(
    "Cumulative share of active origins"
)


ax.set_title(
    "Distribution of origin-level reassigned visits",
    pad=7
)


ax.set_ylim(
    0,
    1.02
)


ax.legend(
    loc="lower right",
    fontsize="small",
    frameon=True,
    framealpha=0.95,
    edgecolor="0.8"
)


fig.subplots_adjust(
    left=0.14,
    right=0.98,
    bottom=0.14,
    top=0.90
)


plt.savefig(
    fig_path,
    format="pdf",
    dpi=dpi,
    bbox_inches="tight",
    pad_inches=0.01,
    transparent=False,
    backend="pdf"
)


print(
    f"\nSaved figure to: "
    f"{fig_path}"
)


plt.show()
