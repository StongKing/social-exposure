# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 14:58:19 2026

@author: JZS
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# Parameters
# ============================================================

ks = np.arange(0.01, 1.01, 0.01)

markers = ['o', 's', '^']


# ============================================================
# Helper function
# ============================================================

# ---------- 每 5 点一个标记 ----------
def sparse_marker(arr, step=5):
    """返回与 arr 同长数组，仅每 step 位保留原值，其余置 nan"""

    out = np.full_like(
        arr,
        np.nan,
        dtype=float
    )

    out[::step] = arr[::step]

    return out


# ============================================================
# Load final network metrics
# ============================================================

df_metrics_datas = pd.read_csv(
    'figure3d_metrics_datas.csv'
)


degree_finals = (
    df_metrics_datas[
        'degree_finals'
    ].values
)

clustering_finals = (
    df_metrics_datas[
        'clustering_finals'
    ].values
)

betweenness_finals = (
    df_metrics_datas[
        'betweenness_finals'
    ].values
)

modularity_finals = (
    df_metrics_datas[
        'modularity_finals'
    ].values
)


# ============================================================
# Load baseline network metrics
# ============================================================

df_metrics_initials = pd.read_csv(
    'figure3d_metrics_initials.csv'
)


initial_degree = (
    df_metrics_initials[
        'initial_degree'
    ].iloc[0]
)

initial_clustering = (
    df_metrics_initials[
        'initial_clustering'
    ].iloc[0]
)

initial_betweenness = (
    df_metrics_initials[
        'initial_betweenness'
    ].iloc[0]
)

initial_modularity = (
    df_metrics_initials[
        'initial_modularity'
    ].iloc[0]
)


# ============================================================
# Network metrics used in Fig.3d
# ============================================================

metrics_titles = [
    'Average Degree',
    'Clustering Coefficient',
    'Mean Betweenness Centrality',
    'Modularity'
]


metrics_datas = [
    degree_finals,
    clustering_finals,
    betweenness_finals,
    modularity_finals
]


metrics_initials = [
    initial_degree,
    initial_clustering,
    initial_betweenness,
    initial_modularity
]


# ============================================================
# Representative budget reported in manuscript
# ============================================================

# 选择正文中报告的代表性预算
# 建议用 0.20，因为前文已经将 20% budget 作为重要政策节点

report_k = 0.20


report_idx = int(
    np.argmin(
        np.abs(
            ks - report_k
        )
    )
)


actual_report_k = float(
    ks[report_idx]
)


# ============================================================
# Plot network metrics + output from-to values
# ============================================================

fig, axs = plt.subplots(
    2,
    2,
    figsize=(10, 10),
    dpi=300
)


axs = axs.flatten()


summary_rows = []


for i, (
    ax,
    data,
    title,
    initial_val
) in enumerate(
    zip(
        axs,
        metrics_datas,
        metrics_titles,
        metrics_initials
    )
):

    # --------------------------------------------------------
    # Calculate representative values
    # --------------------------------------------------------

    final_val = float(
        data[report_idx]
    )

    initial_val = float(
        initial_val
    )


    abs_change = (
        final_val
        - initial_val
    )


    pct_change = (
        abs_change
        / initial_val
        * 100
    ) if initial_val != 0 else np.nan


    min_val = float(
        np.nanmin(data)
    )

    max_val = float(
        np.nanmax(data)
    )


    min_k = float(
        ks[
            int(
                np.nanargmin(data)
            )
        ]
    )


    max_k = float(
        ks[
            int(
                np.nanargmax(data)
            )
        ]
    )


    direction = (
        "increases"
        if abs_change > 0
        else "decreases"
    )


    # --------------------------------------------------------
    # Save summary information
    # --------------------------------------------------------

    summary_rows.append({
        "metric":
            title,

        "initial":
            initial_val,

        f"value_at_k_{actual_report_k:.2f}":
            final_val,

        "absolute_change":
            abs_change,

        "percent_change":
            pct_change,

        "direction_at_report_k":
            direction,

        "min_over_budgets":
            min_val,

        "k_at_min":
            min_k,

        "max_over_budgets":
            max_val,

        "k_at_max":
            max_k
    })


    # --------------------------------------------------------
    # Plot optimized curve
    # --------------------------------------------------------

    ax.plot(
        ks,
        data,
        color='#4C78A8',
        linewidth=2,
        label='Optimized'
    )


    ax.plot(
        ks,
        sparse_marker(data),
        color='#4C78A8',
        marker='o',
        markersize=6,
        markeredgewidth=1,
        markerfacecolor='w',
        linestyle='None',
        clip_on=False
    )


    # --------------------------------------------------------
    # Plot baseline
    # --------------------------------------------------------

    ax.axhline(
        y=initial_val,
        color='#C76B6B',
        linestyle='--',
        linewidth=2,
        label='Baseline'
    )


    # --------------------------------------------------------
    # Report point k = 0.20
    # --------------------------------------------------------

    ax.scatter(
        actual_report_k,
        final_val,
        s=70,
        color='#3498db',
        edgecolor='black',
        linewidth=0.8,
        zorder=5
    )


    # --------------------------------------------------------
    # Annotation: baseline -> optimized
    # --------------------------------------------------------

    annotation = (
        f"Baseline: {initial_val:.3g}\n"
        f"$\\gamma$={actual_report_k:.2f}: {final_val:.3g}\n"
        f"Δ: {abs_change:+.3g} ({pct_change:+.2f}%)"
        if not np.isnan(pct_change)
        else
        f"Baseline: {initial_val:.3g}\n"
        f"k={actual_report_k:.2f}: {final_val:.3g}\n"
        f"Δ: {abs_change:+.3g}"
    )


    ax.text(
        0.98,
        0.45,
        annotation,
        transform=ax.transAxes,
        ha='right',
        va='top',
        fontsize=8,
        bbox=dict(
            boxstyle='round,pad=0.3',
            facecolor='white',
            edgecolor='0.7',
            alpha=0.9
        )
    )


    # --------------------------------------------------------
    # Axes formatting
    # --------------------------------------------------------

    ax.set_xlabel(
        '$budget$'
    )

    ax.set_title(
        title
    )

    ax.grid(
        True,
        linestyle='--',
        alpha=0.4
    )


    # 固定到右上角，不自动调整

    ax.legend(
        loc='center right',
        bbox_to_anchor=(
            0.98,
            0.55
        ),
        fontsize=8,
        frameon=True,
        fancybox=True,
        shadow=False
    )


    ax.set_xlim(
        ks.min(),
        ks.max()
    )


# ============================================================
# Save figure
# ============================================================

plt.tight_layout()


plt.savefig(
    'figure3d.pdf',
    format='pdf',
    dpi=300,
    bbox_inches='tight',
    transparent=False,
    backend='pdf'
)


plt.show()


# ============================================================
# Output manuscript values
# ============================================================

summary_df = pd.DataFrame(
    summary_rows
)


print(
    "\n=== Figure 3d network metric changes ==="
)


print(
    summary_df.to_string(
        index=False
    )
)