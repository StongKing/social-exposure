# -*- coding: utf-8 -*-

"""
Fig.3f:
Stability of top-adjusted origins across budgets

Each cell reports:

    |T_b ∩ T_b'| / top_N

where T_b is the set of top-N CBGs ranked by reassigned visits:

    A_i(b) = 0.5 * sum_j |H_ij(b) - F_ij|

Required public input:
    figure3f_overlap.csv
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================
# Basic settings
# ============================================================

top_N = 100

save_fig = True

fig_path = (
    "figure3f.pdf"
)

dpi = 300


# 是否遮住对角线
# True: 对角线空白，更突出跨 budget overlap
# False: 保留对角线 1.00

mask_diagonal = False


data_path = (
    "figure3f_overlap.csv"
)


# ============================================================
# Load preprocessed overlap matrix
# ============================================================

overlap_df = pd.read_csv(
    data_path,
    index_col=0
)


# 保证数据为 float

overlap_df = (
    overlap_df.astype(float)
)


# ============================================================
# Prepare annotations
# ============================================================

# 避免 applymap 的 FutureWarning

annot_df = overlap_df.map(
    lambda x:
    ""
    if pd.isna(x)
    else f"{x:.2f}"
)


print(
    "\nTop-N overlap share matrix:"
)


print(
    overlap_df
)


# ============================================================
# Plot heatmap
# ============================================================

fig, ax = plt.subplots(
    figsize=(5, 5),
    dpi=dpi
)


sns.set_theme(
    style="white",
    font_scale=0.95
)


sns.set(
    font_scale=0.8
)


# 你当前使用的色带

cmap = sns.color_palette(
    "coolwarm_r",
    as_cmap=True
)


if mask_diagonal:

    mask = np.eye(
        len(overlap_df),
        dtype=bool
    )

else:

    mask = None


hm = sns.heatmap(
    overlap_df,
    mask=mask,
    annot=annot_df,
    fmt="",
    cmap=cmap,
    vmin=0,
    vmax=1,
    linewidths=0.7,
    linecolor="white",
    square=True,
    ax=ax,
    cbar_kws={
        "label":
            "Top-100 overlap share",

        "shrink":
            0.76,

        "aspect":
            25,

        "pad":
            0.025
    }
)


ax.set_title(
    f"Stability of top-{top_N} adjusted origins across budgets",
    fontsize=10.5,
    pad=5
)


# 如果不需要轴标题，就保持注释
# ax.set_xlabel("Reallocation budget", fontsize=10.5)
# ax.set_ylabel("Reallocation budget", fontsize=10.5)


ax.tick_params(
    axis="x",
    labelrotation=0,
    labelsize=10
)


ax.tick_params(
    axis="y",
    labelrotation=0,
    labelsize=10
)

ax.set_ylabel("")

# 压缩图内边距

fig.subplots_adjust(
    left=0.08,
    right=0.94,
    bottom=0.08,
    top=0.91
)


if save_fig:

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