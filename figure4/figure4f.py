# -*- coding: utf-8 -*-
"""
Figure 4f

Lorenz-like curves for high-income visit distribution

Required public input
---------------------
    figure4f_high_income_visit_distribution.csv

The file contains normalized high-income visit
distributions across POIs under:

    baseline
    regularized allocation
    unregularized allocation

Entropy and Gini are recomputed directly in this script.

Output
------
    figure4f.pdf

@author: JZS
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Configuration
# ============================================================

data_path = (
    'figure4f_high_income_visit_distribution.csv'
)


# ============================================================
# Check input
# ============================================================

if not os.path.exists(
    data_path
):

    raise FileNotFoundError(
        f'Cannot find required file:\n'
        f'{data_path}'
    )


# ============================================================
# Load public Fig.4f data
# ============================================================

figure4f_df = pd.read_csv(

    data_path,

    dtype={
        'POI':
            str
    }
)


required_columns = [

    'POI',

    'baseline',

    'regu',

    'no_regu'
]


missing_columns = [

    col

    for col
    in required_columns

    if col
    not in figure4f_df.columns
]


if len(
    missing_columns
) > 0:

    raise ValueError(
        'Missing required columns in '
        'figure4f_high_income_visit_distribution.csv:\n'
        f'{missing_columns}'
    )


figure4f_df = (
    figure4f_df
    .set_index(
        'POI'
    )
)


# ============================================================
# Restore the three distributions
# ============================================================

high_dist_A = (
    pd.to_numeric(
        figure4f_df[
            'baseline'
        ],
        errors='coerce'
    )
    .fillna(
        0.0
    )
)


high_dist_reg = (
    pd.to_numeric(
        figure4f_df[
            'regu'
        ],
        errors='coerce'
    )
    .fillna(
        0.0
    )
)


high_dist_no = (
    pd.to_numeric(
        figure4f_df[
            'no_regu'
        ],
        errors='coerce'
    )
    .fillna(
        0.0
    )
)


# ============================================================
# Original helper functions
# ============================================================

def shannon_entropy(p):

    p = np.array(
        p,
        dtype=float
    )


    if p.sum() == 0:

        return 0.0


    p = (
        p
        /
        p.sum()
    )


    p = (
        p[
            p > 0
        ]
    )


    if p.size == 0:

        return 0.0


    return (
        -
        (
            p
            *
            np.log2(
                p
            )
        )
        .sum()
    )


def gini_from_values(x):

    x = np.array(
        x,
        dtype=float
    ).flatten()


    if x.size == 0:

        return np.nan


    if np.all(
        x == 0
    ):

        return 0.0


    x = (
        x[
            x >= 0
        ]
    )


    x_sorted = np.sort(
        x
    )


    n = (
        x_sorted.size
    )


    index = np.arange(
        1,
        n + 1
    )


    return (

        2.0
        *
        np.sum(
            index
            *
            x_sorted
        )
        /
        (
            n
            *
            np.sum(
                x_sorted
            )
        )

        -

        (
            n + 1
        )
        /
        n
    )


# ============================================================
# Entropy
# ============================================================

entropy_A_high = (
    shannon_entropy(
        high_dist_A.values
    )
)


entropy_no_high = (
    shannon_entropy(
        high_dist_no.values
    )
)


entropy_reg_high = (
    shannon_entropy(
        high_dist_reg.values
    )
)


# ============================================================
# Gini
# ============================================================

gini_A_high = (
    gini_from_values(
        high_dist_A.values
    )
)


gini_no_high = (
    gini_from_values(
        high_dist_no.values
    )
)


gini_reg_high = (
    gini_from_values(
        high_dist_reg.values
    )
)


# ============================================================
# Print results
# ============================================================

print(
    '=============================================='
)


print(
    'FIG.4F HIGH-INCOME VISIT DISTRIBUTION'
)


print(
    '=============================================='
)


print(
    '\nEntropy'
)


print(
    'baseline =',
    entropy_A_high
)


print(
    'regu =',
    entropy_reg_high
)


print(
    'no_regu =',
    entropy_no_high
)


print(
    '\nGini'
)


print(
    'baseline =',
    gini_A_high
)


print(
    'regu =',
    gini_reg_high
)


print(
    'no_regu =',
    gini_no_high
)


# ============================================================
# Original Fig.4f plotting code
# ============================================================

# 颜色
colors = [
    '#3498db',
    '#e74c3f',
    '#7c5bb8'
]


labels = [
    'baseline',
    'regu',
    'no_regu'
]


# 创建独立图
fig, ax = plt.subplots(
    figsize=(5, 5),
    dpi=300
)


def plot_lorenz(
    ax,
    p,
    label,
    color
):

    p_vals = np.array(
        p.fillna(
            0.0
        )
    )


    if p_vals.sum() == 0:

        p_sorted = np.zeros_like(
            p_vals
        )


    else:

        p_sorted = np.sort(
            p_vals
        )[::-1]


    cum = np.cumsum(
        p_sorted
    )


    cum = (
        cum
        /
        (
            cum[-1]
            if
            cum[-1] > 0
            else
            1
        )
    )


    ax.plot(

        np.linspace(
            0,
            1,
            len(
                p_sorted
            )
        ),

        cum,

        label=label,

        linewidth=2,

        color=color
    )


# 画三条曲线
plot_lorenz(

    ax,

    high_dist_A.fillna(
        0
    ),

    labels[
        0
    ],

    colors[
        0
    ]
)


plot_lorenz(

    ax,

    high_dist_reg.fillna(
        0
    ),

    labels[
        1
    ],

    colors[
        1
    ]
)


plot_lorenz(

    ax,

    high_dist_no.fillna(
        0
    ),

    labels[
        2
    ],

    colors[
        2
    ]
)


# 45° 参考线
ax.plot(

    [
        0,
        1
    ],

    [
        0,
        1
    ],

    color='grey',

    linestyle='--',

    linewidth=1,

    zorder=0
)


# 标注 Entropy & Gini
txt = (

    f"Entropy (base/reg/no): "
    f"{entropy_A_high:.3f}/"
    f"{entropy_reg_high:.3f}/"
    f"{entropy_no_high:.3f}\n"

    f"Gini (base/reg/no): "
    f"{gini_A_high:.3f}/"
    f"{gini_reg_high:.3f}/"
    f"{gini_no_high:.3f}"
)


# 右对齐：x=1 表示最右侧，ha='right' 控制文字右边缘对齐
ax.text(

    0.98,

    0.02,

    txt,

    transform=ax.transAxes,

    fontsize=9,

    ha='right',

    bbox=dict(
        facecolor='white',
        alpha=0.8,
        edgecolor='none'
    )
)


# 轴标签与标题
ax.set_xlabel(
    'Cumulative fraction of POIs'
)


ax.set_ylabel(
    'Cumulative fraction of high-income visits'
)


ax.set_title(
    'Lorenz-like curves for high-income visit distribution'
)


ax.legend(
    frameon=False
)


ax.grid(
    alpha=0.3,
    linestyle='--'
)


plt.tight_layout()


plt.savefig(
    'figure4f.pdf',
    format='pdf',
    dpi=300,
    bbox_inches='tight',
    transparent=False,
    backend='pdf'
)


plt.show()