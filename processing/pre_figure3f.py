# -*- coding: utf-8 -*-

"""
Preprocessing for Fig.3f:
Stability of top-adjusted origins across budgets

Each cell reports:

    |T_b ∩ T_b'| / top_N

where T_b is the set of top-N CBGs ranked by reassigned visits:

    A_i(b) = 0.5 * sum_j |H_ij(b) - F_ij|

Private inputs:
    k_matrices_boston_family_budget.pkl

    matrices_A_D_S_Distribution/
        Other_Individual_and_Family_Services/
            flow_matrix.csv

Geographic input:
    geo_data/
        tl_2021_boston_msa_bg/
            tl_2021_boston_msa_bg.shp

Public output:
    figure3f_overlap.csv
"""

import pickle
import numpy as np
import pandas as pd
import geopandas as gpd


# ============================================================
# Basic settings
# ============================================================

city = "boston"
category = "Other Individual and Family Services"

cat_dir = (
    f"matrices_A_D_S_Distribution/"
    f"{category.replace(' ', '_')}"
)

k_matrix_path = (
    "k_matrices_boston_family_budget.pkl"
)

top_N = 100


# 主图展示的关键 budget

plot_ks_target = [
    0.01,
    0.05,
    0.10,
    0.20,
    0.30,
    0.40,
    0.50
]


output_path = (
    "figure3f_overlap.csv"
)


# ============================================================
# Load private data
# ============================================================

with open(
    k_matrix_path,
    "rb"
) as f:

    k_matrices = pickle.load(f)


flow_matrix = pd.read_csv(
    f"{cat_dir}/flow_matrix.csv",
    index_col=0
)


# ============================================================
# Use shapefile to determine GEOID zero-fill length
# ============================================================

boston_msa_cbg = gpd.read_file(
    "geo_data/"
    "tl_2021_boston_msa_bg/"
    "tl_2021_boston_msa_bg.shp"
)


boston_msa_cbg["GEOID"] = (
    boston_msa_cbg["GEOID"]
    .astype(str)
)


pad_len = int(
    boston_msa_cbg[
        "GEOID"
    ]
    .str
    .len()
    .max()
)


# ============================================================
# Align baseline matrix
# ============================================================

baseline = flow_matrix.copy()


baseline.index = [
    str(x).zfill(pad_len)
    for x in baseline.index
]


baseline.columns = [
    str(x)
    for x in baseline.columns
]


# 只保留 baseline 中有实际访问的 CBG

baseline = (
    baseline.loc[
        baseline.sum(axis=1) > 0
    ]
    .copy()
)


print(
    "Baseline shape:",
    baseline.shape
)


print(
    "Baseline total flow:",
    baseline.values.sum()
)


# ============================================================
# Helper functions
# ============================================================

def find_nearest_k_key(
    k_target,
    k_matrices
):
    """
    在 k_matrices 中寻找与目标 k 最接近的 key。
    key 可以是 float、int 或 str。
    """

    best_key = None
    best_diff = np.inf


    for key in k_matrices.keys():

        try:

            key_float = float(key)

            diff = abs(
                key_float
                - float(k_target)
            )


            if diff < best_diff:

                best_diff = diff
                best_key = key

        except Exception:

            continue


    if best_key is None:

        raise RuntimeError(
            f"Cannot find a valid key close to "
            f"k={k_target} in k_matrices."
        )


    return best_key


def standardize_matrix_index_columns(
    mat,
    pad_len
):
    """
    统一 k matrix 的 index 和 columns 格式，
    方便与 baseline 对齐。
    """

    mat = mat.copy()


    mat.index = [
        str(x).zfill(pad_len)
        for x in mat.index
    ]


    mat.columns = [
        str(x)
        for x in mat.columns
    ]


    return mat


def compute_origin_reassigned_visits(
    k_val,
    baseline,
    k_matrices,
    pad_len
):
    """
    计算每个 CBG 在 budget k 下承担的 reassigned visits:

        A_i(k) = 0.5 * sum_j |H_ij(k) - F_ij|

    返回:
        pd.Series
        index = GEOID
        value = reassigned visits
    """

    key = find_nearest_k_key(
        k_val,
        k_matrices
    )


    H = standardize_matrix_index_columns(
        k_matrices[key],
        pad_len
    )


    # 同时对齐 rows 和 columns

    H_aligned = H.reindex(
        index=baseline.index,
        columns=baseline.columns,
        fill_value=0
    )


    F_aligned = baseline.reindex(
        index=baseline.index,
        columns=baseline.columns,
        fill_value=0
    )


    diff_abs = (
        H_aligned
        - F_aligned
    ).abs()


    # 一次 reallocation 会产生一个减少项和一个增加项，因此除以 2

    reassigned_visits = (
        0.5
        * diff_abs.sum(axis=1)
    )


    reassigned_visits.index = (
        reassigned_visits.index
        .astype(str)
    )


    reassigned_visits.name = (
        f"k_{float(k_val):.2f}"
    )


    return reassigned_visits


# ============================================================
# Choose ks for figure
# ============================================================

all_available_ks = sorted(
    [
        float(k)
        for k in k_matrices.keys()
    ]
)


print(
    "Available k range:",
    min(all_available_ks),
    "to",
    max(all_available_ks)
)


print(
    "Number of available k values:",
    len(all_available_ks)
)


plot_ks = []


for target in plot_ks_target:

    nearest_k = min(
        all_available_ks,
        key=lambda x: abs(
            x - target
        )
    )


    plot_ks.append(
        nearest_k
    )


plot_ks = sorted(
    set(plot_ks)
)


print(
    "Using ks for Fig.3f:",
    plot_ks
)


# ============================================================
# Build top-N adjusted-origin sets
# ============================================================

top_sets = {}

flow_change_series = {}


for k in plot_ks:

    s = compute_origin_reassigned_visits(
        k_val=k,
        baseline=baseline,
        k_matrices=k_matrices,
        pad_len=pad_len
    )


    # 只保留真正发生调整的 CBG

    s = (
        s[
            s > 0
        ]
        .sort_values(
            ascending=False
        )
    )


    if len(s) < top_N:

        print(
            f"[WARNING] k={k:.2f}: "
            f"only {len(s)} CBGs have positive adjustment; "
            f"top_N={top_N}."
        )


        top_list = (
            s.index.tolist()
        )

    else:

        top_list = (
            s.index[
                :top_N
            ].tolist()
        )


    top_sets[k] = set(
        top_list
    )


    flow_change_series[k] = s


    max_change = (
        s.iloc[0]
        if len(s) > 0
        else 0
    )


    total_change = (
        s.sum()
    )


    print(
        f"k={k:.2f} | "
        f"top set size={len(top_sets[k])} | "
        f"max reassigned visits={max_change:.3f} | "
        f"total reassigned visits={total_change:.3f}"
    )


# ============================================================
# Compute pairwise overlap share
# ============================================================

K = len(
    plot_ks
)


overlap_share = np.zeros(
    (
        K,
        K
    ),
    dtype=float
)


for i, ki in enumerate(
    plot_ks
):

    for j, kj in enumerate(
        plot_ks
    ):

        A = top_sets[ki]

        B = top_sets[kj]


        inter = len(
            A.intersection(B)
        )


        # 正常情况下 denominator = top_N
        # 如果某个 k 下非零 adjusted CBG 不足 top_N，
        # 则使用较小集合大小作为分母

        denom = min(
            len(A),
            len(B),
            top_N
        )


        overlap_share[i, j] = (
            inter / denom
            if denom > 0
            else np.nan
        )


# ============================================================
# Prepare DataFrame
# ============================================================

ks_labels = [
    f"{int(round(k * 100))}%"
    for k in plot_ks
]


overlap_df = pd.DataFrame(
    overlap_share,
    index=ks_labels,
    columns=ks_labels
)


overlap_df.index.name = (
    "budget"
)


print(
    "\nTop-N overlap share matrix:"
)


print(
    overlap_df
)


# ============================================================
# Save public plotting data
# ============================================================

overlap_df.to_csv(
    output_path
)


print(
    f"\nSaved public Fig.3f data to: "
    f"{output_path}"
)


print(
    "\n========== FIG.3F PREPROCESSING COMPLETE =========="
)