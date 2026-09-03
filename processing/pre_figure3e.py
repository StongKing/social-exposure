# -*- coding: utf-8 -*-
"""
Preprocessing for Fig.3e:
Distribution of origin-level reassigned visits across budgets

For each active origin i and budget k:

    A_i(k) = 0.5 * sum_j |H_ij(k) - F_ij|

Private inputs:
    matrices_A_D_S_Distribution/
        Other_Individual_and_Family_Services/
            flow_matrix.csv

    k_matrices_boston_family_budget.pkl

Geographic input:
    geo_data/
        tl_2021_boston_msa_bg/
            tl_2021_boston_msa_bg.shp

Public output:
    figure3e_data.csv
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import pickle


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

plot_ks = [
    0.01,
    0.05,
    0.20
]

output_path = (
    "figure3e_data.csv"
)


# ============================================================
# Load data
# ============================================================

flow_matrix = pd.read_csv(
    f"{cat_dir}/flow_matrix.csv",
    index_col=0
)


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
    ].str.len().max()
)


with open(
    k_matrix_path,
    "rb"
) as f:

    k_matrices = pickle.load(f)


# ============================================================
# Build baseline matrix
# ============================================================

poi_total_flow = (
    flow_matrix.sum(axis=0)
)


poi_num = (
    flow_matrix.shape[1]
)


selected_pois = (
    poi_total_flow
    .sort_values(ascending=False)
    .head(poi_num)
    .index
    .tolist()
)


selected_cbgs = set()


for poi in selected_pois:

    cbgs_with_flow = (
        flow_matrix.index[
            flow_matrix[poi] > 0
        ]
        .tolist()
    )

    selected_cbgs.update(
        cbgs_with_flow
    )


selected_cbgs = list(
    selected_cbgs
)


baseline = (
    flow_matrix.loc[
        selected_cbgs,
        selected_pois
    ]
    .copy()
)


baseline.index = [
    str(x).zfill(pad_len)
    for x in baseline.index
]


baseline.columns = [
    str(x)
    for x in baseline.columns
]


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

def find_best_key_in_kmat(
    k_val,
    k_matrices
):
    """
    Find the key in k_matrices closest to k_val.
    """

    best = None
    bestd = 1e9


    for kk in k_matrices.keys():

        try:

            d = abs(
                float(kk)
                - float(k_val)
            )

            if d < bestd:

                bestd = d
                best = kk

        except Exception:

            pass


    return best


def compute_reassigned_visits_per_origin(
    F_df,
    H_df
):
    """
    Compute origin-level reassigned visits:

        A_i(k) = 0.5 * sum_j |H_ij(k) - F_ij|

    The 0.5 factor avoids double-counting one reassignment as both
    a decrease at one destination and an increase at another destination.
    """

    F = F_df.copy()
    H = H_df.copy()


    H_aligned = H.reindex(
        index=F.index,
        columns=F.columns,
        fill_value=0.0
    )


    F_aligned = F.reindex(
        index=H_aligned.index,
        columns=H_aligned.columns,
        fill_value=0.0
    )


    diff = (
        H_aligned
        - F_aligned
    ).abs()


    per_origin = (
        0.5
        * diff.sum(axis=1)
    )


    return per_origin.sort_values(
        ascending=False
    )


def get_H_for_k(
    k_val,
    k_matrices,
    baseline,
    pad_len
):
    """
    Load and align optimized matrix for a given k.
    """

    H = None


    if k_val in k_matrices:

        H = k_matrices[k_val]

    else:

        best = find_best_key_in_kmat(
            k_val,
            k_matrices
        )

        if best is not None:

            H = k_matrices[best]


    if H is None:

        return None


    H_local = H.copy()


    try:

        H_local.index = [
            str(x).zfill(pad_len)
            for x in H_local.index
        ]

    except Exception:

        H_local.index = (
            H_local.index.astype(str)
        )


    H_local.columns = [
        str(x)
        for x in H_local.columns
    ]


    H_local = H_local.reindex(
        index=baseline.index,
        columns=baseline.columns,
        fill_value=0.0
    )


    return H_local


# ============================================================
# Compute origin-level reassigned visits
# ============================================================

figure3e_data = pd.DataFrame(
    index=baseline.index
)


figure3e_data.index.name = (
    "GEOID"
)


for k in plot_ks:

    H_local = get_H_for_k(
        k_val=k,
        k_matrices=k_matrices,
        baseline=baseline,
        pad_len=pad_len
    )


    if H_local is None:

        print(
            f"[WARN] skip k={k}, "
            f"matrix not found"
        )

        continue


    per_origin = (
        compute_reassigned_visits_per_origin(
            baseline,
            H_local
        )
    )


    colname = (
        f"reassigned_k_{k:.2f}"
    )


    figure3e_data[colname] = (
        per_origin.reindex(
            figure3e_data.index
        ).values
    )


# ============================================================
# Save public data
# ============================================================

figure3e_data.to_csv(
    output_path
)


print(
    "\n========== FIG.3E PREPROCESSING =========="
)


print(
    f"Number of active origins: "
    f"{len(figure3e_data)}"
)


for k in plot_ks:

    colname = (
        f"reassigned_k_{k:.2f}"
    )


    if colname not in figure3e_data.columns:

        continue


    vals = (
        figure3e_data[
            colname
        ].values
    )


    print(
        f"k={k:.2f}: "
        f"total reassigned visits="
        f"{np.sum(vals):.6f}, "
        f"changed origins="
        f"{int((vals != 0).sum())}, "
        f"unchanged origins="
        f"{int((vals == 0).sum())}"
    )


print(
    f"\nSaved public Fig.3e data to: "
    f"{output_path}"
)


print(
    "=========================================="
)