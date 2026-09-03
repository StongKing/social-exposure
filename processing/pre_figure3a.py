# -*- coding: utf-8 -*-
"""
Preprocessing code for Fig.3a.

This script computes the observed baseline (gamma = 0)
used in Fig.3a.

Private input:
    matrices_A_D_S_Distribution/
        Other_Individual_and_Family_Services/
            flow_matrix.csv

Public inputs:
    matrices_A_D_S_Distribution/
        Other_Individual_and_Family_Services/
            distance_matrix.csv

    matrices_A_D_S_Distribution/
        cbg_income_level_distribution_boston_msa.csv

Output:
    figure3a_baseline.csv

The output contains the baseline values required by figure3a.py.
"""

import os
import numpy as np
import pandas as pd


# ============================================================
# User parameters
# ============================================================

city = "boston"
category = "Other Individual and Family Services"
cat_dir = f"matrices_A_D_S_Distribution/{category.replace(' ', '_')}"

flow_path = f"{cat_dir}/flow_matrix.csv"
distance_path = f"{cat_dir}/distance_matrix.csv"
income_path = r"matrices_A_D_S_Distribution/cbg_income_level_distribution_boston_msa.csv"

OUT_BASELINE_CSV = "figure3a_baseline.csv"

EPS = 1e-10

income_levels = [
    "low_income_pct",
    "lower_middle_income_pct",
    "upper_middle_income_pct",
    "high_income_pct"
]


# ============================================================
# Helper functions
# ============================================================

def normalize_key(x):
    if pd.isna(x):
        return ""
    s = str(x).strip()
    if s.endswith(".0") and s[:-2].isdigit():
        s = s[:-2]
    return s


def normalize_df_index_columns(df):
    df = df.copy()
    df.index = [normalize_key(x) for x in df.index]
    df.columns = [normalize_key(x) for x in df.columns]
    return df


def load_cbg_income_distribution(income_path, cbg_ids, income_levels):
    if not os.path.isfile(income_path):
        raise FileNotFoundError(f"Income file not found: {income_path}")

    income_df = pd.read_csv(income_path)

    if "GEOID" not in income_df.columns:
        raise ValueError(
            f"`GEOID` column not found. Columns: {list(income_df.columns)}"
        )

    income_df["GEOID"] = income_df["GEOID"].map(normalize_key)
    income_df = income_df.set_index("GEOID")

    missing_cols = [
        c for c in income_levels
        if c not in income_df.columns
    ]

    if missing_cols:
        raise ValueError(f"Missing income columns: {missing_cols}")

    cbg_ids = [normalize_key(x) for x in cbg_ids]

    P_df = income_df.reindex(cbg_ids)[income_levels].copy()
    P_df = P_df.apply(pd.to_numeric, errors="coerce")

    if P_df.isna().any().any():
        missing_ids = (
            P_df[P_df.isna().any(axis=1)]
            .index
            .tolist()[:10]
        )

        raise ValueError(
            f"Some CBGs have missing income distribution, "
            f"e.g. {missing_ids}"
        )

    P = P_df.values.astype(float)
    row_sum = P.sum(axis=1, keepdims=True)

    # If income shares are percentages, convert to proportions.
    if np.nanmedian(row_sum) > 1.5:
        P = P / 100.0
        row_sum = P.sum(axis=1, keepdims=True)

    row_sum[row_sum == 0] = np.nan
    P = P / row_sum
    P = np.nan_to_num(P, nan=0.0)

    P_df = pd.DataFrame(
        P,
        index=cbg_ids,
        columns=income_levels
    )

    print("[INCOME] loaded and normalized")
    print(f"[INCOME] n_CBGs = {P_df.shape[0]}")

    return P_df


def calculate_poi_income_distribution(H_df, P_df):
    """
    Q_j(H) = sum_i H_ij P_i / sum_i H_ij
    """
    H = H_df.values.astype(float)
    P = P_df.values.astype(float)

    poi_flow = H.sum(axis=0)

    Q = np.full(
        (H.shape[1], P.shape[1]),
        np.nan,
        dtype=float
    )

    positive_poi = poi_flow > EPS

    Q[positive_poi, :] = (
        H[:, positive_poi].T @ P
    ) / poi_flow[positive_poi, None]

    Q_df = pd.DataFrame(
        Q,
        index=H_df.columns,
        columns=P_df.columns
    )

    return Q_df


def calculate_social_exposure_matrix(H_df, P_df):
    """
    S_ij(H) = 1 - P_i dot Q_j(H)
    """
    Q_df = calculate_poi_income_distribution(
        H_df,
        P_df
    )

    P = P_df.values.astype(float)
    Q = Q_df.values.astype(float)

    S = 1.0 - (P @ Q.T)

    S_df = pd.DataFrame(
        S,
        index=H_df.index,
        columns=H_df.columns
    )

    return S_df, Q_df


def calculate_structural_potential_social_exposure(
    H_df,
    S_df
):
    """
    SPSE(H) = sum_{(i,j): H_ij > 0} S_ij(H)
    """
    H = H_df.values.astype(float)
    S = S_df.values.astype(float)

    positive_mask = H > EPS

    if positive_mask.sum() == 0:
        return np.nan

    return float(
        np.nansum(S[positive_mask])
    )


def calculate_flow_weighted_exposure(
    H_df,
    S_df
):
    """
    E_w(H) = sum_ij H_ij S_ij(H) / sum_ij H_ij
    """
    H = H_df.values.astype(float)
    S = S_df.values.astype(float)

    total_flow = np.nansum(H)

    if total_flow <= EPS:
        return np.nan

    return float(
        np.nansum(H * S) / total_flow
    )


def calculate_total_distance(
    H_df,
    D_df
):
    """
    Total travel distance = sum_ij H_ij D_ij
    """
    H = H_df.values.astype(float)
    D = D_df.values.astype(float)

    return float(
        np.nansum(H * D)
    )


# ============================================================
# Check required files
# ============================================================

if not os.path.isfile(flow_path):
    raise FileNotFoundError(
        f"Flow matrix not found: {flow_path}"
    )

if not os.path.isfile(distance_path):
    raise FileNotFoundError(
        f"Distance matrix not found: {distance_path}"
    )

if not os.path.isfile(income_path):
    raise FileNotFoundError(
        f"Income file not found: {income_path}"
    )


# ============================================================
# Load baseline matrices and compute original baseline
# ============================================================

flow_matrix = pd.read_csv(
    flow_path,
    index_col=0
)

distance_matrix = pd.read_csv(
    distance_path,
    index_col=0
)

flow_matrix = normalize_df_index_columns(
    flow_matrix
)

distance_matrix = normalize_df_index_columns(
    distance_matrix
)

distance_matrix = distance_matrix.reindex(
    index=flow_matrix.index,
    columns=flow_matrix.columns
)

if distance_matrix.isna().any().any():
    raise ValueError(
        "Distance matrix has missing values "
        "after alignment with flow matrix."
    )


# Same selection logic as the budget experiment:
# all POIs and CBGs with at least one observed visit.
poi_total_flow = flow_matrix.sum(axis=0)

selected_pois = (
    poi_total_flow
    .sort_values(ascending=False)
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

selected_cbgs = list(selected_cbgs)


F_df = (
    flow_matrix
    .loc[selected_cbgs, selected_pois]
    .astype(float)
)

D_df = (
    distance_matrix
    .loc[selected_cbgs, selected_pois]
    .astype(float)
)


P_df = load_cbg_income_distribution(
    income_path,
    selected_cbgs,
    income_levels
)

P_df = P_df.reindex(
    F_df.index
)


S_base_df, _ = calculate_social_exposure_matrix(
    F_df,
    P_df
)


baseline_spse = (
    calculate_structural_potential_social_exposure(
        F_df,
        S_base_df
    )
)

baseline_flow_weighted_exposure = (
    calculate_flow_weighted_exposure(
        F_df,
        S_base_df
    )
)

baseline_total_distance = (
    calculate_total_distance(
        F_df,
        D_df
    )
)

baseline_total_flow = float(
    F_df.values.sum()
)


# ============================================================
# Print original baseline
# ============================================================

print(
    "\n========== ORIGINAL BASELINE =========="
)

print(
    f"n_CBGs: {F_df.shape[0]}"
)

print(
    f"n_POIs: {F_df.shape[1]}"
)

print(
    f"baseline total flow: "
    f"{baseline_total_flow:.6f}"
)

print(
    f"baseline SPSE: "
    f"{baseline_spse:.6f}"
)

print(
    f"baseline flow-weighted exposure: "
    f"{baseline_flow_weighted_exposure:.6f}"
)

print(
    f"baseline total travel distance: "
    f"{baseline_total_distance:.6f}"
)


# ============================================================
# Save baseline for public plotting code
# ============================================================

baseline_df = pd.DataFrame([{
    "city": city,
    "category": category,
    "n_cbgs": F_df.shape[0],
    "n_pois": F_df.shape[1],
    "baseline_total_flow": baseline_total_flow,
    "baseline_spse": baseline_spse,
    "baseline_flow_weighted_exposure":
        baseline_flow_weighted_exposure,
    "baseline_total_distance":
        baseline_total_distance
}])


baseline_df.to_csv(
    OUT_BASELINE_CSV,
    index=False
)


print(
    f"\n[SAVE] {OUT_BASELINE_CSV}"
)

print(
    "\n========== PREPROCESSING COMPLETE =========="
)