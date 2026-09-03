# -*- coding: utf-8 -*-
"""
Preprocessing for Fig. 2b.

The original flow_matrix.csv is private.

This script extracts only the baseline information required to
reproduce Fig. 2b:

1. Baseline-positive POI IDs.
2. Baseline system-level metrics.
3. Baseline CBG-level structural exposure contributions.

Outputs
-------
figure2b_active_pois.csv
figure2b_baseline_system.csv
figure2b_baseline_cbg_contribution.csv
"""

import os
import numpy as np
import pandas as pd


# ============================================================
# 0. Configuration
# ============================================================

naics_code = "624190"

cat_dir = r"matrices_A_D_S_Distribution/Other_Individual_and_Family_Services"

# PRIVATE DATA
flow_path = os.path.join(cat_dir, "flow_matrix.csv")

# PUBLIC DATA
distance_path = os.path.join(cat_dir, "distance_matrix.csv")

static_h_path = os.path.join(
    cat_dir,
    f"H_opt_df_static_{naics_code}.pkl"
)

dynamic_h_path = os.path.join(
    cat_dir,
    f"H_opt_df_dynamic_{naics_code}.pkl"
)

income_path = (
    r"matrices_A_D_S_Distribution/"
    r"cbg_income_level_distribution_boston_msa.csv"
)

income_levels = [
    "low_income_pct",
    "lower_middle_income_pct",
    "upper_middle_income_pct",
    "high_income_pct",
]

EPS = 1e-4


# Output files
ACTIVE_POI_FILE = "figure2b_active_pois.csv"

BASELINE_SYSTEM_FILE = (
    "figure2b_baseline_system.csv"
)

BASELINE_CBG_FILE = (
    "figure2b_baseline_cbg_contribution.csv"
)


# ============================================================
# 1. Helpers
# ============================================================

def normalize_key(x):
    s = str(x).strip()

    if s.endswith(".0"):
        s = s[:-2]

    return s


def load_csv_matrix(path):

    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"File not found: {path}"
        )

    df = pd.read_csv(
        path,
        index_col=0
    )

    df.index = [
        normalize_key(x)
        for x in df.index
    ]

    df.columns = [
        normalize_key(x)
        for x in df.columns
    ]

    df = (
        df
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0.0)
    )

    return df


def load_pkl_matrix(path):

    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"File not found: {path}"
        )

    obj = pd.read_pickle(path)

    if isinstance(obj, pd.DataFrame):
        df = obj.copy()

    elif isinstance(obj, np.ndarray):
        df = pd.DataFrame(obj)

    else:
        raise TypeError(
            f"Unsupported pkl object type: {type(obj)}"
        )

    df.index = [
        normalize_key(x)
        for x in df.index
    ]

    df.columns = [
        normalize_key(x)
        for x in df.columns
    ]

    df = (
        df
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0.0)
    )

    return df


def align_matrices(*dfs):

    common_rows = set(dfs[0].index)
    common_cols = set(dfs[0].columns)

    for df in dfs[1:]:

        common_rows &= set(df.index)
        common_cols &= set(df.columns)

    common_rows = sorted(common_rows)
    common_cols = sorted(common_cols)

    if (
        len(common_rows) == 0
        or
        len(common_cols) == 0
    ):
        raise ValueError(
            "No common rows or columns after alignment."
        )

    aligned = [
        df.loc[
            common_rows,
            common_cols
        ].copy()
        for df in dfs
    ]

    return (
        aligned,
        common_rows,
        common_cols
    )


def load_cbg_income_distribution(
        income_path,
        cbg_ids,
        income_levels):

    if not os.path.isfile(income_path):

        raise FileNotFoundError(
            f"Income file not found: {income_path}"
        )

    income_df = pd.read_csv(
        income_path
    )

    if "GEOID" not in income_df.columns:

        raise ValueError(
            "`GEOID` column not found. "
            f"Columns: {list(income_df.columns)}"
        )

    income_df["GEOID"] = (
        income_df["GEOID"]
        .map(normalize_key)
    )

    income_df = (
        income_df
        .set_index("GEOID")
    )

    missing_cols = [
        c
        for c in income_levels
        if c not in income_df.columns
    ]

    if missing_cols:

        raise ValueError(
            f"Missing income columns: {missing_cols}"
        )

    P_df = (
        income_df
        .reindex(cbg_ids)[income_levels]
        .copy()
    )

    P_df = P_df.apply(
        pd.to_numeric,
        errors="coerce"
    )

    if P_df.isna().any().any():

        missing_ids = (
            P_df[
                P_df.isna().any(axis=1)
            ]
            .index
            .tolist()[:10]
        )

        raise ValueError(
            "Some CBGs have missing income "
            f"distribution, e.g. {missing_ids}"
        )

    P = P_df.values.astype(float)

    row_sum = P.sum(
        axis=1,
        keepdims=True
    )

    if np.nanmedian(row_sum) > 1.5:

        P = P / 100.0

        row_sum = P.sum(
            axis=1,
            keepdims=True
        )

    row_sum[
        row_sum == 0
    ] = np.nan

    P = P / row_sum

    P = np.nan_to_num(
        P,
        nan=0.0
    )

    P_df = pd.DataFrame(
        P,
        index=cbg_ids,
        columns=income_levels
    )

    return P_df


# ============================================================
# 2. Exposure calculation
# ============================================================

def calculate_poi_income_distribution(
        H_df,
        P_df):

    H = H_df.values.astype(float)
    P = P_df.values.astype(float)

    poi_flow = H.sum(axis=0)

    Q = np.full(
        (
            H.shape[1],
            P.shape[1]
        ),
        np.nan,
        dtype=float
    )

    positive_poi = (
        poi_flow > EPS
    )

    Q[
        positive_poi,
        :
    ] = (
        H[:, positive_poi].T @ P
    ) / poi_flow[
        positive_poi,
        None
    ]

    return pd.DataFrame(
        Q,
        index=H_df.columns,
        columns=P_df.columns
    )


def calculate_social_exposure_matrix(
        H_df,
        P_df):

    Q_df = (
        calculate_poi_income_distribution(
            H_df,
            P_df
        )
    )

    P = P_df.values.astype(float)
    Q = Q_df.values.astype(float)

    S = 1.0 - (
        P @ Q.T
    )

    S_df = pd.DataFrame(
        S,
        index=H_df.index,
        columns=H_df.columns
    )

    return S_df, Q_df


# ============================================================
# 3. Evaluation metrics
# ============================================================

def calculate_total_structural_exposure(
        H_df,
        S_df):

    H = H_df.values.astype(float)
    S = S_df.values.astype(float)

    positive = H > EPS

    return float(
        np.nansum(
            S[positive]
        )
    )


def calculate_cbg_structural_contribution(
        H_df,
        S_df):

    H = H_df.values.astype(float)
    S = S_df.values.astype(float)

    positive = H > EPS

    contrib = np.nansum(
        np.where(
            positive,
            S,
            0.0
        ),
        axis=1
    )

    return pd.Series(
        contrib,
        index=H_df.index,
        name="baseline_structural_contrib",
    )


def calculate_total_distance(
        H_df,
        D_df):

    return float(
        np.nansum(
            H_df.values.astype(float)
            *
            D_df.values.astype(float)
        )
    )


def calculate_avg_distance_per_visit(
        H_df,
        D_df):

    total_flow = float(
        np.nansum(
            H_df.values.astype(float)
        )
    )

    total_distance = (
        calculate_total_distance(
            H_df,
            D_df
        )
    )

    if total_flow <= EPS:
        return np.nan

    return (
        total_distance
        /
        total_flow
    )


def calculate_flow_weighted_exposure(
        H_df,
        S_df):

    H = H_df.values.astype(float)
    S = S_df.values.astype(float)

    total_flow = np.nansum(H)

    if total_flow <= EPS:
        return np.nan

    return float(
        np.nansum(
            H * S
        )
        /
        total_flow
    )


# ============================================================
# 4. Load matrices
# ============================================================

flow_matrix = load_csv_matrix(
    flow_path
)

distance_matrix = load_csv_matrix(
    distance_path
)

H_static = load_pkl_matrix(
    static_h_path
)

H_dynamic = load_pkl_matrix(
    dynamic_h_path
)


# ============================================================
# 5. Align exactly as in original Fig. 2b
# ============================================================

aligned, cbg_ids, poi_ids_all = (
    align_matrices(
        flow_matrix,
        distance_matrix,
        H_static,
        H_dynamic,
    )
)

(
    flow_matrix,
    distance_matrix,
    H_static,
    H_dynamic
) = aligned


print(
    "\n========== ORIGINAL ALIGNED DOMAIN =========="
)

print(
    f"n_CBGs: {len(cbg_ids)}"
)

print(
    f"n_POIs before filtering: "
    f"{len(poi_ids_all)}"
)


# ============================================================
# 6. Determine baseline-positive POIs
# ============================================================

baseline_poi_flow = (
    flow_matrix.sum(axis=0)
)

active_poi_ids = (
    baseline_poi_flow[
        baseline_poi_flow > EPS
    ]
    .index
    .tolist()
)


print(
    "\n========== ANALYSIS POI DOMAIN =========="
)

print(
    "Baseline-positive POIs retained: "
    f"{len(active_poi_ids)}"
)


assert len(active_poi_ids) == 44, (
    "Expected 44 baseline-positive POIs, "
    f"but found {len(active_poi_ids)}."
)


# Restrict baseline matrices to those 44 POIs.
flow_matrix = (
    flow_matrix.loc[
        :,
        active_poi_ids
    ].copy()
)

distance_matrix = (
    distance_matrix.loc[
        :,
        active_poi_ids
    ].copy()
)


# ============================================================
# 7. Load income distribution
# ============================================================

P_df = load_cbg_income_distribution(
    income_path=income_path,
    cbg_ids=cbg_ids,
    income_levels=income_levels,
)


# ============================================================
# 8. Calculate baseline exposure
# ============================================================

S0, Q0 = (
    calculate_social_exposure_matrix(
        flow_matrix,
        P_df
    )
)


# ============================================================
# 9. Baseline system metrics
# ============================================================

base_avg_dist = (
    calculate_avg_distance_per_visit(
        flow_matrix,
        distance_matrix
    )
)

base_structural = (
    calculate_total_structural_exposure(
        flow_matrix,
        S0
    )
)

base_fw_exp = (
    calculate_flow_weighted_exposure(
        flow_matrix,
        S0
    )
)

base_n_positive_links = int(
    (
        flow_matrix.values > EPS
    ).sum()
)


baseline_system_df = pd.DataFrame(
    [
        {
            "scenario": "Baseline",

            "avg_distance_per_visit":
                base_avg_dist,

            "structural_exposure":
                base_structural,

            "flow_weighted_exposure":
                base_fw_exp,

            "n_positive_links":
                base_n_positive_links,
        }
    ]
)


# ============================================================
# 10. Baseline CBG structural contribution
# ============================================================

cbg_base_contrib = (
    calculate_cbg_structural_contribution(
        flow_matrix,
        S0
    )
)


baseline_cbg_df = pd.DataFrame(
    {
        "cbg": cbg_base_contrib.index,
        "baseline_structural_contrib":
            cbg_base_contrib.values,
    }
)


# ============================================================
# 11. Save derived baseline data
# ============================================================

active_poi_df = pd.DataFrame(
    {
        "poi": active_poi_ids
    }
)


active_poi_df.to_csv(
    ACTIVE_POI_FILE,
    index=False
)

baseline_system_df.to_csv(
    BASELINE_SYSTEM_FILE,
    index=False
)

baseline_cbg_df.to_csv(
    BASELINE_CBG_FILE,
    index=False
)


# ============================================================
# 12. Summary
# ============================================================

print(
    "\n========== SAVED FIG. 2b BASELINE DATA =========="
)

print(
    f"Saved: {ACTIVE_POI_FILE}"
)

print(
    f"Saved: {BASELINE_SYSTEM_FILE}"
)

print(
    f"Saved: {BASELINE_CBG_FILE}"
)


print(
    "\nBaseline system metrics:"
)

print(
    baseline_system_df
    .round(6)
    .to_string(index=False)
)


print(
    "\nNumber of baseline CBG contributions: "
    f"{len(baseline_cbg_df)}"
)