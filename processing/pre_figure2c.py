# -*- coding: utf-8 -*-
"""
Preprocessing for Fig. 2c.

The original flow_matrix.csv is private.

This script generates only the derived data required to reproduce
Fig. 2c without releasing the original CBG--POI flow matrix.

Outputs
-------
figure2c_new_links.csv
figure2c_quadrant.csv
figure2c_origin_metrics.csv
figure2c_poi_metrics.csv
"""

import os
import glob
import pickle
import numpy as np
import pandas as pd


# ============================================================
# 0. USER SETTINGS
# ============================================================

MATRIX_ROOT = r"matrices_A_D_S_Distribution"


SELECTED_POI_CODE = "624190"

DMAX_KM = 50

# If distance matrix is already in km, keep 1.0.
# If distance matrix is in meters, use 1/1000.
DISTANCE_SCALE = 1.0


# Output plotting-data files
OUTPUT_NEW_LINKS = "figure2c_new_links.csv"
OUTPUT_QUADRANT = "figure2c_quadrant.csv"
OUTPUT_ORIGIN = "figure2c_origin_metrics.csv"
OUTPUT_POI = "figure2c_poi_metrics.csv"


# ============================================================
# 1. Metadata
# ============================================================

income_levels = [
    "low_income_pct",
    "lower_middle_income_pct",
    "upper_middle_income_pct",
    "high_income_pct",
]


# ============================================================
# 2. Basic helpers
# ============================================================

def normalize_geoid(x):

    if pd.isna(x):
        return None

    try:
        return str(int(float(x)))

    except Exception:
        return str(x).strip()


def read_matrix_csv(path, distance=False):

    df = pd.read_csv(
        path,
        header=0,
        index_col=0
    )

    df.index = (
        df.index
        .astype(str)
        .map(normalize_geoid)
    )

    df.columns = (
        df.columns
        .astype(str)
        .str.strip()
    )

    df = df.apply(
        pd.to_numeric,
        errors="coerce"
    )

    if df.index.duplicated().any():
        df = df.groupby(level=0).sum()

    if distance:
        df = df * DISTANCE_SCALE

    return df


def read_hopt_pickle(path):

    with open(path, "rb") as f:
        H = pickle.load(f)

    if not isinstance(H, pd.DataFrame):
        H = pd.DataFrame(H)

    H = H.copy()

    H.index = (
        H.index
        .astype(str)
        .map(normalize_geoid)
    )

    H.columns = (
        H.columns
        .astype(str)
        .str.strip()
    )

    H = (
        H
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0)
    )

    if H.index.duplicated().any():
        H = H.groupby(level=0).sum()

    return H


def weighted_mean(x, w):

    x = np.asarray(
        x,
        dtype=float
    )

    w = np.asarray(
        w,
        dtype=float
    )

    mask = (
        np.isfinite(x)
        &
        np.isfinite(w)
        &
        (w > 0)
    )

    if mask.sum() == 0:
        return np.nan

    return float(
        np.sum(
            x[mask] * w[mask]
        )
        /
        np.sum(w[mask])
    )


# ============================================================
# 3. Locate files
# ============================================================

def find_case_dir_by_poi_code(poi_code):
    POI = "Other_Individual_and_Family_Services"
    candidates = sorted(
        glob.glob(
            os.path.join(
                MATRIX_ROOT,
                POI,
                f"H_opt_df_dynamic_{poi_code}.pkl"
            ),
            recursive=True,
        )
    )
    print(candidates)

    if len(candidates) == 0:

        raise FileNotFoundError(
            f"Cannot find "
            f"H_opt_df_dynamic_{poi_code}.pkl "
            f"under {MATRIX_ROOT}\{POI}"
        )

    return (
        os.path.dirname(candidates[0]),
        candidates[0]
    )


# ============================================================
# 4. Income distribution
# ============================================================

def load_income_distribution():

    income_path = "cbg_income_level_distribution_boston_msa.csv"

    print(
        f"[LOAD income] {income_path}"
    )

    df = pd.read_csv(
        income_path
    )

    if "GEOID" not in df.columns:

        raise ValueError(
            "Income file must contain GEOID column."
        )

    missing = [
        c
        for c in income_levels
        if c not in df.columns
    ]

    if missing:

        raise ValueError(
            f"Income file missing columns: {missing}"
        )

    df["GEOID_str"] = (
        df["GEOID"]
        .apply(normalize_geoid)
    )

    P = (
        df
        .set_index("GEOID_str")[income_levels]
        .copy()
    )

    P = (
        P
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0)
    )

    if P.index.duplicated().any():
        P = P.groupby(level=0).mean()

    row_sum = (
        P.sum(axis=1)
        .replace(0, np.nan)
    )

    P = (
        P
        .div(row_sum, axis=0)
        .fillna(0)
    )

    return P


# ============================================================
# 5. Baseline exposure computation
# ============================================================

def compute_all_pair_unmasked_exposure(
        flow_df,
        P_df):

    F = flow_df.copy()

    F.index = (
        F.index
        .astype(str)
        .map(normalize_geoid)
    )

    F.columns = (
        F.columns
        .astype(str)
        .str.strip()
    )

    F = (
        F
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0)
    )


    common_cbgs = sorted(
        set(F.index)
        &
        set(P_df.index)
    )

    if len(common_cbgs) == 0:

        raise ValueError(
            "No common CBGs between flow matrix "
            "and income distribution."
        )


    F = F.loc[
        common_cbgs
    ].copy()

    P = P_df.loc[
        common_cbgs,
        income_levels
    ].copy()


    poi_total_flow = F.sum(axis=0)

    valid_pois = (
        poi_total_flow[
            poi_total_flow > 0
        ]
        .index
        .tolist()
    )


    if len(valid_pois) == 0:

        raise ValueError(
            "No POI has positive total flow."
        )


    F = F[
        valid_pois
    ].copy()


    F_values = F.values.astype(float)
    P_values = P.values.astype(float)

    poi_total_flow = (
        F_values.sum(axis=0)
    )


    Q_values = (
        F_values.T
        @
        P_values
    ) / poi_total_flow[:, None]


    Q_sum = Q_values.sum(
        axis=1,
        keepdims=True
    )


    Q_values = np.divide(
        Q_values,
        Q_sum,
        out=np.zeros_like(Q_values),
        where=Q_sum > 0
    )


    S_values = (
        1.0
        -
        (
            P_values
            @
            Q_values.T
        )
    )


    S = pd.DataFrame(
        S_values,
        index=F.index,
        columns=F.columns
    )


    Q = pd.DataFrame(
        Q_values,
        index=F.columns,
        columns=income_levels
    )


    return (
        S,
        F,
        Q
    )


# ============================================================
# 6. Generate Fig. 2c plotting data
# ============================================================

def build_plotting_data():

    # --------------------------------------------------------
    # Locate case
    # --------------------------------------------------------

    case_dir, h_path = (
        find_case_dir_by_poi_code(
            SELECTED_POI_CODE
        )
    )


    print(
        f"[CASE] {case_dir}"
    )

    print(
        f"[HOPT] {h_path}"
    )


    flow_path = os.path.join(
        case_dir,
        "flow_matrix.csv"
    )

    dist_path = os.path.join(
        case_dir,
        "distance_matrix.csv"
    )


    if not os.path.isfile(flow_path):
        raise FileNotFoundError(
            flow_path
        )

    if not os.path.isfile(dist_path):
        raise FileNotFoundError(
            dist_path
        )


    # --------------------------------------------------------
    # Load input data
    # --------------------------------------------------------

    P_df = (
        load_income_distribution()
    )

    # PRIVATE
    F_raw = read_matrix_csv(
        flow_path,
        distance=False
    )

    # PUBLIC
    D_raw = read_matrix_csv(
        dist_path,
        distance=True
    )

    # PUBLIC
    H_opt = read_hopt_pickle(
        h_path
    )


    # --------------------------------------------------------
    # Baseline exposure
    # --------------------------------------------------------

    S0_full, F_income, Q0_full = (
        compute_all_pair_unmasked_exposure(
            F_raw,
            P_df
        )
    )


    # --------------------------------------------------------
    # Align baseline domain
    # --------------------------------------------------------

    common_rows = sorted(
        set(F_income.index)
        &
        set(D_raw.index)
        &
        set(S0_full.index)
    )


    common_cols = sorted(
        set(F_income.columns)
        &
        set(D_raw.columns)
        &
        set(S0_full.columns)
    )


    F_full = F_income.loc[
        common_rows,
        common_cols
    ].copy()


    D_full = D_raw.loc[
        common_rows,
        common_cols
    ].copy()


    S0_full = S0_full.loc[
        common_rows,
        common_cols
    ].copy()


    # --------------------------------------------------------
    # Construct optimized evaluation matrix
    # --------------------------------------------------------

    H_eval = (
        F_full
        .copy()
        .astype(float)
    )


    rows_h = sorted(
        set(H_opt.index)
        &
        set(H_eval.index)
        &
        set(P_df.index)
    )


    cols_h = sorted(
        set(H_opt.columns)
        &
        set(H_eval.columns)
    )


    if (
        len(rows_h) == 0
        or
        len(cols_h) == 0
    ):

        raise ValueError(
            "No common rows/columns between H_opt "
            "and baseline flow matrix."
        )


    H_eval.loc[
        rows_h,
        cols_h
    ] = H_opt.loc[
        rows_h,
        cols_h
    ]


    F_dom = F_full.loc[
        rows_h,
        cols_h
    ].copy()


    H_dom = H_eval.loc[
        rows_h,
        cols_h
    ].copy()


    # --------------------------------------------------------
    # Full baseline diagnostic domain
    # --------------------------------------------------------

    Fv_full = (
        F_full.values.astype(float)
    )

    Dv_full = (
        D_full.values.astype(float)
    )

    S0v_full = (
        S0_full.values.astype(float)
    )

    Hv_full = (
        H_eval.values.astype(float)
    )


    valid_full = (
        np.isfinite(Fv_full)
        &
        np.isfinite(Dv_full)
        &
        np.isfinite(S0v_full)
        &
        np.isfinite(Hv_full)
    )


    distance_feasible_full = (
        valid_full
        &
        (Dv_full >= 0)
        &
        (Dv_full <= DMAX_KM)
    )


    active_ref_full = (
        distance_feasible_full
        &
        (Fv_full > 0)
    )


    unused_feasible_full = (
        distance_feasible_full
        &
        (Fv_full <= 0)
    )


    if active_ref_full.sum() == 0:

        raise ValueError(
            "No active feasible reference links "
            "in the full diagnostic domain."
        )


    # --------------------------------------------------------
    # Active-link baseline reference
    # --------------------------------------------------------

    active_w_s = weighted_mean(
        S0v_full[
            active_ref_full
        ],
        Fv_full[
            active_ref_full
        ]
    )


    active_w_d = weighted_mean(
        Dv_full[
            active_ref_full
        ],
        Fv_full[
            active_ref_full
        ]
    )


    delta_s_full = (
        S0v_full
        -
        active_w_s
    )


    delta_d_full = (
        Dv_full
        -
        active_w_d
    )


    # --------------------------------------------------------
    # Fig. 1d opportunity domain
    # --------------------------------------------------------

    fig1d_opportunity_full = (
        unused_feasible_full
        &
        (delta_s_full > 0)
        &
        (delta_d_full <= 0)
    )


    # --------------------------------------------------------
    # Newly activated links
    # --------------------------------------------------------

    newly_activated_full = (
        valid_full
        &
        (Fv_full <= 0)
        &
        (Hv_full > 0)
    )


    # --------------------------------------------------------
    # Classify new corridors
    # --------------------------------------------------------

    def classify_link(ds, dd):

        if (
            ds > 0
            and
            dd <= 0
        ):
            return (
                "Higher exposure, no farther"
            )

        if (
            ds > 0
            and
            dd > 0
        ):
            return (
                "Higher exposure, farther"
            )

        if (
            ds <= 0
            and
            dd <= 0
        ):
            return (
                "Lower exposure, no farther"
            )

        return (
            "Lower exposure, farther"
        )


    full_row_ids = (
        F_full.index.tolist()
    )

    full_col_ids = (
        F_full.columns.tolist()
    )


    records = []


    for i, geoid in enumerate(
        full_row_ids
    ):

        for j, poi in enumerate(
            full_col_ids
        ):

            if not newly_activated_full[
                i,
                j
            ]:
                continue


            ds = float(
                delta_s_full[
                    i,
                    j
                ]
            )

            dd = float(
                delta_d_full[
                    i,
                    j
                ]
            )


            records.append(
                {
                    "GEOID":
                        geoid,

                    "poi_id":
                        poi,

                    "new_flow":
                        float(
                            Hv_full[
                                i,
                                j
                            ]
                        ),

                    "link_tradeoff":
                        classify_link(
                            ds,
                            dd
                        ),

                    "is_fig1d_opportunity":
                        bool(
                            fig1d_opportunity_full[
                                i,
                                j
                            ]
                        ),
                }
            )


    new_links_df = pd.DataFrame(
        records,
        columns=[
            "GEOID",
            "poi_id",
            "new_flow",
            "link_tradeoff",
            "is_fig1d_opportunity",
        ]
    )


    # --------------------------------------------------------
    # New-flow trade-off composition
    # --------------------------------------------------------

    tradeoff_order = [
        "Higher exposure, no farther",
        "Higher exposure, farther",
        "Lower exposure, no farther",
        "Lower exposure, farther",
    ]


    quadrant_rows = []


    denom = (
        float(
            new_links_df[
                "new_flow"
            ].sum()
        )
        if len(new_links_df) > 0
        else 0.0
    )


    for label in tradeoff_order:

        if len(new_links_df) > 0:

            sub = new_links_df[
                new_links_df[
                    "link_tradeoff"
                ]
                ==
                label
            ]

        else:

            sub = pd.DataFrame()


        if len(sub) > 0:

            val = float(
                sub[
                    "new_flow"
                ].sum()
            )

        else:

            val = 0.0


        if denom > 0:

            share_pct = (
                val
                /
                denom
                *
                100.0
            )

        else:

            share_pct = np.nan


        quadrant_rows.append(
            {
                "quadrant":
                    label,

                "new_flow":
                    val,

                "share_pct":
                    share_pct,

                "n_links":
                    int(
                        len(sub)
                    ),
            }
        )


    quadrant_df = pd.DataFrame(
        quadrant_rows
    )


    # --------------------------------------------------------
    # Origin-level shifted flow
    # --------------------------------------------------------

    diff = (
        H_dom
        -
        F_dom
    )


    origin_rows = []


    for geoid in rows_h:

        row_diff = diff.loc[
            geoid
        ]

        shifted = (
            0.5
            *
            float(
                np.abs(
                    row_diff
                ).sum()
            )
        )


        origin_rows.append(
            {
                "GEOID":
                    geoid,

                "shifted_flow":
                    shifted,
            }
        )


    origin_metrics = pd.DataFrame(
        origin_rows
    )


    # --------------------------------------------------------
    # POI-level changed flow
    # --------------------------------------------------------

    poi_rows = []


    for poi in cols_h:

        diff_col = diff[
            poi
        ]

        total_changed_flow = float(
            np.abs(
                diff_col
            ).sum()
        )


        poi_rows.append(
            {
                "poi_id":
                    poi,

                "total_changed_flow":
                    total_changed_flow,
            }
        )


    poi_metrics = pd.DataFrame(
        poi_rows
    )


    return (
        new_links_df,
        quadrant_df,
        origin_metrics,
        poi_metrics,
    )


# ============================================================
# 7. Save Fig. 2c plotting data
# ============================================================

def main():

    (
        new_links_df,
        quadrant_df,
        origin_metrics,
        poi_metrics,
    ) = build_plotting_data()


    new_links_df.to_csv(
        OUTPUT_NEW_LINKS,
        index=False
    )


    quadrant_df.to_csv(
        OUTPUT_QUADRANT,
        index=False
    )


    origin_metrics.to_csv(
        OUTPUT_ORIGIN,
        index=False
    )


    poi_metrics.to_csv(
        OUTPUT_POI,
        index=False
    )


    print(
        "\n========== FIG. 2c PLOTTING DATA SAVED =========="
    )


    print(
        f"Saved: {OUTPUT_NEW_LINKS}"
    )

    print(
        f"Saved: {OUTPUT_QUADRANT}"
    )

    print(
        f"Saved: {OUTPUT_ORIGIN}"
    )

    print(
        f"Saved: {OUTPUT_POI}"
    )


    print(
        "\nNewly activated links:"
    )

    print(
        len(new_links_df)
    )


    print(
        "\nTrade-off composition:"
    )

    print(
        quadrant_df
        .round(3)
        .to_string(index=False)
    )


    print(
        "\nOrigins:"
    )

    print(
        len(origin_metrics)
    )


    print(
        "\nPOIs:"
    )

    print(
        len(poi_metrics)
    )


if __name__ == "__main__":
    main()