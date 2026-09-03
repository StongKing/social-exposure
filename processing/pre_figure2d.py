# -*- coding: utf-8 -*-
"""
Preprocessing for Fig.2d.

Purpose
-------
Read the original/restricted CBG-POI mobility inputs and generate
POI-level derived data that can be publicly released and used by
figure2d.py without redistributing the original flow matrix.

Outputs
-------
fig2d_spse_destination_outputs/
    figure2d_destination_summary.csv
    figure2d_top10_reassignment_summary.csv
    figure2d_case_summary.csv
"""

import os
import glob
import pickle
import numpy as np
import pandas as pd


# ============================================================
# 0. USER SETTINGS
# ============================================================

PROJECT_ROOT = r"d:\mobility_social_exposure"
MATRIX_ROOT = os.path.join(PROJECT_ROOT, "matrices_A_D_S_Distribution")

SELECTED_POI_CODE = "624190"
CITY_LABEL = "Boston MSA"

DMAX_KM = 50
DISTANCE_SCALE = 1.0

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "fig2d_spse_destination_outputs"
)

DESTINATION_TIER_BASIS = "baseline_total_visits"

EPS = 1e-9


# ============================================================
# 1. Metadata
# ============================================================

income_levels = [
    "low_income_pct",
    "lower_middle_income_pct",
    "upper_middle_income_pct",
    "high_income_pct",
]

income_score_weights = {
    "low_income_pct": 1.0,
    "lower_middle_income_pct": 2.0,
    "upper_middle_income_pct": 3.0,
    "high_income_pct": 4.0,
}

poi_code_to_full_label = {
    "624190": "Other Individual and Family Services",
    "711310": "Promoters of Performing Arts, Sports, and Similar Events with Facilities",
    "712110": "Museums",
    "713940": "Fitness and Recreational Sports Centers",
    "722410": "Drinking Places (Alcoholic Beverages)",
    "813110": "Religious Organizations",
}


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
        H.apply(
            pd.to_numeric,
            errors="coerce"
        )
        .fillna(0)
    )

    if H.index.duplicated().any():
        H = H.groupby(level=0).sum()

    return H


def safe_div(a, b):
    if (
        b is None
        or not np.isfinite(b)
        or abs(b) < EPS
    ):
        return np.nan

    return float(a) / float(b)


def weighted_mean(x, w):
    x = np.asarray(x, dtype=float)
    w = np.asarray(w, dtype=float)

    mask = (
        np.isfinite(x)
        & np.isfinite(w)
        & (w > 0)
    )

    if mask.sum() == 0:
        return np.nan

    return float(
        np.sum(x[mask] * w[mask])
        / np.sum(w[mask])
    )


def gini_coefficient(x):
    x = np.asarray(x, dtype=float)

    x = x[np.isfinite(x)]
    x = x[x >= 0]

    if len(x) == 0:
        return np.nan

    if np.allclose(x.sum(), 0):
        return 0.0

    x = np.sort(x)
    n = len(x)

    return float(
        (
            2
            * np.sum(
                np.arange(1, n + 1) * x
            )
        )
        / (n * x.sum())
        - (n + 1) / n
    )


# ============================================================
# 3. Locate files
# ============================================================

def find_case_dir_by_poi_code(poi_code):

    candidates = sorted(
        glob.glob(
            os.path.join(
                MATRIX_ROOT,
                "**",
                f"H_opt_df_dynamic_{poi_code}.pkl"
            ),
            recursive=True,
        )
    )

    if len(candidates) == 0:
        raise FileNotFoundError(
            f"Cannot find "
            f"H_opt_df_dynamic_{poi_code}.pkl "
            f"under {MATRIX_ROOT}"
        )

    return (
        os.path.dirname(candidates[0]),
        candidates[0],
    )


def find_income_file():

    candidates = [
        os.path.join(
            MATRIX_ROOT,
            "cbg_income_level_distribution_boston_msa.csv"
        ),
        os.path.join(
            MATRIX_ROOT,
            "cbg_income_level_distribution_boston_core.csv"
        ),
        os.path.join(
            PROJECT_ROOT,
            "cbg_income_level_distribution_boston_msa.csv"
        ),
        os.path.join(
            PROJECT_ROOT,
            "cbg_income_level_distribution_boston_core.csv"
        ),
    ]

    for p in candidates:
        if os.path.isfile(p):
            return p

    matches = glob.glob(
        os.path.join(
            PROJECT_ROOT,
            "**",
            "cbg_income_level_distribution*.csv"
        ),
        recursive=True,
    )

    if len(matches) == 0:
        raise FileNotFoundError(
            "Cannot find "
            "cbg_income_level_distribution*.csv"
        )

    return matches[0]


# ============================================================
# 4. Load income distribution
# ============================================================

def load_income_distribution():

    income_path = find_income_file()

    print(
        f"[LOAD income] {income_path}"
    )

    df = pd.read_csv(income_path)

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
        P.apply(
            pd.to_numeric,
            errors="coerce"
        )
        .fillna(0)
    )

    if P.index.duplicated().any():
        P = P.groupby(level=0).mean()

    row_sum = (
        P.sum(axis=1)
        .replace(0, np.nan)
    )

    P = (
        P.div(
            row_sum,
            axis=0
        )
        .fillna(0)
    )

    return P


# ============================================================
# 5. Exposure computation
# ============================================================

def compute_all_pair_unmasked_exposure(
    flow_df,
    P_df,
):
    """
    Compute all-pair S_ij in the baseline POI domain.

    S_ij = 1 - dot(P_i, Q_j)

    P_i:
        income distribution of origin CBG i.

    Q_j:
        visitor-income distribution of POI j,
        estimated from flow_df.
    """

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
        F.apply(
            pd.to_numeric,
            errors="coerce"
        )
        .fillna(0)
    )

    common_cbgs = [
        g
        for g in F.index
        if g in P_df.index
    ]

    if len(common_cbgs) == 0:
        raise ValueError(
            "No common CBGs between flow matrix "
            "and income distribution."
        )

    F = F.loc[common_cbgs].copy()

    P = (
        P_df
        .loc[
            common_cbgs,
            income_levels
        ]
        .copy()
    )

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
            "No POI has positive baseline flow."
        )

    F = F[valid_pois].copy()

    F_values = F.values.astype(float)
    P_values = P.values.astype(float)

    poi_total_flow = F_values.sum(axis=0)

    Q_values = (
        F_values.T
        @ P_values
    ) / poi_total_flow[:, None]

    Q_sum = Q_values.sum(
        axis=1,
        keepdims=True,
    )

    Q_values = np.divide(
        Q_values,
        Q_sum,
        out=np.zeros_like(Q_values),
        where=Q_sum > 0,
    )

    S_values = (
        1.0
        - (
            P_values
            @ Q_values.T
        )
    )

    S = pd.DataFrame(
        S_values,
        index=F.index,
        columns=F.columns,
    )

    Q = pd.DataFrame(
        Q_values,
        index=F.columns,
        columns=income_levels,
    )

    return S, F, Q


def compute_exposure_on_fixed_domain(
    flow_df,
    P_df,
    fixed_columns,
):
    """
    Compute S_ij on a fixed POI domain.

    Used for post-optimization exposure so baseline
    and optimized exposure are evaluated on the same POI set.
    """

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
        F.apply(
            pd.to_numeric,
            errors="coerce"
        )
        .fillna(0)
    )

    fixed_columns = [
        str(c).strip()
        for c in fixed_columns
    ]

    common_cbgs = [
        g
        for g in F.index
        if g in P_df.index
    ]

    common_cols = [
        c
        for c in fixed_columns
        if c in F.columns
    ]

    if (
        len(common_cbgs) == 0
        or len(common_cols) == 0
    ):
        raise ValueError(
            "No common CBGs or POIs "
            "in fixed exposure domain."
        )

    F = (
        F.loc[
            common_cbgs,
            common_cols
        ]
        .copy()
    )

    P = (
        P_df
        .loc[
            common_cbgs,
            income_levels
        ]
        .copy()
    )

    F_values = F.values.astype(float)
    P_values = P.values.astype(float)

    poi_total_flow = F_values.sum(axis=0)

    Q_values = np.zeros(
        (
            F_values.shape[1],
            len(income_levels),
        ),
        dtype=float,
    )

    pos = poi_total_flow > 0

    if pos.any():
        Q_values[pos, :] = (
            F_values[:, pos].T
            @ P_values
        ) / poi_total_flow[pos, None]

    Q_sum = Q_values.sum(
        axis=1,
        keepdims=True,
    )

    Q_values = np.divide(
        Q_values,
        Q_sum,
        out=np.zeros_like(Q_values),
        where=Q_sum > 0,
    )

    S_values = (
        1.0
        - (
            P_values
            @ Q_values.T
        )
    )

    S = pd.DataFrame(
        S_values,
        index=F.index,
        columns=F.columns,
    )

    Q = pd.DataFrame(
        Q_values,
        index=F.columns,
        columns=income_levels,
    )

    return S, Q


# ============================================================
# 6. Grouping and score helpers
# ============================================================

def compute_income_composition_score(P_model):
    """
    income_score_i =
          1 * low_income_pct
        + 2 * lower_middle_income_pct
        + 3 * upper_middle_income_pct
        + 4 * high_income_pct
    """

    P = P_model[
        income_levels
    ].copy()

    weights = np.array(
        [
            income_score_weights[c]
            for c in income_levels
        ],
        dtype=float,
    )

    score = pd.Series(
        P.values @ weights,
        index=P.index,
        name="income_composition_score",
    )

    return score


def make_destination_tiers(
    poi_metrics,
    basis="baseline_total_visits",
):
    """
    Group POIs into:
        Top 10%
        Middle 40%
        Bottom 50%
    """

    if basis not in [
        "baseline_total_visits",
        "optimized_total_visits",
    ]:
        raise ValueError(
            "basis must be "
            "'baseline_total_visits' "
            "or 'optimized_total_visits'."
        )

    df = poi_metrics[
        [
            "poi_id",
            "baseline_total_visits",
            "optimized_total_visits",
        ]
    ].copy()

    df = (
        df
        .sort_values(
            basis,
            ascending=False
        )
        .reset_index(drop=True)
    )

    n = len(df)

    if n == 0:
        raise ValueError(
            "No POIs available for destination tiering."
        )

    n_top = max(
        1,
        int(
            np.ceil(
                n * 0.10
            )
        )
    )

    n_mid = max(
        1,
        int(
            np.ceil(
                n * 0.40
            )
        )
    )

    n_mid = min(
        n_mid,
        max(
            0,
            n - n_top
        )
    )

    if basis == "baseline_total_visits":

        tier_order = [
            "Top 10% baseline-flow POIs",
            "Middle 40% baseline-flow POIs",
            "Bottom 50% baseline-flow POIs",
        ]

        rank_col = (
            "poi_rank_baseline_flow"
        )

    else:

        tier_order = [
            "Top 10% final-flow POIs",
            "Middle 40% final-flow POIs",
            "Bottom 50% final-flow POIs",
        ]

        rank_col = (
            "poi_rank_final_flow"
        )

    tiers = []

    for pos in range(n):

        if pos < n_top:
            tiers.append(
                tier_order[0]
            )

        elif pos < n_top + n_mid:
            tiers.append(
                tier_order[1]
            )

        else:
            tiers.append(
                tier_order[2]
            )

    df["poi_tier"] = tiers

    df[rank_col] = np.arange(
        1,
        n + 1,
    )

    return df, tier_order


def compute_top10_outflow_reassignment(
    F_dom,
    H_dom,
    poi_tiers,
):
    """
    Estimate where flow removed from baseline Top-10% POIs
    is reassigned.

    For each CBG, flow removed from Top-10% POIs is
    attributed to destination tiers receiving positive
    increments from the same CBG, proportional to those
    positive increments.
    """

    F = F_dom.copy().astype(float)
    H = H_dom.copy().astype(float)

    diff = (
        H.values
        - F.values
    )

    add = np.maximum(
        diff,
        0.0
    )

    loss = np.maximum(
        -diff,
        0.0
    )

    cols = list(
        F.columns
    )

    tier_s = (
        poi_tiers
        .set_index("poi_id")["poi_tier"]
        .reindex(cols)
        .fillna("")
    )

    top_mask = (
        tier_s
        .str.contains(
            "Top 10%",
            case=False,
            regex=False,
        )
        .values
    )

    mid_mask = (
        tier_s
        .str.contains(
            "Middle",
            case=False,
            regex=False,
        )
        .values
    )

    bottom_mask = (
        tier_s
        .str.contains(
            "Bottom",
            case=False,
            regex=False,
        )
        .values
    )

    if (
        top_mask.sum() == 0
        or mid_mask.sum() == 0
        or bottom_mask.sum() == 0
    ):
        return {
            "available": False,
            "reason": (
                "Cannot identify "
                "Top/Middle/Bottom POI tiers."
            ),
        }

    top_loss_by_origin = (
        loss[:, top_mask]
        .sum(axis=1)
    )

    total_add_by_origin = (
        add.sum(axis=1)
    )

    add_top_by_origin = (
        add[:, top_mask]
        .sum(axis=1)
    )

    add_mid_by_origin = (
        add[:, mid_mask]
        .sum(axis=1)
    )

    add_bottom_by_origin = (
        add[:, bottom_mask]
        .sum(axis=1)
    )

    valid = (
        (top_loss_by_origin > EPS)
        & (total_add_by_origin > EPS)
    )

    top_to_top = float(
        np.nansum(
            top_loss_by_origin[valid]
            * add_top_by_origin[valid]
            / total_add_by_origin[valid]
        )
    )

    top_to_middle = float(
        np.nansum(
            top_loss_by_origin[valid]
            * add_mid_by_origin[valid]
            / total_add_by_origin[valid]
        )
    )

    top_to_bottom = float(
        np.nansum(
            top_loss_by_origin[valid]
            * add_bottom_by_origin[valid]
            / total_add_by_origin[valid]
        )
    )

    top_outflow = float(
        np.nansum(
            top_loss_by_origin
        )
    )

    baseline_top_flow = float(
        np.nansum(
            F.values[:, top_mask]
        )
    )

    return {
        "available":
            bool(
                top_outflow > EPS
            ),

        "top_outflow":
            top_outflow,

        "baseline_top_flow":
            baseline_top_flow,

        "top_to_top":
            top_to_top,

        "top_to_middle":
            top_to_middle,

        "top_to_bottom":
            top_to_bottom,

        "top_to_middle_share_of_top_outflow_pct":
            safe_div(
                top_to_middle,
                top_outflow
            ) * 100,

        "top_to_bottom_share_of_top_outflow_pct":
            safe_div(
                top_to_bottom,
                top_outflow
            ) * 100,

        "top_to_top_share_of_top_outflow_pct":
            safe_div(
                top_to_top,
                top_outflow
            ) * 100,

        "top_to_middle_share_of_baseline_top_flow_pct":
            safe_div(
                top_to_middle,
                baseline_top_flow
            ) * 100,

        "top_to_bottom_share_of_baseline_top_flow_pct":
            safe_div(
                top_to_bottom,
                baseline_top_flow
            ) * 100,

        "n_top_pois":
            int(
                top_mask.sum()
            ),

        "n_middle_pois":
            int(
                mid_mask.sum()
            ),

        "n_bottom_pois":
            int(
                bottom_mask.sum()
            ),
    }


# ============================================================
# 7. Build destination-level SPSE summary
# ============================================================

def build_spse_destination_summary(
    F_dom,
    H_dom,
    S0_dom,
    S1_dom,
    D_dom,
    P_model,
    poi_tiers,
):
    """
    Build one-row-per-POI table for Fig.2d.
    """

    rows = F_dom.index.tolist()
    cols = F_dom.columns.tolist()

    F_v = F_dom.values.astype(float)
    H_v = H_dom.values.astype(float)
    S0_v = S0_dom.values.astype(float)
    S1_v = S1_dom.values.astype(float)
    D_v = D_dom.values.astype(float)

    diff_v = H_v - F_v

    add_v = np.maximum(
        diff_v,
        0
    )

    remove_v = np.maximum(
        -diff_v,
        0
    )

    income_score = (
        compute_income_composition_score(
            P_model
        )
        .reindex(rows)
    )

    income_score_v = (
        income_score
        .values
        .astype(float)
    )

    poi_tier_map = (
        poi_tiers
        .set_index("poi_id")["poi_tier"]
        .to_dict()
    )

    records = []

    for j, poi_id in enumerate(cols):

        f = F_v[:, j]
        h = H_v[:, j]
        s0 = S0_v[:, j]
        s1 = S1_v[:, j]
        d = D_v[:, j]

        add = add_v[:, j]
        remove = remove_v[:, j]

        baseline_visits = float(
            np.nansum(f)
        )

        optimized_visits = float(
            np.nansum(h)
        )

        delta_visits = (
            optimized_visits
            - baseline_visits
        )

        delta_visits_pct = (
            safe_div(
                delta_visits,
                baseline_visits
            )
            * 100
        )

        positive_increment_received = float(
            np.nansum(add)
        )

        removed_flow = float(
            np.nansum(remove)
        )

        net_flow_change = (
            optimized_visits
            - baseline_visits
        )

        active_before = (
            f > EPS
        )

        active_after = (
            h > EPS
        )

        active_union = (
            active_before
            | active_after
        )

        new_positive_links = (
            (~active_before)
            & active_after
        )

        removed_positive_links = (
            active_before
            & (~active_after)
        )

        retained_positive_links = (
            active_before
            & active_after
        )

        n_active_links_before = int(
            active_before.sum()
        )

        n_active_links_after = int(
            active_after.sum()
        )

        delta_active_links = (
            n_active_links_after
            - n_active_links_before
        )

        n_new_positive_links = int(
            new_positive_links.sum()
        )

        n_removed_positive_links = int(
            removed_positive_links.sum()
        )

        n_retained_positive_links = int(
            retained_positive_links.sum()
        )

        # ----------------------------------------------------
        # POI-level SPSE decomposition
        # ----------------------------------------------------

        spse_contribution_before = float(
            np.nansum(
                s0[active_before]
            )
        )

        spse_contribution_after = float(
            np.nansum(
                s1[active_after]
            )
        )

        delta_spse_contribution = (
            spse_contribution_after
            - spse_contribution_before
        )

        spse_from_new_links = float(
            np.nansum(
                s1[new_positive_links]
            )
        )

        spse_lost_from_removed_links = float(
            np.nansum(
                s0[removed_positive_links]
            )
        )

        spse_change_on_retained_links = float(
            np.nansum(
                s1[retained_positive_links]
                - s0[retained_positive_links]
            )
        )

        mean_delta_structural_social_exposure_per_visiting_cbg = (
            safe_div(
                delta_spse_contribution,
                n_active_links_after,
            )
        )

        mean_delta_structural_social_exposure_per_baseline_visiting_cbg = (
            safe_div(
                delta_spse_contribution,
                n_active_links_before,
            )
        )

        mean_delta_structural_social_exposure_per_union_visiting_cbg = (
            safe_div(
                delta_spse_contribution,
                int(
                    active_union.sum()
                ),
            )
        )

        # ----------------------------------------------------
        # Weighted exposure contribution
        # ----------------------------------------------------

        weighted_contribution_before = float(
            np.nansum(
                f * s0
            )
        )

        weighted_contribution_after = float(
            np.nansum(
                h * s1
            )
        )

        delta_weighted_contribution = (
            weighted_contribution_after
            - weighted_contribution_before
        )

        weighted_added_contribution = float(
            np.nansum(
                add * s1
            )
        )

        weighted_removed_contribution = float(
            np.nansum(
                remove * s0
            )
        )

        # ----------------------------------------------------
        # Visit-weighted exposure diagnostics
        # ----------------------------------------------------

        flow_weighted_exposure_before = (
            weighted_mean(
                s0,
                f
            )
        )

        flow_weighted_exposure_after = (
            weighted_mean(
                s1,
                h
            )
        )

        if (
            np.isfinite(
                flow_weighted_exposure_before
            )
            and np.isfinite(
                flow_weighted_exposure_after
            )
        ):
            delta_flow_weighted_exposure = (
                flow_weighted_exposure_after
                - flow_weighted_exposure_before
            )

        else:
            delta_flow_weighted_exposure = np.nan

        # ----------------------------------------------------
        # Origin income-composition diagnostics
        # ----------------------------------------------------

        added_origin_income_score = (
            weighted_mean(
                income_score_v,
                add
            )
        )

        baseline_origin_income_score = (
            weighted_mean(
                income_score_v,
                f
            )
        )

        optimized_origin_income_score = (
            weighted_mean(
                income_score_v,
                h
            )
        )

        if (
            np.isfinite(
                baseline_origin_income_score
            )
            and np.isfinite(
                optimized_origin_income_score
            )
        ):
            delta_origin_income_score = (
                optimized_origin_income_score
                - baseline_origin_income_score
            )

        else:
            delta_origin_income_score = np.nan

        # ----------------------------------------------------
        # Distance diagnostics
        # ----------------------------------------------------

        flow_weighted_distance_before = (
            weighted_mean(
                d,
                f
            )
        )

        flow_weighted_distance_after = (
            weighted_mean(
                d,
                h
            )
        )

        if (
            np.isfinite(
                flow_weighted_distance_before
            )
            and np.isfinite(
                flow_weighted_distance_after
            )
        ):
            delta_flow_weighted_distance_km = (
                flow_weighted_distance_after
                - flow_weighted_distance_before
            )

        else:
            delta_flow_weighted_distance_km = np.nan

        # ----------------------------------------------------
        # Exposure-field diagnostics
        # ----------------------------------------------------

        delta_s = (
            s1
            - s0
        )

        finite_delta_s = (
            np.isfinite(
                delta_s
            )
        )

        delta_exposure_field_all_cbgs = float(
            np.nansum(
                delta_s[
                    finite_delta_s
                ]
            )
        )

        n_exposure_field_all_cbgs = int(
            finite_delta_s.sum()
        )

        mean_delta_exposure_field_all_cbgs = (
            safe_div(
                delta_exposure_field_all_cbgs,
                n_exposure_field_all_cbgs,
            )
        )

        delta_exposure_field_active_union = float(
            np.nansum(
                delta_s[
                    active_union
                    & finite_delta_s
                ]
            )
        )

        n_exposure_field_active_union = int(
            (
                active_union
                & finite_delta_s
            ).sum()
        )

        mean_delta_exposure_field_active_union = (
            safe_div(
                delta_exposure_field_active_union,
                n_exposure_field_active_union,
            )
        )

        records.append(
            {
                "poi_id":
                    poi_id,

                "poi_tier":
                    poi_tier_map.get(
                        poi_id,
                        "Unclassified"
                    ),

                "baseline_total_visits":
                    baseline_visits,

                "optimized_total_visits":
                    optimized_visits,

                "delta_visits":
                    delta_visits,

                "delta_visits_pct":
                    delta_visits_pct,

                "positive_increment_received":
                    positive_increment_received,

                "removed_flow":
                    removed_flow,

                "net_flow_change":
                    net_flow_change,

                "n_active_links_before":
                    n_active_links_before,

                "n_active_links_after":
                    n_active_links_after,

                "delta_active_links":
                    delta_active_links,

                "n_new_positive_links":
                    n_new_positive_links,

                "n_removed_positive_links":
                    n_removed_positive_links,

                "n_retained_positive_links":
                    n_retained_positive_links,

                "spse_contribution_before":
                    spse_contribution_before,

                "spse_contribution_after":
                    spse_contribution_after,

                "delta_spse_contribution":
                    delta_spse_contribution,

                "spse_from_new_links":
                    spse_from_new_links,

                "spse_lost_from_removed_links":
                    spse_lost_from_removed_links,

                "spse_change_on_retained_links":
                    spse_change_on_retained_links,

                "mean_delta_structural_social_exposure_per_visiting_cbg":
                    mean_delta_structural_social_exposure_per_visiting_cbg,

                "mean_delta_structural_social_exposure_per_baseline_visiting_cbg":
                    mean_delta_structural_social_exposure_per_baseline_visiting_cbg,

                "mean_delta_structural_social_exposure_per_union_visiting_cbg":
                    mean_delta_structural_social_exposure_per_union_visiting_cbg,

                "weighted_contribution_before":
                    weighted_contribution_before,

                "weighted_contribution_after":
                    weighted_contribution_after,

                "delta_weighted_contribution":
                    delta_weighted_contribution,

                "weighted_added_contribution":
                    weighted_added_contribution,

                "weighted_removed_contribution":
                    weighted_removed_contribution,

                "flow_weighted_exposure_before":
                    flow_weighted_exposure_before,

                "flow_weighted_exposure_after":
                    flow_weighted_exposure_after,

                "delta_flow_weighted_exposure":
                    delta_flow_weighted_exposure,

                "added_origin_income_score":
                    added_origin_income_score,

                "baseline_origin_income_score":
                    baseline_origin_income_score,

                "optimized_origin_income_score":
                    optimized_origin_income_score,

                "delta_origin_income_score":
                    delta_origin_income_score,

                "flow_weighted_distance_before":
                    flow_weighted_distance_before,

                "flow_weighted_distance_after":
                    flow_weighted_distance_after,

                "delta_flow_weighted_distance_km":
                    delta_flow_weighted_distance_km,

                "delta_exposure_field_all_cbgs":
                    delta_exposure_field_all_cbgs,

                "mean_delta_exposure_field_all_cbgs":
                    mean_delta_exposure_field_all_cbgs,

                "delta_exposure_field_active_union":
                    delta_exposure_field_active_union,

                "mean_delta_exposure_field_active_union":
                    mean_delta_exposure_field_active_union,
            }
        )

    scatter_df = pd.DataFrame(
        records
    )

    return scatter_df


# ============================================================
# 8. Build full case output
# ============================================================

def build_case_output():

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

    P_df = (
        load_income_distribution()
    )

    F_raw = read_matrix_csv(
        flow_path,
        distance=False,
    )

    D_raw = read_matrix_csv(
        dist_path,
        distance=True,
    )

    H_opt = read_hopt_pickle(
        h_path
    )

    # --------------------------------------------------------
    # Baseline exposure
    # --------------------------------------------------------

    S0_full, F_income, Q0_full = (
        compute_all_pair_unmasked_exposure(
            F_raw,
            P_df,
        )
    )

    common_rows = [
        g
        for g in F_income.index
        if (
            g in D_raw.index
            and g in S0_full.index
        )
    ]

    common_cols = [
        p
        for p in F_income.columns
        if (
            p in D_raw.columns
            and p in S0_full.columns
        )
    ]

    if (
        len(common_rows) == 0
        or len(common_cols) == 0
    ):
        raise ValueError(
            "No common rows/columns "
            "among F, D, and S0."
        )

    F_full = (
        F_income
        .loc[
            common_rows,
            common_cols
        ]
        .copy()
    )

    D_full = (
        D_raw
        .loc[
            common_rows,
            common_cols
        ]
        .copy()
    )

    S0_full = (
        S0_full
        .loc[
            common_rows,
            common_cols
        ]
        .copy()
    )

    # --------------------------------------------------------
    # Embed optimized H into full baseline matrix
    # --------------------------------------------------------

    H_eval = (
        F_full
        .copy()
        .astype(float)
    )

    rows_h = [
        g
        for g in H_opt.index
        if (
            g in H_eval.index
            and g in P_df.index
        )
    ]

    cols_h = [
        p
        for p in H_opt.columns
        if p in H_eval.columns
    ]

    if (
        len(rows_h) == 0
        or len(cols_h) == 0
    ):
        raise ValueError(
            "No common rows/columns "
            "between H_opt and baseline flow matrix."
        )

    H_eval.loc[
        rows_h,
        cols_h
    ] = (
        H_opt
        .loc[
            rows_h,
            cols_h
        ]
        .values
    )

    # --------------------------------------------------------
    # Model domain
    # --------------------------------------------------------

    F_dom = (
        F_full
        .loc[
            rows_h,
            cols_h
        ]
        .copy()
    )

    H_dom = (
        H_eval
        .loc[
            rows_h,
            cols_h
        ]
        .copy()
    )

    D_dom = (
        D_full
        .loc[
            rows_h,
            cols_h
        ]
        .copy()
    )

    S0_dom = (
        S0_full
        .loc[
            rows_h,
            cols_h
        ]
        .copy()
    )

    P_model = (
        P_df
        .loc[
            rows_h,
            income_levels
        ]
        .copy()
    )

    # --------------------------------------------------------
    # Active-link reference from full diagnostic domain
    # --------------------------------------------------------

    Fv_full = (
        F_full
        .values
        .astype(float)
    )

    Dv_full = (
        D_full
        .values
        .astype(float)
    )

    S0v_full = (
        S0_full
        .values
        .astype(float)
    )

    Hv_full = (
        H_eval
        .values
        .astype(float)
    )

    valid_full = (
        np.isfinite(Fv_full)
        & np.isfinite(Dv_full)
        & np.isfinite(S0v_full)
        & np.isfinite(Hv_full)
    )

    distance_feasible_full = (
        valid_full
        & (Dv_full >= 0)
        & (Dv_full <= DMAX_KM)
    )

    active_ref_full = (
        distance_feasible_full
        & (Fv_full > 0)
    )

    if active_ref_full.sum() == 0:
        raise ValueError(
            "No active feasible reference links "
            "in the full diagnostic domain."
        )

    active_w_s = weighted_mean(
        S0v_full[active_ref_full],
        Fv_full[active_ref_full],
    )

    active_w_d = weighted_mean(
        Dv_full[active_ref_full],
        Fv_full[active_ref_full],
    )

    # --------------------------------------------------------
    # Post-optimization exposure
    # --------------------------------------------------------

    S1_full, Q1_full = (
        compute_exposure_on_fixed_domain(
            H_eval,
            P_df,
            F_full.columns,
        )
    )

    S1_dom = (
        S1_full
        .loc[
            rows_h,
            cols_h
        ]
        .copy()
    )

    F_v = (
        F_dom
        .values
        .astype(float)
    )

    H_v = (
        H_dom
        .values
        .astype(float)
    )

    D_v = (
        D_dom
        .values
        .astype(float)
    )

    S0_v = (
        S0_dom
        .values
        .astype(float)
    )

    S1_v = (
        S1_dom
        .values
        .astype(float)
    )

    baseline_total_flow = float(
        np.nansum(F_v)
    )

    optimized_total_flow = float(
        np.nansum(H_v)
    )

    diff_v = (
        H_v
        - F_v
    )

    add_v = np.maximum(
        diff_v,
        0
    )

    remove_v = np.maximum(
        -diff_v,
        0
    )

    reassigned_visit_equiv = (
        0.5
        * float(
            np.nansum(
                np.abs(diff_v)
            )
        )
    )

    total_positive_increment = float(
        np.nansum(
            add_v
        )
    )

    total_removed_flow = float(
        np.nansum(
            remove_v
        )
    )

    # --------------------------------------------------------
    # Distance
    # --------------------------------------------------------

    distance_before = float(
        np.nansum(
            F_v * D_v
        )
    )

    distance_after = float(
        np.nansum(
            H_v * D_v
        )
    )

    distance_change_pct = (
        safe_div(
            distance_after - distance_before,
            distance_before,
        )
        * 100
    )

    # --------------------------------------------------------
    # Active links and SPSE
    # --------------------------------------------------------

    active_before_mask = (
        F_v > EPS
    )

    active_after_mask = (
        H_v > EPS
    )

    total_active_links_before = int(
        active_before_mask.sum()
    )

    total_active_links_after = int(
        active_after_mask.sum()
    )

    total_delta_active_links = (
        total_active_links_after
        - total_active_links_before
    )

    spse_before = float(
        np.nansum(
            S0_v[
                active_before_mask
            ]
        )
    )

    spse_after = float(
        np.nansum(
            S1_v[
                active_after_mask
            ]
        )
    )

    spse_delta = (
        spse_after
        - spse_before
    )

    spse_change_pct = (
        safe_div(
            spse_delta,
            spse_before,
        )
        * 100
    )

    # --------------------------------------------------------
    # Weighted exposure
    # --------------------------------------------------------

    weighted_exposure_numerator_before = float(
        np.nansum(
            F_v * S0_v
        )
    )

    weighted_exposure_numerator_after = float(
        np.nansum(
            H_v * S1_v
        )
    )

    weighted_exposure_numerator_delta = (
        weighted_exposure_numerator_after
        - weighted_exposure_numerator_before
    )

    fw_before = safe_div(
        weighted_exposure_numerator_before,
        baseline_total_flow,
    )

    fw_after = safe_div(
        weighted_exposure_numerator_after,
        optimized_total_flow,
    )

    fw_change_pct = (
        safe_div(
            fw_after - fw_before,
            fw_before,
        )
        * 100
    )

    # --------------------------------------------------------
    # Link-change diagnostics
    # --------------------------------------------------------

    new_link_mask = (
        (F_v <= EPS)
        & (H_v > EPS)
    )

    removed_link_mask = (
        (F_v > EPS)
        & (H_v <= EPS)
    )

    retained_link_mask = (
        (F_v > EPS)
        & (H_v > EPS)
    )

    positive_increment_mask = (
        diff_v > EPS
    )

    add_mask = (
        add_v > EPS
    )

    if add_mask.any():

        w = add_v[
            add_mask
        ]

        realized_adv = (
            S1_v[add_mask]
            - active_w_s
        )

        distance_adv = (
            D_v[add_mask]
            - active_w_d
        )

        positive_increment_higher_exposure_share_pct = (
            safe_div(
                np.nansum(
                    w[
                        realized_adv > 0
                    ]
                ),
                np.nansum(w),
            )
            * 100
        )

        positive_increment_no_farther_share_pct = (
            safe_div(
                np.nansum(
                    w[
                        distance_adv <= 0
                    ]
                ),
                np.nansum(w),
            )
            * 100
        )

        positive_increment_higher_and_no_farther_share_pct = (
            safe_div(
                np.nansum(
                    w[
                        (
                            realized_adv > 0
                        )
                        & (
                            distance_adv <= 0
                        )
                    ]
                ),
                np.nansum(w),
            )
            * 100
        )

    else:

        positive_increment_higher_exposure_share_pct = np.nan
        positive_increment_no_farther_share_pct = np.nan
        positive_increment_higher_and_no_farther_share_pct = np.nan

    # --------------------------------------------------------
    # Destination tiers
    # --------------------------------------------------------

    poi_metrics = pd.DataFrame(
        {
            "poi_id":
                cols_h,

            "baseline_total_visits":
                F_dom
                .sum(axis=0)
                .reindex(cols_h)
                .values
                .astype(float),

            "optimized_total_visits":
                H_dom
                .sum(axis=0)
                .reindex(cols_h)
                .values
                .astype(float),
        }
    )

    poi_tiers, tier_order = (
        make_destination_tiers(
            poi_metrics=poi_metrics,
            basis=DESTINATION_TIER_BASIS,
        )
    )

    # --------------------------------------------------------
    # Top-10% reassignment
    # --------------------------------------------------------

    top10_reassignment_summary = (
        compute_top10_outflow_reassignment(
            F_dom=F_dom,
            H_dom=H_dom,
            poi_tiers=poi_tiers,
        )
    )

    # --------------------------------------------------------
    # POI-level data
    # --------------------------------------------------------

    scatter_df = (
        build_spse_destination_summary(
            F_dom=F_dom,
            H_dom=H_dom,
            S0_dom=S0_dom,
            S1_dom=S1_dom,
            D_dom=D_dom,
            P_model=P_model,
            poi_tiers=poi_tiers,
        )
    )

    # --------------------------------------------------------
    # Decomposition checks
    # --------------------------------------------------------

    sum_delta_spse_from_pois = float(
        scatter_df[
            "delta_spse_contribution"
        ]
        .sum()
    )

    sum_delta_weighted_contribution_from_pois = float(
        scatter_df[
            "delta_weighted_contribution"
        ]
        .sum()
    )

    sum_delta_active_links_from_pois = int(
        scatter_df[
            "delta_active_links"
        ]
        .sum()
    )

    spse_decomposition_error = (
        sum_delta_spse_from_pois
        - spse_delta
    )

    weighted_decomposition_error = (
        sum_delta_weighted_contribution_from_pois
        - weighted_exposure_numerator_delta
    )

    active_link_decomposition_error = (
        sum_delta_active_links_from_pois
        - total_delta_active_links
    )

    origin_reallocation = (
        0.5
        * np.abs(
            diff_v
        )
        .sum(axis=1)
    )

    poi_positive_increment = (
        add_v
        .sum(axis=0)
    )

    case_summary = {

        "poi_code":
            SELECTED_POI_CODE,

        "poi_full_label":
            poi_code_to_full_label.get(
                SELECTED_POI_CODE,
                SELECTED_POI_CODE
            ),

        "city_label":
            CITY_LABEL,

        "case_dir":
            case_dir,

        "diagnostic_domain":
            "full_baseline_matrix_after_alignment",

        "n_model_cbgs":
            len(rows_h),

        "n_model_pois":
            len(cols_h),

        "active_weighted_exposure_ref_full":
            active_w_s,

        "active_weighted_distance_ref_full":
            active_w_d,

        "baseline_total_flow":
            baseline_total_flow,

        "optimized_total_flow":
            optimized_total_flow,

        "total_positive_increment":
            total_positive_increment,

        "total_removed_flow":
            total_removed_flow,

        "reassigned_visit_equiv":
            reassigned_visit_equiv,

        "distance_before":
            distance_before,

        "distance_after":
            distance_after,

        "distance_change_pct":
            distance_change_pct,

        "total_active_links_before":
            total_active_links_before,

        "total_active_links_after":
            total_active_links_after,

        "total_delta_active_links":
            total_delta_active_links,

        "spse_before":
            spse_before,

        "spse_after":
            spse_after,

        "spse_delta":
            spse_delta,

        "spse_change_pct":
            spse_change_pct,

        "sum_delta_spse_from_pois":
            sum_delta_spse_from_pois,

        "spse_decomposition_error":
            spse_decomposition_error,

        "flow_weighted_exposure_before":
            fw_before,

        "flow_weighted_exposure_after":
            fw_after,

        "fw_change_pct":
            fw_change_pct,

        "weighted_exposure_numerator_before":
            weighted_exposure_numerator_before,

        "weighted_exposure_numerator_after":
            weighted_exposure_numerator_after,

        "weighted_exposure_numerator_delta":
            weighted_exposure_numerator_delta,

        "sum_delta_weighted_contribution_from_pois":
            sum_delta_weighted_contribution_from_pois,

        "weighted_decomposition_error":
            weighted_decomposition_error,

        "sum_delta_active_links_from_pois":
            sum_delta_active_links_from_pois,

        "active_link_decomposition_error":
            active_link_decomposition_error,

        "n_new_links":
            int(
                new_link_mask.sum()
            ),

        "n_removed_links":
            int(
                removed_link_mask.sum()
            ),

        "n_retained_links":
            int(
                retained_link_mask.sum()
            ),

        "n_positive_increment_cells":
            int(
                positive_increment_mask.sum()
            ),

        "positive_increment_higher_exposure_share_pct":
            positive_increment_higher_exposure_share_pct,

        "positive_increment_no_farther_share_pct":
            positive_increment_no_farther_share_pct,

        "positive_increment_higher_and_no_farther_share_pct":
            positive_increment_higher_and_no_farther_share_pct,

        "origin_reallocation_gini":
            gini_coefficient(
                origin_reallocation
            ),

        "destination_positive_increment_gini":
            gini_coefficient(
                poi_positive_increment
            ),

        "row_balance_error_max_abs":
            float(
                np.nanmax(
                    np.abs(
                        np.nansum(
                            diff_v,
                            axis=1
                        )
                    )
                )
            ),

        "top10_outflow":
            top10_reassignment_summary.get(
                "top_outflow",
                np.nan
            ),

        "top10_to_middle_share_of_top_outflow_pct":
            top10_reassignment_summary.get(
                "top_to_middle_share_of_top_outflow_pct",
                np.nan
            ),

        "top10_to_bottom_share_of_top_outflow_pct":
            top10_reassignment_summary.get(
                "top_to_bottom_share_of_top_outflow_pct",
                np.nan
            ),

        "upper_left_outcome":
            bool(
                (
                    distance_change_pct < 0
                )
                and (
                    spse_change_pct > 0
                )
            ),

        "fw_positive":
            bool(
                fw_change_pct > 0
            ),
    }

    return {
        "case_summary":
            case_summary,

        "scatter_df":
            scatter_df,

        "poi_tiers":
            poi_tiers,

        "poi_metrics":
            poi_metrics,

        "top10_reassignment_summary":
            top10_reassignment_summary,

        "case_dir":
            case_dir,
    }


# ============================================================
# 9. Save public Fig.2d inputs
# ============================================================

def save_public_outputs(case_output):

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    scatter_path = os.path.join(
        OUTPUT_DIR,
        "figure2d_destination_summary.csv"
    )

    top10_path = os.path.join(
        OUTPUT_DIR,
        "figure2d_top10_reassignment_summary.csv"
    )

    case_summary_path = os.path.join(
        OUTPUT_DIR,
        "figure2d_case_summary.csv"
    )

    # --------------------------------------------------------
    # POI-level data actually needed to reproduce Fig.2d
    # --------------------------------------------------------

    case_output[
        "scatter_df"
    ].to_csv(
        scatter_path,
        index=False,
    )

    # --------------------------------------------------------
    # Top-10% inset data
    # --------------------------------------------------------

    top10_summary = (
        case_output[
            "top10_reassignment_summary"
        ]
        .copy()
    )

    pd.DataFrame(
        [top10_summary]
    ).to_csv(
        top10_path,
        index=False,
    )

    # --------------------------------------------------------
    # Additional public diagnostic summary
    #
    # case_dir is deliberately excluded because it is only
    # a local machine path and has no reproducibility value.
    # --------------------------------------------------------

    public_case_summary = {
        k: v
        for k, v
        in case_output[
            "case_summary"
        ].items()
        if k != "case_dir"
    }

    pd.DataFrame(
        [public_case_summary]
    ).to_csv(
        case_summary_path,
        index=False,
    )

    print(
        "\n========== PUBLIC FIG.2D FILES SAVED =========="
    )

    print(
        f"[SAVE] {scatter_path}"
    )

    print(
        f"[SAVE] {top10_path}"
    )

    print(
        f"[SAVE] {case_summary_path}"
    )


# ============================================================
# 10. Main
# ============================================================

def main():

    case_output = (
        build_case_output()
    )

    print(
        "\n========== FIG.2D SPSE DESTINATION CASE SUMMARY =========="
    )

    for k, v in (
        case_output[
            "case_summary"
        ].items()
    ):
        print(
            f"{k}: {v}"
        )

    top10_summary = (
        case_output.get(
            "top10_reassignment_summary",
            {}
        )
    )

    if top10_summary.get(
        "available",
        False
    ):

        print(
            "\n========== TOP-10% BASELINE-FLOW POI "
            "OUTFLOW REASSIGNMENT =========="
        )

        print(
            "Top-10% outflow: "
            f"{top10_summary['top_outflow']:.3f} visits; "
            "Top -> Middle: "
            f"{top10_summary['top_to_middle']:.3f} visits "
            f"({top10_summary['top_to_middle_share_of_top_outflow_pct']:.2f}% "
            "of Top outflow); "
            "Top -> Bottom: "
            f"{top10_summary['top_to_bottom']:.3f} visits "
            f"({top10_summary['top_to_bottom_share_of_top_outflow_pct']:.2f}% "
            "of Top outflow)"
        )

    print(
        "\n========== SPSE DESTINATION DATA =========="
    )

    cols_show = [
        "poi_id",
        "poi_tier",

        "baseline_total_visits",
        "optimized_total_visits",
        "delta_visits",

        "n_active_links_before",
        "n_active_links_after",
        "delta_active_links",

        "mean_delta_structural_social_exposure_per_visiting_cbg",

        "n_new_positive_links",
        "n_removed_positive_links",

        "spse_contribution_before",
        "spse_contribution_after",
        "delta_spse_contribution",

        "spse_from_new_links",
        "spse_lost_from_removed_links",
        "spse_change_on_retained_links",

        "weighted_contribution_before",
        "weighted_contribution_after",
        "delta_weighted_contribution",

        "positive_increment_received",
        "added_origin_income_score",

        "delta_flow_weighted_distance_km",
    ]

    print(
        case_output[
            "scatter_df"
        ][cols_show]
        .sort_values(
            "delta_visits",
            ascending=False
        )
        .round(4)
        .to_string(
            index=False
        )
    )

    print(
        "\n========== DECOMPOSITION CHECK =========="
    )

    s = case_output[
        "case_summary"
    ]

    print(
        "SPSE decomposition: "
        f"sum_j ΔSPSE_j = "
        f"{s['sum_delta_spse_from_pois']:.10f}; "
        f"system ΔSPSE = "
        f"{s['spse_delta']:.10f}; "
        f"error = "
        f"{s['spse_decomposition_error']:.10e}"
    )

    print(
        "Weighted exposure contribution decomposition: "
        f"sum_j ΔC_j = "
        f"{s['sum_delta_weighted_contribution_from_pois']:.10f}; "
        f"system ΔC = "
        f"{s['weighted_exposure_numerator_delta']:.10f}; "
        f"error = "
        f"{s['weighted_decomposition_error']:.10e}"
    )

    print(
        "Positive-link support decomposition: "
        f"sum_j Δlinks_j = "
        f"{s['sum_delta_active_links_from_pois']}; "
        f"system Δlinks = "
        f"{s['total_delta_active_links']}; "
        f"error = "
        f"{s['active_link_decomposition_error']}"
    )

    save_public_outputs(
        case_output
    )


if __name__ == "__main__":
    main()