# -*- coding: utf-8 -*-
"""
Preprocessing for Figure 4d:
Behavioural anchoring and retained performance

This script performs the matrix-level preprocessing required by Fig. 4d.

Inputs:
    H_opt_df_no_regu_boston_624190.pkl
    H_opt_df_regu_boston_624190.pkl
    pred_rownorm_int_preserve.csv
    flow_matrix.csv
    distance_matrix.csv
    social_exposure_matrix.csv

Outputs:
    figure4d_origin_pullback.csv
    figure4d_summary.csv

The plotting script figure4d.py subsequently uses these processed
outputs together with:

    results_regu_boston_624190.csv
    results_boston_624190.csv

@author: JZS
"""

import os
import pickle
import numpy as np
import pandas as pd


# ============================================================
# Configuration
# ============================================================

city = "boston"

category = "Other Individual and Family Services"

poi_code = "624190"


cat_dir = (
    f"matrices_A_D_S_Distribution/"
    f"{category.replace(' ', '_')}"
)


no_regu_file = os.path.join(
    cat_dir,
    f"H_opt_df_no_regu_{city}_{poi_code}.pkl"
)


regu_file = os.path.join(
    cat_dir,
    f"H_opt_df_regu_{city}_{poi_code}.pkl"
)


flow_matrix_path = os.path.join(
    cat_dir,
    "flow_matrix.csv"
)


distance_matrix_path = os.path.join(
    cat_dir,
    "distance_matrix.csv"
)


social_matrix_path = os.path.join(
    cat_dir,
    "social_exposure_matrix.csv"
)


R_path = os.path.join(
    cat_dir,
    "pred_rownorm_int_preserve.csv"
)


origin_output_path = (
    "figure4d_origin_pullback.csv"
)


summary_output_path = (
    "figure4d_summary.csv"
)


# ============================================================
# Helper functions
# ============================================================

def check_file(path):

    if not os.path.exists(path):

        raise FileNotFoundError(
            f"Cannot find file: {path}"
        )


def infer_cbg_pad_len(
    index_values
):
    """
    CBG GEOID is usually 12 digits.
    Use 12 as a safe lower bound.
    """

    lengths = [

        len(
            str(x)
        )

        for x
        in index_values
    ]


    return max(
        12,
        max(lengths)
    )


def load_H(
    path,
    baseline_df,
    pad_len
):
    """
    Load optimized H matrix and make
    index/columns string-compatible.
    """

    check_file(
        path
    )


    with open(
        path,
        "rb"
    ) as f:

        H = pickle.load(
            f
        )


    if isinstance(
        H,
        pd.DataFrame
    ):

        H = H.copy()


        H.index = [

            str(x).zfill(
                pad_len
            )

            for x
            in H.index
        ]


        H.columns = [

            str(x)

            for x
            in H.columns
        ]


        return H


    else:

        return pd.DataFrame(
            H,
            index=baseline_df.index,
            columns=baseline_df.columns
        )


def align_to_baseline(
    df,
    baseline_df,
    pad_len,
    fill_value=0.0
):
    """
    Align any matrix to baseline rows and columns.

    Rows are zero-padded CBG GEOIDs;
    columns are POI ids as strings.
    """

    X = df.copy()


    X.index = X.index.map(
        lambda x:
        str(x)
    )


    X.columns = X.columns.map(
        lambda x:
        str(x)
    )


    X_try = X.reindex(
        index=baseline_df.index,
        columns=baseline_df.columns
    )


    if X_try.notna().values.any():

        return X_try.fillna(
            fill_value
        )


    X2 = X.copy()


    try:

        X2.index = X2.index.map(
            lambda x:
            str(x).zfill(
                pad_len
            )
        )

    except Exception:

        pass


    X_try2 = X2.reindex(
        index=baseline_df.index,
        columns=baseline_df.columns
    )


    return X_try2.fillna(
        fill_value
    )


def l1_flow_distance_share(
    H,
    R,
    total_flow
):
    """
    Normalized L1 flow distance:

        0.5 * sum |H - R| / total_flow * 100

    Interpretation:

        percentage of total visits that would need
        to be reassigned to transform one matrix
        into the other.
    """

    if (
        total_flow == 0
        or
        np.isnan(total_flow)
    ):

        return np.nan


    H_arr = H.to_numpy(
        dtype=float
    )


    R_arr = R.to_numpy(
        dtype=float
    )


    return (
        0.5
        *
        np.abs(
            H_arr -
            R_arr
        ).sum()
        /
        total_flow
        *
        100.0
    )


def origin_l1_raw(
    A,
    B
):
    """
    Origin-level raw L1 reassignment volume:

        0.5 * sum_j |A_ij - B_ij|

    Interpretation:

        for each origin i, how many visits would
        need to be reassigned to transform row A_i
        into row B_i.
    """

    return (
        0.5
        *
        (
            A.astype(float)
            -
            B.astype(float)
        )
        .abs()
        .sum(
            axis=1
        )
    )


def top_share(
    sorted_values,
    top_pct
):
    """
    Share of total positive contribution accounted
    for by top_pct origins.

    sorted_values must be sorted in descending order.
    """

    values = np.asarray(
        sorted_values,
        dtype=float
    )


    n = len(
        values
    )


    if (
        n == 0
        or
        values.sum() <= 0
    ):

        return np.nan


    k = int(
        np.ceil(
            n
            *
            top_pct
            /
            100.0
        )
    )


    k = min(
        max(
            k,
            1
        ),
        n
    )


    return (
        values[:k].sum()
        /
        values.sum()
        *
        100.0
    )


def smooth_array(
    values,
    smooth_fraction=0.06
):
    """
    Mild centered rolling mean for cleaner line plots.

    Set smooth_fraction=0 to disable smoothing.
    """

    values = np.asarray(
        values,
        dtype=float
    )


    if len(
        values
    ) == 0:

        return values


    if smooth_fraction <= 0:

        return values


    window = max(
        3,
        int(
            round(
                len(values)
                *
                smooth_fraction
            )
        )
    )


    return (
        pd.Series(
            values
        )
        .rolling(
            window=window,
            center=True,
            min_periods=1
        )
        .mean()
        .to_numpy(
            dtype=float
        )
    )


def build_ranked_departure_curves(
    dev_no,
    dev_reg,
    contrib,
    changed,
    tol=1e-9,
    smooth_fraction=0.06
):
    """
    Build ranked origin-level departure curves for Panel a.

    Selection:
        keep all origins whose allocation changes
        between H_no and H_reg.

    Ranking:
        CBGs are ranked by

            c_i^F = d_i(H_no,F) - d_i(H_reg,F),

        descending.

    Interpretation:
        Left side:
            CBGs with largest positive pullback toward F.

        Right side:
            CBGs with small, zero, or negative pullback
            toward F.

    Returns:
        x_rank
        dev_no_plot
        dev_reg_plot
        dev_no_raw
        dev_reg_raw
        contrib_raw
        changed_raw
    """

    mask_changed = (
        changed > tol
    )


    dev_no_sel = (
        dev_no[
            mask_changed
        ]
        .copy()
    )


    dev_reg_sel = (
        dev_reg[
            mask_changed
        ]
        .copy()
    )


    contrib_sel = (
        contrib[
            mask_changed
        ]
        .copy()
    )


    changed_sel = (
        changed[
            mask_changed
        ]
        .copy()
    )


    if len(
        contrib_sel
    ) == 0:

        return (

            np.array([]),

            np.array([]),

            np.array([]),

            np.array([]),

            np.array([]),

            np.array([]),

            np.array([])
        )


    # --------------------------------------------------------
    # Rank all changed CBGs by anchoring contribution toward F
    # --------------------------------------------------------

    rank_index = (
        contrib_sel
        .sort_values(
            ascending=False
        )
        .index
    )


    dev_no_raw = (
        dev_no_sel
        .loc[
            rank_index
        ]
        .to_numpy(
            dtype=float
        )
    )


    dev_reg_raw = (
        dev_reg_sel
        .loc[
            rank_index
        ]
        .to_numpy(
            dtype=float
        )
    )


    contrib_raw = (
        contrib_sel
        .loc[
            rank_index
        ]
        .to_numpy(
            dtype=float
        )
    )


    changed_raw = (
        changed_sel
        .loc[
            rank_index
        ]
        .to_numpy(
            dtype=float
        )
    )


    x_rank = (
        np.arange(
            1,
            len(dev_no_raw) + 1
        )
        /
        len(dev_no_raw)
        *
        100.0
    )


    dev_no_plot = smooth_array(
        dev_no_raw,
        smooth_fraction=smooth_fraction
    )


    dev_reg_plot = smooth_array(
        dev_reg_raw,
        smooth_fraction=smooth_fraction
    )


    return (

        x_rank,

        dev_no_plot,

        dev_reg_plot,

        dev_no_raw,

        dev_reg_raw,

        contrib_raw,

        changed_raw
    )


# ============================================================
# Check inputs
# ============================================================

for path in [

    no_regu_file,

    regu_file,

    flow_matrix_path,

    distance_matrix_path,

    social_matrix_path,

    R_path

]:

    check_file(
        path
    )


# ============================================================
# Load baseline matrices
# ============================================================

flow_matrix = pd.read_csv(
    flow_matrix_path,
    index_col=0
)


distance_matrix = pd.read_csv(
    distance_matrix_path,
    index_col=0
)


social_exposure_matrix = pd.read_csv(
    social_matrix_path,
    index_col=0
)


R_pre = pd.read_csv(
    R_path,
    index_col=0
)


pad_len = infer_cbg_pad_len(
    flow_matrix.index
)


# ============================================================
# Select POIs used in the optimization domain
# ============================================================

poi_total_flow = (
    flow_matrix
    .sum(
        axis=0
    )
)


selected_pois = (
    poi_total_flow
    .sort_values(
        ascending=False
    )
    .head(
        53
    )
    .index
    .tolist()
)


# ============================================================
# Select all CBGs with positive flow to selected POIs
# ============================================================

selected_cbgs = set()


for poi in selected_pois:

    cbgs_with_flow = (
        flow_matrix.index[
            flow_matrix[
                poi
            ] > 0
        ]
        .tolist()
    )


    selected_cbgs.update(
        cbgs_with_flow
    )


selected_cbgs = list(
    selected_cbgs
)


# ============================================================
# Baseline matrix
# ============================================================

baseline = (
    flow_matrix.loc[
        selected_cbgs,
        selected_pois
    ]
    .copy()
)


baseline.index = [

    str(x).zfill(
        pad_len
    )

    for x
    in baseline.index
]


baseline.columns = [

    str(x)

    for x
    in baseline.columns
]


baseline = (
    baseline
    .apply(
        pd.to_numeric,
        errors="coerce"
    )
    .fillna(
        0.0
    )
)


total_flow = (
    baseline
    .to_numpy(
        dtype=float
    )
    .sum()
)


print(
    "\n========== Matrix domain =========="
)


print(
    f"n_CBGs: "
    f"{baseline.shape[0]}"
)


print(
    f"n_POIs: "
    f"{baseline.shape[1]}"
)


print(
    f"baseline total flow: "
    f"{total_flow:.6f}"
)


# ============================================================
# Load and align H_no, H_reg, R, distance, social matrices
# ============================================================

H_no_pre = load_H(
    no_regu_file,
    baseline,
    pad_len
)


H_reg_pre = load_H(
    regu_file,
    baseline,
    pad_len
)


H_no = align_to_baseline(
    H_no_pre,
    baseline,
    pad_len,
    fill_value=0.0
)


H_reg = align_to_baseline(
    H_reg_pre,
    baseline,
    pad_len,
    fill_value=0.0
)


R = align_to_baseline(
    R_pre,
    baseline,
    pad_len,
    fill_value=0.0
)


D = align_to_baseline(
    distance_matrix,
    baseline,
    pad_len,
    fill_value=np.nan
)


S0 = align_to_baseline(
    social_exposure_matrix,
    baseline,
    pad_len,
    fill_value=np.nan
)


H_no = (
    H_no
    .apply(
        pd.to_numeric,
        errors="coerce"
    )
    .fillna(
        0.0
    )
)


H_reg = (
    H_reg
    .apply(
        pd.to_numeric,
        errors="coerce"
    )
    .fillna(
        0.0
    )
)


R = (
    R
    .apply(
        pd.to_numeric,
        errors="coerce"
    )
    .fillna(
        0.0
    )
)


D = D.apply(
    pd.to_numeric,
    errors="coerce"
)


S0 = S0.apply(
    pd.to_numeric,
    errors="coerce"
)


# ============================================================
# Observed baseline values from matrices
# ============================================================

baseline_distance = float(

    (
        baseline.to_numpy(
            dtype=float
        )
        *
        D.to_numpy(
            dtype=float
        )
    )
    .sum()
)


baseline_social = float(

    np.nansum(

        (
            baseline
            .to_numpy(
                dtype=float
            )
            >
            0
        )
        .astype(
            float
        )

        *

        S0.to_numpy(
            dtype=float
        )
    )
)


print(
    "\n========== Observed baseline values from matrices =========="
)


print(
    f"baseline total distance: "
    f"{baseline_distance:.6f}"
)


print(
    f"baseline structural social exposure: "
    f"{baseline_social:.6f}"
)


# ============================================================
# Matrix-level behavioural anchoring diagnostics
# ============================================================

no_regu_ref_dev_pct = (
    l1_flow_distance_share(
        H_no,
        R,
        total_flow
    )
)


regu_ref_dev_pct = (
    l1_flow_distance_share(
        H_reg,
        R,
        total_flow
    )
)


if (
    no_regu_ref_dev_pct == 0
    or
    np.isnan(
        no_regu_ref_dev_pct
    )
    or
    np.isnan(
        regu_ref_dev_pct
    )
):

    ref_dev_reduction_pct = (
        np.nan
    )


else:

    ref_dev_reduction_pct = (

        (
            no_regu_ref_dev_pct
            -
            regu_ref_dev_pct
        )

        /

        no_regu_ref_dev_pct

        *

        100.0
    )


no_regu_from_F_pct = (
    l1_flow_distance_share(
        H_no,
        baseline,
        total_flow
    )
)


regu_from_F_pct = (
    l1_flow_distance_share(
        H_reg,
        baseline,
        total_flow
    )
)


R_from_F_pct = (
    l1_flow_distance_share(
        R,
        baseline,
        total_flow
    )
)


if (
    no_regu_from_F_pct == 0
    or
    np.isnan(
        no_regu_from_F_pct
    )
    or
    np.isnan(
        regu_from_F_pct
    )
):

    from_F_reduction_pct = (
        np.nan
    )


else:

    from_F_reduction_pct = (

        (
            no_regu_from_F_pct
            -
            regu_from_F_pct
        )

        /

        no_regu_from_F_pct

        *

        100.0
    )


print(
    "\n========== Behavioural anchoring diagnostics =========="
)


print(
    f"Deviation from R, unregularized H_no:  "
    f"{no_regu_ref_dev_pct:.6f}%"
)


print(
    f"Deviation from R, regularized H_reg:   "
    f"{regu_ref_dev_pct:.6f}%"
)


print(
    f"Reduction in deviation from R:         "
    f"{ref_dev_reduction_pct:.6f}%"
)


print(
    f"\nDeviation from observed F, "
    f"unregularized H_no: "
    f"{no_regu_from_F_pct:.6f}%"
)


print(
    f"Deviation from observed F, "
    f"regularized H_reg:  "
    f"{regu_from_F_pct:.6f}%"
)


print(
    f"Reduction in deviation from observed F:        "
    f"{from_F_reduction_pct:.6f}%"
)


print(
    f"Diagnostic only, not shown in figure: "
    f"D(F,R) = "
    f"{R_from_F_pct:.6f}%"
)


# ============================================================
# Origin-level anchoring contribution for Panel a
# ============================================================

# ------------------------------------------------------------
# Raw origin-level departure from F
# ------------------------------------------------------------

dev_no_from_F_raw = (
    origin_l1_raw(
        H_no,
        baseline
    )
)


dev_reg_from_F_raw = (
    origin_l1_raw(
        H_reg,
        baseline
    )
)


contrib_from_F_raw = (
    dev_no_from_F_raw
    -
    dev_reg_from_F_raw
)


# ------------------------------------------------------------
# Raw origin-level departure from R,
# used for diagnostics only
# ------------------------------------------------------------

dev_no_from_R_raw = (
    origin_l1_raw(
        H_no,
        R
    )
)


dev_reg_from_R_raw = (
    origin_l1_raw(
        H_reg,
        R
    )
)


contrib_from_R_raw = (
    dev_no_from_R_raw
    -
    dev_reg_from_R_raw
)


# ============================================================
# Positive contributions
# ============================================================

tol = 1e-9


contrib_from_F_pos = (
    contrib_from_F_raw
    .clip(
        lower=0.0
    )
)


contrib_from_R_pos = (
    contrib_from_R_raw
    .clip(
        lower=0.0
    )
)


# ============================================================
# Negative offsets
# ============================================================

contrib_from_F_neg = (

    -
    contrib_from_F_raw
    .clip(
        upper=0.0
    )
)


contrib_from_R_neg = (

    -
    contrib_from_R_raw
    .clip(
        upper=0.0
    )
)


# ============================================================
# Convert net raw reduction to percentage points of total flow
# ============================================================

net_reduction_from_F_pp = (

    contrib_from_F_raw.sum()
    /
    total_flow
    *
    100.0
)


net_reduction_from_R_pp = (

    contrib_from_R_raw.sum()
    /
    total_flow
    *
    100.0
)


positive_reduction_from_F_pp = (

    contrib_from_F_pos.sum()
    /
    total_flow
    *
    100.0
)


positive_reduction_from_R_pp = (

    contrib_from_R_pos.sum()
    /
    total_flow
    *
    100.0
)


negative_offset_from_F_pp = (

    contrib_from_F_neg.sum()
    /
    total_flow
    *
    100.0
)


negative_offset_from_R_pp = (

    contrib_from_R_neg.sum()
    /
    total_flow
    *
    100.0
)


# ============================================================
# Sorted positive contributions for summary diagnostics
# ============================================================

sorted_F_pos = np.sort(

    contrib_from_F_pos[
        contrib_from_F_pos > tol
    ]
    .to_numpy(
        dtype=float
    )

)[::-1]


sorted_R_pos = np.sort(

    contrib_from_R_pos[
        contrib_from_R_pos > tol
    ]
    .to_numpy(
        dtype=float
    )

)[::-1]


top10_F = top_share(
    sorted_F_pos,
    10
)


top20_F = top_share(
    sorted_F_pos,
    20
)


top10_R = top_share(
    sorted_R_pos,
    10
)


top20_R = top_share(
    sorted_R_pos,
    20
)


share_positive_F = (

    (
        contrib_from_F_pos
        >
        tol
    )
    .mean()

    *

    100.0
)


share_positive_R = (

    (
        contrib_from_R_pos
        >
        tol
    )
    .mean()

    *

    100.0
)


share_positive_both = (

    (
        (
            contrib_from_F_pos
            >
            tol
        )

        &

        (
            contrib_from_R_pos
            >
            tol
        )
    )
    .mean()

    *

    100.0
)


# ============================================================
# Change between H_no and H_reg
# ============================================================

change_no_to_reg_raw = (
    origin_l1_raw(
        H_no,
        H_reg
    )
)


# ============================================================
# Ranked departure curves for Panel a
# ============================================================

(
    x_rank_F,

    dev_no_F_plot,

    dev_reg_F_plot,

    dev_no_F_rank_raw,

    dev_reg_F_rank_raw,

    contrib_F_rank_raw,

    changed_F_rank_raw

) = build_ranked_departure_curves(

    dev_no=dev_no_from_F_raw,

    dev_reg=dev_reg_from_F_raw,

    contrib=contrib_from_F_raw,

    changed=change_no_to_reg_raw,

    tol=tol,

    smooth_fraction=0.06
)


if len(
    x_rank_F
) == 0:

    raise ValueError(
        "No positive-contribution CBGs found for c_i^F."
    )


print(
    "\n========== Panel a anchoring contribution diagnostics =========="
)


print(
    f"Net reduction from F:       "
    f"{net_reduction_from_F_pp:.6f} pp"
)


print(
    f"Positive reduction from F:  "
    f"{positive_reduction_from_F_pp:.6f} pp"
)


print(
    f"Negative offset from F:     "
    f"{negative_offset_from_F_pp:.6f} pp"
)


print(
    f"Origins with positive F contribution: "
    f"{share_positive_F:.2f}%"
)


print(
    f"Top 10% positive F origins account for F reduction: "
    f"{top10_F:.2f}%"
)


print(
    f"Top 20% positive F origins account for F reduction: "
    f"{top20_F:.2f}%"
)


print(
    f"\nNet reduction from R:       "
    f"{net_reduction_from_R_pp:.6f} pp"
)


print(
    f"Positive reduction from R:  "
    f"{positive_reduction_from_R_pp:.6f} pp"
)


print(
    f"Negative offset from R:     "
    f"{negative_offset_from_R_pp:.6f} pp"
)


print(
    f"Origins with positive R contribution: "
    f"{share_positive_R:.2f}%"
)


print(
    f"Top 10% positive R origins account for R reduction: "
    f"{top10_R:.2f}%"
)


print(
    f"Top 20% positive R origins account for R reduction: "
    f"{top20_R:.2f}%"
)


print(
    f"\nOrigins with positive contribution to both F and R: "
    f"{share_positive_both:.2f}%"
)


# ============================================================
# Share of origins changed between H_no and H_reg
# ============================================================

share_changed_no_to_reg = (

    (
        change_no_to_reg_raw
        >
        tol
    )
    .mean()

    *

    100.0
)


# ============================================================
# Save Panel a ranked data
# ============================================================

origin_pullback_df = pd.DataFrame({

    "rank_pct":
        x_rank_F,

    "dev_no_F_plot":
        dev_no_F_plot,

    "dev_reg_F_plot":
        dev_reg_F_plot,

    "dev_no_F_raw":
        dev_no_F_rank_raw,

    "dev_reg_F_raw":
        dev_reg_F_rank_raw,

    "contrib_F_raw":
        contrib_F_rank_raw,

    "changed_Hno_Hreg_raw":
        changed_F_rank_raw
})


origin_pullback_df.to_csv(
    origin_output_path,
    index=False
)


# ============================================================
# Save all scalar quantities required by Fig.4d
# ============================================================

summary_rows = [

    {
        "metric":
            "n_CBGs",

        "value":
            baseline.shape[0]
    },

    {
        "metric":
            "n_POIs",

        "value":
            baseline.shape[1]
    },

    {
        "metric":
            "total_flow",

        "value":
            total_flow
    },

    {
        "metric":
            "baseline_distance",

        "value":
            baseline_distance
    },

    {
        "metric":
            "baseline_social",

        "value":
            baseline_social
    },

    {
        "metric":
            "no_regu_ref_dev_pct",

        "value":
            no_regu_ref_dev_pct
    },

    {
        "metric":
            "regu_ref_dev_pct",

        "value":
            regu_ref_dev_pct
    },

    {
        "metric":
            "ref_dev_reduction_pct",

        "value":
            ref_dev_reduction_pct
    },

    {
        "metric":
            "no_regu_from_F_pct",

        "value":
            no_regu_from_F_pct
    },

    {
        "metric":
            "regu_from_F_pct",

        "value":
            regu_from_F_pct
    },

    {
        "metric":
            "R_from_F_pct",

        "value":
            R_from_F_pct
    },

    {
        "metric":
            "from_F_reduction_pct",

        "value":
            from_F_reduction_pct
    },

    {
        "metric":
            "net_reduction_from_F_pp",

        "value":
            net_reduction_from_F_pp
    },

    {
        "metric":
            "positive_reduction_from_F_pp",

        "value":
            positive_reduction_from_F_pp
    },

    {
        "metric":
            "negative_offset_from_F_pp",

        "value":
            negative_offset_from_F_pp
    },

    {
        "metric":
            "share_positive_F",

        "value":
            share_positive_F
    },

    {
        "metric":
            "top10_F",

        "value":
            top10_F
    },

    {
        "metric":
            "top20_F",

        "value":
            top20_F
    },

    {
        "metric":
            "net_reduction_from_R_pp",

        "value":
            net_reduction_from_R_pp
    },

    {
        "metric":
            "positive_reduction_from_R_pp",

        "value":
            positive_reduction_from_R_pp
    },

    {
        "metric":
            "negative_offset_from_R_pp",

        "value":
            negative_offset_from_R_pp
    },

    {
        "metric":
            "share_positive_R",

        "value":
            share_positive_R
    },

    {
        "metric":
            "top10_R",

        "value":
            top10_R
    },

    {
        "metric":
            "top20_R",

        "value":
            top20_R
    },

    {
        "metric":
            "share_positive_both",

        "value":
            share_positive_both
    },

    {
        "metric":
            "share_changed_no_to_reg",

        "value":
            share_changed_no_to_reg
    }
]


summary_df = pd.DataFrame(
    summary_rows
)


summary_df.to_csv(
    summary_output_path,
    index=False
)


# ============================================================
# Final output summary
# ============================================================

print(
    "\n========== FIGURE 4d PREPROCESSING COMPLETE =========="
)


print(
    f"Origin-level pullback data saved to:\n"
    f"{origin_output_path}"
)


print(
    f"\nSummary data saved to:\n"
    f"{summary_output_path}"
)