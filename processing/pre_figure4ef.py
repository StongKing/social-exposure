# -*- coding: utf-8 -*-
"""
Preprocessing for Fig. 4e and Fig. 4f

Outputs
-------
1. figure4e_cbg_data.csv

   Contains:
       GEOID
       fc_reg
       high_income_pct

   where:
       fc_reg = sum_j |H_reg_ij - F_ij|

2. figure4f_high_income_visit_distribution.csv

   Contains normalized high-income visit distributions
   across POIs under:

       baseline
       regularized allocation
       unregularized allocation

Required inputs
---------------
Private:
    flow_matrix.csv

Public:
    H_opt_df_no_regu_boston_624190.pkl
    H_opt_df_regu_boston_624190.pkl
    cbg_income_level_distribution_boston_msa.csv
    tl_2021_boston_msa_bg.shp

@author: JZS
"""

import os
import pickle

import numpy as np
import pandas as pd
import geopandas as gpd


# ============================================================
# Configuration
# ============================================================

city = 'boston'

category = (
    'Other Individual and Family Services'
)

cat_dir = (
    f'matrices_A_D_S_Distribution/'
    f'{category.replace(" ", "_")}'
)


# ============================================================
# Input files
# ============================================================

flow_matrix_path = os.path.join(
    cat_dir,
    'flow_matrix.csv'
)


no_regu_file = os.path.join(
    cat_dir,
    f'H_opt_df_no_regu_{city}_624190.pkl'
)


regu_file = os.path.join(
    cat_dir,
    f'H_opt_df_regu_{city}_624190.pkl'
)


income_file = os.path.join(
    cat_dir,
    f'cbg_income_level_distribution_{city}_msa.csv'
)


cbg_shapefile = (
    'geo_data/'
    'tl_2021_boston_msa_bg/'
    'tl_2021_boston_msa_bg.shp'
)


# ============================================================
# Output files
# ============================================================

figure4e_output = (
    'figure4e_cbg_data.csv'
)


figure4f_output = (
    'figure4f_high_income_visit_distribution.csv'
)


# ============================================================
# Check input files
# ============================================================

for path in [

    flow_matrix_path,

    no_regu_file,

    regu_file,

    income_file,

    cbg_shapefile

]:

    if not os.path.exists(path):

        raise FileNotFoundError(
            f'Cannot find required file:\n{path}'
        )


# ============================================================
# Load baseline flow matrix
# ============================================================

print(
    '[LOAD flow matrix]'
)

print(
    flow_matrix_path
)


flow_matrix = pd.read_csv(
    flow_matrix_path,
    index_col=0
)


# ============================================================
# Load income composition
# ============================================================

print(
    '\n[LOAD income composition]'
)

print(
    income_file
)


cbg_income_dist_df = pd.read_csv(
    income_file,
    dtype={
        'GEOID':
            np.int64
    }
)


cbg_income_dist_dict = (
    cbg_income_dist_df
    .set_index(
        'GEOID'
    )
    .to_dict(
        orient='index'
    )
)


income_levels = [

    'low_income_pct',

    'lower_middle_income_pct',

    'upper_middle_income_pct',

    'high_income_pct'
]


# ============================================================
# Determine GEOID padding length
# ============================================================

print(
    '\n[LOAD Boston CBG shapefile]'
)

print(
    cbg_shapefile
)


boston_msa_cbg = gpd.read_file(
    cbg_shapefile
)


boston_msa_cbg['GEOID'] = (
    boston_msa_cbg[
        'GEOID'
    ]
    .astype(str)
)


pad_len = int(
    boston_msa_cbg[
        'GEOID'
    ]
    .str.len()
    .max()
)


print(
    'GEOID pad length =',
    pad_len
)


# ============================================================
# Reproduce selected_pois
#
# Keep exactly the same domain construction as the
# original Fig.4e / Fig.4f script.
# ============================================================

poi_total_flow = (
    flow_matrix
    .sum(
        axis=0
    )
)


poi_num = (
    flow_matrix
    .shape[1]
)


selected_pois = (
    poi_total_flow
    .sort_values(
        ascending=False
    )
    .head(
        poi_num
    )
    .index
    .tolist()
)


# ============================================================
# Reproduce selected_cbgs
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
# Baseline F
# ============================================================

A_sub_full = flow_matrix.loc[
    selected_cbgs,
    selected_pois
]


baseline = (
    A_sub_full
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
        errors='coerce'
    )
    .fillna(
        0.0
    )
)


print(
    '\nAnalysis domain'
)


print(
    'Number of CBGs =',
    baseline.shape[0]
)


print(
    'Number of POIs =',
    baseline.shape[1]
)


# ============================================================
# Load H
# ============================================================

def load_H(path):

    if not os.path.exists(path):

        raise FileNotFoundError(
            f'Cannot find file: {path}. '
            f'Please check the H file path.'
        )


    with open(
        path,
        'rb'
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

            index=baseline.index,

            columns=baseline.columns
        )


# ============================================================
# Align H to baseline
# ============================================================

def align_H_to_baseline(
    H_df,
    baseline_df
):
    """
    Align H_df to baseline.

    Rows:
        zero-padded GEOID strings

    Columns:
        POI strings

    First try direct string matching.

    If all entries are NaN after direct matching,
    zero-pad H indices and retry.
    """

    H = H_df.copy()


    H.index = H.index.map(
        lambda x:
        str(x)
    )


    H.columns = H.columns.map(
        lambda x:
        str(x)
    )


    # --------------------------------------------------------
    # Direct matching
    # --------------------------------------------------------

    H_try = H.reindex(
        index=baseline_df.index,
        columns=baseline_df.columns
    )


    if H_try.notna().values.any():

        return H_try.fillna(
            0.0
        )


    # --------------------------------------------------------
    # Retry with padded GEOIDs
    # --------------------------------------------------------

    H_index_padded = (
        H.copy()
    )


    try:

        H_index_padded.index = (
            H_index_padded
            .index
            .map(
                lambda x:
                str(x).zfill(
                    pad_len
                )
            )
        )


    except Exception:

        pass


    H_try2 = H_index_padded.reindex(
        index=baseline_df.index,
        columns=baseline_df.columns
    )


    return H_try2.fillna(
        0.0
    )


# ============================================================
# Load and align H_no and H_reg
# ============================================================

print(
    '\n[LOAD H_no]'
)

print(
    no_regu_file
)


H_no_pre = load_H(
    no_regu_file
)


print(
    '\n[LOAD H_reg]'
)

print(
    regu_file
)


H_reg_pre = load_H(
    regu_file
)


H_no = align_H_to_baseline(
    H_no_pre,
    baseline
)


H_reg = align_H_to_baseline(
    H_reg_pre,
    baseline
)


H_no = (
    H_no
    .apply(
        pd.to_numeric,
        errors='coerce'
    )
    .fillna(
        0.0
    )
)


H_reg = (
    H_reg
    .apply(
        pd.to_numeric,
        errors='coerce'
    )
    .fillna(
        0.0
    )
)


# ============================================================
# Build income matrix P
#
# IMPORTANT:
# This retains the same normalization procedure used
# in the original Fig.4e / Fig.4f code.
# ============================================================

def build_income_matrix(
    baseline_index,
    cbg_income_dist_dict,
    income_levels
):

    rows = []

    idxs = []


    for g in baseline_index:

        # ----------------------------------------------------
        # Try integer key variants
        # ----------------------------------------------------

        try:

            key = int(
                g
            )


        except Exception:

            try:

                key = int(
                    g.lstrip(
                        '0'
                    )
                )


            except Exception:

                key = None


        if (
            key is not None
            and
            key in cbg_income_dist_dict
        ):

            row = (
                cbg_income_dist_dict[
                    key
                ]
            )


            vals = [

                row.get(
                    lvl,
                    np.nan
                )

                for lvl
                in income_levels
            ]


            rows.append(
                vals
            )


            idxs.append(
                g
            )


        else:

            # ------------------------------------------------
            # Same original behavior:
            # missing income data -> zeros
            # ------------------------------------------------

            rows.append(
                [
                    0.0
                ]
                *
                len(
                    income_levels
                )
            )


            idxs.append(
                g
            )


    P_df = pd.DataFrame(

        rows,

        index=idxs,

        columns=income_levels
    )


    # --------------------------------------------------------
    # Convert percentages to fractions when needed
    # --------------------------------------------------------

    row_sums = (
        P_df
        .sum(
            axis=1
        )
    )


    if (
        row_sums > 1.5
    ).any():

        P_df = (
            P_df
            /
            100.0
        )


    # --------------------------------------------------------
    # Normalize each nonzero row to sum to 1
    # --------------------------------------------------------

    rs = (
        P_df
        .sum(
            axis=1
        )
    )


    nonzero = (
        rs > 0
    )


    P_df.loc[
        nonzero,
        :
    ] = (
        P_df.loc[
            nonzero,
            :
        ]
        .div(
            rs[
                nonzero
            ],
            axis=0
        )
    )


    return P_df


P_df = build_income_matrix(

    baseline.index,

    cbg_income_dist_dict,

    income_levels
)


# ============================================================
# Fig.4e:
# absolute regularized flow change
#
# IMPORTANT:
# Preserve original definition exactly:
#
#     fc_reg = sum_j |H_reg_ij - F_ij|
#
# There is NO factor 1/2 here because the original
# Fig.4e code did not use one.
# ============================================================

fc_reg = (
    H_reg
    -
    baseline
).abs().sum(
    axis=1
)


# ============================================================
# Prepare Fig.4e public data
# ============================================================

figure4e_df = pd.DataFrame({

    'GEOID':
        baseline.index,

    'fc_reg':
        fc_reg.reindex(
            baseline.index
        ).values,

    'high_income_pct':
        P_df[
            'high_income_pct'
        ]
        .reindex(
            baseline.index
        )
        .values
})


figure4e_df[
    'GEOID'
] = (
    figure4e_df[
        'GEOID'
    ]
    .astype(str)
    .str.zfill(
        pad_len
    )
)


figure4e_df.to_csv(
    figure4e_output,
    index=False
)


print(
    '\nFig.4e public data saved:'
)


print(
    figure4e_output
)


print(
    'Number of CBG rows =',
    len(
        figure4e_df
    )
)


print(
    'Number of changed CBGs =',
    int(
        (
            figure4e_df[
                'fc_reg'
            ] > 0
        )
        .sum()
    )
)


# ============================================================
# Fig.4f helper:
# high-income visit distribution across POIs
#
# This is copied from the original calculation.
# ============================================================

def compute_high_income_visit_dist(
    H_df,
    income_share_series
):

    H_local = H_df.reindex(

        index=income_share_series.index,

        columns=H_df.columns,

        fill_value=0
    )


    hi_visits = (

        H_local
        .multiply(
            income_share_series,
            axis=0
        )
        .sum(
            axis=0
        )
    )


    total = (
        hi_visits.sum()
    )


    if total == 0:

        return (
            hi_visits
            *
            0.0
        )


    return (
        hi_visits
        /
        total
    )


# ============================================================
# High-income visit distributions
# ============================================================

high_dist_A = (
    compute_high_income_visit_dist(

        baseline,

        P_df[
            'high_income_pct'
        ]
    )
)


high_dist_no = (
    compute_high_income_visit_dist(

        H_no,

        P_df[
            'high_income_pct'
        ]
    )
)


high_dist_reg = (
    compute_high_income_visit_dist(

        H_reg,

        P_df[
            'high_income_pct'
        ]
    )
)


# ============================================================
# Prepare Fig.4f public data
# ============================================================

figure4f_df = pd.DataFrame({

    'POI':
        baseline.columns,

    'baseline':
        high_dist_A
        .reindex(
            baseline.columns
        )
        .values,

    'regu':
        high_dist_reg
        .reindex(
            baseline.columns
        )
        .values,

    'no_regu':
        high_dist_no
        .reindex(
            baseline.columns
        )
        .values
})


figure4f_df[
    'POI'
] = (
    figure4f_df[
        'POI'
    ]
    .astype(str)
)


figure4f_df.to_csv(
    figure4f_output,
    index=False
)


print(
    '\nFig.4f public data saved:'
)


print(
    figure4f_output
)


print(
    'Number of POIs =',
    len(
        figure4f_df
    )
)


print(
    '\nDistribution sums:'
)


print(
    'baseline =',
    figure4f_df[
        'baseline'
    ].sum()
)


print(
    'regu =',
    figure4f_df[
        'regu'
    ].sum()
)


print(
    'no_regu =',
    figure4f_df[
        'no_regu'
    ].sum()
)


print(
    '\n=============================================='
)


print(
    'FIG.4E / FIG.4F PREPROCESSING COMPLETE'
)


print(
    '=============================================='
)