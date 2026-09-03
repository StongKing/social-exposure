# -*- coding: utf-8 -*-
"""
Preprocessing for Fig. 4c

This script prepares the public plotting/statistical input:

    figure4c_income_groups.csv

The preprocessing determines adjusted and unadjusted origins
under two allocation conditions:

    1. No regularization
    2. Behavioral anchoring

Adjustment magnitude for origin i:

    d_i(H,F) = 1/2 * sum_j |H_ij - F_ij|

Adjusted:
    d_i > 1e-10

Unadjusted:
    d_i <= 1e-10

Required inputs
---------------
Private / restricted input:

    matrices_A_D_S_Distribution/
        Other_Individual_and_Family_Services/
            flow_matrix.csv

Public inputs:

    matrices_A_D_S_Distribution/
        Other_Individual_and_Family_Services/
            cbg_income_level_distribution_boston_msa.csv
            H_opt_df_no_regu_boston_624190.pkl
            H_opt_df_regu_boston_624190.pkl

    geo_data/
        tl_2021_boston_msa_bg/
            tl_2021_boston_msa_bg.shp

Output
------
    figure4c_income_groups.csv

@author: JZS
"""

import os
import pickle
import numpy as np
import pandas as pd
import geopandas as gpd


# ============================================================
# 1. Configuration
# ============================================================

city = 'boston'

category = 'Other Individual and Family Services'

cat_dir = (
    f'matrices_A_D_S_Distribution/'
    f'{category.replace(" ", "_")}'
)


# ============================================================
# 2. Input files
# ============================================================

flow_file = os.path.join(
    cat_dir,
    'flow_matrix.csv'
)


income_file = os.path.join(
    cat_dir,
    f'cbg_income_level_distribution_{city}_msa.csv'
)


no_regu_file = os.path.join(
    cat_dir,
    f'H_opt_df_no_regu_{city}_624190.pkl'
)


regu_file = os.path.join(
    cat_dir,
    f'H_opt_df_regu_{city}_624190.pkl'
)


shapefile_path = (
    'geo_data/'
    'tl_2021_boston_msa_bg/'
    'tl_2021_boston_msa_bg.shp'
)


# ============================================================
# 3. Output file
# ============================================================

output_file = (
    'figure4c_income_groups.csv'
)


# ============================================================
# 4. Check required files
# ============================================================

for path in [

    flow_file,

    income_file,

    no_regu_file,

    regu_file,

    shapefile_path

]:

    if not os.path.exists(path):

        raise FileNotFoundError(
            f'Cannot find required file:\n{path}'
        )


# ============================================================
# 5. Load baseline flow matrix
# ============================================================

print(
    '[LOAD baseline flow]'
)

print(
    flow_file
)


flow_matrix = pd.read_csv(
    flow_file,
    index_col=0
)


# ============================================================
# 6. Load CBG income composition
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
        'GEOID': np.int64
    }
)


cbg_income_dist_dict = (
    cbg_income_dist_df
    .set_index('GEOID')
    .to_dict(
        orient='index'
    )
)


# ============================================================
# 7. Determine GEOID length
# ============================================================

print(
    '\n[LOAD shapefile]'
)

print(
    shapefile_path
)


boston_msa_cbg = gpd.read_file(
    shapefile_path
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
    '\nGEOID pad length =',
    pad_len
)


# ============================================================
# 8. Income components
# ============================================================

income_levels = [

    'low_income_pct',

    'lower_middle_income_pct',

    'upper_middle_income_pct',

    'high_income_pct'
]


# ============================================================
# 9. Reconstruct analysis domain
#
# Keep only POIs with positive observed baseline flow.
#
# This corresponds to the active POIs used in Fig.4c.
# ============================================================

poi_total_flow = (
    flow_matrix
    .sum(axis=0)
)


selected_pois = (
    poi_total_flow[
        poi_total_flow > 0
    ]
    .sort_values(
        ascending=False
    )
    .index
    .tolist()
)


print(
    '\nNumber of active POIs =',
    len(selected_pois)
)


# ============================================================
# 10. Select CBGs with positive baseline visits to at least
#     one active POI
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
# 11. Baseline submatrix
# ============================================================

A_sub_full = flow_matrix.loc[
    selected_cbgs,
    selected_pois
]


baseline = (
    A_sub_full.copy()
)


baseline.index = [

    str(x).zfill(
        pad_len
    )

    for x in baseline.index
]


print(
    'Number of CBGs in analysis =',
    len(baseline)
)


# ============================================================
# 12. Helper: load H matrix
# ============================================================

def load_H(path):

    if not os.path.exists(path):

        raise FileNotFoundError(
            f"Cannot find file:\n{path}\n"
            f"Please check the saved H_opt_df file."
        )


    with open(
        path,
        'rb'
    ) as f:

        H = pickle.load(f)


    if isinstance(
        H,
        pd.DataFrame
    ):

        H = H.copy()

    else:

        H = pd.DataFrame(
            H,
            index=baseline.index,
            columns=baseline.columns
        )


    return H


# ============================================================
# 13. Helper:
#
# Origin-level reassignment magnitude
#
# d_i(H,F)
# =
# 1/2 sum_j |H_ij - F_ij|
#
# adjusted:
#     d_i > tolerance
#
# unadjusted:
#     d_i <= tolerance
# ============================================================

def compute_flow_change_and_groups(
    H_df,
    baseline_df,
    tolerance=1e-10
):

    H_local = (
        H_df.copy()
    )


    H_local.index = [

        str(x).zfill(
            pad_len
        )

        for x
        in H_local.index
    ]


    # --------------------------------------------------------
    # Align optimized allocation with baseline analysis domain
    # --------------------------------------------------------

    H_aligned = H_local.reindex(
        index=baseline_df.index,
        columns=baseline_df.columns,
        fill_value=0
    )


    A_aligned = baseline_df.reindex(
        index=baseline_df.index,
        columns=baseline_df.columns,
        fill_value=0
    )


    # --------------------------------------------------------
    # Absolute flow difference
    # --------------------------------------------------------

    diff_abs = (
        H_aligned -
        A_aligned
    ).abs()


    # --------------------------------------------------------
    # One moved visit creates two absolute changes:
    #
    # -1 at source POI
    # +1 at destination POI
    #
    # Hence factor 1/2.
    # --------------------------------------------------------

    flow_change = (
        0.5 *
        diff_abs.sum(
            axis=1
        )
    )


    fc_selected = (
        flow_change
        .reindex(
            baseline_df.index
        )
        .fillna(
            0.0
        )
    )


    # --------------------------------------------------------
    # Adjusted origins
    # --------------------------------------------------------

    changed_geo = (
        fc_selected[
            fc_selected > tolerance
        ]
        .index
        .tolist()
    )


    # --------------------------------------------------------
    # Unadjusted origins
    # --------------------------------------------------------

    unchanged_geo = (
        fc_selected[
            fc_selected <= tolerance
        ]
        .index
        .tolist()
    )


    return (
        fc_selected,
        changed_geo,
        unchanged_geo
    )


# ============================================================
# 14. Helper:
# extract income composition for selected GEOIDs
# ============================================================

def extract_income_df_for_geoids(
    geoids,
    cbg_income_dist_dict,
    income_levels
):

    rows = []

    ids = []


    for g in geoids:

        g_str = str(g).zfill(
            pad_len
        )


        try:

            key = int(
                g_str
            )

        except Exception:

            try:

                key = int(
                    g_str.lstrip(
                        '0'
                    )
                )

            except Exception:

                continue


        if key not in cbg_income_dist_dict:

            continue


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

        ids.append(
            g_str
        )


    if len(rows) == 0:

        return pd.DataFrame(
            columns=income_levels
        )


    return pd.DataFrame(
        rows,
        index=ids,
        columns=income_levels
    )


# ============================================================
# 15. Load optimized allocations
# ============================================================

print(
    '\n[LOAD no-regularization allocation]'
)

print(
    no_regu_file
)


H_no = load_H(
    no_regu_file
)


print(
    '\n[LOAD behavioral-anchoring allocation]'
)

print(
    regu_file
)


H_reg = load_H(
    regu_file
)


# ============================================================
# 16. Determine adjusted / unadjusted origins
# ============================================================

fc_no, changed_no, unchanged_no = (
    compute_flow_change_and_groups(
        H_no,
        baseline
    )
)


fc_reg, changed_reg, unchanged_reg = (
    compute_flow_change_and_groups(
        H_reg,
        baseline
    )
)


print(
    "\nCounts (no regularization): "
    "adjusted =",
    len(changed_no),
    ", unadjusted =",
    len(unchanged_no)
)


print(
    "Counts (behavioral anchoring): "
    "adjusted =",
    len(changed_reg),
    ", unadjusted =",
    len(unchanged_reg)
)


# ============================================================
# 17. Extract income composition
# ============================================================

income_changed_no = (
    extract_income_df_for_geoids(
        changed_no,
        cbg_income_dist_dict,
        income_levels
    )
)


income_unchanged_no = (
    extract_income_df_for_geoids(
        unchanged_no,
        cbg_income_dist_dict,
        income_levels
    )
)


income_changed_reg = (
    extract_income_df_for_geoids(
        changed_reg,
        cbg_income_dist_dict,
        income_levels
    )
)


income_unchanged_reg = (
    extract_income_df_for_geoids(
        unchanged_reg,
        cbg_income_dist_dict,
        income_levels
    )
)


# ============================================================
# 18. Prepare public Fig.4c dataset
# ============================================================

def make_output_block(
    income_df,
    condition,
    status
):

    df = (
        income_df
        .copy()
        .reset_index()
        .rename(
            columns={
                'index':
                    'GEOID'
            }
        )
    )


    df.insert(
        0,
        'status',
        status
    )


    df.insert(
        0,
        'condition',
        condition
    )


    return df


# ------------------------------------------------------------
# No regularization
# ------------------------------------------------------------

output_changed_no = make_output_block(
    income_changed_no,
    'No regularization',
    'adjusted'
)


output_unchanged_no = make_output_block(
    income_unchanged_no,
    'No regularization',
    'unadjusted'
)


# ------------------------------------------------------------
# Behavioral anchoring
# ------------------------------------------------------------

output_changed_reg = make_output_block(
    income_changed_reg,
    'Behavioral anchoring',
    'adjusted'
)


output_unchanged_reg = make_output_block(
    income_unchanged_reg,
    'Behavioral anchoring',
    'unadjusted'
)


# ============================================================
# 19. Combine public data
# ============================================================

figure4c_data = pd.concat(

    [

        output_changed_no,

        output_unchanged_no,

        output_changed_reg,

        output_unchanged_reg

    ],

    axis=0,

    ignore_index=True
)


# ============================================================
# 20. Ensure GEOID is stored as zero-padded string
# ============================================================

figure4c_data['GEOID'] = (
    figure4c_data[
        'GEOID'
    ]
    .astype(str)
    .str.zfill(
        pad_len
    )
)


# ============================================================
# 21. Final checks
# ============================================================

expected_columns = [

    'condition',

    'status',

    'GEOID',

    'low_income_pct',

    'lower_middle_income_pct',

    'upper_middle_income_pct',

    'high_income_pct'
]


figure4c_data = (
    figure4c_data[
        expected_columns
    ]
)


print(
    '\n'
    '=========================================================='
)

print(
    'FIG.4C PUBLIC DATA SUMMARY'
)

print(
    '=========================================================='
)


summary = (
    figure4c_data
    .groupby(
        [
            'condition',
            'status'
        ]
    )
    .size()
)


print(
    summary
)


# ============================================================
# 22. Save public Fig.4c input
# ============================================================

figure4c_data.to_csv(
    output_file,
    index=False
)


print(
    '\nPublic Fig.4c data saved to:'
)

print(
    output_file
)


print(
    '\nPreprocessing complete.'
)