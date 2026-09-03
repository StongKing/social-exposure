# -*- coding: utf-8 -*-

"""
Preprocessing code for Fig.3b.

Private inputs:
    k_matrices_boston_family_budget.pkl

    matrices_A_D_S_Distribution/
        Other_Individual_and_Family_Services/
            flow_matrix.csv

Public geographic input:
    geo_data/
        tl_2021_boston_msa_bg/
            tl_2021_boston_msa_bg.shp

Output:
    figure3b_data.csv

The output contains the aggregated number of reassigned visits
for each CBG under reallocation budgets 0.01, 0.05, and 0.20.

Definition:
    flow_change_i(b)
        = 0.5 * sum_j |H_ij(b) - F_ij|
"""

import pickle
import numpy as np
import pandas as pd
import geopandas as gpd


# ============================================================
# Load private optimization results
# ============================================================

file_path = 'k_matrices_boston_family_budget.pkl'

with open(file_path, 'rb') as f:
    k_matrices = pickle.load(f)


# ============================================================
# User parameters
# ============================================================

city = 'boston'

category = 'Other Individual and Family Services'

cat_dir = (
    f'matrices_A_D_S_Distribution/'
    f'{category.replace(" ", "_")}'
)


flow_matrix = pd.read_csv(
    f'{cat_dir}/flow_matrix.csv',
    index_col=0
)


# ============================================================
# Load Boston MSA CBG geometry
# ============================================================

boston_msa_cbg = gpd.read_file(
    'geo_data/'
    'tl_2021_boston_msa_bg/'
    'tl_2021_boston_msa_bg.shp'
)

boston_msa_cbg['GEOID'] = (
    boston_msa_cbg['GEOID']
    .astype(str)
)

pad_len = int(
    boston_msa_cbg['GEOID']
    .str.len()
    .max()
)


# ============================================================
# Select POIs/CBGs as in original script
# ============================================================

poi_total_flow = flow_matrix.sum(
    axis=0
)

poi_num = flow_matrix.shape[1]

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


A_sub_full = flow_matrix.loc[
    selected_cbgs,
    selected_pois
]


baseline = A_sub_full.copy()

baseline.index = [
    str(x).zfill(pad_len)
    for x in baseline.index
]


# ============================================================
# Parameters
# ============================================================

ks_to_plot = [
    0.01,
    0.05,
    0.20
]


pad_len = (
    pad_len
    if 'pad_len' in globals()
    else None
)


# ============================================================
# Helper: find nearest key in k_matrices if exact not present
# ============================================================

def find_best_key(k_val, k_matrices):

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

        except:

            pass

    return best


# ============================================================
# Compute per-CBG flow_change for ks_to_plot
# ============================================================

flow_change = {}


for k in ks_to_plot:

    mat = k_matrices.get(k)

    if mat is None:

        bk = find_best_key(
            k,
            k_matrices
        )

        mat = (
            k_matrices.get(bk)
            if bk is not None
            else None
        )


    if mat is None:

        raise RuntimeError(
            f"未找到 k={k} 的矩阵"
        )


    H = mat.copy()


    try:

        if pad_len is not None:

            H.index = [
                str(x).zfill(pad_len)
                for x in H.index
            ]

    except:

        H.index = H.index.astype(str)


    H_aligned = H.reindex(
        index=baseline.index,
        columns=baseline.columns,
        fill_value=0.0
    )


    A_aligned = baseline.reindex(
        index=baseline.index,
        columns=baseline.columns,
        fill_value=0.0
    )


    fc = (
        0.5
        * (
            H_aligned
            - A_aligned
        )
        .abs()
        .sum(axis=1)
    )


    flow_change[k] = fc


# ============================================================
# Save aggregated public Fig.3b data
# ============================================================

figure3b_data = pd.DataFrame(
    index=baseline.index
)


for k, series in flow_change.items():

    figure3b_data[
        f'fc_k_{k:.2f}'
    ] = series.reindex(
        figure3b_data.index
    ).fillna(0.0)


figure3b_data.index.name = 'GEOID'


figure3b_data.to_csv(
    'figure3b_data.csv'
)


# ============================================================
# Print summary
# ============================================================

print(
    '\n========== FIG.3B PREPROCESSING =========='
)

print(
    f'Number of CBGs = '
    f'{len(figure3b_data)}'
)

print(
    f'Number of POIs = '
    f'{len(selected_pois)}'
)


for k in ks_to_plot:

    colname = f'fc_k_{k:.2f}'

    print(
        f'k={k:.2f}: '
        f'total reassigned visits = '
        f'{figure3b_data[colname].sum():.6f}, '
        f'max per CBG = '
        f'{figure3b_data[colname].max():.6f}'
    )


print(
    '\n[SAVE] figure3b_data.csv'
)

print(
    '=========================================='
)