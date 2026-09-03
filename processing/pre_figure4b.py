# -*- coding: utf-8 -*-
"""
Preprocessing for Fig. 4b

This script prepares the two public plotting inputs used by figure4b.py:

    figure4b_baseline_flow.csv
    figure4b_reference_flow.csv

Private / intermediate inputs:
    matrices_A_D_S_Distribution/
        Other_Individual_and_Family_Services/
            flow_matrix.csv
            pred_rownorm_int_preserve.csv

Public geographic input:
    geo_data/
        tl_2021_boston_msa_bg/
            tl_2021_boston_msa_bg.shp

The plotting script figure4b.py does NOT require the original
flow_matrix.csv or pred_rownorm_int_preserve.csv.
"""

import os
import numpy as np
import pandas as pd
import geopandas as gpd


# ============================================================
# Basic settings
# ============================================================

city = 'boston'

category = 'Other Individual and Family Services'

cat_dir = (
    f'matrices_A_D_S_Distribution/'
    f'{category.replace(" ", "_")}'
)


# ============================================================
# Input files
# ============================================================

flow_path = os.path.join(
    cat_dir,
    'flow_matrix.csv'
)

reference_path = os.path.join(
    cat_dir,
    'pred_rownorm_int_preserve.csv'
)

shapefile_path = (
    'geo_data/'
    'tl_2021_boston_msa_bg/'
    'tl_2021_boston_msa_bg.shp'
)


# ============================================================
# Public output files
# ============================================================

baseline_output_path = (
    'figure4b_baseline_flow.csv'
)

reference_output_path = (
    'figure4b_reference_flow.csv'
)


# ============================================================
# Check input files
# ============================================================

for path in [
    flow_path,
    reference_path,
    shapefile_path,
]:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f'Cannot find required file:\n{path}'
        )


# ============================================================
# Load original data
# ============================================================

print(
    f'[LOAD baseline flow] '
    f'{flow_path}'
)

flow_matrix = pd.read_csv(
    flow_path,
    index_col=0
)


print(
    f'[LOAD reference flow] '
    f'{reference_path}'
)

R = pd.read_csv(
    reference_path,
    index_col=0
)


# ============================================================
# Load Boston MSA CBG shapefile
#
# Same purpose as the original Fig.4b script:
# determine the GEOID padding length.
# ============================================================

print(
    f'[LOAD shapefile] '
    f'{shapefile_path}'
)

boston_msa_cbg = gpd.read_file(
    shapefile_path
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

print(
    f'[INFO] GEOID pad length = '
    f'{pad_len}'
)


# ============================================================
# Reproduce selected_pois
#
# This is kept consistent with the original Fig.4b code.
# ============================================================

poi_total_flow = (
    flow_matrix
    .sum(axis=0)
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
    .head(poi_num)
    .index
    .tolist()
)


# ============================================================
# Reproduce selected_cbgs
#
# This is also kept consistent with the original Fig.4b code.
# ============================================================

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


# ============================================================
# Construct baseline matrix F
# ============================================================

A_sub_full = (
    flow_matrix.loc[
        selected_cbgs,
        selected_pois
    ]
)

baseline = (
    A_sub_full.copy()
)


# ============================================================
# Standardize baseline index
#
# Same zero-padding operation as the original code.
# ============================================================

baseline.index = [
    str(x).zfill(pad_len)
    for x in baseline.index
]

baseline.columns = [
    str(x)
    for x in baseline.columns
]


# ============================================================
# Align R to baseline
# ============================================================

def align_H_to_baseline(
    H_df,
    baseline_df,
    pad_len
):
    """
    Align another CBG × POI matrix to the baseline matrix.

    First attempt:
        direct string matching.

    If direct matching fails:
        zero-pad the CBG GEOIDs and try again.

    Missing entries after alignment are filled with zero.
    """

    H = H_df.copy()

    # --------------------------------------------------------
    # Convert row / column labels to strings
    # --------------------------------------------------------

    H.index = H.index.map(
        lambda x: str(x)
    )

    H.columns = H.columns.map(
        lambda x: str(x)
    )

    baseline_index = [
        str(x)
        for x in baseline_df.index
    ]

    baseline_columns = [
        str(x)
        for x in baseline_df.columns
    ]


    # --------------------------------------------------------
    # First attempt:
    # direct string matching
    # --------------------------------------------------------

    H_try = H.reindex(
        index=baseline_index,
        columns=baseline_columns
    )

    if H_try.notna().values.any():

        H_aligned = (
            H_try.fillna(0)
        )

        print(
            'align_H_to_baseline: '
            'direct string match found, '
            f'flow sum = '
            f'{H_aligned.values.sum():.4f}'
        )

        return H_aligned


    # --------------------------------------------------------
    # Second attempt:
    # zero-pad CBG GEOIDs
    #
    # This makes the alignment robust if pandas originally
    # interpreted the GEOID column as an integer.
    # --------------------------------------------------------

    H.index = H.index.map(
        lambda x:
        str(x).zfill(pad_len)
    )

    H_try = H.reindex(
        index=baseline_index,
        columns=baseline_columns
    )

    if H_try.notna().values.any():

        H_aligned = (
            H_try.fillna(0)
        )

        print(
            'align_H_to_baseline: '
            'zero-padded GEOID match found, '
            f'flow sum = '
            f'{H_aligned.values.sum():.4f}'
        )

        return H_aligned


    # --------------------------------------------------------
    # If nothing matches, do not silently create a zero matrix
    # --------------------------------------------------------

    raise ValueError(
        'Unable to align the reference-flow matrix '
        'to the baseline matrix.\n'
        'Please check the GEOID and POI labels in:\n'
        f'{reference_path}'
    )


R_aligned = align_H_to_baseline(
    R,
    baseline,
    pad_len
)


# ============================================================
# Final consistency checks
# ============================================================

if baseline.shape != R_aligned.shape:

    raise ValueError(
        'Baseline and reference matrices '
        'have different shapes after alignment:\n'
        f'baseline = {baseline.shape}\n'
        f'R        = {R_aligned.shape}'
    )


if not baseline.index.equals(
    R_aligned.index
):

    raise ValueError(
        'CBG indices are not aligned.'
    )


if not baseline.columns.equals(
    R_aligned.columns
):

    raise ValueError(
        'POI columns are not aligned.'
    )


# ============================================================
# Save public plotting data
# ============================================================

baseline.to_csv(
    baseline_output_path
)

R_aligned.to_csv(
    reference_output_path
)


# ============================================================
# Summary
# ============================================================

print()
print(
    '=' * 60
)

print(
    'FIG.4B PREPROCESSING COMPLETE'
)

print(
    '=' * 60
)

print(
    f'Number of CBGs = '
    f'{baseline.shape[0]}'
)

print(
    f'Number of POIs = '
    f'{baseline.shape[1]}'
)

print(
    f'Baseline total flow F = '
    f'{baseline.values.sum():.4f}'
)

print(
    f'Reference total flow R = '
    f'{R_aligned.values.sum():.4f}'
)

print()

print(
    '[PUBLIC OUTPUT]'
)

print(
    baseline_output_path
)

print(
    reference_output_path
)

print()

print(
    'These two CSV files can now be used '
    'directly by figure4b.py.'
)