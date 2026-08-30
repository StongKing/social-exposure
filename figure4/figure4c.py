# -*- coding: utf-8 -*-
"""
Fig. 4c
Income composition of adjusted and unadjusted origins

Statistical analysis
--------------------
Within EACH allocation condition separately:

    1. Compare adjusted vs unadjusted CBGs
       for each of the four income-share components.

    2. Use two-sided Mann-Whitney U tests.

    3. Apply Holm correction across the FOUR income-component
       comparisons within that allocation condition.

Therefore:

    No regularization:
        4 tests -> one Holm family

    Behavioral anchoring:
        4 tests -> another Holm family

Figure annotation
-----------------
Only statistically significant Holm-adjusted comparisons
are shown:

    *   adjusted p < 0.05
    **  adjusted p < 0.01
    *** adjusted p < 0.001

Non-significant comparisons are not marked and no bracket
is drawn.

@author: JZS
"""

import os
import pickle
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests


# ============================================================
# 1. Configuration
# ============================================================

city = 'boston'

category = 'Other Individual and Family Services'

cat_dir = (
    f'matrices_A_D_S_Distribution/'
    f'{category.replace(" ", "_")}'
)

outdir = 'k_flow_change_outputs'

os.makedirs(
    outdir,
    exist_ok=True
)


# ------------------------------------------------------------
# Reallocation-result files
# ------------------------------------------------------------

no_regu_file = os.path.join(
    cat_dir,
    f'H_opt_df_no_regu_{city}_624190.pkl'
)

regu_file = os.path.join(
    cat_dir,
    f'H_opt_df_regu_{city}_624190.pkl'
)


# ============================================================
# 2. Load baseline flow matrix
# ============================================================

flow_matrix = pd.read_csv(
    f'{cat_dir}/flow_matrix.csv',
    index_col=0
)


# ============================================================
# 3. Load CBG income composition
# ============================================================

cbg_income_dist_df = pd.read_csv(
    f'{cat_dir}/cbg_income_level_distribution_{city}_msa.csv',
    dtype={
        'GEOID': np.int64
    }
)


cbg_income_dist_dict = (
    cbg_income_dist_df
    .set_index('GEOID')
    .to_dict(orient='index')
)


# ============================================================
# 4. Determine GEOID length
# ============================================================

boston_msa_cbg = gpd.read_file(
    'geo_data/tl_2021_boston_msa_bg/'
    'tl_2021_boston_msa_bg.shp'
)


boston_msa_cbg['GEOID'] = (
    boston_msa_cbg['GEOID']
    .astype(str)
)


pad_len = int(
    boston_msa_cbg[
        'GEOID'
    ]
    .str.len()
    .max()
)


# ============================================================
# 5. Income components
# ============================================================

income_levels = [

    'low_income_pct',

    'lower_middle_income_pct',

    'upper_middle_income_pct',

    'high_income_pct'
]


income_labels = {

    'low_income_pct':
        'low',

    'lower_middle_income_pct':
        'lower-mid',

    'upper_middle_income_pct':
        'upper-mid',

    'high_income_pct':
        'high'
}


# ============================================================
# 6. Reconstruct analysis domain
#
# Keep only POIs with positive observed baseline flow.
#
# This should correspond to the 44 active POIs reported
# in the manuscript.
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
    "Number of active POIs =",
    len(selected_pois)
)


# ------------------------------------------------------------
# Select CBGs with positive baseline visits to at least
# one active POI
# ------------------------------------------------------------

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


# ------------------------------------------------------------
# Baseline submatrix
# ------------------------------------------------------------

A_sub_full = flow_matrix.loc[
    selected_cbgs,
    selected_pois
]


baseline = A_sub_full.copy()


baseline.index = [

    str(x).zfill(
        pad_len
    )

    for x in baseline.index
]


print(
    "Number of CBGs in analysis =",
    len(baseline)
)


# ============================================================
# 7. Helper: load H matrix
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
# 8. Helper:
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
# 9. Helper:
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
# 10. Load optimized allocations
# ============================================================

H_no = load_H(
    no_regu_file
)


H_reg = load_H(
    regu_file
)


# ============================================================
# 11. Determine adjusted / unadjusted origins
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
# 12. Extract income composition
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
# 13. Statistical tests
#
# Four two-sided Mann-Whitney U tests within EACH condition.
# ============================================================

def run_four_tests(
    changed_df,
    unchanged_df
):

    results = {}


    for lvl in income_levels:

        arr_changed = (
            changed_df[
                lvl
            ]
            .dropna()
            .astype(float)
            .values
        )


        arr_unchanged = (
            unchanged_df[
                lvl
            ]
            .dropna()
            .astype(float)
            .values
        )


        if (
            len(arr_changed) == 0
            or
            len(arr_unchanged) == 0
        ):

            raise ValueError(
                f"No valid observations for {lvl}."
            )


        # ----------------------------------------------------
        # Two-sided Mann-Whitney U
        #
        # Large samples + possible ties:
        # use asymptotic implementation.
        # ----------------------------------------------------

        u_stat, p_raw = (
            mannwhitneyu(
                arr_changed,
                arr_unchanged,
                alternative='two-sided',
                method='asymptotic'
            )
        )


        n_changed = (
            len(
                arr_changed
            )
        )

        n_unchanged = (
            len(
                arr_unchanged
            )
        )


        # ----------------------------------------------------
        # Rank-biserial effect size
        #
        # Positive:
        # adjusted group tends to have larger income share.
        #
        # Negative:
        # adjusted group tends to have smaller income share.
        # ----------------------------------------------------

        rank_biserial = (

            2.0 *
            u_stat /
            (
                n_changed *
                n_unchanged
            )

            - 1.0
        )


        results[
            lvl
        ] = {

            'U':
                u_stat,

            'p_raw':
                p_raw,

            'n_adjusted':
                n_changed,

            'n_unadjusted':
                n_unchanged,

            'median_adjusted':
                np.median(
                    arr_changed
                ),

            'median_unadjusted':
                np.median(
                    arr_unchanged
                ),

            'mean_adjusted':
                np.mean(
                    arr_changed
                ),

            'mean_unadjusted':
                np.mean(
                    arr_unchanged
                ),

            'rank_biserial':
                rank_biserial
        }


    return results


# ------------------------------------------------------------
# No regularization
# ------------------------------------------------------------

stats_no = run_four_tests(
    income_changed_no,
    income_unchanged_no
)


# ------------------------------------------------------------
# Behavioral anchoring
# ------------------------------------------------------------

stats_reg = run_four_tests(
    income_changed_reg,
    income_unchanged_reg
)


# ============================================================
# 14. Holm correction
#
# IMPORTANT:
#
# DO NOT combine the 8 tests.
#
# Each allocation condition is treated as a separate
# inferential family.
#
# No regularization:
#     4 tests -> Holm correction
#
# Behavioral anchoring:
#     4 tests -> separate Holm correction
# ============================================================

def apply_holm_within_condition(
    stats_dict
):

    raw_p_values = [

        stats_dict[
            lvl
        ][
            'p_raw'
        ]

        for lvl
        in income_levels
    ]


    reject, adjusted_p, _, _ = (
        multipletests(
            raw_p_values,
            alpha=0.05,
            method='holm'
        )
    )


    for i, lvl in enumerate(
        income_levels
    ):

        stats_dict[
            lvl
        ][
            'p_holm'
        ] = adjusted_p[i]


        stats_dict[
            lvl
        ][
            'reject_holm'
        ] = reject[i]


    return stats_dict


# ------------------------------------------------------------
# Apply separately
# ------------------------------------------------------------

stats_no = (
    apply_holm_within_condition(
        stats_no
    )
)


stats_reg = (
    apply_holm_within_condition(
        stats_reg
    )
)


# ============================================================
# 15. Significance stars
#
# Non-significant:
#     return empty string
#
# Therefore:
#     no ns
#     no bracket
# ============================================================

def p_to_stars(p):

    if p < 0.001:

        return '***'

    elif p < 0.01:

        return '**'

    elif p < 0.05:

        return '*'

    else:

        return ''


# ============================================================
# 16. Print statistical results
# ============================================================

print(
    "\n"
    "=========================================================="
)

print(
    "FIG. 4c STATISTICAL TESTS"
)

print(
    "Two-sided Mann-Whitney U tests"
)

print(
    "Holm correction separately within each allocation condition"
)

print(
    "=========================================================="
)


def print_condition_results(
    condition_name,
    stats_dict
):

    print(
        f"\n{condition_name}"
    )

    print(
        "-" * 72
    )


    for lvl in income_levels:

        res = (
            stats_dict[
                lvl
            ]
        )


        print(
            f"\n{lvl}"
        )


        print(
            "  n adjusted =",
            res[
                'n_adjusted'
            ]
        )


        print(
            "  n unadjusted =",
            res[
                'n_unadjusted'
            ]
        )


        print(
            "  median adjusted =",
            res[
                'median_adjusted'
            ]
        )


        print(
            "  median unadjusted =",
            res[
                'median_unadjusted'
            ]
        )


        print(
            "  mean adjusted =",
            res[
                'mean_adjusted'
            ]
        )


        print(
            "  mean unadjusted =",
            res[
                'mean_unadjusted'
            ]
        )


        print(
            "  U =",
            res[
                'U'
            ]
        )


        print(
            "  raw p =",
            res[
                'p_raw'
            ]
        )


        print(
            "  Holm-adjusted p =",
            res[
                'p_holm'
            ]
        )


        print(
            "  rank-biserial r =",
            res[
                'rank_biserial'
            ]
        )


        stars = p_to_stars(
            res[
                'p_holm'
            ]
        )


        if stars == '':

            figure_text = (
                'not displayed'
            )

        else:

            figure_text = (
                stars
            )


        print(
            "  figure annotation =",
            figure_text
        )


print_condition_results(
    'No regularization',
    stats_no
)


print_condition_results(
    'Behavioral anchoring',
    stats_reg
)


# ============================================================
# 17. Save complete statistical results
# ============================================================

stats_rows = []


for condition_name, stats_dict in [

    (
        'No regularization',
        stats_no
    ),

    (
        'Behavioral anchoring',
        stats_reg
    )

]:

    for lvl in income_levels:

        res = (
            stats_dict[
                lvl
            ]
        )


        stats_rows.append({

            'condition':
                condition_name,

            'income_component':
                lvl,

            'income_label':
                income_labels[
                    lvl
                ],

            'n_adjusted':
                res[
                    'n_adjusted'
                ],

            'n_unadjusted':
                res[
                    'n_unadjusted'
                ],

            'median_adjusted':
                res[
                    'median_adjusted'
                ],

            'median_unadjusted':
                res[
                    'median_unadjusted'
                ],

            'mean_adjusted':
                res[
                    'mean_adjusted'
                ],

            'mean_unadjusted':
                res[
                    'mean_unadjusted'
                ],

            'U':
                res[
                    'U'
                ],

            'p_raw':
                res[
                    'p_raw'
                ],

            'p_holm_within_condition':
                res[
                    'p_holm'
                ],

            'rank_biserial':
                res[
                    'rank_biserial'
                ],

            'figure_annotation':
                p_to_stars(
                    res[
                        'p_holm'
                    ]
                )
        })


stats_df = pd.DataFrame(
    stats_rows
)


stats_output_file = os.path.join(
    outdir,
    'figure4c_statistics_holm_within_condition.csv'
)


stats_df.to_csv(
    stats_output_file,
    index=False
)


print(
    "\nStatistical results saved to:"
)

print(
    stats_output_file
)


# ============================================================
# 18. Figure settings
# ============================================================

fig, axes = plt.subplots(
    1,
    2,
    figsize=(
        10,
        5
    ),
    dpi=300,
    constrained_layout=True
)


# ============================================================
# 19. Color helper
# ============================================================

def blend_with_white(
    hex_color,
    alpha
):

    hex_color = (
        hex_color
        .lstrip('#')
    )


    r = int(
        hex_color[
            0:2
        ],
        16
    )


    g = int(
        hex_color[
            2:4
        ],
        16
    )


    b = int(
        hex_color[
            4:6
        ],
        16
    )


    r_new = int(
        round(
            alpha * r +
            (
                1 - alpha
            ) * 255
        )
    )


    g_new = int(
        round(
            alpha * g +
            (
                1 - alpha
            ) * 255
        )
    )


    b_new = int(
        round(
            alpha * b +
            (
                1 - alpha
            ) * 255
        )
    )


    return (
        f'#{r_new:02X}'
        f'{g_new:02X}'
        f'{b_new:02X}'
    )


BASE_BLUE = (
    '#3498DB'
)

BASE_PURPLE = (
    '#7C5BB8'
)

BOX_ALPHA = 0.5


COLOR_UNADJUSTED = (
    blend_with_white(
        BASE_BLUE,
        BOX_ALPHA
    )
)


COLOR_ADJUSTED = (
    blend_with_white(
        BASE_PURPLE,
        BOX_ALPHA
    )
)


# ============================================================
# 20. Plotting function
# ============================================================

def plot_adjusted_vs_unadjusted(
    ax,
    adjusted_df,
    unadjusted_df,
    title,
    n_adjusted,
    n_unadjusted,
    stats_dict
):

    box_data = []

    positions = []

    label_positions = []


    offset = 0.18

    box_width = 0.30


    # ========================================================
    # Prepare boxplot data
    # ========================================================

    for i, lvl in enumerate(
        income_levels
    ):

        arr_unadjusted = (
            unadjusted_df[
                lvl
            ]
            .dropna()
            .astype(float)
            .values
        )


        arr_adjusted = (
            adjusted_df[
                lvl
            ]
            .dropna()
            .astype(float)
            .values
        )


        # unadjusted
        box_data.append(
            arr_unadjusted
        )


        positions.append(
            i - offset
        )


        # adjusted
        box_data.append(
            arr_adjusted
        )


        positions.append(
            i + offset
        )


        label_positions.append(
            i
        )


    # ========================================================
    # Draw boxplots
    # ========================================================

    bp = ax.boxplot(
        box_data,
        positions=positions,
        widths=box_width,
        patch_artist=True,
        showfliers=False,

        medianprops={
            'color':
                'black',

            'linewidth':
                1.2
        },

        whiskerprops={
            'color':
                'black',

            'linewidth':
                0.8
        },

        capprops={
            'color':
                'black',

            'linewidth':
                0.8
        }
    )


    # ========================================================
    # Box colors
    # ========================================================

    colors = []


    for _ in income_levels:

        colors.append(
            COLOR_UNADJUSTED
        )

        colors.append(
            COLOR_ADJUSTED
        )


    for patch, color in zip(
        bp[
            'boxes'
        ],
        colors
    ):

        patch.set_facecolor(
            color
        )

        patch.set_edgecolor(
            'black'
        )

        patch.set_linewidth(
            0.8
        )


    # ========================================================
    # Mean markers
    #
    # White square = arithmetic mean
    # ========================================================

    for pos, arr in zip(
        positions,
        box_data
    ):

        if len(arr) == 0:

            continue


        mean_value = (
            np.mean(
                arr
            )
        )


        ax.scatter(
            pos,
            mean_value,
            marker='s',
            s=55,
            facecolors='white',
            edgecolors='black',
            linewidths=0.8,
            zorder=10
        )


    # ========================================================
    # Axis labels
    # ========================================================

    ax.set_xticks(
        label_positions
    )


    ax.set_xticklabels(
        [
            'low',
            'lower-mid',
            'upper-mid',
            'high'
        ]
    )


    ax.set_ylabel(
        'Income share'
    )


    ax.set_title(
        title
    )


    # ========================================================
    # Sample sizes
    # ========================================================

    ax.text(
        0.04,
        0.85,

        (
            f'adjusted = {n_adjusted}\n'
            f'unadjusted = {n_unadjusted}'
        ),

        transform=ax.transAxes,

        ha='left',

        va='top',

        fontsize=7.5
    )


    # ========================================================
    # Significance annotations
    #
    # IMPORTANT:
    #
    # Non-significant:
    #     nothing is displayed
    #     no "ns"
    #     no bracket
    #
    # Significant:
    #     bracket + stars
    # ========================================================

    ylim_bottom, ylim_top = (
        ax.get_ylim()
    )


    y_range = (
        ylim_top -
        ylim_bottom
    )


    # --------------------------------------------------------
    # Distance between upper whisker and bracket
    # --------------------------------------------------------

    gap = (
        0.018 *
        y_range
    )


    # --------------------------------------------------------
    # Length of short vertical lines
    # --------------------------------------------------------

    bracket_height = (
        0.015 *
        y_range
    )


    # --------------------------------------------------------
    # Distance between bracket and stars
    # --------------------------------------------------------

    star_gap = (
        0.004 *
        y_range
    )


    max_annotation_y = (
        ylim_top
    )


    # ========================================================
    # Loop over four income components
    # ========================================================

    for i, lvl in enumerate(
        income_levels
    ):

        adjusted_p = (
            stats_dict[
                lvl
            ][
                'p_holm'
            ]
        )


        stars = (
            p_to_stars(
                adjusted_p
            )
        )


        # ----------------------------------------------------
        # CRITICAL:
        #
        # If not significant:
        #
        #     skip everything
        #
        # Therefore:
        #
        #     no "ns"
        #     no bracket
        # ----------------------------------------------------

        if stars == '':

            continue


        # ----------------------------------------------------
        # Box indices
        #
        # 2*i:
        #     unadjusted
        #
        # 2*i+1:
        #     adjusted
        # ----------------------------------------------------

        unadjusted_box_index = (
            2 * i
        )


        adjusted_box_index = (
            2 * i + 1
        )


        # ----------------------------------------------------
        # Upper whisker of unadjusted box
        # ----------------------------------------------------

        whisker_unadjusted = (
            bp[
                'whiskers'
            ][
                2 *
                unadjusted_box_index
                + 1
            ]
        )


        # ----------------------------------------------------
        # Upper whisker of adjusted box
        # ----------------------------------------------------

        whisker_adjusted = (
            bp[
                'whiskers'
            ][
                2 *
                adjusted_box_index
                + 1
            ]
        )


        top_unadjusted = (
            np.nanmax(
                whisker_unadjusted
                .get_ydata()
            )
        )


        top_adjusted = (
            np.nanmax(
                whisker_adjusted
                .get_ydata()
            )
        )


        pair_top = max(
            top_unadjusted,
            top_adjusted
        )


        # ----------------------------------------------------
        # Bracket y-position
        # ----------------------------------------------------

        y_line = (
            pair_top +
            gap
        )


        x1 = (
            i -
            offset
        )


        x2 = (
            i +
            offset
        )


        # ----------------------------------------------------
        # Horizontal bracket line
        # ----------------------------------------------------

        ax.hlines(
            y=y_line,
            xmin=x1,
            xmax=x2,
            color='black',
            linewidth=0.9,
            zorder=20,
            clip_on=False
        )


        # ----------------------------------------------------
        # Short vertical lines
        # ----------------------------------------------------

        ax.vlines(
            x=[
                x1,
                x2
            ],

            ymin=(
                y_line -
                bracket_height
            ),

            ymax=y_line,

            color='black',

            linewidth=0.9,

            zorder=20,

            clip_on=False
        )


        # ----------------------------------------------------
        # Significance stars
        # ----------------------------------------------------

        ax.text(
            (
                x1 +
                x2
            ) / 2,

            (
                y_line +
                star_gap
            ),

            stars,

            ha='center',

            va='bottom',

            fontsize=10,

            zorder=21
        )


        max_annotation_y = max(
            max_annotation_y,

            (
                y_line
                +
                star_gap
                +
                0.035 *
                y_range
            )
        )


    # ========================================================
    # Increase y-axis upper limit only when necessary
    # ========================================================

    current_bottom, current_top = (
        ax.get_ylim()
    )


    if (
        max_annotation_y
        >
        current_top
    ):

        ax.set_ylim(
            current_bottom,
            max_annotation_y
        )


# ============================================================
# 21. Left panel:
# No regularization
# ============================================================

plot_adjusted_vs_unadjusted(
    axes[0],

    income_changed_no,

    income_unchanged_no,

    'No regularization',

    len(
        changed_no
    ),

    len(
        unchanged_no
    ),

    stats_no
)


# ============================================================
# 22. Right panel:
# Behavioral anchoring
# ============================================================

plot_adjusted_vs_unadjusted(
    axes[1],

    income_changed_reg,

    income_unchanged_reg,

    'With behavioural anchoring',

    len(
        changed_reg
    ),

    len(
        unchanged_reg
    ),

    stats_reg
)


# ============================================================
# 23. Overall figure title
# ============================================================

fig.suptitle(
    'Income composition of adjusted and unadjusted origins',
    fontsize=14,
    y=1.04
)


# ============================================================
# 24. Legend
# ============================================================

legend_unadjusted = (
    mpatches.Patch(
        facecolor=COLOR_UNADJUSTED,
        edgecolor='black',
        label='unadjusted'
    )
)


legend_adjusted = (
    mpatches.Patch(
        facecolor=COLOR_ADJUSTED,
        edgecolor='black',
        label='adjusted'
    )
)


axes[0].legend(
    handles=[
        legend_unadjusted,
        legend_adjusted
    ],
    loc='upper left',
    frameon=True
)


axes[1].legend(
    handles=[
        legend_unadjusted,
        legend_adjusted
    ],
    loc='upper left',
    frameon=True
)


# ============================================================
# 25. Save figure
# ============================================================

pdf_file = (
    'figure4c.pdf'
)


png_file = (
    'figure4c.png'
)


plt.savefig(
    pdf_file,
    format='pdf',
    dpi=300,
    bbox_inches='tight',
    transparent=False
)


plt.savefig(
    png_file,
    format='png',
    dpi=600,
    bbox_inches='tight',
    transparent=False
)


print(
    "\nFigure saved:"
)


print(
    pdf_file
)


print(
    png_file
)


plt.show()