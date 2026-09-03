# -*- coding: utf-8 -*-
"""
Fig. 2e
Matrix-style split-triangle plot for multiple POIs × cities.

Public plotting version.

Required input:
    figure2e_source_data.csv

Output:
    figure2e.pdf
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from matplotlib import cm
from matplotlib.colors import TwoSlopeNorm
import matplotlib.ticker as mtick
import textwrap


# ============================================================
# 1. CONFIGURATION
# ============================================================

SOURCE_DATA_FILE = "figure2e_source_data.csv"


# City order is kept exactly the same as in the original code.
cities = [
    "newyork",
    "losangeles",
    "chicago",
    "houston",
    "atlanta",
    "seattle",
    "boston",
    "fresno",
    "baltimore",
    "tulsa",
    "tyler",
    "champaign",
    "billings",
    "sebring",
    "cheyenne",
]


pretty_names = {
    k: (
        k.title()
        if k not in (
            "newyork",
            "losangeles",
        )
        else (
            "New York"
            if k == "newyork"
            else "Los Angeles"
        )
    )
    for k in cities
}


# POI mapping is kept exactly the same as in the original code.
naics_map = {
    "624190": "Other_Individual_and_Family_Services",
    "711310": "Promoters_of_Performing_Arts,_Sports,_and_Similar_Events_with_Facilities",
    "813110": "Religious_Organizations_catholic",
    "712110": "Museums",
    "713940": "Fitness_and_Recreational_Sports_Centers",
    "722410": "Drinking_Places_(Alcoholic_Beverages)",
}


poi_codes = list(
    naics_map.keys()
)

poi_labels = [
    naics_map[c]
    for c in poi_codes
]


# Replace underscores with spaces.
poi_labels = [
    label.replace("_", " ")
    for label in poi_labels
]


# Keep the original final plotting display name.
poi_labels = [
    "Religious Organizations (Catholic)"
    if label
    == "Religious Organizations catholic"
    else label
    for label in poi_labels
]


# ============================================================
# 2. READ PUBLIC SOURCE DATA
# ============================================================

if not os.path.exists(
    SOURCE_DATA_FILE
):

    raise FileNotFoundError(
        "\nCannot find Fig. 2e source data:\n"
        f"    {SOURCE_DATA_FILE}\n\n"
        "Please place figure2e_source_data.csv "
        "in the same directory as figure2e.py."
    )


df_source = pd.read_csv(
    SOURCE_DATA_FILE,
    dtype={
        "city": str,
        "poi_code": str,
    },
)


# Strip whitespace.
df_source.columns = [
    str(c).strip()
    for c in df_source.columns
]


required_columns = [
    "city",
    "poi_code",
    "social_relative_change",
    "distance_relative_change",
]


missing_columns = [
    c
    for c in required_columns
    if c not in df_source.columns
]


if missing_columns:

    raise ValueError(
        "\nfigure2e_source_data.csv is missing "
        "required columns:\n"
        f"    {missing_columns}\n"
    )


# Normalize identifiers.
df_source["city"] = (
    df_source["city"]
    .astype(str)
    .str.strip()
    .str.lower()
)


df_source["poi_code"] = (
    df_source["poi_code"]
    .astype(str)
    .str.strip()
)


# Handle the case where Excel or another program may have
# converted codes to strings such as "624190.0".
df_source["poi_code"] = (
    df_source["poi_code"]
    .str.replace(
        r"\.0$",
        "",
        regex=True,
    )
)


df_source[
    "social_relative_change"
] = pd.to_numeric(
    df_source[
        "social_relative_change"
    ],
    errors="coerce",
)


df_source[
    "distance_relative_change"
] = pd.to_numeric(
    df_source[
        "distance_relative_change"
    ],
    errors="coerce",
)


# Check for duplicate city × POI rows.
duplicate_mask = (
    df_source.duplicated(
        subset=[
            "city",
            "poi_code",
        ],
        keep=False,
    )
)


if duplicate_mask.any():

    duplicate_rows = (
        df_source.loc[
            duplicate_mask,
            [
                "city",
                "poi_code",
            ],
        ]
        .sort_values(
            [
                "poi_code",
                "city",
            ]
        )
    )

    raise ValueError(
        "\nDuplicate city × POI combinations were "
        "found in figure2e_source_data.csv:\n\n"
        + duplicate_rows.to_string(
            index=False
        )
    )


# ============================================================
# 3. CONVERT SOURCE DATA TO ORIGINAL MATRICES
# ============================================================

n_pois = len(
    poi_codes
)

n_cities = len(
    cities
)


social_matrix = np.full(
    (
        n_pois,
        n_cities,
    ),
    np.nan,
    dtype=float,
)


dist_matrix = np.full(
    (
        n_pois,
        n_cities,
    ),
    np.nan,
    dtype=float,
)


missing = []


for pi, poi_code in enumerate(
    poi_codes
):

    for ci, city in enumerate(
        cities
    ):

        subset = df_source[
            (
                df_source["city"]
                == city
            )
            &
            (
                df_source["poi_code"]
                == poi_code
            )
        ]

        if subset.empty:

            missing.append(
                (
                    city,
                    poi_code,
                )
            )

            continue

        row = subset.iloc[0]

        social_matrix[
            pi,
            ci
        ] = row[
            "social_relative_change"
        ]

        dist_matrix[
            pi,
            ci
        ] = row[
            "distance_relative_change"
        ]


if missing:

    print(
        f"Warning: {len(missing)} "
        "city × POI combinations are missing "
        "from the source-data file."
    )

    for (
        city,
        poi_code,
    ) in missing:

        print(
            f"  Missing: "
            f"{city}, {poi_code}"
        )


# ============================================================
# 4. FIG.2b-CONSISTENT LIGHT-HALF COLOR NORMALIZATION
# ============================================================

# Use light 0%--50% visual range only.
# This keeps Fig.2e consistent with Fig.2b and avoids
# overly dark colorbar tails.

color_static = "#4C78A8"   # Fig.2b blue
color_dynamic = "#8E5EA2"  # Fig.2b purple
neutral_light = "#F6F8FC"


def make_light_half_cmap(
    name,
    end_color,
    n=256,
):
    """
    Build a light-half sequential colormap.

    It mimics using only the first 50% of a sequential palette:
    light neutral -> pale color -> Fig.2b reference color.

    No extra dark tail is added.
    """

    return (
        mcolors.LinearSegmentedColormap.from_list(
            name,
            [
                neutral_light,
                mcolors.to_rgba(
                    end_color,
                    alpha=0.18,
                ),
                mcolors.to_rgba(
                    end_color,
                    alpha=0.36,
                ),
                mcolors.to_rgba(
                    end_color,
                    alpha=0.58,
                ),
                end_color,
            ],
            N=n,
        )
    )


# Social exposure:
# larger increase = stronger purple,
# but only up to Fig.2b purple.
cmap_social = make_light_half_cmap(
    name="fig2b_light_purple_half",
    end_color=color_dynamic,
)


# Distance:
# larger distance reduction = stronger blue,
# but only up to Fig.2b blue.
#
# Because distance changes are usually negative,
# we normalize the magnitude of reduction.
cmap_dist = make_light_half_cmap(
    name="fig2b_light_blue_half",
    end_color=color_static,
)


# ============================================================
# 5. DATA TRANSFORMATION FOR COLOR MAPPING
# ============================================================

# Social exposure uses the original relative change.
social_plot_matrix = (
    social_matrix.copy()
)


# Distance is converted to reduction magnitude:
#
#   distance change = -20% -> plotted value = +20%
#   distance change =  +5% -> plotted value = 0
#
# This makes darker blue mean larger distance saving.
dist_reduction_matrix = np.where(
    np.isfinite(
        dist_matrix
    ),
    np.maximum(
        -dist_matrix,
        0,
    ),
    np.nan,
)


def safe_norm_from_data(
    values,
    start_at_zero=True,
):

    vals = np.asarray(
        values,
        dtype=float,
    )

    vals = vals[
        np.isfinite(vals)
    ]

    if vals.size == 0:

        return mcolors.Normalize(
            vmin=0.0,
            vmax=1.0,
        )

    vmax = float(
        np.nanmax(vals)
    )

    if start_at_zero:

        vmin = 0.0

    else:

        vmin = float(
            np.nanmin(vals)
        )

    if np.isclose(
        vmin,
        vmax,
    ):

        vmax = (
            vmin + 1e-6
        )

    return mcolors.Normalize(
        vmin=vmin,
        vmax=vmax,
    )


# Colorbar only covers the useful data range.
# No unused diverging side is included.
norm_social = safe_norm_from_data(
    social_plot_matrix,
    start_at_zero=True,
)


norm_dist = safe_norm_from_data(
    dist_reduction_matrix,
    start_at_zero=True,
)


# ============================================================
# 6. PLOTTING
# ============================================================

fig_w = 17

fig_h = (
    7
    + n_pois * 0.19
)


fig = plt.figure(
    figsize=(
        fig_w,
        fig_h,
    ),
    dpi=300,
)


ax = fig.add_subplot(
    111
)


ax.set_xlim(
    -0.5,
    n_cities - 0.5,
)


ax.set_ylim(
    -0.5,
    n_pois - 0.5,
)


ax.set_xticks(
    np.arange(
        n_cities
    )
)


# Use textwrap.wrap() to limit the number of
# characters per line.
poi_labels_wrapped = [
    "\n".join(
        textwrap.wrap(
            label,
            width=20,
        )
    )
    for label in poi_labels
]


# Keep the original reversed POI display order.
ax.set_yticks(
    np.arange(
        n_pois
    )
)


ax.set_yticklabels(
    poi_labels_wrapped[::-1],
    fontsize=12,
)


ax.invert_yaxis()


# Cell size.
cell_size = 0.8

half = (
    cell_size / 2.0
)


# ============================================================
# 7. DRAW SPLIT-TRIANGLE CELLS
# ============================================================

for i in range(
    n_pois
):

    for j in range(
        n_cities
    ):

        y = i
        x = j

        bl = (
            x - half,
            y + half,
        )

        tl = (
            x - half,
            y - half,
        )

        tr = (
            x + half,
            y - half,
        )

        br = (
            x + half,
            y + half,
        )


        # Because the y-axis display is reversed,
        # use the reversed POI matrix index exactly
        # as in the original Fig. 2e code.
        soc_val = social_matrix[
            n_pois - 1 - i,
            j,
        ]

        dis_val = dist_matrix[
            n_pois - 1 - i,
            j,
        ]


        # ----------------------------------------------------
        # Social-exposure triangle
        # ----------------------------------------------------
        if not np.isnan(
            soc_val
        ):

            color = cmap_social(
                norm_social(
                    soc_val
                )
            )

            tri_soc = plt.Polygon(
                [
                    tl,
                    tr,
                    br,
                ],
                closed=True,
                facecolor=color,
                edgecolor="k",
                linewidth=0.3,
            )

            ax.add_patch(
                tri_soc
            )


            centroid_soc = np.mean(
                np.array(
                    [
                        tl,
                        tr,
                        br,
                    ]
                ),
                axis=0,
            )


            # Original text annotation remains disabled.
            #
            # ax.text(
            #     centroid_soc[0],
            #     centroid_soc[1],
            #     f"{soc_val * 100:+.0f}%",
            #     ha="center",
            #     va="center",
            #     fontsize=6,
            #     color="k",
            # )


        # ----------------------------------------------------
        # Distance triangle
        # ----------------------------------------------------
        if not np.isnan(
            dis_val
        ):

            dis_reduction_val = max(
                -dis_val,
                0,
            )

            color = cmap_dist(
                norm_dist(
                    dis_reduction_val
                )
            )

            tri_dis = plt.Polygon(
                [
                    tl,
                    bl,
                    br,
                ],
                closed=True,
                facecolor=color,
                edgecolor="k",
                linewidth=0.3,
            )

            ax.add_patch(
                tri_dis
            )


            centroid_dis = np.mean(
                np.array(
                    [
                        tl,
                        bl,
                        br,
                    ]
                ),
                axis=0,
            )


            # Original text annotation remains disabled.
            #
            # ax.text(
            #     centroid_dis[0],
            #     centroid_dis[1],
            #     f"{dis_val * 100:+.0f}%",
            #     ha="center",
            #     va="center",
            #     fontsize=6,
            #     color="k",
            # )


        # ----------------------------------------------------
        # Missing data:
        # draw the same small grey dot at cell center.
        # ----------------------------------------------------
        if (
            np.isnan(soc_val)
            and
            np.isnan(dis_val)
        ):

            ax.plot(
                x,
                y,
                marker="o",
                markersize=4,
                color="lightgray",
                markeredgecolor="k",
                zorder=6,
            )


# ============================================================
# 8. APPEARANCE
# ============================================================

fig.suptitle(
    "POI × City — Social Exposure (upper triangle) & "
    "Distance (lower triangle) relative change",
    fontsize=14,
    y=1,
)


ax.set_aspect(
    "equal"
)


ax.set_xticks(
    np.arange(
        n_cities
    )
)


ax.set_xticklabels(
    [
        pretty_names.get(
            c,
            c,
        )
        for c in cities
    ],
    rotation=0,
    ha="center",
    fontsize=12,
)


# ============================================================
# 9. COLORBARS
# ============================================================

sm_s = cm.ScalarMappable(
    norm=norm_social,
    cmap=cmap_social,
)

sm_s.set_array([])


sm_d = cm.ScalarMappable(
    norm=norm_dist,
    cmap=cmap_dist,
)

sm_d.set_array([])


# ------------------------------------------------------------
# Social-exposure colorbar
# ------------------------------------------------------------

cax1 = fig.add_axes(
    [
        0.12,
        0.92,
        0.38,
        0.02,
    ]
)


cb1 = fig.colorbar(
    sm_s,
    cax=cax1,
    orientation="horizontal",
)


cb1.set_label(
    "Social Exposure relative change (%)",
    fontsize=12,
    labelpad=3,
)


cb1.ax.xaxis.set_label_position(
    "top"
)


cb1.ax.xaxis.set_ticks_position(
    "bottom"
)


cb1.ax.xaxis.set_major_formatter(
    mtick.PercentFormatter(
        xmax=1.0
    )
)


cb1.ax.tick_params(
    axis="x",
    labelsize=12,
)


# ------------------------------------------------------------
# Distance colorbar
# ------------------------------------------------------------

cax2 = fig.add_axes(
    [
        0.55,
        0.92,
        0.33,
        0.02,
    ]
)


cb2 = fig.colorbar(
    sm_d,
    cax=cax2,
    orientation="horizontal",
)


cb2.set_label(
    "Distance relative change (%)",
    fontsize=12,
)


cb2.ax.xaxis.set_label_position(
    "top"
)


cb2.ax.xaxis.set_ticks_position(
    "bottom"
)


cb2.ax.xaxis.set_major_formatter(
    mtick.FuncFormatter(
        lambda x, pos:
        "0%"
        if abs(x) < 1e-12
        else f"{-100 * x:.0f}%"
    )
)


cb2.ax.tick_params(
    axis="x",
    labelsize=12,
)


# ============================================================
# 10. SAVE FIGURE
# ============================================================

plt.subplots_adjust(
    top=0.9,
    left=0.08,
    right=0.98,
    bottom=0.12,
)


plt.savefig(
    "figure2e.pdf",
    format="pdf",
    dpi=300,
    bbox_inches="tight",
    transparent=True,
)


plt.show()