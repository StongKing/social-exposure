# -*- coding: utf-8 -*-
"""
Standalone public Fig.2d:
destination-level SPSE contribution scatter.

This public plotting script does NOT require the restricted
CBG-POI baseline mobility flow matrix.

Required public derived files
-----------------------------
fig2d_spse_destination_outputs/
    figure2d_destination_summary.csv
    figure2d_top10_reassignment_summary.csv

Optional diagnostic file
------------------------
    figure2d_case_summary.csv

Each point = one POI.

x-axis:
    Δ POI visits

y-axis:
    Average Δ structural social exposure per visiting CBG

point size:
    |Δ POI visits|

point color:
    Δ weighted exposure contribution

point marker:
    baseline-flow POI tier
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.cm import ScalarMappable
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


# ============================================================
# 0. USER SETTINGS
# ============================================================

# The script is assumed to be placed in the repository root.
PROJECT_ROOT = os.path.dirname(
    os.path.abspath(__file__)
)

INPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "fig2d_spse_destination_outputs"
)

DESTINATION_SUMMARY_FILE = os.path.join(
    INPUT_DIR,
    "figure2d_destination_summary.csv"
)

TOP10_SUMMARY_FILE = os.path.join(
    INPUT_DIR,
    "figure2d_top10_reassignment_summary.csv"
)

CASE_SUMMARY_FILE = os.path.join(
    INPUT_DIR,
    "figure2d_case_summary.csv"
)

# Set to "" for final composite figure if you add panel label externally.
PANEL_TITLE = "Destination-level social exposure contributions"

EPS = 1e-9


# ============================================================
# 1. Load public derived data
# ============================================================

def normalize_bool_value(x):

    if isinstance(
        x,
        (bool, np.bool_)
    ):
        return bool(x)

    if pd.isna(x):
        return False

    return (
        str(x)
        .strip()
        .lower()
        in {
            "true",
            "1",
            "yes",
            "y",
        }
    )


def load_public_case_output():

    if not os.path.isfile(
        DESTINATION_SUMMARY_FILE
    ):
        raise FileNotFoundError(
            DESTINATION_SUMMARY_FILE
        )

    if not os.path.isfile(
        TOP10_SUMMARY_FILE
    ):
        raise FileNotFoundError(
            TOP10_SUMMARY_FILE
        )

    print(
        f"[LOAD] {DESTINATION_SUMMARY_FILE}"
    )

    scatter_df = pd.read_csv(
        DESTINATION_SUMMARY_FILE,
        dtype={
            "poi_id": str,
            "poi_tier": str,
        },
    )

    print(
        f"[LOAD] {TOP10_SUMMARY_FILE}"
    )

    top10_df = pd.read_csv(
        TOP10_SUMMARY_FILE
    )

    if len(top10_df) == 0:
        top10_summary = {
            "available": False
        }

    else:
        top10_summary = (
            top10_df
            .iloc[0]
            .to_dict()
        )

        top10_summary[
            "available"
        ] = normalize_bool_value(
            top10_summary.get(
                "available",
                False
            )
        )

    case_summary = {}

    if os.path.isfile(
        CASE_SUMMARY_FILE
    ):

        print(
            f"[LOAD] {CASE_SUMMARY_FILE}"
        )

        case_summary_df = (
            pd.read_csv(
                CASE_SUMMARY_FILE
            )
        )

        if len(
            case_summary_df
        ) > 0:

            case_summary = (
                case_summary_df
                .iloc[0]
                .to_dict()
            )

    return {
        "scatter_df":
            scatter_df,

        "top10_reassignment_summary":
            top10_summary,

        "case_summary":
            case_summary,
    }


# ============================================================
# 2. Plotting helpers
# ============================================================

def set_nature_style():
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = [
        "Times New Roman",
        "Times",
        "DejaVu Serif"
    ]
    plt.rcParams["font.size"] = 10.5
    plt.rcParams["axes.linewidth"] = 0.75
    plt.rcParams["axes.labelsize"] = 10.5
    plt.rcParams["axes.titlesize"] = 10.5
    plt.rcParams["xtick.labelsize"] = 10.5
    plt.rcParams["ytick.labelsize"] = 10.5
    plt.rcParams["legend.fontsize"] = 10.5
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42


def make_weighted_contribution_cmap():
    """
    Purple sequential colormap for weighted exposure contribution.

    Uses the lower half of matplotlib's Purples colormap to avoid overly dark
    points while keeping a coherent purple visual language.
    """
    cmap_social = cm.get_cmap("Purples")
    social_half_range = cmap_social(
        np.linspace(
            0.20,
            0.65,
            cmap_social.N // 2
        )
    )

    cmap_social_half = (
        mcolors.LinearSegmentedColormap.from_list(
            "weighted_contribution_purples_half",
            social_half_range,
        )
    )

    return cmap_social_half


def scatter_size_from_abs_delta_visits(
    delta_visits,
    vmax
):
    """
    Marker size is proportional to the absolute magnitude of POI visit change.

    This treats large visit gains and large visit losses symmetrically:
        size_j ~ |optimized visits_j - baseline visits_j|
    """
    delta_visits = np.asarray(
        delta_visits,
        dtype=float
    )

    abs_change = np.abs(
        delta_visits
    )

    vmax = max(
        1.0,
        float(vmax)
    )

    return (
        100
        + 390
        * np.sqrt(
            abs_change
            / vmax
        )
    )


def choose_size_legend_values(
    size_var,
    max_n=3
):
    """
    Return readable representative values
    for the |Δ visits| legend.
    """

    x = np.asarray(
        size_var,
        dtype=float
    )

    x = x[
        np.isfinite(x)
        & (x > EPS)
    ]

    if len(x) == 0:
        return [1.0]

    qs = np.nanpercentile(
        x,
        [35, 65, 90]
    )

    vals = []

    for q in qs:

        if q <= 10:
            rounded = max(
                1.0,
                round(q)
            )

        elif q <= 100:
            rounded = max(
                5.0,
                round(q / 5) * 5
            )

        else:
            rounded = max(
                10.0,
                round(q / 10) * 10
            )

        vals.append(
            float(rounded)
        )

    vals_unique = []

    for v in vals:

        if v not in vals_unique:
            vals_unique.append(v)

    if (
        len(vals_unique) < 2
        and len(x) >= 2
    ):

        fallback = [
            float(
                np.nanmin(x)
            ),
            float(
                np.nanmedian(x)
            ),
            float(
                np.nanmax(x)
            ),
        ]

        vals_unique = []

        for v in fallback:

            if v <= 10:
                vv = max(
                    1.0,
                    round(v)
                )

            elif v <= 100:
                vv = max(
                    5.0,
                    round(v / 5) * 5
                )

            else:
                vv = max(
                    10.0,
                    round(v / 10) * 10
                )

            if vv not in vals_unique:
                vals_unique.append(
                    float(vv)
                )

    return vals_unique[:max_n]


def add_top10_reassignment_inset(
    ax,
    top10_summary
):
    """
    Add an upper-left inset showing inferred Top-10% -> Middle/Bottom flow.
    """

    if (
        not top10_summary
        or not top10_summary.get(
            "available",
            False
        )
    ):
        return None

    mid_pct = top10_summary.get(
        "top_to_middle_share_of_top_outflow_pct",
        np.nan
    )

    bot_pct = top10_summary.get(
        "top_to_bottom_share_of_top_outflow_pct",
        np.nan
    )

    top_outflow = top10_summary.get(
        "top_outflow",
        np.nan
    )

    inax = ax.inset_axes(
        [
            0.045,
            0.710,
            0.445,
            0.245
        ],
        zorder=8
    )

    inax.set_xlim(
        0,
        1
    )

    inax.set_ylim(
        0,
        1
    )

    inax.axis(
        "off"
    )

    bg = FancyBboxPatch(
        (0.00, 0.00),
        1.00,
        1.00,
        boxstyle=(
            "round,pad=0.015,"
            "rounding_size=0.035"
        ),
        facecolor="white",
        edgecolor="#D7DCE5",
        linewidth=0.75,
        alpha=0.92,
        transform=inax.transAxes,
        zorder=0,
    )

    inax.add_patch(bg)

    c_arrow = "#6D5BA6"
    c_text = "#243447"
    c_muted = "#697386"

    inax.text(
        0.03,
        1.05,
        "Reassigned Top-10% outflow",
        ha="left",
        va="top",
        fontsize=10.5,
        color=c_text,
        transform=inax.transAxes,
        zorder=5,
    )

    inax.text(
        0.045,
        0.935,
        f"denom. = {top_outflow:.0f} visits",
        ha="left",
        va="top",
        fontsize=10.5,
        color=c_muted,
        transform=inax.transAxes,
        zorder=5,
    )

    def node(xy, label):

        inax.text(
            xy[0],
            xy[1],
            label,
            ha="center",
            va="center",
            fontsize=7.8,
            color=c_text,
            bbox=dict(
                boxstyle=(
                    "round,pad=0.24,"
                    "rounding_size=0.10"
                ),
                facecolor="#F8FAFC",
                edgecolor="#D7DCE5",
                linewidth=0.70,
                alpha=0.98,
            ),
            transform=inax.transAxes,
            zorder=6,
        )

    p_top = (
        0.10,
        0.460
    )

    p_mid = (
        0.410,
        0.460
    )

    p_bot = (
        0.810,
        0.460
    )

    node(
        p_top,
        "Top 10%"
    )

    node(
        p_mid,
        "Middle 40%"
    )

    node(
        p_bot,
        "Bottom 50%"
    )

    arrow_mid = FancyArrowPatch(
        (0.10, 0.40),
        (0.42, 0.40),
        connectionstyle="arc3,rad=0.43",
        arrowstyle="-|>",
        mutation_scale=15,
        linewidth=3,
        facecolor="none",
        edgecolor=c_arrow,
        alpha=0.98,
        transform=inax.transAxes,
        zorder=4,
    )

    arrow_bot = FancyArrowPatch(
        (0.10, 0.5),
        (0.83, 0.5),
        connectionstyle="arc3,rad=-0.33",
        arrowstyle="-|>",
        mutation_scale=15,
        linewidth=3,
        facecolor="none",
        edgecolor=c_arrow,
        alpha=0.98,
        transform=inax.transAxes,
        zorder=4,
    )

    inax.add_patch(
        arrow_mid
    )

    inax.add_patch(
        arrow_bot
    )

    pct_bbox = dict(
        boxstyle=(
            "round,pad=0.15,"
            "rounding_size=0.08"
        ),
        facecolor="white",
        edgecolor="none",
        alpha=0.88,
    )

    inax.text(
        0.26,
        0.22,
        f"{mid_pct:.2f}%",
        ha="center",
        va="center",
        fontsize=8.2,
        color=c_arrow,
        fontweight="bold",
        bbox=pct_bbox,
        transform=inax.transAxes,
        zorder=7,
    )

    inax.text(
        0.50,
        0.77,
        f"{bot_pct:.2f}%",
        ha="center",
        va="center",
        fontsize=8.2,
        color=c_arrow,
        fontweight="bold",
        bbox=pct_bbox,
        transform=inax.transAxes,
        zorder=7,
    )

    return inax


# ============================================================
# 3. Fig.2d plotting
# ============================================================

def add_quadrant_background(
    ax,
    xlim,
    ylim
):
    """
    Very subtle quadrant background for publication-style plotting.
    """

    x0, x1 = xlim
    y0, y1 = ylim

    zero_y = (
        (0 - y0)
        / (y1 - y0)
    )

    zero_y = np.clip(
        zero_y,
        0,
        1
    )

    # More visits + SPSE gain
    ax.axvspan(
        0,
        x1,
        ymin=zero_y,
        ymax=1,
        facecolor="#EAF7F3",
        alpha=0.24,
        zorder=0,
    )

    # Fewer visits + SPSE gain
    ax.axvspan(
        x0,
        0,
        ymin=zero_y,
        ymax=1,
        facecolor="#EEF3FC",
        alpha=0.22,
        zorder=0,
    )

    # More visits + SPSE loss
    ax.axvspan(
        0,
        x1,
        ymin=0,
        ymax=zero_y,
        facecolor="#FBF1F1",
        alpha=0.18,
        zorder=0,
    )

    # Fewer visits + SPSE loss
    ax.axvspan(
        x0,
        0,
        ymin=0,
        ymax=zero_y,
        facecolor="#F5F6FA",
        alpha=0.20,
        zorder=0,
    )


def make_symmetric_xlim(x):

    finite_x = x[
        np.isfinite(x)
    ]

    if len(finite_x) == 0:
        return (
            -1,
            1
        )

    xmax = float(
        np.nanmax(
            np.abs(
                finite_x
            )
        )
    )

    xmax = max(
        5,
        np.ceil(
            (xmax + 5)
            / 10
        ) * 10
    )

    return (
        -xmax,
        xmax
    )


def make_zero_including_ylim(y):

    finite_y = y[
        np.isfinite(y)
    ]

    if len(finite_y) == 0:
        return (
            -1,
            1
        )

    ymin = float(
        np.nanmin(
            finite_y
        )
    )

    ymax = float(
        np.nanmax(
            finite_y
        )
    )

    if np.isclose(
        ymin,
        ymax
    ):
        ymin -= 1
        ymax += 1

    yrange = (
        ymax
        - ymin
    )

    ymin = (
        ymin
        - 0.12 * yrange
    )

    ymax = (
        ymax
        + 0.12 * yrange
    )

    ymin = min(
        ymin,
        -0.2
    )

    ymax = max(
        ymax,
        0.5
    )

    return (
        ymin,
        ymax
    )


def make_contribution_norm(
    color_var
):
    """
    Sequential normalization for the purple colorbar.

    Low values are light purple; high values are darker purple.
    This no longer treats zero as a diverging center.
    """

    finite_color = color_var[
        np.isfinite(
            color_var
        )
    ]

    if len(finite_color) == 0:
        return mcolors.Normalize(
            vmin=0.0,
            vmax=1.0,
            clip=True
        )

    vmin = float(
        np.nanpercentile(
            finite_color,
            5
        )
    )

    vmax = float(
        np.nanpercentile(
            finite_color,
            95
        )
    )

    if (
        (not np.isfinite(vmin))
        or (not np.isfinite(vmax))
        or np.isclose(
            vmin,
            vmax
        )
    ):

        vmin = float(
            np.nanmin(
                finite_color
            )
        )

        vmax = float(
            np.nanmax(
                finite_color
            )
        )

    if (
        (not np.isfinite(vmin))
        or (not np.isfinite(vmax))
        or np.isclose(
            vmin,
            vmax
        )
    ):
        vmin, vmax = (
            0.0,
            1.0
        )

    return mcolors.Normalize(
        vmin=vmin,
        vmax=vmax,
        clip=True
    )


def plot_fused_fig2d(
    case_output
):
    set_nature_style()

    df = (
        case_output[
            "scatter_df"
        ]
        .copy()
    )

    top10_summary = (
        case_output.get(
            "top10_reassignment_summary",
            {}
        )
    )

    c_grid = "#E1E7F0"
    c_axis = "#9CA3AF"
    c_text = "#243447"

    cmap = (
        make_weighted_contribution_cmap()
    )

    # Whether to show quadrant-level summed structural social exposure contribution.
    # If False, only POI counts are displayed.
    SHOW_QUADRANT_SUMS = False

    # --------------------------------------------------------
    # Plotted variables.
    # --------------------------------------------------------

    x = (
        df[
            "delta_visits"
        ]
        .astype(float)
        .values
    )

    # Average structural social exposure gain per CBG visiting the POI.
    y = (
        df[
            "mean_delta_structural_social_exposure_per_visiting_cbg"
        ]
        .astype(float)
        .values
    )

    # Marker size represents absolute POI visit change.
    size_var = np.abs(
        df[
            "delta_visits"
        ]
        .astype(float)
        .values
    )

    # Marker color represents POI-level weighted exposure contribution.
    color_var = (
        df[
            "delta_weighted_contribution"
        ]
        .astype(float)
        .values
    )

    xlim = make_symmetric_xlim(
        x
    )

    ylim = make_zero_including_ylim(
        y
    )

    norm = make_contribution_norm(
        color_var
    )

    size_vmax = (
        max(
            1.0,
            float(
                np.nanmax(
                    size_var
                )
            )
        )
        if len(size_var)
        else 1.0
    )

    # --------------------------------------------------------
    # Figure layout.
    # Colorbar height is aligned with the main axes height.
    # --------------------------------------------------------

    fig = plt.figure(
        figsize=(8, 8),
        dpi=300
    )

    fig.patch.set_facecolor(
        "white"
    )

    ax_left = 0.120
    ax_bottom = 0.155
    ax_width = 0.675
    ax_height = 0.720

    ax = fig.add_axes(
        [
            ax_left,
            ax_bottom,
            ax_width,
            ax_height
        ]
    )

    cax = fig.add_axes(
        [
            ax_left
            + ax_width
            + 0.045,

            ax_bottom,

            0.020,

            ax_height,
        ]
    )

    ax.set_xlim(
        *xlim
    )

    ax.set_ylim(
        *ylim
    )

    add_quadrant_background(
        ax,
        xlim,
        ylim
    )

    ax.axhline(
        0,
        color=c_axis,
        linewidth=0.85,
        linestyle="--",
        zorder=1
    )

    ax.axvline(
        0,
        color=c_axis,
        linewidth=0.85,
        linestyle="--",
        zorder=1
    )

    ax.grid(
        color=c_grid,
        linewidth=0.65,
        zorder=0.5
    )

    tier_order_short = [
        (
            "Top 10% baseline-flow POIs",
            "Top 10%",
            "^"
        ),
        (
            "Middle 40% baseline-flow POIs",
            "Middle 40%",
            "o"
        ),
        (
            "Bottom 50% baseline-flow POIs",
            "Bottom 50%",
            "s"
        ),
    ]

    for (
        tier,
        _,
        marker
    ) in tier_order_short:

        sub = (
            df[
                df[
                    "poi_tier"
                ]
                == tier
            ]
            .copy()
        )

        if len(sub) == 0:
            continue

        sub_x = (
            sub[
                "delta_visits"
            ]
            .astype(float)
            .values
        )

        sub_y = (
            sub[
                "mean_delta_structural_social_exposure_per_visiting_cbg"
            ]
            .astype(float)
            .values
        )

        sub_s = (
            scatter_size_from_abs_delta_visits(
                sub[
                    "delta_visits"
                ]
                .astype(float)
                .values,
                size_vmax,
            )
        )

        sub_c = (
            sub[
                "delta_weighted_contribution"
            ]
            .astype(float)
            .values
        )

        color_mask = (
            np.isfinite(
                sub_c
            )
        )

        if color_mask.any():

            ax.scatter(
                sub_x[
                    color_mask
                ],
                sub_y[
                    color_mask
                ],
                s=sub_s[
                    color_mask
                ],
                c=sub_c[
                    color_mask
                ],
                cmap=cmap,
                norm=norm,
                marker=marker,
                edgecolor="white",
                linewidth=0.70,
                alpha=0.90,
                zorder=3,
            )

        if (
            ~color_mask
        ).any():

            ax.scatter(
                sub_x[
                    ~color_mask
                ],
                sub_y[
                    ~color_mask
                ],
                s=sub_s[
                    ~color_mask
                ],
                color="#CBD5E1",
                marker=marker,
                edgecolor="white",
                linewidth=0.70,
                alpha=0.85,
                zorder=3,
            )

    # --------------------------------------------------------
    # Quadrant counts and quadrant-level contribution sums.
    # --------------------------------------------------------

    m_pp = (
        (x > 0)
        & (y > 0)
    )

    m_mp = (
        (x < 0)
        & (y > 0)
    )

    m_mm = (
        (x < 0)
        & (y < 0)
    )

    m_pm = (
        (x > 0)
        & (y < 0)
    )

    q_plus_plus = int(
        m_pp.sum()
    )

    q_minus_plus = int(
        m_mp.sum()
    )

    q_minus_minus = int(
        m_mm.sum()
    )

    q_plus_minus = int(
        m_pm.sum()
    )

    sum_plus_plus = float(
        np.nansum(
            y[
                m_pp
            ]
        )
    )

    sum_minus_plus = float(
        np.nansum(
            y[
                m_mp
            ]
        )
    )

    sum_minus_minus = float(
        np.nansum(
            y[
                m_mm
            ]
        )
    )

    sum_plus_minus = float(
        np.nansum(
            y[
                m_pm
            ]
        )
    )

    print(
        "\n========== FIG.2D QUADRANT COUNTS =========="
    )

    print(
        "More visits + structural social exposure gain: "
        f"{q_plus_plus}; "
        f"sum Δ = {sum_plus_plus:.3f}"
    )

    print(
        "Fewer visits + structural social exposure gain: "
        f"{q_minus_plus}; "
        f"sum Δ = {sum_minus_plus:.3f}"
    )

    print(
        "Fewer visits + structural social exposure loss: "
        f"{q_minus_minus}; "
        f"sum Δ = {sum_minus_minus:.3f}"
    )

    print(
        "More visits + structural social exposure loss: "
        f"{q_plus_minus}; "
        f"sum Δ = {sum_plus_minus:.3f}"
    )

    def quadrant_label(
        n,
        s
    ):

        if SHOW_QUADRANT_SUMS:

            return (
                f"n = {n}\n"
                f"ΣΔ = {s:+.1f}"
            )

        return (
            f"n = {n}"
        )

    quad_bbox = dict(
        facecolor="white",
        edgecolor="none",
        alpha=0.78,
        pad=1.35,
    )

    # Upper-left: fewer visits, structural social exposure gain.
    # Moved downward to leave room for the Top-10% reassignment inset.
    ax.text(
        0.025,
        0.675,
        quadrant_label(
            q_minus_plus,
            sum_minus_plus
        ),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=12,
        color=c_text,
        bbox=quad_bbox,
        linespacing=1.08,
        zorder=5,
    )

    # Upper-right: more visits, structural social exposure gain.
    ax.text(
        0.975,
        0.965,
        quadrant_label(
            q_plus_plus,
            sum_plus_plus
        ),
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=12,
        color=c_text,
        bbox=quad_bbox,
        linespacing=1.08,
        zorder=5,
    )

    # Lower-left: fewer visits, structural social exposure loss.
    ax.text(
        0.025,
        0.040,
        quadrant_label(
            q_minus_minus,
            sum_minus_minus
        ),
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=12,
        color=c_text,
        bbox=quad_bbox,
        linespacing=1.08,
        zorder=5,
    )

    # Lower-right: more visits, structural social exposure loss.
    # Placed slightly above the legend to avoid overlap.
    ax.text(
        0.975,
        0.040,
        quadrant_label(
            q_plus_minus,
            sum_plus_minus
        ),
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=12,
        color=c_text,
        bbox=quad_bbox,
        linespacing=1.08,
        zorder=5,
    )

    # --------------------------------------------------------
    # Upper-left inset: inferred Top-10% -> Middle/Bottom reassignment.
    # --------------------------------------------------------

    add_top10_reassignment_inset(
        ax,
        top10_summary
    )

    # --------------------------------------------------------
    # Centered title.
    # --------------------------------------------------------

    if PANEL_TITLE:

        ax.set_title(
            PANEL_TITLE,
            loc="center",
            pad=7,
            fontsize=14,
        )

    ax.set_xlabel(
        "Δ POI visits",
        labelpad=6,
        fontsize=14
    )

    ax.set_ylabel(
        "Average Δ structural social exposure per visiting CBG",
        labelpad=7,
        fontsize=14
    )

    ax.spines[
        [
            "top",
            "right"
        ]
    ].set_visible(
        False
    )

    ax.tick_params(
        axis="both",
        length=2.8,
        color="#6B7280"
    )

    # --------------------------------------------------------
    # Purple colorbar: same height as main axes.
    # --------------------------------------------------------

    sm = ScalarMappable(
        norm=norm,
        cmap=cmap
    )

    sm.set_array(
        []
    )

    cbar = fig.colorbar(
        sm,
        cax=cax,
        orientation="vertical"
    )

    cbar.set_label(
        "Δ weighted contribution",
        fontsize=14,
        labelpad=7,
    )

    cbar.ax.tick_params(
        labelsize=12,
        length=2
    )

    # --------------------------------------------------------
    # Marker-shape legend inside the plotting area.
    # --------------------------------------------------------

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="^",
            linestyle="none",
            markerfacecolor="#C7B9E8",
            markeredgecolor="white",
            markeredgewidth=0.7,
            markersize=6.4,
            label="Top 10%",
        ),

        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor="#C7B9E8",
            markeredgecolor="white",
            markeredgewidth=0.7,
            markersize=6.4,
            label="Middle 40%",
        ),

        Line2D(
            [0],
            [0],
            marker="s",
            linestyle="none",
            markerfacecolor="#C7B9E8",
            markeredgecolor="white",
            markeredgewidth=0.7,
            markersize=6.4,
            label="Bottom 50%",
        ),
    ]

    tier_legend = ax.legend(
        handles=legend_handles,
        title="Baseline-flow POI tier",
        loc="lower right",
        bbox_to_anchor=(
            0.975,
            0.045
        ),
        ncol=1,
        frameon=True,
        fancybox=False,
        framealpha=0.84,
        facecolor="white",
        edgecolor="none",
        fontsize=12,
        title_fontsize=12,
        borderpad=0.35,
        handletextpad=0.40,
        labelspacing=0.28,
    )

    ax.add_artist(
        tier_legend
    )

    # --------------------------------------------------------
    # Marker-size legend: absolute POI visit change.
    # --------------------------------------------------------

    size_legend_values = (
        choose_size_legend_values(
            size_var,
            max_n=3
        )
    )

    size_handles = []

    for v in size_legend_values:

        s_legend = (
            scatter_size_from_abs_delta_visits(
                np.array(
                    [v]
                ),
                size_vmax
            )[0]
        )

        size_handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="none",
                markerfacecolor="#C7B9E8",
                markeredgecolor="white",
                markeredgewidth=0.7,
                markersize=(
                    np.sqrt(
                        s_legend
                    )
                    / 1.65
                ),
                alpha=0.90,
                label=f"{v:.0f}",
            )
        )

    ax.legend(
        handles=size_handles,
        title="|Δ POI visits|",
        loc="upper right",
        bbox_to_anchor=(
            0.855,
            0.38
        ),
        ncol=1,
        frameon=True,
        fancybox=False,
        framealpha=0.84,
        facecolor="white",
        edgecolor="none",
        fontsize=12,
        title_fontsize=12,
        borderpad=0.35,
        handletextpad=0.55,
        labelspacing=0.40,
    )

    fig.savefig(
        "figure2d.pdf",
        format="pdf",
        dpi=300,
        bbox_inches="tight",
        transparent=False,
        backend="pdf"
    )

    plt.show()

    return fig


# ============================================================
# 4. Main
# ============================================================

def main():

    case_output = (
        load_public_case_output()
    )

    case_summary = (
        case_output.get(
            "case_summary",
            {}
        )
    )

    if case_summary:

        print(
            "\n========== FIG.2D SPSE DESTINATION CASE SUMMARY =========="
        )

        for k, v in case_summary.items():
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
            f"{float(top10_summary['top_outflow']):.3f} visits; "
            "Top -> Middle: "
            f"{float(top10_summary['top_to_middle']):.3f} visits "
            f"({float(top10_summary['top_to_middle_share_of_top_outflow_pct']):.2f}% "
            "of Top outflow); "
            "Top -> Bottom: "
            f"{float(top10_summary['top_to_bottom']):.3f} visits "
            f"({float(top10_summary['top_to_bottom_share_of_top_outflow_pct']):.2f}% "
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

    if case_summary:

        required_keys = [
            "sum_delta_spse_from_pois",
            "spse_delta",
            "spse_decomposition_error",

            "sum_delta_weighted_contribution_from_pois",
            "weighted_exposure_numerator_delta",
            "weighted_decomposition_error",

            "sum_delta_active_links_from_pois",
            "total_delta_active_links",
            "active_link_decomposition_error",
        ]

        if all(
            k in case_summary
            for k in required_keys
        ):

            print(
                "\n========== DECOMPOSITION CHECK =========="
            )

            s = case_summary

            print(
                "SPSE decomposition: "
                f"sum_j ΔSPSE_j = "
                f"{float(s['sum_delta_spse_from_pois']):.10f}; "
                f"system ΔSPSE = "
                f"{float(s['spse_delta']):.10f}; "
                f"error = "
                f"{float(s['spse_decomposition_error']):.10e}"
            )

            print(
                "Weighted exposure contribution decomposition: "
                f"sum_j ΔC_j = "
                f"{float(s['sum_delta_weighted_contribution_from_pois']):.10f}; "
                f"system ΔC = "
                f"{float(s['weighted_exposure_numerator_delta']):.10f}; "
                f"error = "
                f"{float(s['weighted_decomposition_error']):.10e}"
            )

            print(
                "Positive-link support decomposition: "
                f"sum_j Δlinks_j = "
                f"{int(float(s['sum_delta_active_links_from_pois']))}; "
                f"system Δlinks = "
                f"{int(float(s['total_delta_active_links']))}; "
                f"error = "
                f"{int(float(s['active_link_decomposition_error']))}"
            )

    plot_fused_fig2d(
        case_output
    )


if __name__ == "__main__":
    main()