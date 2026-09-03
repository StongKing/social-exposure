# -*- coding: utf-8 -*-
"""
Figure 4d: Behavioural anchoring and retained performance

Final design:
    Panel a:
        Main: origin-level behavioural pullback toward observed matrix F.
              CBGs are ranked by c_i^F = d_i(H_no,F) - d_i(H_reg,F).
              The two curves show d_i(H_no,F) and d_i(H_reg,F).

        Inset:
              matrix-level pullback after L1 regularization,
              excluding D(F,R).

    Panel b:
        Total travel distance trajectory.

    Panel c:
        Structural potential social exposure trajectory.

        Left axis:
            raw SPSE.

        Right axis:
            percentage change relative to observed baseline.

Required public inputs:
    figure4d_origin_pullback.csv
    figure4d_summary.csv

    matrices_A_D_S_Distribution/
        Other_Individual_and_Family_Services/
            results_regu_boston_624190.csv
            results_boston_624190.csv

Output:
    figure4d.pdf

@author: JZS
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.ticker import FuncFormatter


# ============================================================
# Style
# ============================================================

try:

    plt.rcParams[
        "font.sans-serif"
    ] = [

        "Microsoft YaHei",

        "Arial Unicode MS",

        "DejaVu Sans"
    ]


    plt.rcParams[
        "axes.unicode_minus"
    ] = False


except Exception:

    pass


plt.rcParams.update({

    "font.size":
        11,

    "axes.titlesize":
        12.5,

    "axes.labelsize":
        11,

    "xtick.labelsize":
        10,

    "ytick.labelsize":
        10,

    "legend.fontsize":
        9.5,

    "pdf.fonttype":
        42,

    "ps.fonttype":
        42,
})


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


regu_csv_path = os.path.join(
    cat_dir,
    f"results_regu_{city}_{poi_code}.csv"
)


dynamic_csv_path = os.path.join(
    cat_dir,
    f"results_{city}_{poi_code}.csv"
)


origin_pullback_path = (
    "figure4d_origin_pullback.csv"
)


summary_path = (
    "figure4d_summary.csv"
)


out_pdf = (
    "figure4d.pdf"
)


out_png = (
    "figure4d.png"
)


dpi = 300


figsize = (
    15.5,
    5.0
)


dynamic_color = (
    "#3498db"
)


regu_color = (
    "#7C5BB8"
)


baseline_color = (
    "0.45"
)


curve_F_color = (
    "#5C5C5C"
)


curve_R_color = (
    "#7E57C2"
)


# ============================================================
# Helper functions
# ============================================================

def check_file(
    path
):

    if not os.path.exists(
        path
    ):

        raise FileNotFoundError(
            f"Cannot find file: {path}"
        )


def to_numeric_array(
    df,
    col
):

    return (
        pd.to_numeric(
            df[
                col
            ],
            errors="coerce"
        )
        .to_numpy(
            dtype=float
        )
    )


def pct_change(
    new,
    base
):

    if (
        base == 0
        or
        np.isnan(base)
        or
        np.isnan(new)
    ):

        return np.nan


    return (

        (
            new
            -
            base
        )

        /

        abs(
            base
        )

        *

        100.0
    )


def pct_axis_formatter(
    x,
    pos
):

    return (
        f"{x:.0f}%"
    )


def add_percent_secondary_axis(
    ax,
    baseline,
    label="Change from baseline (%)"
):
    """
    Add a right y-axis tied to the raw-value left y-axis.
    """

    if (
        baseline == 0
        or
        np.isnan(
            baseline
        )
    ):

        return None


    def value_to_pct(
        y
    ):

        y = np.asarray(
            y,
            dtype=float
        )


        return (

            (
                y
                -
                baseline
            )

            /

            abs(
                baseline
            )

            *

            100.0
        )


    def pct_to_value(
        p
    ):

        p = np.asarray(
            p,
            dtype=float
        )


        return (

            baseline

            +

            p
            /
            100.0
            *
            abs(
                baseline
            )
        )


    secax = ax.secondary_yaxis(

        "right",

        functions=(
            value_to_pct,
            pct_to_value
        )
    )


    secax.set_ylabel(
        label,
        labelpad=9
    )


    secax.yaxis.set_major_formatter(
        FuncFormatter(
            pct_axis_formatter
        )
    )


    secax.tick_params(
        axis="y",
        labelsize=10
    )


    return secax


def fmt_num(
    x
):

    if pd.isna(
        x
    ):

        return "NA"


    if abs(
        x
    ) >= 100:

        return (
            f"{x:.2f}"
        )


    elif abs(
        x
    ) >= 1:

        return (
            f"{x:.4f}"
        )


    else:

        return (
            f"{x:.6f}"
        )


def fmt_pct(
    x
):

    if pd.isna(
        x
    ):

        return "NA"


    return (
        f"{x:.2f}\\%"
    )


def gain_retention(
    lower_is_better,
    baseline,
    dynamic_final,
    regu_final
):
    """
    Retained share of the unregularized gain
    after regularization.
    """

    if lower_is_better:

        dynamic_gain = (
            baseline
            -
            dynamic_final
        )


        regu_gain = (
            baseline
            -
            regu_final
        )


    else:

        dynamic_gain = (
            dynamic_final
            -
            baseline
        )


        regu_gain = (
            regu_final
            -
            baseline
        )


    if (
        dynamic_gain == 0
        or
        np.isnan(
            dynamic_gain
        )
        or
        np.isnan(
            regu_gain
        )
    ):

        return (
            np.nan,
            dynamic_gain,
            regu_gain
        )


    retention = (

        regu_gain
        /
        dynamic_gain
        *
        100.0
    )


    return (
        retention,
        dynamic_gain,
        regu_gain
    )


# ============================================================
# Check inputs
# ============================================================

for path in [

    regu_csv_path,

    dynamic_csv_path,

    origin_pullback_path,

    summary_path

]:

    check_file(
        path
    )


# ============================================================
# Load preprocessed matrix-level Fig.4d data
# ============================================================

origin_pullback_df = pd.read_csv(
    origin_pullback_path
)


summary_df = pd.read_csv(
    summary_path
)


required_origin_columns = [

    "rank_pct",

    "dev_no_F_plot",

    "dev_reg_F_plot",

    "dev_no_F_raw",

    "dev_reg_F_raw",

    "contrib_F_raw",

    "changed_Hno_Hreg_raw"
]


missing_origin_columns = [

    col

    for col
    in required_origin_columns

    if col
    not in origin_pullback_df.columns
]


if len(
    missing_origin_columns
) > 0:

    raise ValueError(
        "figure4d_origin_pullback.csv "
        "is missing columns:\n"
        f"{missing_origin_columns}"
    )


if (
    "metric"
    not in summary_df.columns
    or
    "value"
    not in summary_df.columns
):

    raise ValueError(
        "figure4d_summary.csv must contain "
        "'metric' and 'value' columns."
    )


summary_values = (

    summary_df
    .set_index(
        "metric"
    )[
        "value"
    ]
    .to_dict()
)


# ============================================================
# Restore Panel a arrays
# ============================================================

x_rank_F = (
    pd.to_numeric(
        origin_pullback_df[
            "rank_pct"
        ],
        errors="coerce"
    )
    .to_numpy(
        dtype=float
    )
)


dev_no_F_plot = (
    pd.to_numeric(
        origin_pullback_df[
            "dev_no_F_plot"
        ],
        errors="coerce"
    )
    .to_numpy(
        dtype=float
    )
)


dev_reg_F_plot = (
    pd.to_numeric(
        origin_pullback_df[
            "dev_reg_F_plot"
        ],
        errors="coerce"
    )
    .to_numpy(
        dtype=float
    )
)


dev_no_F_rank_raw = (
    pd.to_numeric(
        origin_pullback_df[
            "dev_no_F_raw"
        ],
        errors="coerce"
    )
    .to_numpy(
        dtype=float
    )
)


dev_reg_F_rank_raw = (
    pd.to_numeric(
        origin_pullback_df[
            "dev_reg_F_raw"
        ],
        errors="coerce"
    )
    .to_numpy(
        dtype=float
    )
)


contrib_F_rank_raw = (
    pd.to_numeric(
        origin_pullback_df[
            "contrib_F_raw"
        ],
        errors="coerce"
    )
    .to_numpy(
        dtype=float
    )
)


changed_F_rank_raw = (
    pd.to_numeric(
        origin_pullback_df[
            "changed_Hno_Hreg_raw"
        ],
        errors="coerce"
    )
    .to_numpy(
        dtype=float
    )
)


if len(
    x_rank_F
) == 0:

    raise ValueError(
        "No Panel a origin-level data found."
    )


# ============================================================
# Restore scalar matrix-level quantities
# ============================================================

n_CBGs = int(
    summary_values[
        "n_CBGs"
    ]
)


n_POIs = int(
    summary_values[
        "n_POIs"
    ]
)


total_flow = float(
    summary_values[
        "total_flow"
    ]
)


baseline_distance = float(
    summary_values[
        "baseline_distance"
    ]
)


baseline_social = float(
    summary_values[
        "baseline_social"
    ]
)


no_regu_ref_dev_pct = float(
    summary_values[
        "no_regu_ref_dev_pct"
    ]
)


regu_ref_dev_pct = float(
    summary_values[
        "regu_ref_dev_pct"
    ]
)


ref_dev_reduction_pct = float(
    summary_values[
        "ref_dev_reduction_pct"
    ]
)


no_regu_from_F_pct = float(
    summary_values[
        "no_regu_from_F_pct"
    ]
)


regu_from_F_pct = float(
    summary_values[
        "regu_from_F_pct"
    ]
)


R_from_F_pct = float(
    summary_values[
        "R_from_F_pct"
    ]
)


from_F_reduction_pct = float(
    summary_values[
        "from_F_reduction_pct"
    ]
)


net_reduction_from_F_pp = float(
    summary_values[
        "net_reduction_from_F_pp"
    ]
)


positive_reduction_from_F_pp = float(
    summary_values[
        "positive_reduction_from_F_pp"
    ]
)


negative_offset_from_F_pp = float(
    summary_values[
        "negative_offset_from_F_pp"
    ]
)


share_positive_F = float(
    summary_values[
        "share_positive_F"
    ]
)


top10_F = float(
    summary_values[
        "top10_F"
    ]
)


top20_F = float(
    summary_values[
        "top20_F"
    ]
)


net_reduction_from_R_pp = float(
    summary_values[
        "net_reduction_from_R_pp"
    ]
)


positive_reduction_from_R_pp = float(
    summary_values[
        "positive_reduction_from_R_pp"
    ]
)


negative_offset_from_R_pp = float(
    summary_values[
        "negative_offset_from_R_pp"
    ]
)


share_positive_R = float(
    summary_values[
        "share_positive_R"
    ]
)


top10_R = float(
    summary_values[
        "top10_R"
    ]
)


top20_R = float(
    summary_values[
        "top20_R"
    ]
)


share_positive_both = float(
    summary_values[
        "share_positive_both"
    ]
)


share_changed_no_to_reg = float(
    summary_values[
        "share_changed_no_to_reg"
    ]
)


# ============================================================
# Print original matrix diagnostics
# ============================================================

print(
    "\n========== Matrix domain =========="
)


print(
    f"n_CBGs: "
    f"{n_CBGs}"
)


print(
    f"n_POIs: "
    f"{n_POIs}"
)


print(
    f"baseline total flow: "
    f"{total_flow:.6f}"
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
# Load trajectory CSVs
# ============================================================

df_regu = pd.read_csv(
    regu_csv_path
)


df_dynamic = pd.read_csv(
    dynamic_csv_path
)


required_cols = [

    "f_values_iter",

    "distances_iter",

    "social_iter"
]


for name, df in [

    (
        "Regularized",
        df_regu
    ),

    (
        "Unregularized continuous",
        df_dynamic
    )

]:

    miss = [

        c

        for c
        in required_cols

        if c
        not in df.columns
    ]


    if miss:

        print(
            f"[ERROR] {name} file missing columns: "
            f"{miss}"
        )


        print(
            f"Current columns: "
            f"{list(df.columns)}"
        )


        sys.exit(
            1
        )


regu_f = to_numeric_array(
    df_regu,
    "f_values_iter"
)


regu_d = to_numeric_array(
    df_regu,
    "distances_iter"
)


regu_s = to_numeric_array(
    df_regu,
    "social_iter"
)


dyn_f = to_numeric_array(
    df_dynamic,
    "f_values_iter"
)


dyn_d = to_numeric_array(
    df_dynamic,
    "distances_iter"
)


dyn_s = to_numeric_array(
    df_dynamic,
    "social_iter"
)


valid_regu = ~(
    np.isnan(
        regu_d
    )
    &
    np.isnan(
        regu_s
    )
)


valid_dyn = ~(
    np.isnan(
        dyn_d
    )
    &
    np.isnan(
        dyn_s
    )
)


regu_f = regu_f[
    valid_regu
]


regu_d = regu_d[
    valid_regu
]


regu_s = regu_s[
    valid_regu
]


dyn_f = dyn_f[
    valid_dyn
]


dyn_d = dyn_d[
    valid_dyn
]


dyn_s = dyn_s[
    valid_dyn
]


if (
    len(
        regu_d
    ) == 0
    or
    len(
        dyn_d
    ) == 0
):

    print(
        "[ERROR] One trajectory file has no valid data rows."
    )


    sys.exit(
        1
    )


x_dynamic = np.arange(
    1,
    len(
        dyn_d
    ) + 1
)


if len(
    regu_d
) == 100:

    x_regu = np.arange(
        1,
        101
    )


else:

    x_regu = np.linspace(
        1,
        100,
        len(
            regu_d
        )
    )


# ============================================================
# Final values and gain retention
# ============================================================

dynamic_final_f = float(
    dyn_f[
        -1
    ]
)


dynamic_final_d = float(
    dyn_d[
        -1
    ]
)


dynamic_final_s = float(
    dyn_s[
        -1
    ]
)


regu_final_f = float(
    regu_f[
        -1
    ]
)


regu_final_d = float(
    regu_d[
        -1
    ]
)


regu_final_s = float(
    regu_s[
        -1
    ]
)


(
    distance_retention_pct,

    dynamic_d_gain,

    regu_d_gain

) = gain_retention(

    lower_is_better=True,

    baseline=baseline_distance,

    dynamic_final=dynamic_final_d,

    regu_final=regu_final_d
)


(
    social_retention_pct,

    dynamic_s_gain,

    regu_s_gain

) = gain_retention(

    lower_is_better=False,

    baseline=baseline_social,

    dynamic_final=dynamic_final_s,

    regu_final=regu_final_s
)


print(
    "\n========== Benefit retention relative to observed baseline =========="
)


print(
    f"Dynamic distance gain:       "
    f"{dynamic_d_gain:.6f}"
)


print(
    f"Regularized distance gain:   "
    f"{regu_d_gain:.6f}"
)


print(
    f"Distance retained:           "
    f"{distance_retention_pct:.6f}%"
)


print(
    f"\nDynamic social gain:         "
    f"{dynamic_s_gain:.6f}"
)


print(
    f"Regularized social gain:     "
    f"{regu_s_gain:.6f}"
)


print(
    f"Social exposure retained:    "
    f"{social_retention_pct:.6f}%"
)


# ============================================================
# Plot Figure 4d
# ============================================================

fig, axes = plt.subplots(

    1,

    3,

    figsize=figsize,

    dpi=dpi
)


# ------------------------------------------------------------
# Panel a:
# origin-level behavioural pullback toward F
# ------------------------------------------------------------

ax = axes[
    0
]


ax.fill_between(

    x_rank_F,

    dev_reg_F_plot,

    dev_no_F_plot,

    where=(
        dev_no_F_plot
        >=
        dev_reg_F_plot
    ),

    color="0.82",

    alpha=0.55,

    zorder=1,

    label=r"Pullback $c_i^F$"
)


ax.plot(

    x_rank_F,

    dev_no_F_plot,

    color=dynamic_color,

    linewidth=2.5,

    zorder=3,

    label=r"Unregularized $H_{\mathrm{no}}$"
)


ax.plot(

    x_rank_F,

    dev_reg_F_plot,

    color=regu_color,

    linewidth=2.5,

    linestyle="--",

    zorder=4,

    label=r"$L_1$-regularized $H_{\mathrm{reg}}$"
)


ax.set_title(
    r"Origin-level behavioral pullback toward $F$"
)


ax.set_xlabel(
    r"Changed CBGs ranked by pullback $c_i^F$ (%)"
)


ax.set_ylabel(
    r"Origin-level departure $d_i(H,F)$"
)


ax.set_xlim(
    0,
    100
)


ymax_a = max(

    np.nanmax(
        dev_no_F_plot
    ),

    np.nanmax(
        dev_reg_F_plot
    )
)


ax.set_ylim(
    0,
    ymax_a
    *
    1.12
)


ax.grid(
    True,
    linestyle=":",
    linewidth=0.55,
    alpha=0.75
)


# Main legend
ax.legend(

    loc="upper right",

    fontsize=7.8,

    frameon=False,

    handlelength=2.1
)


# ============================================================
# Inset:
# matrix-level pullback, excluding D(F,R)
# ============================================================

inset = ax.inset_axes(
    [
        0.24,
        0.48,
        0.47,
        0.35
    ]
)


metrics = [

    r"from $F$",

    r"from $R$"
]


unreg_vals = np.array(

    [

        no_regu_from_F_pct,

        no_regu_ref_dev_pct

    ],

    dtype=float
)


regu_vals = np.array(

    [

        regu_from_F_pct,

        regu_ref_dev_pct

    ],

    dtype=float
)


reduction_vals = np.array(

    [

        from_F_reduction_pct,

        ref_dev_reduction_pct

    ],

    dtype=float
)


y = np.arange(
    len(
        metrics
    )
)


bar_h = 0.32


inset.barh(

    y
    +
    bar_h
    /
    2,

    unreg_vals,

    height=bar_h,

    color=dynamic_color,

    alpha=0.85,

    label="Unregularized",

    zorder=2
)


inset.barh(

    y
    -
    bar_h
    /
    2,

    regu_vals,

    height=bar_h,

    color=regu_color,

    alpha=0.85,

    label=r"$L_1$-regularized",

    zorder=3
)


for i, (
    u,
    r,
    red
) in enumerate(
    zip(
        unreg_vals,
        regu_vals,
        reduction_vals
    )
):

    inset.text(

        u
        +
        1.0,

        y[
            i
        ]
        +
        bar_h
        /
        2,

        f"{u:.1f}",

        ha="left",

        va="center",

        fontsize=7.1,

        color=dynamic_color
    )


    inset.text(

        r
        +
        1.0,

        y[
            i
        ]
        -
        bar_h
        /
        2,

        f"{r:.1f}",

        ha="left",

        va="center",

        fontsize=7.1,

        color=regu_color
    )


inset.set_yticks(
    y
)


inset.set_yticklabels(
    metrics,
    fontsize=7.8
)


inset.invert_yaxis()


inset.set_xlim(
    0,
    70
)


inset.set_ylim(
    -0.45,
    1.45
)


inset.set_xlabel(
    "Matrix departure (% visits)",
    fontsize=7.7,
    labelpad=1
)


inset.set_title(
    "Global pullback",
    fontsize=8.4,
    pad=3
)


inset.tick_params(
    axis="x",
    labelsize=7.2,
    length=2
)


inset.tick_params(
    axis="y",
    labelsize=7.6,
    length=0
)


inset.grid(
    True,
    axis="x",
    linestyle=":",
    linewidth=0.45,
    alpha=0.7
)


inset.grid(
    False,
    axis="y"
)


inset.legend(

    loc="lower right",

    fontsize=6.6,

    frameon=False,

    handlelength=1.1,

    borderaxespad=0.2
)


for spine in inset.spines.values():

    spine.set_linewidth(
        0.7
    )


    spine.set_edgecolor(
        "0.55"
    )


try:

    ax.set_box_aspect(
        1
    )


except Exception:

    pass


# ------------------------------------------------------------
# Panel b:
# distance trajectory
# ------------------------------------------------------------

ax = axes[
    1
]


ax.plot(

    x_dynamic,

    dyn_d,

    label="Unregularized continuous",

    linewidth=2.4,

    color=dynamic_color
)


ax.plot(

    x_regu,

    regu_d,

    label=r"$L_1$-regularized continuous",

    linewidth=2.4,

    linestyle="--",

    color=regu_color
)


ax.scatter(

    [
        x_dynamic[
            -1
        ]
    ],

    [
        dynamic_final_d
    ],

    color=dynamic_color,

    s=38,

    zorder=4
)


ax.scatter(

    [
        x_regu[
            -1
        ]
    ],

    [
        regu_final_d
    ],

    color=regu_color,

    s=38,

    zorder=4
)


ax.set_title(
    "Travel-distance trajectory"
)


ax.set_xlabel(
    "Iteration"
)


ax.set_ylabel(
    "Total travel distance"
)


ax.set_xlim(
    1,
    100
)


ax.grid(
    True,
    linestyle=":",
    linewidth=0.55,
    alpha=0.8
)


add_percent_secondary_axis(

    ax,

    baseline=baseline_distance,

    label=""
)


ax.legend(

    loc="upper left",

    fontsize=8.8,

    frameon=False,

    handlelength=2.0
)


try:

    ax.set_box_aspect(
        1
    )


except Exception:

    pass


# ------------------------------------------------------------
# Panel c:
# structural social exposure trajectory
# ------------------------------------------------------------

ax = axes[
    2
]


ax.plot(

    x_dynamic,

    dyn_s,

    label="Unregularized continuous",

    linewidth=2.4,

    color=dynamic_color
)


ax.plot(

    x_regu,

    regu_s,

    label=r"$L_1$-regularized continuous",

    linewidth=2.4,

    linestyle="--",

    color=regu_color
)


ax.scatter(

    [
        x_dynamic[
            -1
        ]
    ],

    [
        dynamic_final_s
    ],

    color=dynamic_color,

    s=38,

    zorder=4
)


ax.scatter(

    [
        x_regu[
            -1
        ]
    ],

    [
        regu_final_s
    ],

    color=regu_color,

    s=38,

    zorder=4
)


ax.set_title(
    "Structural-exposure trajectory"
)


ax.set_xlabel(
    "Iteration"
)


ax.set_ylabel(
    "Structural potential social exposure"
)


ax.set_xlim(
    1,
    100
)


ax.grid(
    True,
    linestyle=":",
    linewidth=0.55,
    alpha=0.8
)


add_percent_secondary_axis(

    ax,

    baseline=baseline_social,

    label="Change from baseline (%)"
)


ax.legend(

    loc="upper left",

    fontsize=8.8,

    frameon=False,

    handlelength=2.0
)


try:

    ax.set_box_aspect(
        1
    )


except Exception:

    pass


# ============================================================
# Layout and output
# ============================================================

plt.tight_layout()


plt.subplots_adjust(
    wspace=0.25,
    bottom=0.02
)


plt.savefig(

    'figure4d.pdf',

    format='pdf',

    dpi=300,

    bbox_inches='tight',

    transparent=False,

    backend='pdf'
)


plt.show()


# ============================================================
# Tables and LaTeX-ready outputs
# ============================================================

final_compare = {

    "Travel distance": {

        "Observed baseline":
            baseline_distance,

        "Dynamic final":
            dynamic_final_d,

        "Regularized final":
            regu_final_d,

        "Regu_vs_Dynamic_percent":
            pct_change(
                regu_final_d,
                dynamic_final_d
            ),

        "Dynamic_vs_baseline_percent":
            pct_change(
                dynamic_final_d,
                baseline_distance
            ),

        "Regularized_vs_baseline_percent":
            pct_change(
                regu_final_d,
                baseline_distance
            ),

        "Gain_retention_percent":
            distance_retention_pct,
    },


    "Structural social exposure": {

        "Observed baseline":
            baseline_social,

        "Dynamic final":
            dynamic_final_s,

        "Regularized final":
            regu_final_s,

        "Regu_vs_Dynamic_percent":
            pct_change(
                regu_final_s,
                dynamic_final_s
            ),

        "Dynamic_vs_baseline_percent":
            pct_change(
                dynamic_final_s,
                baseline_social
            ),

        "Regularized_vs_baseline_percent":
            pct_change(
                regu_final_s,
                baseline_social
            ),

        "Gain_retention_percent":
            social_retention_pct,
    }
}


anchoring_compare = {

    "Deviation from R": {

        "Unregularized final":
            no_regu_ref_dev_pct,

        "Regularized final":
            regu_ref_dev_pct,

        "Reduction_percent":
            ref_dev_reduction_pct,
    },


    "Deviation from observed F": {

        "Unregularized final":
            no_regu_from_F_pct,

        "Regularized final":
            regu_from_F_pct,

        "Reduction_percent":
            from_F_reduction_pct,
    },


    "Diagnostic D(F,R)": {

        "Unregularized final":
            np.nan,

        "Regularized final":
            np.nan,

        "Reduction_percent":
            R_from_F_pct,
    },


    "Anchoring concentration from F": {

        "Top10_positive_origin_percent":
            top10_F,

        "Top20_positive_origin_percent":
            top20_F,

        "Positive_origin_share":
            share_positive_F,
    },


    "Anchoring concentration from R": {

        "Top10_positive_origin_percent":
            top10_R,

        "Top20_positive_origin_percent":
            top20_R,

        "Positive_origin_share":
            share_positive_R,
    }
}


final_compare_df = pd.DataFrame(
    final_compare
).T


anchoring_compare_df = pd.DataFrame(
    anchoring_compare
).T


print(
    "\n=== Figure 4d final performance comparison ==="
)


print(

    final_compare_df.to_string(

        float_format=lambda x:
        f"{x:.6f}"
    )
)


print(
    "\n=== Figure 4d behavioural anchoring and concentration comparison ==="
)


print(

    anchoring_compare_df.to_string(

        float_format=lambda x:
        f"{x:.6f}"
    )
)


latex_sentence = (

    f"The $L_1$-regularized solution produces a measurable behavioural pullback: "

    f"departure from the observed baseline $F$ decreases from "

    f"{no_regu_from_F_pct:.2f}\\% to {regu_from_F_pct:.2f}\\% of total visits "

    f"({from_F_reduction_pct:.2f}\\% lower), while departure from the behavioural "

    f"reference $R$ decreases from {no_regu_ref_dev_pct:.2f}\\% to "

    f"{regu_ref_dev_pct:.2f}\\% ({ref_dev_reduction_pct:.2f}\\% lower). "

    f"At the origin level, only {share_positive_F:.2f}\\% of CBGs make a positive "

    f"contribution to the pullback toward $F$, and the top 10\\% of these "

    f"positive-contribution origins account for {top10_F:.2f}\\% of the positive "

    f"reduction in departure from $F$. "

    f"At the same time, the regularized solution retains {distance_retention_pct:.2f}\\% "

    f"of the travel-distance reduction and {social_retention_pct:.2f}\\% of the "

    f"structural social-exposure gain achieved by the unregularized continuous optimization."
)


print(
    "\n=== LaTeX-ready sentence for Results ==="
)


print(
    latex_sentence
)