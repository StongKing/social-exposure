# -*- coding: utf-8 -*-
"""
Fig. 1c plotting-only script

Purpose
-------
Read the aggregated CSV produced by fig1c_aggregate_data.py and generate
the figure without accessing any original CBG, POI, OD, income, distance,
or flow-matrix data.

Required input in the current working directory
------------------------------------------------
fig1c_aggregated_data.csv

Output in the current working directory
---------------------------------------
figure1c.pdf
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# ============================================================
# 0. Adjustable parameters
# ============================================================

INPUT_CSV = os.path.join(
    os.getcwd(),
    "fig1c_aggregated_data.csv",
)
OUTPUT_PDF = os.path.join(
    os.getcwd(),
    "figure1c.pdf",
)

FIG_DPI = 300

AUTO_XMAX = True
DEFAULT_XMAX = 45
RIGHT_MARGIN = 2.5

# Legend positions.
LEGEND1_LOC = "lower left"
LEGEND1_BBOX_TO_ANCHOR = (0.6, 0.02)

LEGEND2_LOC = "lower left"
LEGEND2_BBOX_TO_ANCHOR = (0.6, 0.18)

# City aggregate point-size range.
AGG_SIZE_MIN = 35
AGG_SIZE_MAX = 100


# ============================================================
# 1. Labels and colors
# ============================================================

POI_CODE_TO_LABEL = {
    "624190": "Individual & Family Services",
    "711310": "Performing Arts Facilities",
    "712110": "Museums",
    "713940": "Fitness Centers",
    "722410": "Drinking Places",
    "813110": "Religious Organizations",
}

BASE_COLORS = {
    "713940": "#984EA3",
    "813110": "#377EB8",
    "722410": "#FF7F00",
    "712110": "#4DAF4A",
    "711310": "#A65628",
    "624190": "#E41A1C",
}

POI_ORDER = [
    "624190",
    "711310",
    "712110",
    "713940",
    "722410",
    "813110",
]


# ============================================================
# 2. Read and validate aggregated data
# ============================================================

def load_aggregated_data(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(
            "Aggregated CSV not found:\n"
            f"{os.path.abspath(path)}\n\n"
            "Run fig1c_aggregate_data.py first and place the resulting "
            "fig1c_aggregated_data.csv in the current directory."
        )

    data = pd.read_csv(
        path,
        dtype={
            "city": str,
            "city_label": str,
            "poi_category": str,
            "poi_code": str,
        },
    )

    required_columns = {
        "city",
        "city_label",
        "poi_category",
        "poi_code",
        "share_second_quadrant",
        "total_active_ref_flow",
        "city_aggregate_share",
        "city_total_active_ref_flow",
        "city_n_unused_feasible",
        "city_aggregate_weight",
    }

    missing = sorted(required_columns - set(data.columns))
    if missing:
        raise ValueError(
            "Aggregated CSV is missing required columns: "
            + ", ".join(missing)
        )

    if data.empty:
        raise ValueError("Aggregated CSV contains no rows.")

    unexpected_weights = set(
        data["city_aggregate_weight"].dropna().astype(str)
    ) - {"n_unused_feasible"}

    if unexpected_weights:
        raise ValueError(
            "Unexpected city aggregation rule found in the CSV: "
            + ", ".join(sorted(unexpected_weights))
        )

    numeric_columns = [
        "share_second_quadrant",
        "total_active_ref_flow",
        "city_aggregate_share",
        "city_total_active_ref_flow",
        "city_n_unused_feasible",
    ]

    for column in numeric_columns:
        data[column] = pd.to_numeric(
            data[column],
            errors="coerce",
        )

    if data["share_second_quadrant"].isna().all():
        raise ValueError(
            "Column share_second_quadrant contains no valid values."
        )

    if data["city_aggregate_share"].isna().all():
        raise ValueError(
            "Column city_aggregate_share contains no valid values."
        )

    return data


def extract_city_summary(data):
    """
    The city-level values were already computed by the aggregation script.
    They are repeated across category rows, so retain one row per city.
    """
    city_columns = [
        "city",
        "city_label",
        "city_aggregate_share",
        "city_total_active_ref_flow",
        "city_n_unused_feasible",
    ]

    city_summary = (
        data[city_columns]
        .drop_duplicates(subset=["city", "city_label"])
        .reset_index(drop=True)
    )

    # Verify that repeated city values are internally consistent.
    consistency = (
        data.groupby(["city", "city_label"], observed=True)
        .agg(
            aggregate_min=("city_aggregate_share", "min"),
            aggregate_max=("city_aggregate_share", "max"),
            flow_min=("city_total_active_ref_flow", "min"),
            flow_max=("city_total_active_ref_flow", "max"),
        )
        .reset_index()
    )

    inconsistent = consistency[
        ~np.isclose(
            consistency["aggregate_min"],
            consistency["aggregate_max"],
            equal_nan=True,
        )
        | ~np.isclose(
            consistency["flow_min"],
            consistency["flow_max"],
            equal_nan=True,
        )
    ]

    if not inconsistent.empty:
        raise ValueError(
            "The aggregated CSV contains inconsistent city-level "
            "values across category rows."
        )

    return city_summary


# ============================================================
# 3. Plotting helpers
# ============================================================

def make_percent_ticks(data_xmax):
    if data_xmax <= 45:
        ticks = [0, 2, 5, 10, 15, 20, 25, 30, 35, 40, 45]
        return [tick for tick in ticks if tick <= data_xmax]

    if data_xmax <= 60:
        ticks = [0, 5, 10, 20, 30, 40, 50, 60]
        return [tick for tick in ticks if tick <= data_xmax]

    return list(np.arange(0, data_xmax + 1e-9, 10))


def scale_aggregate_point_sizes(city_summary):
    flow = (
        city_summary["city_total_active_ref_flow"]
        .replace(0, np.nan)
    )
    size_raw = np.log10(flow)

    finite = size_raw[np.isfinite(size_raw)]

    if finite.empty:
        return pd.Series(
            np.full(
                len(city_summary),
                (AGG_SIZE_MIN + AGG_SIZE_MAX) / 2,
            ),
            index=city_summary.index,
        )

    size_min = finite.min()
    size_max = finite.max()

    if np.isclose(size_min, size_max):
        return pd.Series(
            np.full(
                len(city_summary),
                (AGG_SIZE_MIN + AGG_SIZE_MAX) / 2,
            ),
            index=city_summary.index,
        )

    scaled = (
        (size_raw - size_min)
        / (size_max - size_min)
        * (AGG_SIZE_MAX - AGG_SIZE_MIN)
        + AGG_SIZE_MIN
    )

    return scaled.fillna(AGG_SIZE_MIN)


# ============================================================
# 4. Figure
# ============================================================

def plot_fig1c_second_quadrant_dotrange(
    category_data,
    city_summary,
):
    category_data = category_data.copy()
    city_summary = city_summary.copy()

    category_data["q2_pct"] = (
        category_data["share_second_quadrant"] * 100
    )
    city_summary["q2_pct"] = (
        city_summary["city_aggregate_share"] * 100
    )

    category_data["poi_color"] = (
        category_data["poi_code"].map(BASE_COLORS)
    )

    max_value = np.nanmax([
        category_data["q2_pct"].max(),
        city_summary["q2_pct"].max(),
        DEFAULT_XMAX,
    ])

    if AUTO_XMAX:
        data_xmax = int(np.ceil(max_value / 5.0) * 5)
        data_xmax = max(data_xmax, DEFAULT_XMAX)
    else:
        data_xmax = DEFAULT_XMAX

    # Sort cities from low to high aggregate value.
    city_summary = (
        city_summary
        .sort_values("q2_pct", ascending=True)
        .reset_index(drop=True)
    )

    city_order = city_summary["city_label"].tolist()
    y_position = {
        city_label: index
        for index, city_label in enumerate(city_order)
    }

    category_data = category_data[
        category_data["city_label"].isin(city_order)
    ].copy()

    category_data["y"] = (
        category_data["city_label"].map(y_position)
    )
    city_summary["y"] = (
        city_summary["city_label"].map(y_position)
    )

    city_range = (
        category_data
        .groupby("city_label", observed=True)
        .agg(
            x_min=("q2_pct", "min"),
            x_max=("q2_pct", "max"),
            n_cat=("q2_pct", "count"),
        )
        .reset_index()
    )
    city_range["y"] = city_range["city_label"].map(y_position)

    city_summary["point_size"] = scale_aggregate_point_sizes(
        city_summary
    )

    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = [
        "Times New Roman",
        "Times",
        "DejaVu Serif",
    ]
    plt.rcParams["font.size"] = 9
    plt.rcParams["axes.linewidth"] = 0.8
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42

    fig, ax = plt.subplots(
        figsize=(8, 6.0),
        dpi=FIG_DPI,
    )

    # Category-level min–max range.
    for _, row in city_range.iterrows():
        ax.plot(
            [row["x_min"], row["x_max"]],
            [row["y"], row["y"]],
            color="#BDBDBD",
            linewidth=1.3,
            zorder=1,
        )

    # POI-category points.
    for code in POI_ORDER:
        subset = category_data[
            category_data["poi_code"] == code
        ]

        if subset.empty:
            continue

        ax.scatter(
            subset["q2_pct"],
            subset["y"],
            s=28,
            color=BASE_COLORS[code],
            alpha=0.86,
            edgecolor="white",
            linewidth=0.45,
            zorder=2,
        )

    # City aggregate points.
    ax.scatter(
        city_summary["q2_pct"],
        city_summary["y"],
        s=city_summary["point_size"],
        color="#4B2E83",
        alpha=0.96,
        edgecolor="white",
        linewidth=0.8,
        zorder=3,
    )

    # Right-side city aggregate annotation.
    text_x = data_xmax - 1.2

    for _, row in city_summary.iterrows():
        ax.text(
            text_x + 1,
            row["y"],
            f"{row['q2_pct']:.1f}%",
            va="center",
            ha="right",
            fontsize=9.5,
            color="#4D4D4D",
            zorder=4,
        )

    ax.set_yticks(np.arange(len(city_order)))
    ax.set_yticklabels(city_order)
    ax.set_ylim(-0.5, len(city_order) - 0.3)

    ax.set_xlim(0, data_xmax + RIGHT_MARGIN)

    xticks = make_percent_ticks(data_xmax)
    ax.set_xticks(xticks)
    ax.set_xticklabels([f"{tick:g}%" for tick in xticks])

    ax.set_xlabel(
        "Unused feasible links with higher exposure "
        "and no longer distance",
        fontsize=10.5,
    )

    ax.set_title(
        "Latent exposure opportunity varies across "
        "city cores and POI categories",
        fontsize=10.5,
        pad=8,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.grid(
        axis="x",
        linestyle="-",
        linewidth=0.4,
        alpha=0.22,
    )
    ax.set_axisbelow(True)

    aggregate_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="#4B2E83",
            markeredgecolor="white",
            markersize=7,
            label="City aggregate\nunused-link weighted",
        ),
        Line2D(
            [0],
            [0],
            color="#BDBDBD",
            linewidth=1.3,
            label="Category range",
        ),
    ]

    legend1 = ax.legend(
        handles=aggregate_handles,
        loc=LEGEND1_LOC,
        bbox_to_anchor=LEGEND1_BBOX_TO_ANCHOR,
        frameon=False,
        fontsize=9.5,
        handlelength=1.8,
    )
    ax.add_artist(legend1)

    category_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=BASE_COLORS[code],
            markeredgecolor="white",
            markersize=6,
            label=POI_CODE_TO_LABEL[code],
        )
        for code in POI_ORDER
    ]

    legend2 = ax.legend(
        handles=category_handles,
        loc=LEGEND2_LOC,
        bbox_to_anchor=LEGEND2_BBOX_TO_ANCHOR,
        frameon=False,
        fontsize=9.5,
        handlelength=1.2,
    )
    ax.add_artist(legend2)

    fig.tight_layout(rect=[0.02, 0.04, 0.98, 1])

    fig.savefig(
        OUTPUT_PDF,
        format="pdf",
        dpi=FIG_DPI,
        bbox_inches="tight",
        transparent=False,
    )

    return fig, ax


# ============================================================
# 5. Main workflow
# ============================================================

def main():
    aggregated_data = load_aggregated_data(INPUT_CSV)
    city_summary = extract_city_summary(aggregated_data)

    print("========== Aggregated input loaded ==========")
    print(f"Input: {os.path.abspath(INPUT_CSV)}")
    print(
        f"Cities: {city_summary['city'].nunique()} | "
        f"city-category rows: {len(aggregated_data)}"
    )

    plot_fig1c_second_quadrant_dotrange(
        category_data=aggregated_data,
        city_summary=city_summary,
    )

    print(f"Saved figure: {os.path.abspath(OUTPUT_PDF)}")
    plt.show()


if __name__ == "__main__":
    main()
