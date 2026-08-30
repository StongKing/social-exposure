# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import Normalize
from shapely.geometry import Point
import cartopy.crs as ccrs
import cartopy.feature as cfeature

AGGREGATED_CSV = "us_15_cities_aggregated.csv"
STATE_SHP_PATH = "geo_data/tl_2021_us_state/tl_2021_us_state.shp"
OUTPUT_PDF = "figure1a.pdf"

MAX_MARKER_AREA = 400
MIN_MARKER_AREA = 40

CMAP_NAME = "Purples"
CMAP_START = 0.5
CMAP_END = 1.0

FIGSIZE = (10, 8)
FIG_DPI = 300

MAP_EXTENT = [-125, -70, 25, 50]

EXCLUDE_STATE_CODES = [
    "AK",
    "HI",
    "PR",
    "VI",
    "GU",
    "MP",
    "AS",
]


def build_truncated_colormap(name, start=0.5, end=1.0, n=256):
    base_cmap = plt.get_cmap(name)
    colors = base_cmap(np.linspace(start, end, n))

    return mpl.colors.LinearSegmentedColormap.from_list(
        f"truncated_{name}",
        colors,
    )


def calculate_marker_sizes(values, min_area, max_area):
    values = np.asarray(values, dtype=float)

    values = np.nan_to_num(
        values,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    values = np.clip(
        values,
        a_min=0.0,
        a_max=None,
    )

    sqrt_values = np.sqrt(values)

    if sqrt_values.size == 0:
        return np.array([], dtype=float)

    if np.nanmax(sqrt_values) == 0:
        return np.full(
            sqrt_values.shape,
            min_area,
            dtype=float,
        )

    return (
        sqrt_values / np.nanmax(sqrt_values)
    ) * (max_area - min_area) + min_area


def build_normalizer(values):
    values = np.asarray(values, dtype=float)
    finite_values = values[np.isfinite(values)]

    if finite_values.size == 0:
        raise ValueError(
            "No valid values were found in the total_se variable."
        )

    vmin = float(np.min(finite_values))
    vmax = float(np.max(finite_values))

    if np.isclose(vmin, vmax):
        delta = (
            1.0
            if np.isclose(vmin, 0.0)
            else abs(vmin) * 0.01
        )

        vmin -= delta
        vmax += delta

    return Normalize(
        vmin=vmin,
        vmax=vmax,
    )


if not os.path.isfile(AGGREGATED_CSV):
    raise FileNotFoundError(
        f"Aggregated city-level file not found: {AGGREGATED_CSV}\n"
        "Run the aggregation script before running this plotting script."
    )

df_agg = pd.read_csv(
    AGGREGATED_CSV,
    encoding="utf-8-sig",
)

required_columns = {
    "city",
    "city_name",
    "lon",
    "lat",
    "total_distance",
    "total_se",
}

missing_columns = required_columns.difference(
    df_agg.columns
)

if missing_columns:
    raise KeyError(
        "The aggregated CSV file is missing required columns: "
        f"{sorted(missing_columns)}"
    )

numeric_columns = [
    "lon",
    "lat",
    "total_distance",
    "total_se",
]

for column in numeric_columns:
    df_agg[column] = pd.to_numeric(
        df_agg[column],
        errors="coerce",
    )

if "data_available" in df_agg.columns:
    data_available = (
        df_agg["data_available"]
        .astype(str)
        .str.strip()
        .str.lower()
        .isin(["true", "1", "yes"])
    )

    df_agg = df_agg[
        data_available
    ].copy()

df_agg = df_agg.dropna(
    subset=[
        "lon",
        "lat",
        "total_distance",
        "total_se",
    ]
).copy()

if df_agg.empty:
    raise RuntimeError(
        "The aggregated file contains no valid city records for plotting."
    )

if not os.path.isfile(STATE_SHP_PATH):
    raise FileNotFoundError(
        f"State boundary shapefile not found: {STATE_SHP_PATH}"
    )

usa = gpd.read_file(
    STATE_SHP_PATH
).to_crs(
    "EPSG:4326"
)

if "STUSPS" in usa.columns:
    usa_contig = usa[
        ~usa["STUSPS"].isin(
            EXCLUDE_STATE_CODES
        )
    ].copy()
else:
    usa_contig = usa.copy()

if hasattr(
    usa_contig.geometry,
    "union_all",
):
    contiguous_geometry = (
        usa_contig.geometry.union_all()
    )
else:
    contiguous_geometry = (
        usa_contig.geometry.unary_union
    )

df_agg["in_contiguous_us"] = df_agg.apply(
    lambda row: contiguous_geometry.covers(
        Point(
            row["lon"],
            row["lat"],
        )
    ),
    axis=1,
)

df_plot = df_agg[
    df_agg["in_contiguous_us"]
].copy()

if df_plot.empty:
    raise RuntimeError(
        "No city points are located within the contiguous United States."
    )

color_values = df_plot[
    "total_se"
].to_numpy(
    dtype=float
)

size_values = df_plot[
    "total_distance"
].to_numpy(
    dtype=float
)

norm = build_normalizer(
    color_values
)

cmap = build_truncated_colormap(
    CMAP_NAME,
    start=CMAP_START,
    end=CMAP_END,
)

marker_sizes = calculate_marker_sizes(
    size_values,
    min_area=MIN_MARKER_AREA,
    max_area=MAX_MARKER_AREA,
)

projection = ccrs.AlbersEqualArea(
    central_longitude=-96,
    central_latitude=37.5,
    standard_parallels=(
        29.5,
        45.5,
    ),
)

data_crs = ccrs.PlateCarree()

fig = plt.figure(
    figsize=FIGSIZE,
    dpi=FIG_DPI,
)

ax = fig.add_subplot(
    1,
    1,
    1,
    projection=projection,
)

ax.add_feature(
    cfeature.OCEAN.with_scale("50m"),
    facecolor="#b3d9f5",
    zorder=0,
)

ax.add_feature(
    cfeature.LAND.with_scale("50m"),
    facecolor="whitesmoke",
    edgecolor="#cccccc",
    zorder=1,
)

ax.add_feature(
    cfeature.COASTLINE.with_scale("50m"),
    linewidth=0.5,
    zorder=2,
)

for geometry in usa_contig.geometry:
    ax.add_geometries(
        [geometry],
        crs=data_crs,
        facecolor="none",
        edgecolor="#999999",
        linewidth=0.5,
        zorder=3,
    )

ax.set_extent(
    MAP_EXTENT,
    crs=data_crs,
)

ax.scatter(
    df_plot["lon"].to_numpy(),
    df_plot["lat"].to_numpy(),
    s=marker_sizes,
    c=color_values,
    cmap=cmap,
    norm=norm,
    transform=data_crs,
    zorder=6,
    edgecolors="black",
    linewidths=1.0,
    alpha=0.9,
)

# ------------------------------------------------------------
# Circle-size legend: Total Distance
# ------------------------------------------------------------

# 选取分位数作为三个代表值
legend_values = np.percentile(size_values, [1, 50, 90])

# 按照主图完全相同的方式换算圆圈面积
legend_sizes = (
    np.sqrt(legend_values) / np.sqrt(np.max(size_values))
) * (MAX_MARKER_AREA - MIN_MARKER_AREA) + MIN_MARKER_AREA+2

# 图例中心位置：ax 坐标，左下角为 (0, 0)，右上角为 (1, 1)
legend_x = 0.05
legend_y = 0.08

# 三个同心圆，从大到小画
for s in legend_sizes[::-1]:
    ax.scatter(
        legend_x,
        legend_y,
        s=s,
        facecolors="none",
        edgecolors="black",
        linewidths=0.8,
        transform=ax.transAxes,
        zorder=20,
    )

# 图例标题
ax.text(
    legend_x+0.040,
    legend_y + 0.055,
    "Total Distance",
    transform=ax.transAxes,
    fontsize=10.5,
    ha="center",
    va="bottom",
    zorder=20,
)

# 三个圆对应的数值
ax.text(
    legend_x + 0.025,
    legend_y + 0.025,
    f"{legend_values[2]:,.0f}",
    transform=ax.transAxes,
    fontsize=7,
    va="center",
)

ax.text(
    legend_x + 0.025,
    legend_y,
    f"{legend_values[1]:,.0f}",
    transform=ax.transAxes,
    fontsize=7,
    va="center",
)

ax.text(
    legend_x + 0.025,
    legend_y - 0.025,
    f"{legend_values[0]:,.0f}",
    transform=ax.transAxes,
    fontsize=7,
    va="center",
)
    

for _, row in df_plot.iterrows():
    ax.text(
        row["lon"] + 1.0,
        row["lat"],
        row["city_name"],
        transform=data_crs,
        fontsize=7,
        ha="left",
        va="center",
        bbox={
            "boxstyle": "round,pad=0.25",
            "facecolor": "white",
            "edgecolor": "#a6d0e8",
        },
        zorder=7,
    )

ax.set_title(
    "US 15 Cities",
    fontsize=11,
)

fig.canvas.draw()

ax_position = ax.get_position()

colorbar_ax = fig.add_axes(
    [
        ax_position.x1 + 0.01,
        ax_position.y0,
        0.02,
        ax_position.height,
    ]
)

scalar_mappable = mpl.cm.ScalarMappable(
    norm=norm,
    cmap=cmap,
)

scalar_mappable.set_array([])

colorbar = fig.colorbar(
    scalar_mappable,
    cax=colorbar_ax,
    orientation="vertical",
)

colorbar.set_label(
    "Total Social Exposure"
)

plt.savefig(
    OUTPUT_PDF,
    format="pdf",
    dpi=FIG_DPI,
    bbox_inches="tight",
    transparent=False,
    backend="pdf",
)

print(
    f"Figure saved to: {os.path.abspath(OUTPUT_PDF)}"
)

plt.show()