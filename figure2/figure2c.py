# -*- coding: utf-8 -*-
"""
Fig. 2c — plotting script.

This script does NOT require the private flow_matrix.csv.

Required derived plotting files
-------------------------------
figure2c_new_links.csv
figure2c_quadrant.csv
figure2c_origin_metrics.csv
figure2c_poi_metrics.csv

Required spatial files
----------------------
tl_2021_25_bg/
tl_2021_33_bg/
tl_2021_us_cbsa/
poi_boston_msa_all.csv
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.patches import FancyArrowPatch
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D


try:
    import geopandas as gpd

except Exception as e:

    raise ImportError(
        "This script requires geopandas. "
        "Please install geopandas first."
    ) from e


try:
    from shapely import affinity

except Exception as e:

    raise ImportError(
        "This script requires shapely. "
        "Please install shapely first."
    ) from e


# ============================================================
# 0. USER SETTINGS
# ============================================================


CITY_LABEL = "Boston"

# ------------------------------------------------------------
# Derived plotting data
# ------------------------------------------------------------

NEW_LINKS_CSV = "figure2c_new_links.csv"
QUADRANT_CSV = "figure2c_quadrant.csv"
ORIGIN_METRICS_CSV = "figure2c_origin_metrics.csv"
POI_METRICS_CSV = "figure2c_poi_metrics.csv"



# ------------------------------------------------------------
# Public spatial data
# ------------------------------------------------------------

CBG_SHP_PATHS = [
    os.path.join(
        "geo_data",
        "tl_2021_25_bg",
        "tl_2021_25_bg.shp"
    ),

    os.path.join(
        "geo_data",
        "tl_2021_33_bg",
        "tl_2021_33_bg.shp"
    ),
]


CBG_CENTROID_CSV = None


CBSA_SHP_PATH = os.path.join(
    "geo_data",
    "tl_2021_us_cbsa",
    "tl_2021_us_cbsa.shp"
)


BOSTON_CBSAFP = "14460"
POI_METADATA_CSV = "poi_boston_msa_all.csv"


# ------------------------------------------------------------
# Plot settings
# ------------------------------------------------------------

MAX_NEW_LINKS_TO_DRAW = 1000

EDGE_RAD_POS = 0.36
EDGE_RAD_NEG = -0.32

SHOW_FIGURES = True
SAVE_FIGURES = False


MAP_TARGET_HEIGHT_TO_WIDTH = 1.20

MAP_Y_SCALE_MIN = 0.42
MAP_Y_SCALE_MAX = 1.00

MAP_PAD_FRAC = 0.015


RIGHT_CROP_FRAC = 0.12


FIG_HEIGHT_IN = 8.0


DONUT_HATCH = "////"

DONUT_WIDTH = 0.38

DONUT_HATCH_LINEWIDTH = 1.05


# ============================================================
# 1. Basic helpers
# ============================================================

def normalize_geoid(x):

    if pd.isna(x):
        return None

    try:
        return str(int(float(x)))

    except Exception:
        return str(x).strip()


def set_nature_style():

    plt.rcParams["font.family"] = "serif"

    plt.rcParams["font.serif"] = [
        "Times New Roman",
        "Times",
        "DejaVu Serif"
    ]

    plt.rcParams["font.size"] = 8.5

    plt.rcParams["axes.linewidth"] = 0.75

    plt.rcParams["axes.labelsize"] = 8.8

    plt.rcParams["axes.titlesize"] = 9.6

    plt.rcParams["xtick.labelsize"] = 8

    plt.rcParams["ytick.labelsize"] = 8

    plt.rcParams["legend.fontsize"] = 7.3

    plt.rcParams["pdf.fonttype"] = 42

    plt.rcParams["ps.fonttype"] = 42

    plt.rcParams[
        "hatch.linewidth"
    ] = DONUT_HATCH_LINEWIDTH


def detect_lat_lon_columns(df):

    lat_candidates = [
        "latitude",
        "lat",
        "poi_lat",
        "centroid_lat",
        "y"
    ]

    lon_candidates = [
        "longitude",
        "lon",
        "lng",
        "poi_lon",
        "centroid_lon",
        "x"
    ]


    cols_lower = {
        c.lower(): c
        for c in df.columns
    }


    lat_col = next(
        (
            cols_lower[c]
            for c in lat_candidates
            if c in cols_lower
        ),
        None
    )


    lon_col = next(
        (
            cols_lower[c]
            for c in lon_candidates
            if c in cols_lower
        ),
        None
    )


    return (
        lat_col,
        lon_col
    )


def detect_id_columns(df):

    candidates = [
        "poi_id",
        "safegraph_place_id",
        "placekey",
        "location_name",
        "poi",
        "id"
    ]


    cols_lower = {
        c.lower(): c
        for c in df.columns
    }


    return next(
        (
            cols_lower[c]
            for c in candidates
            if c in cols_lower
        ),
        None
    )


def detect_name_columns(df):

    candidates = [
        "location_name",
        "name",
        "poi_name",
        "brands",
        "brand",
        "top_category"
    ]


    cols_lower = {
        c.lower(): c
        for c in df.columns
    }


    return next(
        (
            cols_lower[c]
            for c in candidates
            if c in cols_lower
        ),
        None
    )


# ============================================================
# 2. Load derived Fig. 2c plotting data
# ============================================================

def load_case_output():

    # --------------------------------------------------------
    # New links
    # --------------------------------------------------------

    new_links_df = pd.read_csv(
        NEW_LINKS_CSV,
        dtype={
            "GEOID": str,
            "poi_id": str,
        }
    )


    new_links_df["GEOID"] = (
        new_links_df[
            "GEOID"
        ]
        .map(normalize_geoid)
    )


    new_links_df["poi_id"] = (
        new_links_df[
            "poi_id"
        ]
        .astype(str)
        .str.strip()
    )


    new_links_df["new_flow"] = (
        pd.to_numeric(
            new_links_df["new_flow"],
            errors="coerce"
        )
        .fillna(0)
    )


    if (
        "is_fig1d_opportunity"
        in
        new_links_df.columns
    ):

        new_links_df[
            "is_fig1d_opportunity"
        ] = (
            new_links_df[
                "is_fig1d_opportunity"
            ]
            .astype(str)
            .str.lower()
            .map(
                {
                    "true": True,
                    "false": False,
                    "1": True,
                    "0": False,
                }
            )
            .fillna(False)
        )


    # --------------------------------------------------------
    # Quadrant composition
    # --------------------------------------------------------

    quadrant_df = pd.read_csv(
        QUADRANT_CSV
    )


    quadrant_df[
        "share_pct"
    ] = pd.to_numeric(
        quadrant_df[
            "share_pct"
        ],
        errors="coerce"
    )


    # --------------------------------------------------------
    # Origin metrics
    # --------------------------------------------------------

    origin_metrics = pd.read_csv(
        ORIGIN_METRICS_CSV,
        dtype={
            "GEOID": str
        }
    )


    origin_metrics["GEOID"] = (
        origin_metrics[
            "GEOID"
        ]
        .map(normalize_geoid)
    )


    origin_metrics[
        "shifted_flow"
    ] = pd.to_numeric(
        origin_metrics[
            "shifted_flow"
        ],
        errors="coerce"
    ).fillna(0)


    # --------------------------------------------------------
    # POI metrics
    # --------------------------------------------------------

    poi_metrics = pd.read_csv(
        POI_METRICS_CSV,
        dtype={
            "poi_id": str
        }
    )


    poi_metrics[
        "poi_id"
    ] = (
        poi_metrics[
            "poi_id"
        ]
        .astype(str)
        .str.strip()
    )


    poi_metrics[
        "total_changed_flow"
    ] = pd.to_numeric(
        poi_metrics[
            "total_changed_flow"
        ],
        errors="coerce"
    ).fillna(0)


    cbg_ids = (
        origin_metrics[
            "GEOID"
        ]
        .tolist()
    )


    poi_ids = (
        poi_metrics[
            "poi_id"
        ]
        .tolist()
    )


    return {
        # Kept for compatibility with original plotting function.
        "summary": {},

        "quadrant_df":
            quadrant_df,

        "new_links_df":
            new_links_df,

        "poi_metrics":
            poi_metrics,

        "origin_metrics":
            origin_metrics,

        "cbg_ids":
            cbg_ids,

        "poi_ids":
            poi_ids,
    }


# ============================================================
# 3. Load spatial reference data
# ============================================================

def load_poi_coordinates(
        poi_ids_needed):

    if not os.path.isfile(
        POI_METADATA_CSV
    ):

        raise FileNotFoundError(
            f"Cannot find POI metadata: "
            f"{POI_METADATA_CSV}"
        )


    print(
        f"[LOAD POI metadata] "
        f"{POI_METADATA_CSV}"
    )


    df = pd.read_csv(
        POI_METADATA_CSV,
        low_memory=False
    )


    id_col = (
        detect_id_columns(df)
    )


    lat_col, lon_col = (
        detect_lat_lon_columns(df)
    )


    name_col = (
        detect_name_columns(df)
    )


    if (
        id_col is None
        or
        lat_col is None
        or
        lon_col is None
    ):

        raise ValueError(
            "POI metadata CSV must contain "
            "POI id and latitude/longitude columns."
        )


    columns = [
        id_col,
        lat_col,
        lon_col
    ]


    if name_col is not None:
        columns.append(
            name_col
        )


    out = df[
        columns
    ].copy()


    if name_col is not None:

        out.columns = [
            "poi_id",
            "lat",
            "lon",
            "poi_name"
        ]

    else:

        out.columns = [
            "poi_id",
            "lat",
            "lon"
        ]


    out[
        "poi_id"
    ] = (
        out[
            "poi_id"
        ]
        .astype(str)
        .str.strip()
    )


    out[
        "lat"
    ] = pd.to_numeric(
        out["lat"],
        errors="coerce"
    )


    out[
        "lon"
    ] = pd.to_numeric(
        out["lon"],
        errors="coerce"
    )


    out = (
        out
        .dropna(
            subset=[
                "lat",
                "lon"
            ]
        )
        .drop_duplicates(
            subset=[
                "poi_id"
            ]
        )
    )


    needed = [
        str(x).strip()
        for x in poi_ids_needed
    ]


    out = out[
        out[
            "poi_id"
        ].isin(needed)
    ].copy()


    if "poi_name" not in out.columns:

        out[
            "poi_name"
        ] = out[
            "poi_id"
        ]

    else:

        out[
            "poi_name"
        ] = (
            out[
                "poi_name"
            ]
            .fillna(
                out[
                    "poi_id"
                ]
            )
            .astype(str)
        )


    return gpd.GeoDataFrame(
        out,

        geometry=gpd.points_from_xy(
            out["lon"],
            out["lat"]
        ),

        crs="EPSG:4326",
    )


def load_cbg_geometries(
        cbg_ids_needed=None):

    if cbg_ids_needed is None:

        cbg_ids_needed = None

    else:

        cbg_ids_needed = set(
            [
                normalize_geoid(x)
                for x in cbg_ids_needed
            ]
        )


    if (
        CBG_CENTROID_CSV
        is not None
        and
        os.path.isfile(
            CBG_CENTROID_CSV
        )
    ):

        df = pd.read_csv(
            CBG_CENTROID_CSV
        )


        lat_col, lon_col = (
            detect_lat_lon_columns(
                df
            )
        )


        if (
            "GEOID"
            not in
            df.columns
            or
            lat_col is None
            or
            lon_col is None
        ):

            raise ValueError(
                "CBG centroid CSV must contain "
                "GEOID and latitude/longitude columns"
            )


        df["GEOID"] = (
            df["GEOID"]
            .apply(normalize_geoid)
        )


        if cbg_ids_needed is not None:

            df = df[
                df[
                    "GEOID"
                ].isin(
                    cbg_ids_needed
                )
            ].copy()


        return gpd.GeoDataFrame(
            df[
                ["GEOID"]
            ].copy(),

            geometry=gpd.points_from_xy(
                df[lon_col],
                df[lat_col]
            ),

            crs="EPSG:4326",
        )


    shp_existing = [
        p
        for p in CBG_SHP_PATHS
        if os.path.isfile(p)
    ]


    if len(shp_existing) == 0:

        raise FileNotFoundError(
            "No CBG shapefile found. "
            "Please set CBG_SHP_PATHS "
            "or CBG_CENTROID_CSV."
        )


    gdfs = []


    for shp in shp_existing:

        print(
            f"[LOAD CBG shp] {shp}"
        )


        g = gpd.read_file(
            shp
        )


        geoid_col = next(
            (
                cand
                for cand in [
                    "GEOID",
                    "geoid",
                    "GEOID20"
                ]
                if cand in g.columns
            ),
            None
        )


        if geoid_col is None:

            raise ValueError(
                f"Cannot find GEOID "
                f"column in {shp}"
            )


        g["GEOID"] = (
            g[
                geoid_col
            ]
            .apply(
                normalize_geoid
            )
        )


        if cbg_ids_needed is not None:

            g = g[
                g[
                    "GEOID"
                ].isin(
                    cbg_ids_needed
                )
            ].copy()


        if len(g) > 0:

            gdfs.append(
                g[
                    [
                        "GEOID",
                        "geometry"
                    ]
                ].copy()
            )


    if len(gdfs) == 0:

        if cbg_ids_needed is None:

            raise ValueError(
                "No CBG geometries were found "
                "in CBG shapefiles."
            )


        raise ValueError(
            "None of the requested CBG ids "
            "were found in CBG shapefiles."
        )


    cbg = pd.concat(
        gdfs,
        axis=0,
        ignore_index=True
    )


    cbg = gpd.GeoDataFrame(
        cbg,
        geometry="geometry",
        crs=gdfs[0].crs
    )


    return cbg


def load_boston_msa_boundary(
        cbg_crs):

    if not os.path.isfile(
        CBSA_SHP_PATH
    ):

        return None


    print(
        f"[LOAD CBSA shp] "
        f"{CBSA_SHP_PATH}"
    )


    cbsa = gpd.read_file(
        CBSA_SHP_PATH
    )


    col = (
        "CBSAFP"
        if "CBSAFP" in cbsa.columns
        else None
    )


    if col is None:
        return None


    msa = cbsa[
        cbsa[
            col
        ].astype(str)
        ==
        str(BOSTON_CBSAFP)
    ].copy()


    if len(msa) == 0:
        return None


    if msa.crs != cbg_crs:

        msa = msa.to_crs(
            cbg_crs
        )


    return msa


# ============================================================
# 4. Spatial plotting data
# ============================================================

def prepare_spatial_layers(
        case_output):

    # Full CBG layer for map background.
    cbg_all = (
        load_cbg_geometries(
            cbg_ids_needed=None
        )
    )


    # Boston MSA boundary.
    msa = load_boston_msa_boundary(
        cbg_all.crs
    )


    # Clip CBG polygons to Boston MSA.
    if (
        msa is not None
        and
        len(msa) > 0
    ):

        try:

            cbg_bg = gpd.clip(
                cbg_all,
                msa
            )

        except Exception:

            cbg_bg = cbg_all.copy()

            cbg_bg[
                "geometry"
            ] = (
                cbg_bg.geometry
                .intersection(
                    msa.unary_union
                )
            )

            cbg_bg = cbg_bg[
                cbg_bg.geometry.notna()
                &
                (
                    ~cbg_bg.geometry.is_empty
                )
            ].copy()

    else:

        cbg_bg = cbg_all.copy()


    if len(cbg_bg) == 0:

        raise ValueError(
            "No CBG polygons remain after "
            "clipping to Boston MSA."
        )


    # Model CBGs for origin centroids.
    model_ids = set(
        [
            normalize_geoid(x)
            for x
            in case_output[
                "cbg_ids"
            ]
        ]
    )


    cbg_model = cbg_bg[
        cbg_bg[
            "GEOID"
        ].isin(
            model_ids
        )
    ].copy()


    if len(cbg_model) == 0:

        cbg_model = (
            load_cbg_geometries(
                case_output[
                    "cbg_ids"
                ]
            )
        )


    # POIs
    poi = load_poi_coordinates(
        case_output[
            "poi_ids"
        ]
    )


    # Project everything.
    if cbg_bg.crs is None:

        cbg_bg = cbg_bg.set_crs(
            "EPSG:4326"
        )


    if cbg_model.crs is None:

        cbg_model = (
            cbg_model
            .set_crs(
                cbg_bg.crs
            )
        )


    if poi.crs is None:

        poi = poi.set_crs(
            "EPSG:4326"
        )


    cbg_bg = cbg_bg.to_crs(
        3857
    )

    cbg_model = cbg_model.to_crs(
        3857
    )

    poi = poi.to_crs(
        3857
    )


    if (
        msa is not None
        and
        len(msa) > 0
    ):

        msa = msa.to_crs(
            3857
        )


    # Model-CBG centroids.
    if (
        cbg_model
        .geom_type
        .iloc[0]
        .lower()
        ==
        "point"
    ):

        cbg_cent = cbg_model.copy()

    else:

        cbg_cent = (
            cbg_model[
                ["GEOID"]
            ]
            .copy()
        )

        cbg_cent[
            "geometry"
        ] = (
            cbg_model
            .geometry
            .centroid
        )

        cbg_cent = (
            gpd.GeoDataFrame(
                cbg_cent,
                geometry="geometry",
                crs=cbg_model.crs
            )
        )


    return (
        cbg_bg,
        cbg_cent,
        poi,
        msa
    )


def draw_curved_edge(
        ax,
        x0,
        y0,
        x1,
        y1,
        color,
        lw,
        alpha=0.5,
        rad=0.15,
        zorder=2):

    patch = FancyArrowPatch(
        (x0, y0),
        (x1, y1),

        arrowstyle='-',

        connectionstyle=(
            f"arc3,rad={rad}"
        ),

        linewidth=lw,

        color=color,

        alpha=alpha,

        zorder=zorder,

        capstyle='round',

        joinstyle='round',
    )


    ax.add_patch(
        patch
    )


# ============================================================
# 5. Display-only aspect compression
# ============================================================

def _valid_total_bounds(gdf):

    if (
        gdf is None
        or
        len(gdf) == 0
    ):

        return None


    b = np.asarray(
        gdf.total_bounds,
        dtype=float
    )


    if (
        b.shape[0] != 4
        or
        not np.all(
            np.isfinite(b)
        )
    ):

        return None


    if (
        b[2] <= b[0]
        or
        b[3] <= b[1]
    ):

        return None


    return b


def compress_layers_to_square_display(
        cbg_poly,
        cbg_cent,
        poi_gdf,
        msa_gdf=None):

    bounds = []


    for g in [
        msa_gdf,
        cbg_poly,
        cbg_cent,
        poi_gdf
    ]:

        b = _valid_total_bounds(
            g
        )

        if b is not None:
            bounds.append(b)


    if len(bounds) == 0:

        return (
            cbg_poly,
            cbg_cent,
            poi_gdf,
            msa_gdf,
            1.0
        )


    bounds = np.vstack(
        bounds
    )


    xmin = np.nanmin(
        bounds[:, 0]
    )

    ymin = np.nanmin(
        bounds[:, 1]
    )

    xmax = np.nanmax(
        bounds[:, 2]
    )

    ymax = np.nanmax(
        bounds[:, 3]
    )


    raw_xspan = (
        xmax
        -
        xmin
    )

    raw_yspan = (
        ymax
        -
        ymin
    )


    if (
        raw_xspan <= 0
        or
        raw_yspan <= 0
    ):

        y_scale = 1.0

    else:

        y_scale = (
            raw_xspan
            /
            raw_yspan
        ) * MAP_TARGET_HEIGHT_TO_WIDTH


        y_scale = float(
            np.clip(
                y_scale,
                MAP_Y_SCALE_MIN,
                MAP_Y_SCALE_MAX
            )
        )


    origin = (
        (
            xmin
            +
            xmax
        )
        /
        2.0,

        (
            ymin
            +
            ymax
        )
        /
        2.0
    )


    def _scale_gdf(gdf):

        if gdf is None:
            return None


        if len(gdf) == 0:
            return gdf.copy()


        out = gdf.copy()


        out[
            "geometry"
        ] = (
            out.geometry
            .apply(
                lambda geom:
                affinity.scale(
                    geom,

                    xfact=1.0,

                    yfact=y_scale,

                    origin=origin,
                )
                if (
                    geom is not None
                    and
                    not geom.is_empty
                )
                else geom
            )
        )


        return out


    return (
        _scale_gdf(
            cbg_poly
        ),

        _scale_gdf(
            cbg_cent
        ),

        _scale_gdf(
            poi_gdf
        ),

        _scale_gdf(
            msa_gdf
        ),

        y_scale,
    )


# ============================================================
# 6. Plot Fig. 2c
# ============================================================

def plot_fig2c_realized_corridors(
        case_output,
        cbg_poly,
        cbg_cent,
        poi_gdf,
        msa_gdf=None):

    set_nature_style()


    summary = (
        case_output[
            "summary"
        ]
    )

    quadrant_df = (
        case_output[
            "quadrant_df"
        ].copy()
    )

    new_links = (
        case_output[
            "new_links_df"
        ].copy()
    )

    origins = (
        case_output[
            "origin_metrics"
        ].copy()
    )

    pois = (
        case_output[
            "poi_metrics"
        ].copy()
    )


    # -----------------------------
    # Colors
    # -----------------------------

    c_origin = "#3498DB"

    c_poi = "#6E5AA8"


    c_ocean = "whitesmoke"

    c_cbg_fill = "whitesmoke"

    c_cbg_edge = "#D2D2D2"

    c_msa_edge = "#AFAFAF"

    c_text = "#243447"


    tradeoff_colors = {

        "Higher exposure, no farther":
            "#6A51A3",

        "Higher exposure, farther":
            "#C7C2E0",

        "Lower exposure, no farther":
            "#1B9E77",

        "Lower exposure, farther":
            "#B8E3B2",
    }


    tradeoff_alpha = {

        "Higher exposure, no farther":
            1,

        "Higher exposure, farther":
            1,

        "Lower exposure, no farther":
            1,

        "Lower exposure, farther":
            1,
    }


    compact_labels = {

        "Higher exposure, no farther":
            "Higher exp., no farther",

        "Higher exposure, farther":
            "Higher exp., farther",

        "Lower exposure, no farther":
            "Lower exp., no farther",

        "Lower exposure, farther":
            "Lower exp., farther",
    }


    # --------------------------------------------------------
    # Display geometry transform
    # --------------------------------------------------------

    (
        cbg_poly,
        cbg_cent,
        poi_gdf,
        msa_gdf,
        map_y_scale
    ) = (
        compress_layers_to_square_display(
            cbg_poly,
            cbg_cent,
            poi_gdf,
            msa_gdf
        )
    )


    print(
        "[DISPLAY] map y-axis "
        f"compression factor = "
        f"{map_y_scale:.3f}"
    )


    # --------------------------------------------------------
    # Link sampling
    # --------------------------------------------------------

    if len(new_links) > 0:

        new_links = (
            new_links
            .sort_values(
                [
                    "new_flow",
                    "is_fig1d_opportunity"
                ],
                ascending=[
                    False,
                    False
                ]
            )
            .head(
                MAX_NEW_LINKS_TO_DRAW
            )
            .copy()
        )


    # --------------------------------------------------------
    # Coordinates
    # --------------------------------------------------------

    cbg_cent_xy = (
        cbg_cent.copy()
    )


    cbg_cent_xy[
        "x"
    ] = (
        cbg_cent_xy
        .geometry
        .x
    )


    cbg_cent_xy[
        "y"
    ] = (
        cbg_cent_xy
        .geometry
        .y
    )


    poi_xy = (
        poi_gdf.copy()
    )


    poi_xy[
        "x"
    ] = (
        poi_xy
        .geometry
        .x
    )


    poi_xy[
        "y"
    ] = (
        poi_xy
        .geometry
        .y
    )


    if len(new_links) > 0:

        new_links = (
            new_links
            .merge(
                cbg_cent_xy[
                    [
                        "GEOID",
                        "x",
                        "y"
                    ]
                ],
                on="GEOID",
                how="left"
            )
        )


        new_links = (
            new_links.rename(
                columns={
                    "x": "x0",
                    "y": "y0"
                }
            )
        )


        new_links = (
            new_links
            .merge(
                poi_xy[
                    [
                        "poi_id",
                        "x",
                        "y"
                    ]
                ],
                on="poi_id",
                how="left"
            )
        )


        new_links = (
            new_links.rename(
                columns={
                    "x": "x1",
                    "y": "y1"
                }
            )
        )


    origin_plot = (
        cbg_cent_xy
        .merge(
            origins,
            on="GEOID",
            how="left"
        )
        .fillna(0)
    )


    poi_plot = (
        poi_xy
        .merge(
            pois,
            on="poi_id",
            how="left"
        )
        .fillna(0)
    )


    modified_poi = poi_plot[
        poi_plot[
            "total_changed_flow"
        ]
        >
        0
    ].copy()


    # --------------------------------------------------------
    # Plot bounds
    # --------------------------------------------------------

    bounds = []


    if (
        msa_gdf is not None
        and
        len(msa_gdf) > 0
    ):

        bounds.append(
            msa_gdf.total_bounds
        )


    if len(cbg_poly) > 0:

        bounds.append(
            cbg_poly.total_bounds
        )


    if len(origin_plot) > 0:

        bounds.append(
            np.array(
                [
                    origin_plot["x"].min(),
                    origin_plot["y"].min(),
                    origin_plot["x"].max(),
                    origin_plot["y"].max()
                ],
                dtype=float
            )
        )


    if len(poi_plot) > 0:

        bounds.append(
            np.array(
                [
                    poi_plot["x"].min(),
                    poi_plot["y"].min(),
                    poi_plot["x"].max(),
                    poi_plot["y"].max()
                ],
                dtype=float
            )
        )


    if (
        len(new_links) > 0
        and
        {
            "x0",
            "y0",
            "x1",
            "y1"
        }.issubset(
            new_links.columns
        )
    ):

        xy = (
            new_links[
                [
                    "x0",
                    "y0",
                    "x1",
                    "y1"
                ]
            ]
            .replace(
                [
                    np.inf,
                    -np.inf
                ],
                np.nan
            )
        )


        if (
            xy
            .notna()
            .any()
            .any()
        ):

            bounds.append(
                np.array(
                    [
                        np.nanmin(
                            [
                                xy["x0"].min(),
                                xy["x1"].min()
                            ]
                        ),

                        np.nanmin(
                            [
                                xy["y0"].min(),
                                xy["y1"].min()
                            ]
                        ),

                        np.nanmax(
                            [
                                xy["x0"].max(),
                                xy["x1"].max()
                            ]
                        ),

                        np.nanmax(
                            [
                                xy["y0"].max(),
                                xy["y1"].max()
                            ]
                        ),
                    ],
                    dtype=float
                )
            )


    if len(bounds) == 0:

        raise ValueError(
            "Cannot determine map bounds."
        )


    bounds = np.vstack(
        bounds
    )


    xmin = np.nanmin(
        bounds[:, 0]
    )

    ymin = np.nanmin(
        bounds[:, 1]
    )

    xmax = np.nanmax(
        bounds[:, 2]
    )

    ymax = np.nanmax(
        bounds[:, 3]
    )


    side = max(
        xmax - xmin,
        ymax - ymin
    )


    pad = (
        side
        *
        MAP_PAD_FRAC
    )


    xmin_p = xmin - pad
    xmax_p = xmax + pad

    ymin_p = ymin - pad
    ymax_p = ymax + pad


    xspan_p = (
        xmax_p
        -
        xmin_p
    )

    yspan_p = (
        ymax_p
        -
        ymin_p
    )


    side_p = max(
        xspan_p,
        yspan_p
    )


    xmid_p = (
        0.5
        *
        (
            xmin_p
            +
            xmax_p
        )
    )


    ymid_p = (
        0.5
        *
        (
            ymin_p
            +
            ymax_p
        )
    )


    xmin_sq = (
        xmid_p
        -
        side_p / 2.0
    )

    xmax_sq = (
        xmid_p
        +
        side_p / 2.0
    )

    ymin_sq = (
        ymid_p
        -
        side_p / 2.0
    )

    ymax_sq = (
        ymid_p
        +
        side_p / 2.0
    )


    xspan_sq = (
        xmax_sq
        -
        xmin_sq
    )


    xmin_plot = xmin_sq

    xmax_plot = (
        xmax_sq
        -
        RIGHT_CROP_FRAC
        *
        xspan_sq
    )

    ymin_plot = ymin_sq

    ymax_plot = ymax_sq


    fig_width_in = (
        FIG_HEIGHT_IN
        *
        (
            (
                xmax_plot
                -
                xmin_plot
            )
            /
            (
                ymax_plot
                -
                ymin_plot
            )
        )
    )


    # --------------------------------------------------------
    # Figure
    # --------------------------------------------------------

    fig = plt.figure(
        figsize=(
            fig_width_in,
            FIG_HEIGHT_IN
        ),
        dpi=300
    )


    fig.patch.set_facecolor(
        "white"
    )


    ax_map = fig.add_axes(
        [
            0.00,
            0.00,
            1,
            1
        ]
    )


    ax_donut = ax_map.inset_axes(
        [
            0.020,
            0.700,
            0.260,
            0.260
        ],
        transform=ax_map.transAxes
    )


    ax_donut.set_facecolor(
        "none"
    )


    ax_leg = ax_map.inset_axes(
        [
            0.03,
            0.025,
            0.500,
            0.285
        ],
        transform=ax_map.transAxes
    )


    ax_leg.axis(
        "off"
    )


    ax_leg.set_facecolor(
        "none"
    )


    # --------------------------------------------------------
    # Map background
    # --------------------------------------------------------

    ax_map.set_facecolor(
        c_ocean
    )


    if (
        len(cbg_poly) > 0
        and
        cbg_poly
        .geom_type
        .iloc[0]
        .lower()
        !=
        "point"
    ):

        cbg_poly.plot(
            ax=ax_map,

            facecolor=c_cbg_fill,

            edgecolor=c_cbg_edge,

            linewidth=0.42,

            zorder=0
        )


    if (
        msa_gdf is not None
        and
        len(msa_gdf) > 0
    ):

        msa_gdf.boundary.plot(
            ax=ax_map,

            color=c_msa_edge,

            linewidth=0.95,

            zorder=0.5
        )


    if len(poi_plot) > 0:

        ax_map.scatter(
            poi_plot["x"],
            poi_plot["y"],

            s=4.0,

            color="#C9C9C9",

            alpha=0.20,

            zorder=1
        )


    # --------------------------------------------------------
    # Corridors
    # --------------------------------------------------------

    if len(new_links) > 0:

        max_flow = max(
            1.0,
            new_links[
                "new_flow"
            ].max()
        )


        edge_draw_order = [
            "Lower exposure, farther",
            "Lower exposure, no farther",
            "Higher exposure, farther",
            "Higher exposure, no farther",
        ]


        edge_zorder = {
            "Lower exposure, farther":
                2.10,

            "Lower exposure, no farther":
                2.20,

            "Higher exposure, farther":
                2.30,

            "Higher exposure, no farther":
                2.80,
        }


        for klass in edge_draw_order:

            sub_links = new_links[
                new_links[
                    "link_tradeoff"
                ]
                ==
                klass
            ].copy()


            for local_idx, (_, r) in enumerate(
                sub_links.iterrows()
            ):

                if (
                    pd.isna(
                        r.get("x0")
                    )
                    or
                    pd.isna(
                        r.get("x1")
                    )
                ):

                    continue


                color = (
                    tradeoff_colors
                    .get(
                        klass,
                        "#BFC7D5"
                    )
                )


                alpha = (
                    tradeoff_alpha
                    .get(
                        klass,
                        0.60
                    )
                )


                lw = (
                    0.42
                    +
                    2.20
                    *
                    np.sqrt(
                        r["new_flow"]
                        /
                        max_flow
                    )
                )


                rad = (
                    EDGE_RAD_POS
                    if local_idx % 2 == 0
                    else EDGE_RAD_NEG
                )


                draw_curved_edge(
                    ax_map,

                    r["x0"],
                    r["y0"],

                    r["x1"],
                    r["y1"],

                    color=color,

                    lw=lw,

                    alpha=alpha,

                    rad=rad,

                    zorder=(
                        edge_zorder
                        .get(
                            klass,
                            2.5
                        )
                    )
                )


    # --------------------------------------------------------
    # Active origins
    # --------------------------------------------------------

    active_origins = origin_plot[
        origin_plot[
            "shifted_flow"
        ]
        >
        0
    ].copy()


    if len(active_origins) > 0:

        max_shift = max(
            1.0,
            active_origins[
                "shifted_flow"
            ].max()
        )


        s = 16


        ax_map.scatter(
            active_origins["x"],
            active_origins["y"],

            s=s,

            facecolor=c_origin,

            edgecolor="white",

            linewidth=0.30,

            alpha=0.86,

            zorder=3.4
        )


    # --------------------------------------------------------
    # Modified POIs
    # --------------------------------------------------------

    if len(modified_poi) > 0:

        ax_map.scatter(
            modified_poi["x"],
            modified_poi["y"],

            s=28,

            marker="o",

            facecolor=c_poi,

            edgecolor="none",

            linewidth=0.0,

            alpha=0.96,

            zorder=4
        )


    MAP_VIEW_X_SHIFT_FRAC = 0.04

    MAP_VIEW_Y_SHIFT_FRAC = 0.00


    x_shift = (
        (
            xmax_plot
            -
            xmin_plot
        )
        *
        MAP_VIEW_X_SHIFT_FRAC
    )


    y_shift = (
        (
            ymax_plot
            -
            ymin_plot
        )
        *
        MAP_VIEW_Y_SHIFT_FRAC
    )


    ax_map.set_xlim(
        xmin_plot + x_shift,
        xmax_plot + x_shift
    )


    ax_map.set_ylim(
        ymin_plot + y_shift,
        ymax_plot + y_shift
    )


    ax_map.set_aspect(
        "equal",
        adjustable="box"
    )


    ax_map.set_anchor(
        "C"
    )


    ax_map.axis(
        "off"
    )


    ax_map.text(
        0.01,
        0.98,
        CITY_LABEL,

        transform=ax_map.transAxes,

        ha="left",

        va="top",

        fontsize=18.5,

        fontweight="bold",

        color=c_text
    )


    # --------------------------------------------------------
    # Donut chart
    # --------------------------------------------------------

    qd = quadrant_df.set_index(
        "quadrant"
    )


    donut_labels = list(
        tradeoff_colors.keys()
    )


    donut_values = [
        (
            float(
                qd.loc[
                    label,
                    "share_pct"
                ]
            )
            if (
                label in qd.index
                and
                np.isfinite(
                    qd.loc[
                        label,
                        "share_pct"
                    ]
                )
            )
            else
            0.0
        )
        for label in donut_labels
    ]


    donut_colors = [
        tradeoff_colors[
            label
        ]
        for label
        in donut_labels
    ]


    if np.nansum(
        donut_values
    ) <= 0:

        donut_values = [
            1.0
        ]

        donut_colors = [
            "#BDBDBD"
        ]


    def _autopct(pct):

        return (
            f"{pct:.1f}"
            if pct >= 7
            else ""
        )


    wedges, _, autotexts = (
        ax_donut.pie(
            donut_values,

            colors=[
                "white"
            ]
            *
            len(
                donut_values
            ),

            startangle=90,

            counterclock=False,

            autopct=_autopct,

            pctdistance=0.78,

            textprops={
                "fontsize":
                    12,

                "color":
                    "#1F2937",
            },

            wedgeprops={
                "width":
                    DONUT_WIDTH,

                "linewidth":
                    0.95,

                "antialiased":
                    True,
            },
        )
    )


    for w, color in zip(
        wedges,
        donut_colors
    ):

        w.set_facecolor(
            "white"
        )

        w.set_edgecolor(
            color
        )

        w.set_hatch(
            DONUT_HATCH
        )

        w.set_linewidth(
            0.95
        )

        w.set_alpha(
            0.98
        )


    for t in autotexts:

        t.set_fontsize(
            12
        )

        t.set_color(
            "#1F2937"
        )


    ax_donut.text(
        0.0,
        0.0,

        "New-flow\ncomposition",

        ha="center",

        va="center",

        fontsize=12,

        color=c_text
    )


    ax_donut.set_aspect(
        "equal"
    )


    ax_donut.axis(
        "off"
    )


    # --------------------------------------------------------
    # Legend
    # --------------------------------------------------------

    legend_handles = [
        Line2D(
            [0],
            [0],

            color=tradeoff_colors[k],

            lw=2.6,

            alpha=0.88,

            label=compact_labels[k]
        )

        for k in tradeoff_colors
    ]


    legend_handles += [

        Line2D(
            [0],
            [0],

            marker='o',

            linestyle='none',

            markerfacecolor=c_origin,

            markeredgecolor="white",

            markersize=7.0,

            label="CBG"
        ),


        Line2D(
            [0],
            [0],

            marker='o',

            linestyle='none',

            markerfacecolor=c_poi,

            markeredgecolor="none",

            markersize=7.0,

            label="POI"
        ),
    ]


    ax_leg.legend(
        handles=legend_handles,

        loc="lower left",

        frameon=True,

        facecolor="white",

        edgecolor="none",

        framealpha=0.82,

        bbox_to_anchor=(
            0.00,
            0.00
        ),

        handlelength=1.30,

        handletextpad=0.50,

        labelspacing=0.58,

        fontsize=14,

        ncol=1,

        borderaxespad=0.0,

        borderpad=0.55
    )


    fig.savefig(
        "figure2c.pdf",

        format="pdf",

        dpi=300,

        bbox_inches="tight",

        transparent=False,

        backend="pdf"
    )


    if SHOW_FIGURES:

        plt.show()

    else:

        plt.close(
            fig
        )


    return fig


# ============================================================
# 7. Main
# ============================================================

def main():

    case_output = (
        load_case_output()
    )


    print(
        "\n========== FIG.2C PLOTTING DATA =========="
    )


    print(
        f"Model CBGs: "
        f"{len(case_output['cbg_ids'])}"
    )


    print(
        f"Model POIs: "
        f"{len(case_output['poi_ids'])}"
    )


    print(
        f"Newly activated links: "
        f"{len(case_output['new_links_df'])}"
    )


    print(
        "\n========== NEW-FLOW TRADE-OFF DECOMPOSITION =========="
    )


    print(
        case_output[
            "quadrant_df"
        ]
        .round(3)
        .to_string(index=False)
    )


    (
        cbg_poly,
        cbg_cent,
        poi_gdf,
        msa_gdf
    ) = (
        prepare_spatial_layers(
            case_output
        )
    )


    plot_fig2c_realized_corridors(
        case_output,
        cbg_poly,
        cbg_cent,
        poi_gdf,
        msa_gdf
    )


if __name__ == "__main__":
    main()