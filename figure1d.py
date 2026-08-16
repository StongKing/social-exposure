# -*- coding: utf-8 -*-
"""
Plot Fig. 1d from the released binary-flow and minimal POI-location package.

This script never reads poi_boston_msa_all.csv, filtered_boston_msa.csv, or
original flow magnitudes. Public shapefiles, income distributions, distance
matrices, and the prepared fig1d_binary_flow_package are used for plotting.
"""

import os
import glob
import warnings
import numpy as np
import pandas as pd
import geopandas as gpd

from shapely.geometry import Point, LineString

import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle, Circle, ConnectionPatch
from matplotlib.colors import to_rgb
import textwrap
from pyproj import Transformer


# ============================================================
# 0. Adjustable parameters
# ============================================================

PROJECT_ROOT = r"d:\mobility_social_exposure"

city = "boston"

DMAX_KM = 50
DISTANCE_SCALE = 1.0

FIG_DPI = 300

SAVE_FIG = False
SAVE_CSV = False

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "fig1d_boston_msa_second_quadrant_network"
)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 如果 distance_matrix.csv 单位已经是 km，保持 1.0；
# 如果是 meter，改成 1 / 1000。
USE_ACTIVE_WITHIN_DMAX_FOR_BASELINE = True


# ------------------------------------------------------------
# Network-edge display settings
# ------------------------------------------------------------

# 是否显示全部 second-quadrant links
# True = 不抽样、不筛选，全部显示
SHOW_ALL_SECOND_QUADRANT_EDGES = True

# 当 SHOW_ALL_SECOND_QUADRANT_EDGES = False 时才生效
MAX_EDGES_PER_CATEGORY = 2000

# 可选：
#   "top_delta" : 每类选择 ΔS 最大的边
#   "shortest"  : 每类选择距离最短的边
#   "random"    : 每类随机抽样
EDGE_SELECT_MODE = "random"

RANDOM_SEED = 42

# 节点显示
DRAW_POI_NODES = True
DRAW_CBG_NODES = False
DRAW_ALL_POI_BACKGROUND = True

# POI 点统一大小
UNIFORM_POI_NODE_SIZE = 3

# 曲线边设置
DRAW_CURVED_EDGES = True
EDGE_ALPHA = 0.055       # 边多时保持较低；如果仍然太淡可调到 0.07
EDGE_LINEWIDTH = 0.22

# 倒 U 型曲线强度
# 数值越大，弯曲越明显；0.12–0.25 通常比较合适
CURVE_STRENGTH = 0.36

# 每条曲线用多少个点拟合；越大越平滑，但绘制越慢
CURVE_N_POINTS = 36

# ------------------------------------------------------------
# Circular inset display settings
# ------------------------------------------------------------
# 只是在原图左上角额外添加一个 10 km 圆形放大子图；
# 默认不在主图上画定位圆，避免改变原图视觉。
DRAW_10KM_INSET = True
DRAW_INSET_LOCATOR_ON_MAIN = True

# 固定放大中心点。当前默认值为 Boston downtown 附近；
# 你可以改成任意 Boston MSA 内的经纬度。
INSET_CENTER_LON = -71.410
INSET_CENTER_LAT = 42.2851
INSET_RADIUS_KM = 10

# 左上角 inset 的位置与大小，使用 figure 坐标：
# [left, bottom, width, height]
INSET_AX_BOUNDS = [0.03, 0.665, 0.25, 0.25]

# inset 内部是否继续使用原来的曲线边。
# 保持 True 时，小图与原图的边绘制方式一致。
DRAW_CURVED_EDGES_IN_INSET = DRAW_CURVED_EDGES

# inset 默认沿用原图线宽和透明度；如果局部太淡，可以只调这里，
# 不会影响主图。
INSET_EDGE_ALPHA = 0.3
INSET_EDGE_LINEWIDTH = 0.2
INSET_POI_NODE_SIZE = UNIFORM_POI_NODE_SIZE
INSET_ALL_POI_BACKGROUND_SIZE = 0.8
INSET_FRAME_LINEWIDTH = 0.90
INSET_FRAME_COLOR = "#222222"

# inset 内最多显示多少条边。None 表示不额外抽样，完全显示局部范围内边。
# 如果局部边过密或出图太慢，可改为 1000、2000 等。
INSET_MAX_EDGES = None


# 主图定位圆与 inset 连接线样式
INSET_LOCATOR_COLOR = "#222222"
INSET_LOCATOR_LINEWIDTH = 0.85
INSET_LOCATOR_ALPHA = 0.80
INSET_LOCATOR_DASH = (0, (3.0, 2.2))

# 两条放射虚线：
# 主图圆上的连接点，用经纬度椭圆参数角度控制。
# 这里 135° 和 45° 大致对应主图圆的左上、右上边缘。
CONNECTOR_MAIN_ANGLE_1 = 180
CONNECTOR_MAIN_ANGLE_2 = 0

# inset 圆上的连接点，使用 inset axes 坐标。
# 这里连接到 inset 圆的左、右边缘。
CONNECTOR_INSET_POINT_1 = (0.00, 0.5)
CONNECTOR_INSET_POINT_2 = (1.00, 0.5)

DRAW_INSET_CONNECTORS = True



# CBG centroid 用美国等面积投影计算，再转回 WGS84。
CENTROID_CRS = "EPSG:5070"

# 曲线计算也在投影坐标系中完成，否则在经纬度中抬高控制点时视觉上容易像直线。
CURVE_CRS = "EPSG:5070"
CURVE_MIN_LIFT_M = 800.0
CURVE_MAX_LIFT_M = 18000.0

# Boston MSA boundary and CBG shapefiles
MA_CBG_SHP = os.path.join("geo_data", "tl_2021_25_bg", "tl_2021_25_bg.shp")
NH_CBG_SHP = os.path.join("geo_data", "tl_2021_33_bg", "tl_2021_33_bg.shp")
CBSA_SHP = os.path.join("geo_data","tl_2021_us_cbsa", "tl_2021_us_cbsa.shp")
BOSTON_CBSA_GEOID = "14460"

# The plotting script does not read the original POI attribute or visit files.
# It loads only the minimal coordinate table created by
# pre_figure1d_with_poi_locations.py.
POI_ID_COL = "poi_id"

# MSA matrix folder
# 你的 Boston MSA 版本之前使用的是：
#   matrices_A_D_S_Distribution/
#       POI category/
#           flow_matrix.csv
#           distance_matrix.csv
MSA_MATRIX_DIR = os.path.join(PROJECT_ROOT, "matrices_A_D_S_Distribution")

# Released package created by pre_figure1d_with_poi_locations.py.
# It is expected in the current working directory.
BINARY_FLOW_DIR = os.path.join(
    os.getcwd(),
    "fig1d_binary_flow_package"
)
FLOW_METADATA_PATH = os.path.join(
    BINARY_FLOW_DIR,
    "category_flow_metadata.csv"
)
POI_LOCATION_PATH = os.path.join(
    BINARY_FLOW_DIR,
    "poi_boston_msa_plot_locations.csv"
)

print_progress = True


# ============================================================
# 1. POI and income settings
# ============================================================

income_levels = [
    "low_income_pct",
    "lower_middle_income_pct",
    "upper_middle_income_pct",
    "high_income_pct"
]

# ------------------------------------------------------------
# Drawing / calculation order
# ------------------------------------------------------------
# 这里的顺序同时控制：
#   1) 数据计算循环顺序；
#   2) 地图上边和点的绘制顺序；
#   3) 图例顺序；
#   4) 右侧横向条形图的显示顺序。
# Matplotlib 中后绘制的图层会压在先绘制图层上方；
# 因此下面的 Other Individual & Family Services 会最后绘制、位于最上层。

poi_names = [
    "Fitness_and_Recreational_Sports_Centers",
    "Religious_Organizations",
    "Drinking_Places_(Alcoholic_Beverages)",
    "Museums",
    "Promoters_of_Performing_Arts,_Sports,_and_Similar_Events_with_Facilities",
    "Other_Individual_and_Family_Services"
]

poi_pretty = {
    "Fitness_and_Recreational_Sports_Centers": "Fitness\nCenters",
    "Religious_Organizations": "Religious\nOrganizations",
    "Drinking_Places_(Alcoholic_Beverages)": "Drinking\nPlaces",
    "Museums": "Museums",
    "Promoters_of_Performing_Arts,_Sports,_and_Similar_Events_with_Facilities": "Performing Arts\nFacilities",
    "Other_Individual_and_Family_Services": "Individual &\nFamily Services"
}

poi_pretty_one_line = {
    "Fitness_and_Recreational_Sports_Centers": "Fitness Centers",
    "Religious_Organizations": "Religious Organizations",
    "Drinking_Places_(Alcoholic_Beverages)": "Drinking Places",
    "Museums": "Museums",
    "Promoters_of_Performing_Arts,_Sports,_and_Similar_Events_with_Facilities": "Performing Arts Facilities",
    "Other_Individual_and_Family_Services": "Individual & Family Services"
}

poi_to_code = {
    "Fitness_and_Recreational_Sports_Centers": "713940",
    "Religious_Organizations": "813110",
    "Drinking_Places_(Alcoholic_Beverages)": "722410",
    "Museums": "712110",
    "Promoters_of_Performing_Arts,_Sports,_and_Similar_Events_with_Facilities": "711310",
    "Other_Individual_and_Family_Services": "624190"
}

poi_code_to_label = {
    "713940": "Fitness Centers",
    "813110": "Religious Organizations",
    "722410": "Drinking Places",
    "712110": "Museums",
    "711310": "Performing Arts Facilities",
    "624190": "Individual & Family Services"
}

base_colors = {
    "713940": "#984EA3",
    "813110": "#A65628",
    "722410": "#FF7F00",
    "712110": "#4DAF4A",
    "711310": "#377EB8",
    "624190": "#E41A1C"
}

# 专门用于网络边的颜色，不影响 POI 点颜色。
# 如果要改边的颜色，只改这里；POI 点颜色仍由 base_colors 控制。

edge_colors = {
    "713940": "#984EA3",
    "813110": "#A65628",
    "722410": "#FF7F00",
    "712110": "#4DAF4A",
    "711310": "#377EB8",
    "624190": "#E41A1C"
}


base_colors = {
    "713940": "#984EA3",
    "813110": "#377EB8",
    "722410": "#FF7F00",
    "712110": "#4DAF4A",
    "711310": "#A65628",
    "624190": "#E41A1C"
}


edge_colors = {
    "713940": "#984EA3",
    "813110": "#377EB8",
    "722410": "#FF7F00",
    "712110": "#4DAF4A",
    "711310": "#A65628",
    "624190": "#E41A1C"
}

poi_order = ["713940", "813110", "722410", "712110", "711310", "624190"]
poi_order_rank = {code: i for i, code in enumerate(poi_order)}
poi_name_order_rank = {name: i for i, name in enumerate(poi_names)}


# ============================================================
# 2. Helper functions
# ============================================================

def lighten_color(color, amount=0.55):
    try:
        r, g, b = to_rgb(color)
    except Exception:
        r, g, b = (0.5, 0.5, 0.5)

    r = r + (1.0 - r) * amount
    g = g + (1.0 - g) * amount
    b = b + (1.0 - b) * amount

    return (r, g, b)


soft_color_map = {
    k: lighten_color(v, amount=0.52)
    for k, v in base_colors.items()
}


def normalize_geoid(x):
    """
    Normalize CBG GEOID:
        250250001011.0 -> '250250001011'
    """
    if pd.isna(x):
        return None
    try:
        return str(int(float(x)))
    except Exception:
        return str(x)


def read_matrix_csv(path):
    df = pd.read_csv(path, header=0, index_col=0)

    df.index = df.index.astype(str).map(normalize_geoid)
    df.columns = df.columns.astype(str)

    df = df.apply(pd.to_numeric, errors="coerce")

    if df.index.duplicated().any():
        df = df.groupby(level=0).sum()

    return df


def weighted_mean(x, w):
    x = np.asarray(x, dtype=float)
    w = np.asarray(w, dtype=float)

    mask = np.isfinite(x) & np.isfinite(w) & (w > 0)

    if mask.sum() == 0:
        return np.nan

    return np.sum(x[mask] * w[mask]) / np.sum(w[mask])


def find_income_file_boston_msa():
    candidates = [
        os.path.join(MSA_MATRIX_DIR, "cbg_income_level_distribution_boston_msa.csv"),
        os.path.join(MSA_MATRIX_DIR, "cbg_income_level_distribution_boston_core.csv"),
        os.path.join(PROJECT_ROOT, "cbg_income_level_distribution_boston_msa.csv"),
        os.path.join(PROJECT_ROOT, "cbg_income_level_distribution_boston_core.csv"),
    ]

    for p in candidates:
        if os.path.isfile(p):
            return p

    patterns = [
        os.path.join(PROJECT_ROOT, "**", "cbg_income_level_distribution_boston_msa.csv"),
        os.path.join(PROJECT_ROOT, "**", "cbg_income_level_distribution_boston_core.csv"),
    ]

    for pattern in patterns:
        matches = glob.glob(pattern, recursive=True)
        if len(matches) > 0:
            return matches[0]

    raise FileNotFoundError(
        "Cannot find Boston income distribution file. Checked common paths and recursive patterns."
    )


def load_income_distribution_boston_msa():
    income_path = find_income_file_boston_msa()

    if print_progress:
        print(f"[LOAD income] {income_path}")

    df = pd.read_csv(income_path)

    if "GEOID" not in df.columns:
        raise ValueError(f"Income file must contain GEOID column: {income_path}")

    missing_cols = [c for c in income_levels if c not in df.columns]
    if len(missing_cols) > 0:
        raise ValueError(f"Income file missing columns {missing_cols}: {income_path}")

    df["GEOID_str"] = df["GEOID"].apply(normalize_geoid)

    P_df = df.set_index("GEOID_str")[income_levels].copy()
    P_df = P_df.apply(pd.to_numeric, errors="coerce").fillna(0)

    if P_df.index.duplicated().any():
        P_df = P_df.groupby(level=0).mean()

    row_sum = P_df.sum(axis=1).replace(0, np.nan)
    P_df = P_df.div(row_sum, axis=0).fillna(0)

    return P_df


def get_poi_dir_msa(poi_name):
    return os.path.join(MSA_MATRIX_DIR, poi_name)


_flow_metadata_cache = None


def _parse_bool(value):
    if isinstance(value, (bool, np.bool_)):
        return bool(value)

    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False

    raise ValueError(f"Cannot interpret Boolean value: {value!r}")


def load_flow_metadata():
    global _flow_metadata_cache

    if _flow_metadata_cache is None:
        if not os.path.isfile(FLOW_METADATA_PATH):
            raise FileNotFoundError(
                "Binary-flow metadata not found:\n"
                f"{os.path.abspath(FLOW_METADATA_PATH)}\n"
                "Run fig1d_prepare_binary_flow.py first."
            )

        metadata = pd.read_csv(FLOW_METADATA_PATH)

        required = {
            "poi_category",
            "poi_code",
            "package_file",
            "n_cbgs",
            "n_pois",
            "n_active_ref",
            "n_unused_feasible",
            "total_active_ref_flow",
            "total_all_active_flow",
            "active_weighted_exposure_ref",
            "active_weighted_distance_ref",
            "dmax_km",
            "distance_scale",
            "use_active_within_dmax_for_baseline",
        }
        missing = sorted(required - set(metadata.columns))
        if missing:
            raise ValueError(
                "Binary-flow metadata is missing columns: "
                + ", ".join(missing)
            )

        _flow_metadata_cache = metadata

    return _flow_metadata_cache.copy()


def load_boston_msa_poi_case(poi_name):
    """
    Load binary flow presence, its validity mask, saved POI composition Q,
    the original non-sensitive distance matrix, and category-level derived
    flow-weighted baselines.
    """
    metadata = load_flow_metadata()
    row = metadata[metadata["poi_category"] == poi_name].copy()

    if len(row) != 1:
        raise ValueError(
            f"Expected one metadata row for {poi_name}, found {len(row)}."
        )

    meta = row.iloc[0]
    package_path = os.path.join(
        BINARY_FLOW_DIR,
        str(meta["package_file"]),
    )

    if not os.path.isfile(package_path):
        raise FileNotFoundError(
            f"Binary-flow package not found: {package_path}"
        )

    with np.load(package_path, allow_pickle=False) as package:
        required_arrays = {
            "flow_binary",
            "flow_valid",
            "cbg_ids",
            "poi_ids",
            "q_values",
        }
        missing_arrays = sorted(required_arrays - set(package.files))
        if missing_arrays:
            raise ValueError(
                f"Package {package_path} is missing arrays: "
                + ", ".join(missing_arrays)
            )

        cbg_ids = package["cbg_ids"].astype(str)
        poi_ids = package["poi_ids"].astype(str)
        binary_values = package["flow_binary"].astype(np.uint8)
        valid_values = package["flow_valid"].astype(bool)
        q_values = package["q_values"].astype(float)

    expected_shape = (len(cbg_ids), len(poi_ids))
    if binary_values.shape != expected_shape:
        raise ValueError(
            f"Invalid binary matrix shape for {poi_name}: "
            f"{binary_values.shape} != {expected_shape}"
        )
    if valid_values.shape != expected_shape:
        raise ValueError(
            f"Invalid flow-valid mask shape for {poi_name}."
        )
    if q_values.shape != (len(poi_ids), len(income_levels)):
        raise ValueError(
            f"Invalid q_values shape for {poi_name}: {q_values.shape}"
        )

    B = pd.DataFrame(
        binary_values,
        index=cbg_ids,
        columns=poi_ids,
    )
    V = pd.DataFrame(
        valid_values,
        index=cbg_ids,
        columns=poi_ids,
    )
    Q_df = pd.DataFrame(
        q_values,
        index=poi_ids,
        columns=income_levels,
    )

    poi_dir = get_poi_dir_msa(poi_name)
    dist_path = os.path.join(poi_dir, "distance_matrix.csv")
    if not os.path.isfile(dist_path):
        raise FileNotFoundError(
            f"distance_matrix.csv not found: {dist_path}"
        )

    D = read_matrix_csv(dist_path) * DISTANCE_SCALE

    if print_progress:
        print(f"\n[LOAD] Boston MSA | {poi_name}")
        print(f"  binary flow: {package_path} | shape={B.shape}")
        print(f"  distance   : {dist_path} | shape={D.shape}")

    return B, V, D, Q_df, meta


# ============================================================
# 3. Exposure functions using saved original-flow derivatives
# ============================================================

def compute_all_pair_unmasked_exposure(B, V, P_df, Q_df):
    """
    Reconstruct S_ij from the saved original-flow-derived POI composition Q_j:

        S_ij = 1 - dot(P_i, Q_j)

    B contains only 0/1 flow presence and does not contain flow magnitudes.
    """
    B = B.copy()
    V = V.copy()
    B.index = B.index.map(normalize_geoid)
    V.index = V.index.map(normalize_geoid)

    common_cbgs = sorted(set(B.index) & set(V.index) & set(P_df.index))
    common_pois = sorted(set(B.columns) & set(V.columns) & set(Q_df.index))

    if len(common_cbgs) == 0:
        raise ValueError(
            "No common CBGs among binary flow, validity, and income data."
        )
    if len(common_pois) == 0:
        raise ValueError(
            "No common POIs between binary flow and saved Q data."
        )

    B = B.loc[common_cbgs, common_pois].copy()
    V = V.loc[common_cbgs, common_pois].copy()
    P = P_df.loc[common_cbgs, income_levels].copy()
    Q = Q_df.loc[common_pois, income_levels].copy()

    P_values = P.values.astype(float)
    Q_values = Q.values.astype(float)
    S_values = 1.0 - (P_values @ Q_values.T)

    S = pd.DataFrame(
        S_values,
        index=B.index,
        columns=B.columns,
    )

    return S, B, V, Q


def align_F_D_S(B, V, D, S):
    B = B.copy()
    V = V.copy()
    D = D.copy()
    S = S.copy()

    B.index = B.index.map(normalize_geoid)
    V.index = V.index.map(normalize_geoid)
    D.index = D.index.map(normalize_geoid)
    S.index = S.index.map(normalize_geoid)

    common_rows = sorted(
        set(B.index) & set(V.index) & set(D.index) & set(S.index)
    )
    common_cols = sorted(
        set(B.columns) & set(V.columns) & set(D.columns) & set(S.columns)
    )

    if len(common_rows) == 0:
        raise ValueError("No common CBG rows among B, V, D, and S.")

    if len(common_cols) == 0:
        raise ValueError("No common POI columns among B, V, D, and S.")

    return (
        B.loc[common_rows, common_cols].copy(),
        V.loc[common_rows, common_cols].copy(),
        D.loc[common_rows, common_cols].copy(),
        S.loc[common_rows, common_cols].copy(),
    )


# ============================================================
# 4. Load Boston MSA map layers
# ============================================================

def load_boston_msa_boundary_and_cbgs():
    if not os.path.isfile(CBSA_SHP):
        raise FileNotFoundError(f"CBSA shapefile not found: {CBSA_SHP}")

    if not os.path.isfile(MA_CBG_SHP):
        raise FileNotFoundError(f"MA CBG shapefile not found: {MA_CBG_SHP}")

    if not os.path.isfile(NH_CBG_SHP):
        raise FileNotFoundError(f"NH CBG shapefile not found: {NH_CBG_SHP}")

    cbsa = gpd.read_file(CBSA_SHP).to_crs("EPSG:4326")

    if "GEOID" in cbsa.columns:
        bos_msa = cbsa[cbsa["GEOID"].astype(str) == BOSTON_CBSA_GEOID].copy()
    else:
        bos_msa = cbsa[
            cbsa["NAME"].str.contains("Boston", case=False, na=False)
        ].copy()

    if bos_msa.empty:
        raise RuntimeError(
            "Boston MSA not found in CBSA shapefile. Check BOSTON_CBSA_GEOID or CBSA_SHP."
        )

    bos_union = bos_msa.unary_union

    ma_cbg = gpd.read_file(MA_CBG_SHP).to_crs("EPSG:4326")
    nh_cbg = gpd.read_file(NH_CBG_SHP).to_crs("EPSG:4326")

    cbg_all = pd.concat([ma_cbg, nh_cbg], ignore_index=True)
    cbg_all = gpd.GeoDataFrame(cbg_all, crs="EPSG:4326")

    try:
        cbg_inside = gpd.clip(cbg_all, bos_msa)
    except Exception:
        cbg_inside = cbg_all.copy()
        cbg_inside["geometry"] = cbg_inside.geometry.intersection(bos_union)

    cbg_inside = cbg_inside[
        cbg_inside.geometry.notna() &
        (~cbg_inside.geometry.is_empty)
    ].copy()

    if cbg_inside.empty:
        raise RuntimeError("After clipping, no CBG geometries remain inside Boston MSA.")

    if "GEOID" not in cbg_inside.columns:
        raise ValueError("CBG shapefile must contain GEOID column.")

    cbg_inside["GEOID_str"] = cbg_inside["GEOID"].apply(normalize_geoid)

    if print_progress:
        print(f"[INFO] CBGs after clipping to Boston MSA: {len(cbg_inside):,}")

    return bos_msa, bos_union, cbg_inside


def load_boston_msa_pois(bos_union):
    """
    Load the minimal POI plotting table created by the preparation script.

    Required released fields:
        poi_id, longitude, latitude, poi_code

    The original poi_boston_msa_all.csv and filtered_boston_msa.csv files are
    neither required nor accessed here.
    """
    if not os.path.isfile(POI_LOCATION_PATH):
        raise FileNotFoundError(
            "Minimal POI plotting file not found:\n"
            f"{os.path.abspath(POI_LOCATION_PATH)}\n"
            "Run pre_figure1d_with_poi_locations.py first."
        )

    poi = pd.read_csv(
        POI_LOCATION_PATH,
        dtype={"poi_id": str, "poi_code": str},
        low_memory=False,
    )

    required = {"poi_id", "longitude", "latitude", "poi_code"}
    missing = sorted(required - set(poi.columns))
    if missing:
        raise ValueError(
            "Minimal POI plotting file is missing columns: "
            + ", ".join(missing)
        )

    poi["poi_id"] = poi["poi_id"].astype(str).str.strip()
    poi["poi_code"] = poi["poi_code"].astype(str).str.replace(
        r"\.0$", "", regex=True
    )
    poi["longitude"] = pd.to_numeric(poi["longitude"], errors="coerce")
    poi["latitude"] = pd.to_numeric(poi["latitude"], errors="coerce")

    poi = poi[
        poi["poi_code"].isin(poi_order)
        & poi["poi_id"].ne("")
        & poi["poi_id"].ne("nan")
    ].copy()
    poi = poi.dropna(subset=["longitude", "latitude"]).copy()
    poi = poi.drop_duplicates(subset=["poi_id"], keep="first").copy()

    poi_geom = [
        Point(xy)
        for xy in zip(
            poi["longitude"].astype(float),
            poi["latitude"].astype(float),
        )
    ]

    poi_gdf = gpd.GeoDataFrame(
        poi,
        crs="EPSG:4326",
        geometry=poi_geom,
    )

    # Retain only points inside the public Boston MSA boundary.
    poi_gdf = poi_gdf[poi_gdf.geometry.within(bos_union)].copy()

    # Keep the in-memory alias used by the existing coordinate merge helper.
    poi_gdf["naics_prefix"] = poi_gdf["poi_code"]

    if print_progress:
        print(
            f"[INFO] POIs loaded from released location table and inside "
            f"Boston MSA: {len(poi_gdf):,}"
        )

    return poi_gdf


# ============================================================
# 5. Edge selection helper
# ============================================================

def select_second_quadrant_edges(
    row_idx,
    col_idx,
    delta_s,
    delta_d,
    Dv,
    max_edges,
    mode,
    random_seed
):
    """
    Select second-quadrant edges for plotting.

    If SHOW_ALL_SECOND_QUADRANT_EDGES = True:
        return all second-quadrant edges.

    Otherwise:
        select at most max_edges edges per POI category.
    """

    n = len(row_idx)

    if n == 0:
        return row_idx, col_idx

    # --------------------------------------------------------
    # 显示全部 second-quadrant links
    # --------------------------------------------------------
    if SHOW_ALL_SECOND_QUADRANT_EDGES:
        return row_idx, col_idx

    if n <= max_edges:
        return row_idx, col_idx

    rng = np.random.default_rng(random_seed)

    if mode == "random":
        keep = rng.choice(n, size=max_edges, replace=False)

    elif mode == "top_delta":
        score = delta_s[row_idx, col_idx]
        keep_unsorted = np.argpartition(score, -max_edges)[-max_edges:]
        keep = keep_unsorted[np.argsort(score[keep_unsorted])[::-1]]

    elif mode == "shortest":
        score = Dv[row_idx, col_idx]
        keep_unsorted = np.argpartition(score, max_edges)[:max_edges]
        keep = keep_unsorted[np.argsort(score[keep_unsorted])]

    else:
        raise ValueError("EDGE_SELECT_MODE must be 'top_delta', 'shortest', or 'random'.")

    return row_idx[keep], col_idx[keep]

# ============================================================
# 6. Core calculation for Fig. 1d
# ============================================================

def summarize_boston_msa_category_and_select_edges(poi_name, P_df):
    """
    For one Boston MSA POI category:
        1. read binary active/unused status;
        2. reconstruct exposure from saved original-flow-derived Q;
        3. apply the saved original flow-weighted baselines;
        4. compute statistics and retain map-display edges.
    """
    B_raw, V_raw, D_raw, Q_df, flow_meta = load_boston_msa_poi_case(
        poi_name
    )

    # Baseline settings must remain identical to those used when the package
    # was prepared. Otherwise the saved weighted baselines are not compatible.
    saved_dmax = float(flow_meta["dmax_km"])
    saved_distance_scale = float(flow_meta["distance_scale"])
    saved_mode = _parse_bool(
        flow_meta["use_active_within_dmax_for_baseline"]
    )

    if not np.isclose(saved_dmax, DMAX_KM):
        raise ValueError(
            f"DMAX_KM mismatch for {poi_name}: package={saved_dmax}, "
            f"plotting code={DMAX_KM}."
        )
    if not np.isclose(saved_distance_scale, DISTANCE_SCALE):
        raise ValueError(
            f"DISTANCE_SCALE mismatch for {poi_name}: "
            f"package={saved_distance_scale}, plotting code={DISTANCE_SCALE}."
        )
    if saved_mode != bool(USE_ACTIVE_WITHIN_DMAX_FOR_BASELINE):
        raise ValueError(
            f"USE_ACTIVE_WITHIN_DMAX_FOR_BASELINE mismatch for {poi_name}."
        )

    S_all, B_income, V_income, Q_aligned = (
        compute_all_pair_unmasked_exposure(
            B_raw,
            V_raw,
            P_df,
            Q_df,
        )
    )
    B, V, D, S = align_F_D_S(
        B_income,
        V_income,
        D_raw,
        S_all,
    )

    Bv = B.values.astype(np.uint8)
    Vv = V.values.astype(bool)
    Dv = D.values.astype(float)
    Sv = S.values.astype(float)

    valid = Vv & np.isfinite(Dv) & np.isfinite(Sv)
    distance_feasible = valid & (Dv >= 0) & (Dv <= DMAX_KM)
    active_all = valid & (Bv > 0)

    if USE_ACTIVE_WITHIN_DMAX_FOR_BASELINE:
        active_ref = active_all & distance_feasible
    else:
        active_ref = active_all

    unused_feasible = distance_feasible & (Bv == 0)

    n_active_ref = int(active_ref.sum())
    n_unused_feasible = int(unused_feasible.sum())

    # These quantities were calculated once from the confidential original
    # flow magnitudes. No original magnitude is loaded by this script.
    total_active_ref_flow = float(flow_meta["total_active_ref_flow"])
    total_all_active_flow = float(flow_meta["total_all_active_flow"])
    active_w_s = float(flow_meta["active_weighted_exposure_ref"])
    active_w_d = float(flow_meta["active_weighted_distance_ref"])

    expected_active_ref = int(flow_meta["n_active_ref"])
    expected_unused = int(flow_meta["n_unused_feasible"])

    if n_active_ref != expected_active_ref:
        raise ValueError(
            f"Active-reference count mismatch for {poi_name}: "
            f"current={n_active_ref}, package={expected_active_ref}. "
            "Check the distance and income files."
        )
    if n_unused_feasible != expected_unused:
        raise ValueError(
            f"Unused-feasible count mismatch for {poi_name}: "
            f"current={n_unused_feasible}, package={expected_unused}. "
            "Check the distance and income files."
        )

    if n_active_ref == 0:
        raise ValueError(f"No active reference links for {poi_name}.")
    if n_unused_feasible == 0:
        raise ValueError(f"No unused feasible links for {poi_name}.")
    if not np.isfinite(active_w_s) or not np.isfinite(active_w_d):
        raise ValueError(
            f"Invalid saved weighted baseline for {poi_name}."
        )

    delta_s = Sv - active_w_s
    delta_d = Dv - active_w_d

    second_quadrant = unused_feasible & (delta_s > 0) & (delta_d <= 0)

    n_second_quadrant = int(second_quadrant.sum())
    share_second_quadrant = float(
        n_second_quadrant / n_unused_feasible
    )

    cbg_ids = np.array(B.index.astype(str))
    poi_ids = np.array(B.columns.astype(str))

    category_row = {
        "city": "boston_msa",
        "city_label": "Boston MSA",
        "poi_category": poi_name,
        "poi_label": poi_pretty.get(poi_name, poi_name),
        "poi_label_one_line": poi_pretty_one_line.get(
            poi_name,
            poi_name,
        ),
        "poi_code": poi_to_code.get(poi_name, poi_name),
        "n_active_ref": n_active_ref,
        "n_unused_feasible": n_unused_feasible,
        "n_second_quadrant": n_second_quadrant,
        "total_active_ref_flow": total_active_ref_flow,
        "total_all_active_flow": total_all_active_flow,
        "active_weighted_exposure_ref": active_w_s,
        "active_weighted_distance_ref": active_w_d,
        "share_second_quadrant": share_second_quadrant,
    }

    unused_by_origin = unused_feasible.sum(axis=1)
    q2_by_origin = second_quadrant.sum(axis=1)

    origin_df = pd.DataFrame({
        "cbg_id": cbg_ids,
        "poi_category": poi_name,
        "poi_code": poi_to_code.get(poi_name, poi_name),
        "n_unused_feasible": unused_by_origin.astype(int),
        "n_second_quadrant": q2_by_origin.astype(int),
    })
    origin_df["share_second_quadrant"] = (
        origin_df["n_second_quadrant"]
        / origin_df["n_unused_feasible"].replace(0, np.nan)
    )

    unused_by_dest = unused_feasible.sum(axis=0)
    q2_by_dest = second_quadrant.sum(axis=0)

    dest_df = pd.DataFrame({
        "poi_id": poi_ids.astype(str),
        "poi_category": poi_name,
        "poi_code": poi_to_code.get(poi_name, poi_name),
        "n_unused_feasible": unused_by_dest.astype(int),
        "n_second_quadrant": q2_by_dest.astype(int),
    })
    dest_df["share_second_quadrant"] = (
        dest_df["n_second_quadrant"]
        / dest_df["n_unused_feasible"].replace(0, np.nan)
    )

    row_idx, col_idx = np.where(second_quadrant)
    row_sel, col_sel = select_second_quadrant_edges(
        row_idx=row_idx,
        col_idx=col_idx,
        delta_s=delta_s,
        delta_d=delta_d,
        Dv=Dv,
        max_edges=MAX_EDGES_PER_CATEGORY,
        mode=EDGE_SELECT_MODE,
        random_seed=(
            RANDOM_SEED
            + poi_name_order_rank.get(poi_name, 0)
        ),
    )

    edge_df = pd.DataFrame({
        "cbg_id": cbg_ids[row_sel],
        "poi_id": poi_ids[col_sel],
        "poi_category": poi_name,
        "poi_code": poi_to_code.get(poi_name, poi_name),
        "distance_km": Dv[row_sel, col_sel],
        "exposure": Sv[row_sel, col_sel],
        "delta_exposure": delta_s[row_sel, col_sel],
        "delta_distance": delta_d[row_sel, col_sel],
    })
    edge_df["selection_mode"] = EDGE_SELECT_MODE

    if print_progress:
        print(
            f"  {poi_pretty_one_line.get(poi_name, poi_name)} | "
            f"unused feasible={n_unused_feasible:,} | "
            f"second quadrant={n_second_quadrant:,} | "
            f"share={share_second_quadrant * 100:.2f}% | "
            f"displayed edges={len(edge_df):,}"
        )

    return category_row, origin_df, dest_df, edge_df


def build_fig1d_network_data():
    P_df = load_income_distribution_boston_msa()

    category_rows = []
    origin_dfs = []
    dest_dfs = []
    edge_dfs = []

    for poi_name in poi_names:
        try:
            category_row, origin_df, dest_df, edge_df = summarize_boston_msa_category_and_select_edges(
                poi_name=poi_name,
                P_df=P_df
            )

            category_rows.append(category_row)
            origin_dfs.append(origin_df)
            dest_dfs.append(dest_df)
            edge_dfs.append(edge_df)

        except Exception as e:
            warnings.warn(f"Skipped Boston MSA | {poi_name}: {e}")

    if len(category_rows) == 0:
        raise RuntimeError("No Boston MSA POI category loaded successfully.")

    category_summary = pd.DataFrame(category_rows)
    origin_all = pd.concat(origin_dfs, ignore_index=True)
    dest_all = pd.concat(dest_dfs, ignore_index=True)
    edge_selected = pd.concat(edge_dfs, ignore_index=True)

    # --------------------------------------------------------
    # Origin-side pooled summary across six categories
    # --------------------------------------------------------

    origin_pooled = (
        origin_all
        .groupby("cbg_id", as_index=False)
        .agg(
            n_unused_feasible=("n_unused_feasible", "sum"),
            n_second_quadrant=("n_second_quadrant", "sum")
        )
    )

    origin_pooled["share_second_quadrant"] = (
        origin_pooled["n_second_quadrant"] /
        origin_pooled["n_unused_feasible"].replace(0, np.nan)
    )

    # --------------------------------------------------------
    # Destination-side pooled summary across six categories
    # --------------------------------------------------------

    dest_pooled = (
        dest_all
        .groupby("poi_id", as_index=False)
        .agg(
            n_unused_feasible=("n_unused_feasible", "sum"),
            n_second_quadrant=("n_second_quadrant", "sum")
        )
    )

    dest_pooled["share_second_quadrant"] = (
        dest_pooled["n_second_quadrant"] /
        dest_pooled["n_unused_feasible"].replace(0, np.nan)
    )

    # --------------------------------------------------------
    # Boston MSA pooled share across categories
    # Use unused-link weighting because denominator is unused feasible links.
    # --------------------------------------------------------

    total_unused = category_summary["n_unused_feasible"].sum()
    total_q2 = category_summary["n_second_quadrant"].sum()

    city_summary = pd.DataFrame([{
        "city_label": "Boston MSA",
        "n_categories": len(category_summary),
        "n_unused_feasible": int(total_unused),
        "n_second_quadrant": int(total_q2),
        "share_second_quadrant": total_q2 / total_unused if total_unused > 0 else np.nan,
        "total_active_ref_flow": category_summary["total_active_ref_flow"].sum()
    }])

    return (
        category_summary,
        origin_all,
        origin_pooled,
        dest_all,
        dest_pooled,
        edge_selected,
        city_summary
    )


# ============================================================
# 7. Build edge GeoDataFrame
# ============================================================

def make_cbg_centroid_df(cbg_inside):
    cbg = cbg_inside.copy()

    if "GEOID" not in cbg.columns:
        raise ValueError("cbg_inside must contain GEOID column.")

    cbg["cbg_id"] = cbg["GEOID"].apply(normalize_geoid)

    cbg_proj = cbg.to_crs(CENTROID_CRS).copy()
    cent_proj = cbg_proj.geometry.centroid

    cent_wgs = gpd.GeoSeries(
        cent_proj,
        crs=CENTROID_CRS
    ).to_crs("EPSG:4326")

    out = pd.DataFrame({
        "cbg_id": cbg["cbg_id"].astype(str).values,
        "cbg_lon": cent_wgs.x.values,
        "cbg_lat": cent_wgs.y.values
    })

    return out


def make_poi_coord_df(poi_gdf):
    if "poi_id" not in poi_gdf.columns:
        if POI_ID_COL not in poi_gdf.columns:
            raise ValueError(
                f"poi_gdf must contain either 'poi_id' or {POI_ID_COL!r}."
            )
        poi_gdf = poi_gdf.copy()
        poi_gdf["poi_id"] = poi_gdf[POI_ID_COL].astype(str)

    poi = poi_gdf.copy()
    poi["poi_id"] = poi["poi_id"].astype(str)

    poi = poi.drop_duplicates(subset=["poi_id"]).copy()

    out = pd.DataFrame({
        "poi_id": poi["poi_id"].astype(str).values,
        "poi_lon": poi.geometry.x.values,
        "poi_lat": poi.geometry.y.values,
        "naics_prefix": poi["naics_prefix"].values
    })

    return out


def build_edge_gdf(edge_selected, cbg_inside, poi_gdf):
    cbg_coord = make_cbg_centroid_df(cbg_inside)
    poi_coord = make_poi_coord_df(poi_gdf)

    edges = edge_selected.copy()

    edges["cbg_id"] = edges["cbg_id"].astype(str)
    edges["poi_id"] = edges["poi_id"].astype(str)

    edges = edges.merge(
        cbg_coord,
        on="cbg_id",
        how="left"
    )

    edges = edges.merge(
        poi_coord,
        on="poi_id",
        how="left"
    )

    before = len(edges)

    edges = edges.dropna(
        subset=["cbg_lon", "cbg_lat", "poi_lon", "poi_lat"]
    ).copy()

    after = len(edges)

    if print_progress:
        print(f"[INFO] Selected edges matched to coordinates: {after:,} / {before:,}")

    if after == 0:
        raise RuntimeError(
            "No selected edge can be matched to CBG and POI coordinates. "
            "Most likely matrix columns are not the same as POI_ID_COL. "
            "Check whether flow_matrix columns correspond to placekey."
        )

    edges["geometry"] = [
        LineString([(r.cbg_lon, r.cbg_lat), (r.poi_lon, r.poi_lat)])
        for r in edges.itertuples(index=False)
    ]

    edge_gdf = gpd.GeoDataFrame(
        edges,
        crs="EPSG:4326",
        geometry="geometry"
    )

    return edge_gdf


def build_poi_node_gdf(edge_gdf):
    poi_nodes = (
        edge_gdf
        .groupby(["poi_id", "poi_code", "poi_lon", "poi_lat"], as_index=False)
        .agg(
            n_displayed_edges=("cbg_id", "size"),
            mean_delta_exposure=("delta_exposure", "mean"),
            mean_distance_km=("distance_km", "mean")
        )
    )

    poi_nodes["geometry"] = [
        Point(xy)
        for xy in zip(poi_nodes["poi_lon"], poi_nodes["poi_lat"])
    ]

    poi_nodes_gdf = gpd.GeoDataFrame(
        poi_nodes,
        crs="EPSG:4326",
        geometry="geometry"
    )

    return poi_nodes_gdf


def build_cbg_node_gdf(edge_gdf):
    cbg_nodes = (
        edge_gdf
        .groupby(["cbg_id", "cbg_lon", "cbg_lat"], as_index=False)
        .agg(
            n_displayed_edges=("poi_id", "size"),
            mean_delta_exposure=("delta_exposure", "mean"),
            mean_distance_km=("distance_km", "mean")
        )
    )

    cbg_nodes["geometry"] = [
        Point(xy)
        for xy in zip(cbg_nodes["cbg_lon"], cbg_nodes["cbg_lat"])
    ]

    cbg_nodes_gdf = gpd.GeoDataFrame(
        cbg_nodes,
        crs="EPSG:4326",
        geometry="geometry"
    )

    return cbg_nodes_gdf


from matplotlib.collections import LineCollection


def make_curved_segments(edge_gdf, curve_strength=0.36, n_points=36):
    """
    Convert CBG-POI links into inverted-U quadratic Bezier curves.

    Key change relative to the previous version:
        The curve is generated in a projected CRS, not directly in lon/lat.
        This makes the lift distance meaningful in meters and prevents the
        curves from visually collapsing into straight lines.

    Curve definition in projected coordinates:
        P0 = CBG centroid
        P2 = POI point
        P1 = midpoint shifted upward along projected y-axis

    The result is transformed back to EPSG:4326 for plotting on the existing map.
    """

    segments_by_code = {code: [] for code in poi_order}

    transformer_to_curve = Transformer.from_crs(
        "EPSG:4326",
        CURVE_CRS,
        always_xy=True
    )
    transformer_to_wgs = Transformer.from_crs(
        CURVE_CRS,
        "EPSG:4326",
        always_xy=True
    )

    t = np.linspace(0, 1, int(n_points))

    for r in edge_gdf.itertuples(index=False):
        lon0 = float(r.cbg_lon)
        lat0 = float(r.cbg_lat)
        lon2 = float(r.poi_lon)
        lat2 = float(r.poi_lat)

        # lon/lat -> projected meters
        x0, y0 = transformer_to_curve.transform(lon0, lat0)
        x2, y2 = transformer_to_curve.transform(lon2, lat2)

        dx = x2 - x0
        dy = y2 - y0
        span_m = float(np.sqrt(dx * dx + dy * dy))

        if not np.isfinite(span_m) or span_m <= 0:
            continue

        xm = 0.5 * (x0 + x2)
        ym = 0.5 * (y0 + y2)

        # 倒 U 型控制点：向地图上方抬高。
        # clip 的作用是避免短边完全看不出弯曲，同时避免长边弯得过分夸张。
        lift_m = np.clip(
            curve_strength * span_m,
            CURVE_MIN_LIFT_M,
            CURVE_MAX_LIFT_M
        )

        x1 = xm
        y1 = ym + lift_m

        xs_m = (1 - t) ** 2 * x0 + 2 * (1 - t) * t * x1 + t ** 2 * x2
        ys_m = (1 - t) ** 2 * y0 + 2 * (1 - t) * t * y1 + t ** 2 * y2

        # projected meters -> lon/lat
        xs, ys = transformer_to_wgs.transform(xs_m, ys_m)
        seg = np.column_stack([xs, ys])

        code = str(getattr(r, "poi_code"))
        if code in segments_by_code:
            segments_by_code[code].append(seg)

    return segments_by_code


def draw_curved_edges(
    ax,
    edge_gdf,
    curve_strength=0.36,
    n_points=36,
    linewidth=0.22,
    alpha=0.055
):
    """
    Draw curved network edges using LineCollection.

    注意：这里不再调用 GeoDataFrame.plot() 绘制 edge_gdf 的原始 LineString，
    因此不会再出现曲线被直线覆盖的问题。
    """

    segments_by_code = make_curved_segments(
        edge_gdf=edge_gdf,
        curve_strength=curve_strength,
        n_points=n_points
    )

    for draw_i, code in enumerate(poi_order):
        segs = segments_by_code.get(code, [])

        if len(segs) == 0:
            continue

        lc = LineCollection(
            segs,
            colors=[edge_colors.get(code, "#666666")],
            linewidths=linewidth,
            alpha=alpha,
            zorder=3 + draw_i * 0.01,
            rasterized=True
        )

        ax.add_collection(lc)


# ============================================================
# 7A. Circular 10-km inset helper
# ============================================================

def _segment_intersects_circle_mask(edge_df, center_lon, center_lat, radius_km, crs=CURVE_CRS):
    """
    Vectorized selection of edges whose straight CBG-POI segment intersects
    the radius-km circle around the fixed center. This is only used to reduce
    what is drawn in the inset; the full main map is unchanged.
    """

    if len(edge_df) == 0:
        return np.zeros(0, dtype=bool)

    transformer = Transformer.from_crs(
        "EPSG:4326",
        crs,
        always_xy=True
    )

    cx, cy = transformer.transform(float(center_lon), float(center_lat))

    x0, y0 = transformer.transform(
        edge_df["cbg_lon"].to_numpy(dtype=float),
        edge_df["cbg_lat"].to_numpy(dtype=float)
    )
    x1, y1 = transformer.transform(
        edge_df["poi_lon"].to_numpy(dtype=float),
        edge_df["poi_lat"].to_numpy(dtype=float)
    )

    x0 = np.asarray(x0, dtype=float)
    y0 = np.asarray(y0, dtype=float)
    x1 = np.asarray(x1, dtype=float)
    y1 = np.asarray(y1, dtype=float)

    vx = x1 - x0
    vy = y1 - y0
    seg_len2 = vx * vx + vy * vy

    wx = cx - x0
    wy = cy - y0

    t = np.divide(
        wx * vx + wy * vy,
        seg_len2,
        out=np.zeros_like(seg_len2, dtype=float),
        where=seg_len2 > 0
    )
    t = np.clip(t, 0.0, 1.0)

    closest_x = x0 + t * vx
    closest_y = y0 + t * vy

    dist2 = (closest_x - cx) ** 2 + (closest_y - cy) ** 2
    radius_m = float(radius_km) * 1000.0

    return dist2 <= radius_m ** 2


def _points_within_circle_mask(gdf, center_lon, center_lat, radius_km, crs=CURVE_CRS):
    """
    Vectorized point-in-radius mask for POIs or point-like GeoDataFrames.
    """

    if len(gdf) == 0:
        return np.zeros(0, dtype=bool)

    transformer = Transformer.from_crs(
        "EPSG:4326",
        crs,
        always_xy=True
    )

    cx, cy = transformer.transform(float(center_lon), float(center_lat))
    xs, ys = transformer.transform(
        gdf.geometry.x.to_numpy(dtype=float),
        gdf.geometry.y.to_numpy(dtype=float)
    )

    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)

    radius_m = float(radius_km) * 1000.0
    return ((xs - cx) ** 2 + (ys - cy) ** 2) <= radius_m ** 2


def _get_wgs84_buffer_bounds(center_lon, center_lat, radius_km, crs=CURVE_CRS):
    """
    Compute the true radius-km buffer in a projected CRS and return its
    WGS84 bounds for setting the inset extent.
    """

    center = gpd.GeoSeries(
        [Point(float(center_lon), float(center_lat))],
        crs="EPSG:4326"
    )
    buffer_wgs = center.to_crs(crs).buffer(float(radius_km) * 1000.0).to_crs("EPSG:4326")
    return buffer_wgs.total_bounds


def add_10km_circular_inset(
    fig,
    ax_map,
    cbg_inside,
    bos_msa,
    poi_gdf,
    edge_gdf
):
    """
    Add a circular 10-km zoom-in inset without modifying the main map.

    This function intentionally repeats the same layer order used in the
    original main map:
        CBG background -> all POI background -> network edges -> optional CBG nodes
        -> POI nodes -> MSA boundary.
    """

    if not DRAW_10KM_INSET:
        return None

    # --------------------------------------------------------
    # Local data selection for the inset only
    # --------------------------------------------------------

    minx, miny, maxx, maxy = _get_wgs84_buffer_bounds(
        INSET_CENTER_LON,
        INSET_CENTER_LAT,
        INSET_RADIUS_KM,
        crs=CURVE_CRS
    )

    edge_mask = _segment_intersects_circle_mask(
        edge_gdf,
        INSET_CENTER_LON,
        INSET_CENTER_LAT,
        INSET_RADIUS_KM,
        crs=CURVE_CRS
    )
    edge_local = edge_gdf.loc[edge_mask].copy()

    if INSET_MAX_EDGES is not None and len(edge_local) > int(INSET_MAX_EDGES):
        edge_local = (
            edge_local
            .sort_values("delta_exposure", ascending=False)
            .head(int(INSET_MAX_EDGES))
            .copy()
        )

    poi_mask = _points_within_circle_mask(
        poi_gdf,
        INSET_CENTER_LON,
        INSET_CENTER_LAT,
        INSET_RADIUS_KM,
        crs=CURVE_CRS
    )
    poi_local_all = poi_gdf.loc[poi_mask].copy()

    # --------------------------------------------------------
    # Locator circle on the main map.
    # This marks the selected local area on the original map.
    # --------------------------------------------------------

    cx = float(INSET_CENTER_LON)
    cy = float(INSET_CENTER_LAT)
    rx = (maxx - minx) / 2.0
    ry = (maxy - miny) / 2.0

    if DRAW_INSET_LOCATOR_ON_MAIN:
        # In lon/lat axes this appears as an ellipse, but it corresponds to
        # the true projected radius-km buffer bounds.
        theta = np.linspace(0, 2 * np.pi, 240)

        ax_map.plot(
            cx + rx * np.cos(theta),
            cy + ry * np.sin(theta),
            color=INSET_LOCATOR_COLOR,
            linewidth=INSET_LOCATOR_LINEWIDTH,
            linestyle=INSET_LOCATOR_DASH,
            alpha=INSET_LOCATOR_ALPHA,
            zorder=70
        )

    # --------------------------------------------------------
    # Create the inset axis. No title is added.
    # --------------------------------------------------------

    ax_inset = fig.add_axes(INSET_AX_BOUNDS, zorder=40)
    ax_inset.set_aspect("equal")
    ax_inset.axis("off")
    ax_inset.patch.set_visible(False)

    # ========================================================
    # Same layer order as the original main map
    # ========================================================

    cbg_inside.plot(
        ax=ax_inset,
        color="#F3F3F3",
        edgecolor="#D0D0D0",
        linewidth=1,
        zorder=1
    )

    if DRAW_ALL_POI_BACKGROUND and len(poi_local_all) > 0:
        poi_local_all.plot(
            ax=ax_inset,
            marker="o",
            markersize=INSET_ALL_POI_BACKGROUND_SIZE,
            color="#555555",
            alpha=0.06,
            edgecolor="none",
            zorder=2,
            rasterized=True
        )

    if len(edge_local) > 0:
        if DRAW_CURVED_EDGES_IN_INSET:
            draw_curved_edges(
                ax=ax_inset,
                edge_gdf=edge_local,
                curve_strength=CURVE_STRENGTH,
                n_points=CURVE_N_POINTS,
                linewidth=float(INSET_EDGE_LINEWIDTH),
                alpha=INSET_EDGE_ALPHA
            )
        else:
            for code in poi_order:
                sub = edge_local[edge_local["poi_code"] == code].copy()

                if len(sub) == 0:
                    continue

                sub.plot(
                    ax=ax_inset,
                    color=edge_colors.get(code, "#666666"),
                    linewidth=float(INSET_EDGE_LINEWIDTH),
                    alpha=INSET_EDGE_ALPHA,
                    zorder=3,
                    rasterized=True
                )

    if DRAW_CBG_NODES and len(edge_local) > 0:
        cbg_nodes_gdf = build_cbg_node_gdf(edge_local)

        if len(cbg_nodes_gdf) > 0:
            max_n = cbg_nodes_gdf["n_displayed_edges"].max()
            cbg_nodes_gdf["plot_size"] = 2.0 + 10.0 * np.sqrt(
                cbg_nodes_gdf["n_displayed_edges"] / max_n
            )

            cbg_nodes_gdf.plot(
                ax=ax_inset,
                marker="o",
                markersize=cbg_nodes_gdf["plot_size"],
                color="#2B2B2B",
                edgecolor="none",
                alpha=0.30,
                zorder=4,
                rasterized=True
            )

    if DRAW_POI_NODES and len(edge_local) > 0:
        poi_nodes_gdf = build_poi_node_gdf(edge_local)

        if len(poi_nodes_gdf) > 0:
            for code in poi_order:
                subp = poi_nodes_gdf[poi_nodes_gdf["poi_code"] == code].copy()

                if len(subp) == 0:
                    continue

                subp.plot(
                    ax=ax_inset,
                    marker="o",
                    markersize=INSET_POI_NODE_SIZE,
                    color=base_colors.get(code, "#666666"),
                    edgecolor="white",
                    linewidth=0.20,
                    alpha=0.72,
                    zorder=5,
                    rasterized=True
                )

    bos_msa.boundary.plot(
        ax=ax_inset,
        color="black",
        linewidth=0.90,
        zorder=6
    )

    ax_inset.set_xlim(minx, maxx)
    ax_inset.set_ylim(miny, maxy)

    # --------------------------------------------------------
    # Circular clipping frame. This makes the inset appear as a
    # circular cut-out while leaving the main map untouched.
    # --------------------------------------------------------

    clip_circle = Circle(
        (0.5, 0.5),
        0.5,
        transform=ax_inset.transAxes,
        facecolor="none",
        edgecolor="none"
    )

    # --------------------------------------------------------
    # Apply circular clipping to every layer already drawn
    # inside the inset. This removes everything outside the
    # circular cut-out.
    # --------------------------------------------------------

    for artist in ax_inset.collections:
        artist.set_clip_path(clip_circle)

    for artist in ax_inset.lines:
        artist.set_clip_path(clip_circle)

    for artist in ax_inset.patches:
        artist.set_clip_path(clip_circle)

    # inset 轴本身不要显示矩形背景
    ax_inset.set_facecolor("none")
    ax_inset.patch.set_alpha(0.0)

    # --------------------------------------------------------
    # Add circular frame after clipping, so the frame itself is
    # not clipped away.
    # --------------------------------------------------------

    frame_circle = Circle(
        (0.5, 0.5),
        0.5,
        transform=ax_inset.transAxes,
        facecolor="none",
        edgecolor=INSET_FRAME_COLOR,
        linewidth=INSET_FRAME_LINEWIDTH,
        zorder=100,
        clip_on=False
    )

    ax_inset.add_patch(frame_circle)

    # --------------------------------------------------------
    # Two dashed connectors from the main-map locator circle
    # to the circular inset.
    # --------------------------------------------------------

    if DRAW_INSET_CONNECTORS and DRAW_INSET_LOCATOR_ON_MAIN:
        a1 = np.deg2rad(CONNECTOR_MAIN_ANGLE_1)
        a2 = np.deg2rad(CONNECTOR_MAIN_ANGLE_2)

        main_xy_1 = (
            cx + rx * np.cos(a1),
            cy + ry * np.sin(a1)
        )
        main_xy_2 = (
            cx + rx * np.cos(a2),
            cy + ry * np.sin(a2)
        )

        con1 = ConnectionPatch(
            xyA=main_xy_1,
            coordsA=ax_map.transData,
            xyB=CONNECTOR_INSET_POINT_1,
            coordsB=ax_inset.transAxes,
            axesA=ax_map,
            axesB=ax_inset,
            color=INSET_LOCATOR_COLOR,
            linewidth=INSET_LOCATOR_LINEWIDTH,
            linestyle=INSET_LOCATOR_DASH,
            alpha=INSET_LOCATOR_ALPHA,
            zorder=90,
            clip_on=False
        )

        con2 = ConnectionPatch(
            xyA=main_xy_2,
            coordsA=ax_map.transData,
            xyB=CONNECTOR_INSET_POINT_2,
            coordsB=ax_inset.transAxes,
            axesA=ax_map,
            axesB=ax_inset,
            color=INSET_LOCATOR_COLOR,
            linewidth=INSET_LOCATOR_LINEWIDTH,
            linestyle=INSET_LOCATOR_DASH,
            alpha=INSET_LOCATOR_ALPHA,
            zorder=90,
            clip_on=False
        )

        fig.add_artist(con1)
        fig.add_artist(con2)

    return ax_inset


# ============================================================
# 8. Plot Fig. 1d network version
# ============================================================

def plot_fig1d_second_quadrant_network(
    cbg_inside,
    bos_msa,
    poi_gdf,
    edge_gdf,
    category_summary,
    city_summary,
    output_path=None
):
    """
    Main Fig. 1d network plot.

    Map:
        selected CBG centroid → POI links satisfying:
            F_ij <= 0
            D_ij <= DMAX_KM
            ΔS > 0
            ΔD <= 0

    Right-side legend:
        full category-level second-quadrant share.

    Notes on edge width:
        All network edges are forced to use the same EDGE_LINEWIDTH.
        Any remaining visual difference mainly comes from transparent line
        overlap, not from different linewidth values.
    """

    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman", "Times", "DejaVu Serif"]
    plt.rcParams["font.size"] = 10.5
    plt.rcParams["axes.linewidth"] = 0.8
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42

    # --------------------------------------------------------
    # 单图版本：不再使用右侧 bar subplot。
    # right=0.76 给地图右侧图例预留空间。
    # --------------------------------------------------------
    fig = plt.figure(figsize=(9, 10), dpi=FIG_DPI)
    ax_map = fig.add_subplot(1, 1, 1)

    # ========================================================
    # Map background
    # ========================================================

    cbg_inside.plot(
        ax=ax_map,
        color="#F3F3F3",
        edgecolor="#D0D0D0",
        linewidth=1,
        zorder=1
    )

    if DRAW_ALL_POI_BACKGROUND:
        poi_gdf.plot(
            ax=ax_map,
            marker="o",
            markersize=0.8,
            color="#555555",
            alpha=0.06,
            edgecolor="none",
            zorder=2,
            rasterized=True
        )

    # ========================================================
    # Network edges
    # ========================================================
    # 强制所有边使用同一个 linewidth。
    # 注意：如果某些位置看起来更粗/更深，主要是大量透明边叠加造成的视觉效果。
    # ========================================================

    if DRAW_CURVED_EDGES:
        draw_curved_edges(
            ax=ax_map,
            edge_gdf=edge_gdf,
            curve_strength=CURVE_STRENGTH,
            n_points=CURVE_N_POINTS,
            linewidth=float(EDGE_LINEWIDTH),
            alpha=EDGE_ALPHA
        )
    else:
        for code in poi_order:
            sub = edge_gdf[edge_gdf["poi_code"] == code].copy()

            if len(sub) == 0:
                continue

            sub.plot(
                ax=ax_map,
                color=edge_colors.get(code, "#666666"),
                linewidth=float(EDGE_LINEWIDTH),
                alpha=EDGE_ALPHA,
                zorder=3,
                rasterized=True
            )

    # ========================================================
    # Optional CBG nodes
    # ========================================================

    if DRAW_CBG_NODES:
        cbg_nodes_gdf = build_cbg_node_gdf(edge_gdf)

        if len(cbg_nodes_gdf) > 0:
            max_n = cbg_nodes_gdf["n_displayed_edges"].max()
            cbg_nodes_gdf["plot_size"] = 2.0 + 10.0 * np.sqrt(
                cbg_nodes_gdf["n_displayed_edges"] / max_n
            )

            cbg_nodes_gdf.plot(
                ax=ax_map,
                marker="o",
                markersize=cbg_nodes_gdf["plot_size"],
                color="#2B2B2B",
                edgecolor="none",
                alpha=0.30,
                zorder=4,
                rasterized=True
            )

    # ========================================================
    # POI nodes
    # ========================================================

    if DRAW_POI_NODES:
        poi_nodes_gdf = build_poi_node_gdf(edge_gdf)

        if len(poi_nodes_gdf) > 0:
            for code in poi_order:
                subp = poi_nodes_gdf[poi_nodes_gdf["poi_code"] == code].copy()

                if len(subp) == 0:
                    continue

                subp.plot(
                    ax=ax_map,
                    marker="o",
                    markersize=UNIFORM_POI_NODE_SIZE,
                    color=base_colors.get(code, "#666666"),
                    edgecolor="white",
                    linewidth=0.20,
                    alpha=0.72,
                    zorder=5,
                    rasterized=True
                )

    # ========================================================
    # MSA boundary
    # ========================================================

    bos_msa.boundary.plot(
        ax=ax_map,
        color="black",
        linewidth=0.90,
        zorder=6
    )

    minx, miny, maxx, maxy = cbg_inside.total_bounds
    dx = maxx - minx
    dy = maxy - miny

    ax_map.set_xlim(minx - dx * 0.025, maxx + dx * 0.025)
    ax_map.set_ylim(miny - dy * 0.025, maxy + dy * 0.025)

    ax_map.set_aspect("equal")
    ax_map.axis("off")

    # --------------------------------------------------------
    # Added only: 10-km circular inset in the upper-left corner.
    # This does not alter the original main-map drawing sequence above.
    # --------------------------------------------------------
    add_10km_circular_inset(
        fig=fig,
        ax_map=ax_map,
        cbg_inside=cbg_inside,
        bos_msa=bos_msa,
        poi_gdf=poi_gdf,
        edge_gdf=edge_gdf
    )


    # ========================================================
    # Right-side vertical legend with full category-level percentages
    # ========================================================
    # 参照 Fig. 1c 的 legend 写法：
    #   1) 使用 Patch 色块；
    #   2) 使用 textwrap 控制长标签换行；
    #   3) 在地图右侧预留空间；
    #   4) 每个类别后保留 full category-level second-quadrant share 数字。
    #
    # 这里使用 category_summary 的完整统计结果，而不是地图显示边的抽样结果。
    # 即使 SHOW_ALL_SECOND_QUADRANT_EDGES=False，百分比仍然是全量 unused feasible links 的统计。
    # ========================================================

    cat = category_summary.copy()
    cat["share_pct"] = cat["share_second_quadrant"] * 100
    cat["plot_order"] = cat["poi_code"].map(poi_order_rank)
    cat = cat.sort_values("plot_order", ascending=True).reset_index(drop=True)

    share_lookup = {
        str(row["poi_code"]): float(row["share_pct"])
        for _, row in cat.iterrows()
    }

    # 官方类别名；用于右侧图例换行显示。
    # 如果你想更短，可以把 label_base 改回 poi_code_to_label[code]。
    poi_code_to_full_label = {
        "713940": "Fitness and Recreational Sports Centers",
        "813110": "Religious Organizations",
        "722410": "Drinking Places (Alcoholic Beverages)",
        "712110": "Museums",
        "711310": "Promoters of Performing Arts, Sports, and Similar Events with Facilities",
        "624190": "Other Individual and Family Services"
    }

    wrap_width = 25      # 每行字符数，控制图例标签换行；想更紧凑可改为 16–18
    percent_indent = ""  # 若想让百分比缩进，可设为 "  "

    # ========================================================
    # Custom right-side legend with percentage progress bars
    # ========================================================
    # 不再使用 ax_map.legend()，而是用 fig.text + Rectangle 手动画图例。
    # 每个类别包含：
    #   1) 类别色块；
    #   2) 官方类别名；
    #   3) share_second_quadrant 百分比数字；
    #   4) 黑色背景进度条 + 对应类别颜色填充。
    #
    # 百分比口径保持你原来的计算逻辑不变：
    #   share_second_quadrant = n_second_quadrant / n_unused_feasible
    # 即：当前 POI 类别中，满足 ΔS > 0 且 ΔD <= 0 的 unused feasible
    # CBG-POI links 占该类别所有 unused feasible CBG-POI links 的比例。
    # ========================================================

    # 给右侧自定义图例预留空间。
    # 后面的 tight_layout(rect=[..., 0.80, ...]) 会进一步控制地图占用宽度。
    fig.subplots_adjust(right=0.82)

    # ---------- 手动图例整体布局参数 ----------
    # 图例整体越靠近地图：减小 legend_x，例如 0.685。
    # 图例整体越靠右：增大 legend_x，例如 0.720。
    legend_x = 0.720
    legend_y_top = 0.775
    row_gap = 0.105

    # ---------- 色块、文字、进度条参数 ----------
    swatch_size = 0.018
    swatch_x = legend_x

    text_x = legend_x + 0.030
    bar_x = legend_x + 0.030
    bar_w = 0.190
    bar_h = 0.010

    label_fontsize = 12
    pct_fontsize = 12
    title_fontsize = 12

    # ---------- 图例背景框 ----------
    # 如果不想要白色背景框，把 DRAW_LEGEND_BACKGROUND 改成 False。
    DRAW_LEGEND_BACKGROUND = False

    if DRAW_LEGEND_BACKGROUND:
        legend_bg = Rectangle(
            (legend_x - 0.018, legend_y_top - 0.665),
            0.300,
            0.720,
            transform=fig.transFigure,
            facecolor="white",
            edgecolor="white",
            linewidth=0.4,
            alpha=0.90,
            zorder=20,
            clip_on=False
        )
        fig.add_artist(legend_bg)

    # ---------- 图例标题 ----------
    fig.text(
        legend_x,
        legend_y_top + 0.048,
        "POI category share",
        transform=fig.transFigure,
        ha="left",
        va="top",
        fontsize=title_fontsize,
        fontweight="bold",
        color="#222222",
        zorder=30
    )

    # ---------- 每一类 POI 的图例行 ----------
    for i, code in enumerate(poi_order):
        if code not in share_lookup:
            continue

        y = legend_y_top - i * row_gap
        share_pct = float(share_lookup[code])
        fill_ratio = float(np.clip(share_pct / 100.0, 0.0, 1.0))

        # 色块使用浅色，进度条填充使用原始强色，便于识别。
        swatch_col = soft_color_map.get(code, base_colors.get(code, "#666666"))
        fill_col = base_colors.get(code, "#666666")

        label_base = poi_code_to_full_label.get(
            code,
            poi_code_to_label.get(code, code)
        )
        wrapped_label = textwrap.fill(label_base, width=wrap_width)

        # 类别色块
        swatch = Rectangle(
            (swatch_x, y - 0.006),
            swatch_size,
            swatch_size,
            transform=fig.transFigure,
            facecolor=swatch_col,
            edgecolor="white",
            linewidth=0.45,
            alpha=0.95,
            zorder=30,
            clip_on=False
        )
        fig.add_artist(swatch)

        # 类别名称
        fig.text(
            text_x,
            y+ 0.024,
            wrapped_label,
            transform=fig.transFigure,
            ha="left",
            va="top",
            fontsize=label_fontsize,
            color="#222222",
            linespacing=0.95,
            zorder=30
        )

        # 百分比数字
        pct_y = y -0.02
        fig.text(
            text_x,
            pct_y,
            f"{share_pct:.1f}%",
            transform=fig.transFigure,
            ha="left",
            va="top",
            fontsize=pct_fontsize,
            color="#222222",
            zorder=30
        )

        # 小长条背景：完整长度代表 100%。
        bar_y = pct_y - 0.024
        bar_bg = Rectangle(
            (bar_x, bar_y),
            bar_w,
            bar_h,
            transform=fig.transFigure,
            facecolor="white",
            edgecolor="black",
            linewidth=0.35,
            alpha=0.95,
            zorder=30,
            clip_on=False
        )
        fig.add_artist(bar_bg)

        # 对应类别颜色填充：长度 = 当前百分比 / 100。
        bar_fill = Rectangle(
            (bar_x, bar_y),
            bar_w * fill_ratio,
            bar_h,
            transform=fig.transFigure,
            facecolor=fill_col,
            edgecolor="none",
            alpha=0.95,
            zorder=31,
            clip_on=False
        )
        fig.add_artist(bar_fill)

        # 细白色外框，防止黑色条与深色背景混在一起；如果不需要可注释。
        bar_border = Rectangle(
            (bar_x, bar_y),
            bar_w,
            bar_h,
            transform=fig.transFigure,
            facecolor="none",
            edgecolor="white",
            linewidth=0.25,
            alpha=0.85,
            zorder=32,
            clip_on=False
        )
        fig.add_artist(bar_border)

    # ========================================================
    # Figure title and notes
    # ========================================================

    fig.suptitle(
        "Latent exposure capacity as feasible unused CBG-POI network links",
        fontsize=16,
        y=0.94
    )

    if SHOW_ALL_SECOND_QUADRANT_EDGES:
        note_text = (
            f"Notes: All displayed edges are unused feasible CBG–POI links satisfying "
            f"Fij = 0, Dij ≤ {DMAX_KM} km, ΔS > 0, and ΔD ≤ 0. "
            f"The map displays all such second-quadrant links; curved edges are used only for visualization. "
            f"The right-side legend reports the full category-level share across all unused feasible links. "
            f"All edges use the same linewidth; darker/thicker-looking areas reflect overlap of transparent edges."
        )
    else:
        note_text = (
            f"Notes: Every displayed edge satisfies Fij = 0, Dij ≤ {DMAX_KM} km, "
            f"ΔS > 0, and ΔD ≤ 0. "
            f"For legibility, the map displays at most {MAX_EDGES_PER_CATEGORY} selected links "
            f"per POI category using the '{EDGE_SELECT_MODE}' rule. "
            f"The right-side legend reports the full category-level share across all unused feasible links. "
            f"All edges use the same linewidth; darker/thicker-looking areas reflect overlap of transparent edges."
        )
    fig.text(
        0.025, 0.945,
        "Boston",
        ha="left",
        va="top",
        fontsize=20,
        fontweight="bold",
        color="#222222"
    )
    # fig.text(
    #     0.012,
    #     0.018,
    #     note_text,
    #     ha="left",
    #     va="bottom",
    #     fontsize=7.25,
    #     color="#333333"
    # )

    fig.tight_layout(rect=[0.00, 0.1, 0.8, 0.955])

    # if output_path is not None:
    #     fig.savefig(
    #         output_path,
    #         dpi=FIG_DPI,
    #         bbox_inches="tight",
    #         transparent=False
    #     )
    fig.savefig(os.path.join(os.getcwd(), 'figure1d.pdf'),
                format='pdf',
                dpi=300,             # 仅影响位图元素
                bbox_inches='tight',
                transparent=False,   # 关闭透明，兼容性最好
                backend='pdf') 

    return fig, (ax_map, None)

# ============================================================
# 9. Run
# ============================================================

def main():

    bos_msa, bos_union, cbg_inside = load_boston_msa_boundary_and_cbgs()

    poi_gdf = load_boston_msa_pois(bos_union)

    (
        category_summary,
        origin_all,
        origin_pooled,
        dest_all,
        dest_pooled,
        edge_selected,
        city_summary
    ) = build_fig1d_network_data()

    edge_gdf = build_edge_gdf(
        edge_selected=edge_selected,
        cbg_inside=cbg_inside,
        poi_gdf=poi_gdf
    )
    print("\n========== Displayed / plotted second-quadrant edges ==========")
    print(f"Total displayed edges: {len(edge_gdf):,}")

    print(
        edge_gdf
        .groupby("poi_code")
        .size()
        .rename("n_displayed_edges")
        .reset_index()
        .assign(
            poi_label=lambda x: x["poi_code"].map(poi_code_to_label)
        )[["poi_code", "poi_label", "n_displayed_edges"]]
    )

    fig1d_network, axes1d_network = plot_fig1d_second_quadrant_network(
        cbg_inside=cbg_inside,
        bos_msa=bos_msa,
        poi_gdf=poi_gdf,
        edge_gdf=edge_gdf,
        category_summary=category_summary,
        city_summary=city_summary,
        output_path=os.path.join(
            OUTPUT_DIR,
            "figure1d_boston_msa_second_quadrant_network_curved_ordered_patch_legend_300dpi.png"
        ) if SAVE_FIG else None
    )

    plt.show()


    # ============================================================
    # 10. Print and save summary tables
    # ============================================================

    print("\n========== Boston MSA category-level second-quadrant summary ==========")

    cat_print = category_summary[
        [
            "poi_label_one_line",
            "n_active_ref",
            "n_unused_feasible",
            "n_second_quadrant",
            "total_active_ref_flow",
            "active_weighted_exposure_ref",
            "active_weighted_distance_ref",
            "share_second_quadrant"
        ]
    ].copy()

    cat_print["share_second_quadrant_pct"] = cat_print["share_second_quadrant"] * 100
    cat_print = cat_print.drop(columns=["share_second_quadrant"])

    print(cat_print.round(3))


    print("\n========== Boston MSA pooled second-quadrant summary ==========")

    city_print = city_summary.copy()
    city_print["share_second_quadrant_pct"] = city_print["share_second_quadrant"] * 100
    city_print = city_print.drop(columns=["share_second_quadrant"])

    print(city_print.round(3))


    print("\n========== Displayed network edges preview ==========")

    edge_print = edge_selected[
        [
            "cbg_id",
            "poi_id",
            "poi_code",
            "distance_km",
            "exposure",
            "delta_exposure",
            "delta_distance",
            "selection_mode"
        ]
    ].copy()

    print(edge_print.round(4).head(20))


    print("\n========== Matched edge GeoDataFrame preview ==========")

    print(
        edge_gdf[
            [
                "cbg_id",
                "poi_id",
                "poi_code",
                "distance_km",
                "delta_exposure",
                "delta_distance"
            ]
        ].round(4).head(20)
    )


if __name__ == "__main__":
    main()
