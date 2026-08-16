# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 11:08:38 2026

@author: JZS
"""

# -*- coding: utf-8 -*-
"""
Standalone Fig.2c script (Boston MSA version with improved map background).

Purpose
-------
Draw realized newly activated CBG--POI corridors after reallocation.
Edges are colored by exposure-distance trade-off class computed against
the pre-optimization active-link baseline.

Main update in this version
---------------------------
Use a Boston-style polygon background:
    1) light filled CBG polygons inside Boston MSA
    2) a slightly darker MSA outer outline
    3) softer background points
while keeping the compact Fig.2c layout:
    single square map + upper-right donut chart + lower-left legend
"""

import os
import glob
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

try:
    import geopandas as gpd
except Exception as e:
    raise ImportError("This script requires geopandas. Please install geopandas first.") from e

try:
    from shapely import affinity
except Exception as e:
    raise ImportError("This script requires shapely. Please install shapely first.") from e


# ============================================================
# 0. USER SETTINGS
# ============================================================

PROJECT_ROOT = r"d:\mobility_poi_core_place"
MATRIX_ROOT = os.path.join(PROJECT_ROOT, "matrices_A_D_S_Distribution")

SELECTED_POI_CODE = "624190"          # Other Individual and Family Services
CITY_LABEL = "Boston"

DMAX_KM = 50
DISTANCE_SCALE = 1.0                  # if your distance matrix is meters, use 1/1000

CBG_SHP_PATHS = [
    os.path.join(PROJECT_ROOT, "tl_2021_25_bg", "tl_2021_25_bg.shp"),
    os.path.join(PROJECT_ROOT, "tl_2021_33_bg", "tl_2021_33_bg.shp"),
]
CBG_CENTROID_CSV = None
CBSA_SHP_PATH = os.path.join(PROJECT_ROOT, "tl_2021_us_cbsa", "tl_2021_us_cbsa.shp")
BOSTON_CBSAFP = "14460"
POI_METADATA_CSV = os.path.join(PROJECT_ROOT, "poi_boston_msa_all.csv")

MAX_NEW_LINKS_TO_DRAW = 1000
EDGE_RAD_POS = 0.36
EDGE_RAD_NEG = -0.32

SHOW_FIGURES = True
SAVE_FIGURES = False

# Display-only geometry scaling.
# The geographic Boston MSA is naturally north-south elongated.  Here the
# y-axis is compressed in display coordinates so the map itself fills a 1:1
# square panel, rather than merely being centered inside a square canvas.
# Set MAP_TARGET_HEIGHT_TO_WIDTH < 1.0 for a flatter map or > 1.0 for a taller map.
MAP_TARGET_HEIGHT_TO_WIDTH = 1.20
MAP_Y_SCALE_MIN = 0.42
MAP_Y_SCALE_MAX = 1.00
MAP_PAD_FRAC = 0.015

# Crop the right side of the final map panel.
# 0.00 = keep square panel; 0.10 = crop 10% from the right side.
RIGHT_CROP_FRAC = 0.12

# Keep the figure height fixed and shrink the width after right-side cropping.
FIG_HEIGHT_IN = 8.0

# Donut style: diagonal hatch fill rather than solid fill.
DONUT_HATCH = "////"
DONUT_WIDTH = 0.38
DONUT_HATCH_LINEWIDTH = 1.05


# ============================================================
# 1. Metadata
# ============================================================

income_levels = [
    "low_income_pct",
    "lower_middle_income_pct",
    "upper_middle_income_pct",
    "high_income_pct",
]

poi_code_to_full_label = {
    "624190": "Other Individual and Family Services",
    "711310": "Promoters of Performing Arts, Sports, and Similar Events with Facilities",
    "712110": "Museums",
    "713940": "Fitness and Recreational Sports Centers",
    "722410": "Drinking Places (Alcoholic Beverages)",
    "813110": "Religious Organizations",
}


# ============================================================
# 2. Basic helpers
# ============================================================

def normalize_geoid(x):
    if pd.isna(x):
        return None
    try:
        return str(int(float(x)))
    except Exception:
        return str(x).strip()


def read_matrix_csv(path, distance=False):
    df = pd.read_csv(path, header=0, index_col=0)
    df.index = df.index.astype(str).map(normalize_geoid)
    df.columns = df.columns.astype(str).str.strip()
    df = df.apply(pd.to_numeric, errors="coerce")
    if df.index.duplicated().any():
        df = df.groupby(level=0).sum()
    if distance:
        df = df * DISTANCE_SCALE
    return df


def read_hopt_pickle(path):
    with open(path, "rb") as f:
        H = pickle.load(f)
    if not isinstance(H, pd.DataFrame):
        H = pd.DataFrame(H)
    H = H.copy()
    H.index = H.index.astype(str).map(normalize_geoid)
    H.columns = H.columns.astype(str).str.strip()
    H = H.apply(pd.to_numeric, errors="coerce").fillna(0)
    if H.index.duplicated().any():
        H = H.groupby(level=0).sum()
    return H


def weighted_mean(x, w):
    x = np.asarray(x, dtype=float)
    w = np.asarray(w, dtype=float)
    mask = np.isfinite(x) & np.isfinite(w) & (w > 0)
    if mask.sum() == 0:
        return np.nan
    return float(np.sum(x[mask] * w[mask]) / np.sum(w[mask]))


def safe_div(a, b):
    if b is None or not np.isfinite(b) or b == 0:
        return np.nan
    return float(a) / float(b)


def set_nature_style():
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman", "Times", "DejaVu Serif"]
    plt.rcParams["font.size"] = 8.5
    plt.rcParams["axes.linewidth"] = 0.75
    plt.rcParams["axes.labelsize"] = 8.8
    plt.rcParams["axes.titlesize"] = 9.6
    plt.rcParams["xtick.labelsize"] = 8
    plt.rcParams["ytick.labelsize"] = 8
    plt.rcParams["legend.fontsize"] = 7.3
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42
    plt.rcParams["hatch.linewidth"] = DONUT_HATCH_LINEWIDTH


def make_bgp_cmap():
    return LinearSegmentedColormap.from_list(
        "blue_green_purple",
        ["#5E60CE", "#F6F8FC", "#2A9D8F"]
    )


# ============================================================
# 3. Locate files
# ============================================================

def find_case_dir_by_poi_code(poi_code):
    candidates = sorted(glob.glob(
        os.path.join(MATRIX_ROOT, "**", f"H_opt_df_dynamic_{poi_code}.pkl"),
        recursive=True,
    ))
    if len(candidates) == 0:
        raise FileNotFoundError(f"Cannot find H_opt_df_dynamic_{poi_code}.pkl under {MATRIX_ROOT}")
    return os.path.dirname(candidates[0]), candidates[0]


def find_income_file():
    candidates = [
        os.path.join(MATRIX_ROOT, "cbg_income_level_distribution_boston_msa.csv"),
        os.path.join(MATRIX_ROOT, "cbg_income_level_distribution_boston_core.csv"),
        os.path.join(PROJECT_ROOT, "cbg_income_level_distribution_boston_msa.csv"),
        os.path.join(PROJECT_ROOT, "cbg_income_level_distribution_boston_core.csv"),
    ]
    for p in candidates:
        if os.path.isfile(p):
            return p
    matches = glob.glob(os.path.join(PROJECT_ROOT, "**", "cbg_income_level_distribution*.csv"), recursive=True)
    if len(matches) == 0:
        raise FileNotFoundError("Cannot find cbg_income_level_distribution*.csv")
    return matches[0]


def find_poi_metadata(case_dir):
    if POI_METADATA_CSV is not None and os.path.isfile(POI_METADATA_CSV):
        return POI_METADATA_CSV
    candidates = [
        os.path.join(case_dir, "poi_metadata.csv"),
        os.path.join(case_dir, "poi_info.csv"),
        os.path.join(case_dir, "poi_df.csv"),
        os.path.join(PROJECT_ROOT, "poi_metadata.csv"),
        os.path.join(PROJECT_ROOT, "poi_info.csv"),
        os.path.join(PROJECT_ROOT, "all_pois.csv"),
    ]
    for p in candidates:
        if os.path.isfile(p):
            return p
    raise FileNotFoundError("Cannot find POI metadata CSV. Please set POI_METADATA_CSV.")


# ============================================================
# 4. Load reference data
# ============================================================

def load_income_distribution():
    income_path = find_income_file()
    print(f"[LOAD income] {income_path}")
    df = pd.read_csv(income_path)
    if "GEOID" not in df.columns:
        raise ValueError("Income file must contain GEOID column.")
    missing = [c for c in income_levels if c not in df.columns]
    if missing:
        raise ValueError(f"Income file missing columns: {missing}")

    df["GEOID_str"] = df["GEOID"].apply(normalize_geoid)
    P = df.set_index("GEOID_str")[income_levels].copy()
    P = P.apply(pd.to_numeric, errors="coerce").fillna(0)
    if P.index.duplicated().any():
        P = P.groupby(level=0).mean()
    row_sum = P.sum(axis=1).replace(0, np.nan)
    P = P.div(row_sum, axis=0).fillna(0)
    return P


def detect_lat_lon_columns(df):
    lat_candidates = ["latitude", "lat", "poi_lat", "centroid_lat", "y"]
    lon_candidates = ["longitude", "lon", "lng", "poi_lon", "centroid_lon", "x"]
    cols_lower = {c.lower(): c for c in df.columns}
    lat_col = next((cols_lower[c] for c in lat_candidates if c in cols_lower), None)
    lon_col = next((cols_lower[c] for c in lon_candidates if c in cols_lower), None)
    return lat_col, lon_col


def detect_id_columns(df):
    candidates = ["poi_id", "safegraph_place_id", "placekey", "location_name", "poi", "id"]
    cols_lower = {c.lower(): c for c in df.columns}
    return next((cols_lower[c] for c in candidates if c in cols_lower), None)


def detect_name_columns(df):
    candidates = ["location_name", "name", "poi_name", "brands", "brand", "top_category"]
    cols_lower = {c.lower(): c for c in df.columns}
    return next((cols_lower[c] for c in candidates if c in cols_lower), None)


def load_poi_coordinates(case_dir, poi_ids_needed):
    poi_meta_path = find_poi_metadata(case_dir)
    print(f"[LOAD POI metadata] {poi_meta_path}")
    df = pd.read_csv(poi_meta_path, low_memory=False)
    id_col = detect_id_columns(df)
    lat_col, lon_col = detect_lat_lon_columns(df)
    name_col = detect_name_columns(df)
    if id_col is None or lat_col is None or lon_col is None:
        raise ValueError("POI metadata CSV must contain POI id and latitude/longitude columns.")

    out = df[[id_col, lat_col, lon_col] + ([name_col] if name_col is not None else [])].copy()
    out.columns = ["poi_id", "lat", "lon"] + (["poi_name"] if name_col is not None else [])
    out["poi_id"] = out["poi_id"].astype(str).str.strip()
    out["lat"] = pd.to_numeric(out["lat"], errors="coerce")
    out["lon"] = pd.to_numeric(out["lon"], errors="coerce")
    out = out.dropna(subset=["lat", "lon"]).drop_duplicates(subset=["poi_id"])
    out = out[out["poi_id"].isin([str(x).strip() for x in poi_ids_needed])].copy()
    if "poi_name" not in out.columns:
        out["poi_name"] = out["poi_id"]
    else:
        out["poi_name"] = out["poi_name"].fillna(out["poi_id"]).astype(str)

    return gpd.GeoDataFrame(
        out,
        geometry=gpd.points_from_xy(out["lon"], out["lat"]),
        crs="EPSG:4326",
    )


def load_cbg_geometries(cbg_ids_needed=None):
    """
    Load CBG polygons from the configured state shapefiles.

    Parameters
    ----------
    cbg_ids_needed : iterable or None
        If an iterable is provided, only these CBGs are kept.
        If None, all CBGs from the configured shapefiles are loaded.

    Why this matters
    ----------------
    The map background should be drawn from the full Boston MSA CBG layer,
    not only from the model CBGs.  If the plot bounds are based only on model
    CBGs, the MSA outline and peripheral polygons can look clipped on the
    left/right edges.
    """
    if cbg_ids_needed is None:
        cbg_ids_needed = None
    else:
        cbg_ids_needed = set([normalize_geoid(x) for x in cbg_ids_needed])

    if CBG_CENTROID_CSV is not None and os.path.isfile(CBG_CENTROID_CSV):
        df = pd.read_csv(CBG_CENTROID_CSV)
        lat_col, lon_col = detect_lat_lon_columns(df)
        if "GEOID" not in df.columns or lat_col is None or lon_col is None:
            raise ValueError("CBG centroid CSV must contain GEOID and latitude/longitude columns")
        df["GEOID"] = df["GEOID"].apply(normalize_geoid)
        if cbg_ids_needed is not None:
            df = df[df["GEOID"].isin(cbg_ids_needed)].copy()
        return gpd.GeoDataFrame(
            df[["GEOID"]].copy(),
            geometry=gpd.points_from_xy(df[lon_col], df[lat_col]),
            crs="EPSG:4326",
        )

    shp_existing = [p for p in CBG_SHP_PATHS if os.path.isfile(p)]
    if len(shp_existing) == 0:
        raise FileNotFoundError("No CBG shapefile found. Please set CBG_SHP_PATHS or CBG_CENTROID_CSV.")

    gdfs = []
    for shp in shp_existing:
        print(f"[LOAD CBG shp] {shp}")
        g = gpd.read_file(shp)
        geoid_col = next((cand for cand in ["GEOID", "geoid", "GEOID20"] if cand in g.columns), None)
        if geoid_col is None:
            raise ValueError(f"Cannot find GEOID column in {shp}")
        g["GEOID"] = g[geoid_col].apply(normalize_geoid)
        if cbg_ids_needed is not None:
            g = g[g["GEOID"].isin(cbg_ids_needed)].copy()
        if len(g) > 0:
            gdfs.append(g[["GEOID", "geometry"]].copy())

    if len(gdfs) == 0:
        if cbg_ids_needed is None:
            raise ValueError("No CBG geometries were found in CBG shapefiles.")
        raise ValueError("None of the requested CBG ids were found in CBG shapefiles.")

    cbg = pd.concat(gdfs, axis=0, ignore_index=True)
    cbg = gpd.GeoDataFrame(cbg, geometry="geometry", crs=gdfs[0].crs)
    return cbg

def load_boston_msa_boundary(cbg_crs):
    if not os.path.isfile(CBSA_SHP_PATH):
        return None
    print(f"[LOAD CBSA shp] {CBSA_SHP_PATH}")
    cbsa = gpd.read_file(CBSA_SHP_PATH)
    col = "CBSAFP" if "CBSAFP" in cbsa.columns else None
    if col is None:
        return None
    msa = cbsa[cbsa[col].astype(str) == str(BOSTON_CBSAFP)].copy()
    if len(msa) == 0:
        return None
    if msa.crs != cbg_crs:
        msa = msa.to_crs(cbg_crs)
    return msa


# ============================================================
# 5. Exposure computation
# ============================================================

def compute_all_pair_unmasked_exposure(flow_df, P_df):
    F = flow_df.copy()
    F.index = F.index.astype(str).map(normalize_geoid)
    F.columns = F.columns.astype(str).str.strip()
    F = F.apply(pd.to_numeric, errors="coerce").fillna(0)

    common_cbgs = sorted(set(F.index) & set(P_df.index))
    if len(common_cbgs) == 0:
        raise ValueError("No common CBGs between flow matrix and income distribution.")
    F = F.loc[common_cbgs].copy()
    P = P_df.loc[common_cbgs, income_levels].copy()

    poi_total_flow = F.sum(axis=0)
    valid_pois = poi_total_flow[poi_total_flow > 0].index.tolist()
    if len(valid_pois) == 0:
        raise ValueError("No POI has positive total flow.")
    F = F[valid_pois].copy()

    F_values = F.values.astype(float)
    P_values = P.values.astype(float)
    poi_total_flow = F_values.sum(axis=0)
    Q_values = (F_values.T @ P_values) / poi_total_flow[:, None]
    Q_sum = Q_values.sum(axis=1, keepdims=True)
    Q_values = np.divide(Q_values, Q_sum, out=np.zeros_like(Q_values), where=Q_sum > 0)
    S_values = 1.0 - (P_values @ Q_values.T)

    S = pd.DataFrame(S_values, index=F.index, columns=F.columns)
    Q = pd.DataFrame(Q_values, index=F.columns, columns=income_levels)
    return S, F, Q


def compute_exposure_on_fixed_domain(flow_df, P_df, fixed_columns):
    F = flow_df.copy()
    F.index = F.index.astype(str).map(normalize_geoid)
    F.columns = F.columns.astype(str).str.strip()
    F = F.apply(pd.to_numeric, errors="coerce").fillna(0)

    fixed_columns = [str(c).strip() for c in fixed_columns]
    common_cbgs = sorted(set(F.index) & set(P_df.index))
    common_cols = [c for c in fixed_columns if c in F.columns]
    if len(common_cbgs) == 0 or len(common_cols) == 0:
        raise ValueError("No common CBGs or POIs in fixed exposure domain.")

    F = F.loc[common_cbgs, common_cols].copy()
    P = P_df.loc[common_cbgs, income_levels].copy()
    F_values = F.values.astype(float)
    P_values = P.values.astype(float)

    poi_total_flow = F_values.sum(axis=0)
    Q_values = np.zeros((F_values.shape[1], len(income_levels)), dtype=float)
    pos = poi_total_flow > 0
    if pos.any():
        Q_values[pos, :] = (F_values[:, pos].T @ P_values) / poi_total_flow[pos, None]
    Q_sum = Q_values.sum(axis=1, keepdims=True)
    Q_values = np.divide(Q_values, Q_sum, out=np.zeros_like(Q_values), where=Q_sum > 0)
    S_values = 1.0 - (P_values @ Q_values.T)

    S = pd.DataFrame(S_values, index=F.index, columns=F.columns)
    Q = pd.DataFrame(Q_values, index=F.columns, columns=income_levels)
    return S, Q


# ============================================================
# 6. Build evaluation domain and metrics
# ============================================================

edge_draw_order = [
    "Lower exposure, farther",
    "Lower exposure, no farther",
    "Higher exposure, farther",
    "Higher exposure, no farther",
]

def build_case_metrics():
    case_dir, h_path = find_case_dir_by_poi_code(SELECTED_POI_CODE)
    print(f"[CASE] {case_dir}")
    print(f"[HOPT] {h_path}")

    flow_path = os.path.join(case_dir, "flow_matrix.csv")
    dist_path = os.path.join(case_dir, "distance_matrix.csv")
    if not os.path.isfile(flow_path):
        raise FileNotFoundError(flow_path)
    if not os.path.isfile(dist_path):
        raise FileNotFoundError(dist_path)

    P_df = load_income_distribution()
    F_raw = read_matrix_csv(flow_path, distance=False)
    D_raw = read_matrix_csv(dist_path, distance=True)
    H_opt = read_hopt_pickle(h_path)

    S0_full, F_income, Q0_full = compute_all_pair_unmasked_exposure(F_raw, P_df)

    common_rows = sorted(set(F_income.index) & set(D_raw.index) & set(S0_full.index))
    common_cols = sorted(set(F_income.columns) & set(D_raw.columns) & set(S0_full.columns))
    F_full = F_income.loc[common_rows, common_cols].copy()
    D_full = D_raw.loc[common_rows, common_cols].copy()
    S0_full = S0_full.loc[common_rows, common_cols].copy()

    H_eval = F_full.copy().astype(float)
    rows_h = sorted(set(H_opt.index) & set(H_eval.index) & set(P_df.index))
    cols_h = sorted(set(H_opt.columns) & set(H_eval.columns))
    if len(rows_h) == 0 or len(cols_h) == 0:
        raise ValueError("No common rows/columns between H_opt and baseline flow matrix.")
    H_eval.loc[rows_h, cols_h] = H_opt.loc[rows_h, cols_h]

    F_dom = F_full.loc[rows_h, cols_h].copy()
    D_dom = D_full.loc[rows_h, cols_h].copy()
    S0_dom = S0_full.loc[rows_h, cols_h].copy()
    H_dom = H_eval.loc[rows_h, cols_h].copy()

    Fv_full = F_full.values.astype(float)
    Dv_full = D_full.values.astype(float)
    S0v_full = S0_full.values.astype(float)
    Hv_full = H_eval.values.astype(float)

    valid_full = np.isfinite(Fv_full) & np.isfinite(Dv_full) & np.isfinite(S0v_full) & np.isfinite(Hv_full)
    distance_feasible_full = valid_full & (Dv_full >= 0) & (Dv_full <= DMAX_KM)
    active_ref_full = distance_feasible_full & (Fv_full > 0)
    unused_feasible_full = distance_feasible_full & (Fv_full <= 0)
    if active_ref_full.sum() == 0:
        raise ValueError("No active feasible reference links in the full diagnostic domain.")

    active_w_s = weighted_mean(S0v_full[active_ref_full], Fv_full[active_ref_full])
    active_w_d = weighted_mean(Dv_full[active_ref_full], Fv_full[active_ref_full])

    delta_s_full = S0v_full - active_w_s
    delta_d_full = Dv_full - active_w_d
    fig1d_opportunity_full = unused_feasible_full & (delta_s_full > 0) & (delta_d_full <= 0)
    newly_activated_full = valid_full & (Fv_full <= 0) & (Hv_full > 0)
    newly_activated_from_fig1d = newly_activated_full & fig1d_opportunity_full

    S1_full, Q1_full = compute_exposure_on_fixed_domain(H_eval, P_df, F_full.columns)
    S1_dom = S1_full.loc[rows_h, cols_h].copy()

    F2v = F_dom.values.astype(float)
    D2v = D_dom.values.astype(float)
    H2v = H_dom.values.astype(float)
    S0_2v = S0_dom.values.astype(float)
    S1_2v = S1_dom.values.astype(float)

    baseline_total_flow = float(np.nansum(F2v))
    optimized_total_flow = float(np.nansum(H2v))
    reassigned_visit_equiv = 0.5 * float(np.nansum(np.abs(H2v - F2v)))

    distance_before = float(np.nansum(F2v * D2v))
    distance_after = float(np.nansum(H2v * D2v))
    distance_change_pct = safe_div(distance_after - distance_before, distance_before) * 100

    spse_before = float(np.nansum(S0_2v[F2v > 0]))
    spse_after = float(np.nansum(S1_2v[H2v > 0]))
    spse_change_pct = safe_div(spse_after - spse_before, spse_before) * 100

    fw_before = safe_div(np.nansum(F2v * S0_2v), baseline_total_flow)
    fw_after = safe_div(np.nansum(H2v * S1_2v), optimized_total_flow)
    fw_change_pct = safe_div(fw_after - fw_before, fw_before) * 100

    full_row_ids = F_full.index.tolist()
    full_col_ids = F_full.columns.tolist()
    records = []

    def classify_link(ds, dd):
        if ds > 0 and dd <= 0:
            return "Higher exposure, no farther"
        if ds > 0 and dd > 0:
            return "Higher exposure, farther"
        if ds <= 0 and dd <= 0:
            return "Lower exposure, no farther"
        return "Lower exposure, farther"

    for i, geoid in enumerate(full_row_ids):
        for j, poi in enumerate(full_col_ids):
            if not newly_activated_full[i, j]:
                continue
            ds = float(delta_s_full[i, j])
            dd = float(delta_d_full[i, j])
            records.append({
                "GEOID": geoid,
                "poi_id": poi,
                "baseline_flow": float(Fv_full[i, j]),
                "optimized_flow": float(Hv_full[i, j]),
                "new_flow": float(Hv_full[i, j]),
                "distance_km": float(Dv_full[i, j]),
                "S0": float(S0v_full[i, j]),
                "delta_S_against_active_ref": ds,
                "delta_D_against_active_ref": dd,
                "link_tradeoff": classify_link(ds, dd),
                "is_fig1d_opportunity": bool(fig1d_opportunity_full[i, j]),
                "is_opportunity": bool(fig1d_opportunity_full[i, j]),
            })
    new_links_df = pd.DataFrame(records)

    n_unused_feasible_full = int(unused_feasible_full.sum())
    n_fig1d_opportunity_full = int(fig1d_opportunity_full.sum())
    n_new_links = int(newly_activated_full.sum())
    n_new_links_from_fig1d = int(newly_activated_from_fig1d.sum())
    new_flow_total = float(np.nansum(Hv_full[newly_activated_full]))
    new_flow_from_fig1d = float(np.nansum(Hv_full[newly_activated_from_fig1d]))

    summary = {
        "poi_code": SELECTED_POI_CODE,
        "poi_full_label": poi_code_to_full_label.get(SELECTED_POI_CODE, SELECTED_POI_CODE),
        "case_dir": case_dir,
        "diagnostic_domain": "full_baseline_matrix_after_alignment",
        "n_unused_feasible": n_unused_feasible_full,
        "n_opportunity": n_fig1d_opportunity_full,
        "opportunity_share_pct": safe_div(n_fig1d_opportunity_full, n_unused_feasible_full) * 100,
        "n_new_links": n_new_links,
        "n_new_links_from_opp": n_new_links_from_fig1d,
        "eta_link_pct": safe_div(n_new_links_from_fig1d, n_new_links) * 100,
        "new_flow_total": new_flow_total,
        "new_flow_from_opp": new_flow_from_fig1d,
        "eta_flow_pct": safe_div(new_flow_from_fig1d, new_flow_total) * 100,
        "n_model_cbgs": len(rows_h),
        "n_model_pois": len(cols_h),
        "active_weighted_exposure_ref_full": active_w_s,
        "active_weighted_distance_ref_full": active_w_d,
        "distance_change_pct": distance_change_pct,
        "spse_change_pct": spse_change_pct,
        "fw_change_pct": fw_change_pct,
        "reassigned_visit_equiv": reassigned_visit_equiv,
        "baseline_total_flow": baseline_total_flow,
        "optimized_total_flow": optimized_total_flow,
        "upper_left_outcome": bool((distance_change_pct < 0) and (spse_change_pct > 0)),
        "fw_positive": bool(fw_change_pct > 0),
    }

    tradeoff_order = [
        "Higher exposure, no farther",
        "Higher exposure, farther",
        "Lower exposure, no farther",
        "Lower exposure, farther",
    ]
    quad_rows = []
    denom = float(new_links_df["new_flow"].sum()) if len(new_links_df) > 0 else 0.0
    for label in tradeoff_order:
        sub = new_links_df[new_links_df["link_tradeoff"] == label] if len(new_links_df) else pd.DataFrame()
        val = float(sub["new_flow"].sum()) if len(sub) > 0 else 0.0
        quad_rows.append({
            "quadrant": label,
            "new_flow": val,
            "share_pct": safe_div(val, denom) * 100,
            "n_links": int(len(sub)),
        })
    quadrant_df = pd.DataFrame(quad_rows)

    poi_base_total = F_dom.sum(axis=0)
    poi_opt_total = H_dom.sum(axis=0)
    Q0 = Q0_full.loc[cols_h, income_levels].copy()
    Q1 = Q1_full.loc[cols_h, income_levels].copy()

    def income_entropy(row_values, normalize=True, eps=1e-12):
        p = np.asarray(row_values, dtype=float)
        p = np.where(np.isfinite(p), p, 0.0)
        p = np.maximum(p, 0.0)
        s = p.sum()
        if s <= 0:
            return np.nan
        p = p / s
        p_pos = p[p > eps]
        H = -np.sum(p_pos * np.log(p_pos))
        if normalize and len(p) > 1:
            H = H / np.log(len(p))
        return float(H)

    poi_rows = []
    diff = H_dom - F_dom
    for poi in cols_h:
        base_v = float(poi_base_total.loc[poi])
        opt_v = float(poi_opt_total.loc[poi])
        diff_col = diff[poi]
        total_changed_flow = float(np.abs(diff_col).sum())

        H0 = income_entropy(Q0.loc[poi, income_levels].values)
        H1 = income_entropy(Q1.loc[poi, income_levels].values)

        sub = new_links_df[new_links_df["poi_id"] == poi].copy() if len(new_links_df) else pd.DataFrame()
        new_flow_received = float(sub["new_flow"].sum()) if len(sub) else 0.0

        poi_rows.append({
            "poi_id": poi,
            "baseline_total_visits": base_v,
            "optimized_total_visits": opt_v,
            "total_changed_flow": total_changed_flow,
            "entropy_before": H0,
            "entropy_after": H1,
            "new_flow_received": new_flow_received,
        })
    poi_metrics = pd.DataFrame(poi_rows)

    origin_rows = []
    for geoid in rows_h:
        row_diff = diff.loc[geoid]
        shifted = 0.5 * float(np.abs(row_diff).sum())
        origin_rows.append({
            "GEOID": geoid,
            "shifted_flow": shifted,
        })
    origin_metrics = pd.DataFrame(origin_rows)

    return {
        "summary": summary,
        "quadrant_df": quadrant_df,
        "new_links_df": new_links_df,
        "poi_metrics": poi_metrics,
        "origin_metrics": origin_metrics,
        "cbg_ids": rows_h,
        "poi_ids": cols_h,
        "case_dir": case_dir,
    }


# ============================================================
# 7. Spatial plotting data
# ============================================================

def prepare_spatial_layers(case_output):
    """
    Prepare spatial layers for Fig.2c.

    Important change:
        cbg_poly is now the full Boston MSA CBG background, while cbg_cent is
        still restricted to the model CBGs used for optimized corridors.

    This prevents the map from being clipped because of using only model-CBG
    bounds for the background and axis limits.
    """
    # 1) Full CBG layer for map background and plot bounds.
    cbg_all = load_cbg_geometries(cbg_ids_needed=None)

    # 2) Boston MSA boundary in the same CRS as the CBG shapefile.
    msa = load_boston_msa_boundary(cbg_all.crs)

    # 3) Clip full CBG layer to Boston MSA.  Use clip rather than intersects
    #    so the background follows the MSA boundary instead of retaining parts
    #    outside the MSA.
    if msa is not None and len(msa) > 0:
        try:
            cbg_bg = gpd.clip(cbg_all, msa)
        except Exception:
            cbg_bg = cbg_all.copy()
            cbg_bg["geometry"] = cbg_bg.geometry.intersection(msa.unary_union)
            cbg_bg = cbg_bg[cbg_bg.geometry.notna() & (~cbg_bg.geometry.is_empty)].copy()
    else:
        cbg_bg = cbg_all.copy()

    if len(cbg_bg) == 0:
        raise ValueError("No CBG polygons remain after clipping to Boston MSA.")

    # 4) Model CBGs only for origin centroids.
    model_ids = set([normalize_geoid(x) for x in case_output["cbg_ids"]])
    cbg_model = cbg_bg[cbg_bg["GEOID"].isin(model_ids)].copy()

    # Fallback: if clipping removed all model CBGs because of an ID/state mismatch,
    # reload only the model CBGs so the corridor map can still be drawn.
    if len(cbg_model) == 0:
        cbg_model = load_cbg_geometries(case_output["cbg_ids"])

    # 5) POI layer.
    poi = load_poi_coordinates(case_output["case_dir"], case_output["poi_ids"])

    # 6) Project all layers to Web Mercator for plotting.
    if cbg_bg.crs is None:
        cbg_bg = cbg_bg.set_crs("EPSG:4326")
    if cbg_model.crs is None:
        cbg_model = cbg_model.set_crs(cbg_bg.crs)
    if poi.crs is None:
        poi = poi.set_crs("EPSG:4326")

    cbg_bg = cbg_bg.to_crs(3857)
    cbg_model = cbg_model.to_crs(3857)
    poi = poi.to_crs(3857)

    if msa is not None and len(msa) > 0:
        msa = msa.to_crs(3857)

    # 7) Centroids only for model CBGs.
    if cbg_model.geom_type.iloc[0].lower() == "point":
        cbg_cent = cbg_model.copy()
    else:
        cbg_cent = cbg_model[["GEOID"]].copy()
        cbg_cent["geometry"] = cbg_model.geometry.centroid
        cbg_cent = gpd.GeoDataFrame(cbg_cent, geometry="geometry", crs=cbg_model.crs)

    return cbg_bg, cbg_cent, poi, msa

def draw_curved_edge(ax, x0, y0, x1, y1, color, lw, alpha=0.5, rad=0.15, zorder=2):
    patch = FancyArrowPatch(
        (x0, y0), (x1, y1),
        arrowstyle='-',
        connectionstyle=f"arc3,rad={rad}",
        linewidth=lw,
        color=color,
        alpha=alpha,
        zorder=zorder,
        capstyle='round',
        joinstyle='round',
    )
    ax.add_patch(patch)


# ============================================================
# 7b. Display-only aspect compression
# ============================================================

def _valid_total_bounds(gdf):
    """Return finite total bounds for a GeoDataFrame, or None."""
    if gdf is None or len(gdf) == 0:
        return None
    b = np.asarray(gdf.total_bounds, dtype=float)
    if b.shape[0] != 4 or not np.all(np.isfinite(b)):
        return None
    if b[2] <= b[0] or b[3] <= b[1]:
        return None
    return b


def compress_layers_to_square_display(cbg_poly, cbg_cent, poi_gdf, msa_gdf=None):
    """
    Compress the map vertically in display coordinates so the final map panel
    is visually close to 1:1 without merely centering a tall map in a square.

    This is a cartographic display transform only. It does not alter any
    distance, exposure, or optimization calculation; all analytical quantities
    are computed before this plotting step.
    """
    bounds = []
    for g in [msa_gdf, cbg_poly, cbg_cent, poi_gdf]:
        b = _valid_total_bounds(g)
        if b is not None:
            bounds.append(b)

    if len(bounds) == 0:
        return cbg_poly, cbg_cent, poi_gdf, msa_gdf, 1.0

    bounds = np.vstack(bounds)
    xmin, ymin = np.nanmin(bounds[:, 0]), np.nanmin(bounds[:, 1])
    xmax, ymax = np.nanmax(bounds[:, 2]), np.nanmax(bounds[:, 3])
    raw_xspan = xmax - xmin
    raw_yspan = ymax - ymin

    if raw_xspan <= 0 or raw_yspan <= 0:
        y_scale = 1.0
    else:
        y_scale = (raw_xspan / raw_yspan) * MAP_TARGET_HEIGHT_TO_WIDTH
        y_scale = float(np.clip(y_scale, MAP_Y_SCALE_MIN, MAP_Y_SCALE_MAX))

    origin = ((xmin + xmax) / 2.0, (ymin + ymax) / 2.0)

    def _scale_gdf(gdf):
        if gdf is None:
            return None
        if len(gdf) == 0:
            return gdf.copy()
        out = gdf.copy()
        out["geometry"] = out.geometry.apply(
            lambda geom: affinity.scale(
                geom,
                xfact=1.0,
                yfact=y_scale,
                origin=origin,
            ) if geom is not None and not geom.is_empty else geom
        )
        return out

    return (
        _scale_gdf(cbg_poly),
        _scale_gdf(cbg_cent),
        _scale_gdf(poi_gdf),
        _scale_gdf(msa_gdf),
        y_scale,
    )

# ============================================================
# 8. Plot Fig.2c: realized corridors
# ============================================================

def plot_fig2c_realized_corridors(case_output, cbg_poly, cbg_cent, poi_gdf, msa_gdf=None):
    set_nature_style()

    summary = case_output["summary"]
    quadrant_df = case_output["quadrant_df"].copy()
    new_links = case_output["new_links_df"].copy()
    origins = case_output["origin_metrics"].copy()
    pois = case_output["poi_metrics"].copy()

    # -----------------------------
    # Colors
    # -----------------------------
    c_origin = "#3498DB"      # active origins: blue points
    c_poi = "#6E5AA8"         # modified POIs: purple points

    # map background colors
    c_ocean = "whitesmoke"
    c_cbg_fill = "whitesmoke"
    c_cbg_edge = "#D2D2D2"
    c_msa_edge = "#AFAFAF"
    c_text = "#243447"

    # Trade-off colors are shared by map corridors, donut chart, and legend.
    # Higher exposure classes use purple; lower exposure classes use green.
    tradeoff_colors = {
        "Higher exposure, no farther": "#6A51A3",   # purple
        "Higher exposure, farther": "#C7C2E0",      # light purple
        "Lower exposure, no farther": "#1B9E77",    # green
        "Lower exposure, farther": "#B8E3B2",       # light green
    }

    tradeoff_alpha = {
        "Higher exposure, no farther": 1,
        "Higher exposure, farther": 1,
        "Lower exposure, no farther": 1,
        "Lower exposure, farther": 1,
    }

    compact_labels = {
        "Higher exposure, no farther": "Higher exp., no farther",
        "Higher exposure, farther": "Higher exp., farther",
        "Lower exposure, no farther": "Lower exp., no farther",
        "Lower exposure, farther": "Lower exp., farther",
    }

    # Compress the displayed map geometry vertically so the map itself, not only
    # the canvas, becomes close to 1:1.
    cbg_poly, cbg_cent, poi_gdf, msa_gdf, map_y_scale = compress_layers_to_square_display(
        cbg_poly, cbg_cent, poi_gdf, msa_gdf
    )
    print(f"[DISPLAY] map y-axis compression factor = {map_y_scale:.3f}")

    # -----------------------------
    # Link sampling and coordinates
    # -----------------------------
    if len(new_links) > 0:
        new_links = (
            new_links
            .sort_values(["new_flow", "is_fig1d_opportunity"], ascending=[False, False])
            .head(MAX_NEW_LINKS_TO_DRAW)
            .copy()
        )

    cbg_cent_xy = cbg_cent.copy()
    cbg_cent_xy["x"] = cbg_cent_xy.geometry.x
    cbg_cent_xy["y"] = cbg_cent_xy.geometry.y

    poi_xy = poi_gdf.copy()
    poi_xy["x"] = poi_xy.geometry.x
    poi_xy["y"] = poi_xy.geometry.y

    if len(new_links) > 0:
        new_links = new_links.merge(cbg_cent_xy[["GEOID", "x", "y"]], on="GEOID", how="left")
        new_links = new_links.rename(columns={"x": "x0", "y": "y0"})
        new_links = new_links.merge(poi_xy[["poi_id", "x", "y"]], on="poi_id", how="left")
        new_links = new_links.rename(columns={"x": "x1", "y": "y1"})

    origin_plot = cbg_cent_xy.merge(origins, on="GEOID", how="left").fillna(0)
    poi_plot = poi_xy.merge(pois, on="poi_id", how="left").fillna(0)
    modified_poi = poi_plot[poi_plot["total_changed_flow"] > 0].copy()

    # --------------------------------------------------------
    # Plot bounds
    # --------------------------------------------------------
    bounds = []

    if msa_gdf is not None and len(msa_gdf) > 0:
        bounds.append(msa_gdf.total_bounds)

    if len(cbg_poly) > 0:
        bounds.append(cbg_poly.total_bounds)

    if len(origin_plot) > 0:
        bounds.append(np.array([
            origin_plot["x"].min(), origin_plot["y"].min(),
            origin_plot["x"].max(), origin_plot["y"].max()
        ], dtype=float))

    if len(poi_plot) > 0:
        bounds.append(np.array([
            poi_plot["x"].min(), poi_plot["y"].min(),
            poi_plot["x"].max(), poi_plot["y"].max()
        ], dtype=float))

    if len(new_links) > 0 and {"x0", "y0", "x1", "y1"}.issubset(new_links.columns):
        xy = new_links[["x0", "y0", "x1", "y1"]].replace([np.inf, -np.inf], np.nan)
        if xy.notna().any().any():
            bounds.append(np.array([
                np.nanmin([xy["x0"].min(), xy["x1"].min()]),
                np.nanmin([xy["y0"].min(), xy["y1"].min()]),
                np.nanmax([xy["x0"].max(), xy["x1"].max()]),
                np.nanmax([xy["y0"].max(), xy["y1"].max()]),
            ], dtype=float))

    if len(bounds) == 0:
        raise ValueError("Cannot determine map bounds.")

    bounds = np.vstack(bounds)
    xmin, ymin = np.nanmin(bounds[:, 0]), np.nanmin(bounds[:, 1])
    xmax, ymax = np.nanmax(bounds[:, 2]), np.nanmax(bounds[:, 3])

    side = max(xmax - xmin, ymax - ymin)
    pad = side * MAP_PAD_FRAC

    xmin_p, xmax_p = xmin - pad, xmax + pad
    ymin_p, ymax_p = ymin - pad, ymax + pad

    # Keep a square analytical extent first, then crop only the right side
    # for the final displayed panel. This removes right-side blank area without
    # distorting the map geometry.
    xspan_p = xmax_p - xmin_p
    yspan_p = ymax_p - ymin_p
    side_p = max(xspan_p, yspan_p)
    xmid_p = 0.5 * (xmin_p + xmax_p)
    ymid_p = 0.5 * (ymin_p + ymax_p)
    
    xmin_sq, xmax_sq = xmid_p - side_p / 2.0, xmid_p + side_p / 2.0
    ymin_sq, ymax_sq = ymid_p - side_p / 2.0, ymid_p + side_p / 2.0
    
    # Crop the right side of the displayed x-range.
    xspan_sq = xmax_sq - xmin_sq

    
    xmin_plot = xmin_sq
    xmax_plot = xmax_sq - RIGHT_CROP_FRAC * xspan_sq
    ymin_plot = ymin_sq
    ymax_plot = ymax_sq
    
    # Match figure width to the cropped data aspect ratio.
    # This makes the output no longer square and avoids unused right-side canvas.
    fig_width_in = FIG_HEIGHT_IN * ((xmax_plot - xmin_plot) / (ymax_plot - ymin_plot))
    
    # -----------------------------
    # Figure layout: cropped rectangular panel
    # -----------------------------
    fig = plt.figure(figsize=(fig_width_in, FIG_HEIGHT_IN), dpi=300)
    fig.patch.set_facecolor("white")

    # A single square map canvas. Donut and legend are inset inside this map.
    ax_map = fig.add_axes([0.00, 0.00, 1, 1])

    # Donut chart: upper-right corner inside the map.
    ax_donut = ax_map.inset_axes([0.020, 0.700, 0.260, 0.260], transform=ax_map.transAxes)
    ax_donut.set_facecolor("none")

    # Legend: lower-left corner inside the map.
    ax_leg = ax_map.inset_axes([0.03, 0.025, 0.500, 0.285], transform=ax_map.transAxes)
    ax_leg.axis("off")
    ax_leg.set_facecolor("none")

    # -----------------------------
    # Map background
    # -----------------------------
    ax_map.set_facecolor(c_ocean)

    # filled CBG polygons
    if len(cbg_poly) > 0 and cbg_poly.geom_type.iloc[0].lower() != "point":
        cbg_poly.plot(
            ax=ax_map,
            facecolor=c_cbg_fill,
            edgecolor=c_cbg_edge,
            linewidth=0.42,
            zorder=0
        )

    # stronger outer MSA outline
    if msa_gdf is not None and len(msa_gdf) > 0:
        msa_gdf.boundary.plot(
            ax=ax_map,
            color=c_msa_edge,
            linewidth=0.95,
            zorder=0.5
        )

    # Soft background POI points
    if len(poi_plot) > 0:
        ax_map.scatter(
            poi_plot["x"], poi_plot["y"],
            s=4.0, color="#C9C9C9", alpha=0.20, zorder=1
        )

    # corridors from CBG centroids to POIs
    # Draw lower-priority classes first and draw the 42% class last.
    if len(new_links) > 0:
        max_flow = max(1.0, new_links["new_flow"].max())
        
        edge_draw_order = [
            "Lower exposure, farther",
            "Lower exposure, no farther",
            "Higher exposure, farther",
            "Higher exposure, no farther",   # 42.4% class; draw last
            ]
        
        edge_zorder = {
            "Lower exposure, farther": 2.10,
            "Lower exposure, no farther": 2.20,
            "Higher exposure, farther": 2.30,
            "Higher exposure, no farther": 2.80,
            }
        
        for klass in edge_draw_order:
            sub_links = new_links[new_links["link_tradeoff"] == klass].copy()
            
            for local_idx, (_, r) in enumerate(sub_links.iterrows()):
                if pd.isna(r.get("x0")) or pd.isna(r.get("x1")):
                    continue
                
                color = tradeoff_colors.get(klass, "#BFC7D5")
                alpha = tradeoff_alpha.get(klass, 0.60)
                
                # 仍然按 new_flow 调整线宽；如果也想统一线宽，见下面第 6 点。
                lw = 0.42 + 2.20 * np.sqrt(r["new_flow"] / max_flow)
                
                # 正负弧度交替，减少重叠
                rad = EDGE_RAD_POS if (local_idx % 2 == 0) else EDGE_RAD_NEG
                
                draw_curved_edge(
                    ax_map,
                    r["x0"], r["y0"], r["x1"], r["y1"],
                    color=color,
                    lw=lw,
                    alpha=alpha,
                    rad=rad,
                    zorder=edge_zorder.get(klass, 2.5)
                    )

    # active origins: blue scatter points
    active_origins = origin_plot[origin_plot["shifted_flow"] > 0].copy()
    if len(active_origins) > 0:
        max_shift = max(1.0, active_origins["shifted_flow"].max())
        s = 16
        ax_map.scatter(
            active_origins["x"], active_origins["y"],
            s=s,
            facecolor=c_origin,
            edgecolor="white",
            linewidth=0.30,
            alpha=0.86,
            zorder=3.4
        )

    # modified POIs
    if len(modified_poi) > 0:
        ax_map.scatter(
            modified_poi["x"], modified_poi["y"],
            s=28,
            marker="o",
            facecolor=c_poi,
            edgecolor="none",
            linewidth=0.0,
            alpha=0.96,
            zorder=4
        )
    MAP_VIEW_X_SHIFT_FRAC = 0.04   # 正数：地图内容向左；负数：地图内容向右
    MAP_VIEW_Y_SHIFT_FRAC = 0.00   # 正数：地图内容向下；负数：地图内容向上
    
    x_shift = (xmax_plot - xmin_plot) * MAP_VIEW_X_SHIFT_FRAC
    y_shift = (ymax_plot - ymin_plot) * MAP_VIEW_Y_SHIFT_FRAC
    
    ax_map.set_xlim(xmin_plot + x_shift, xmax_plot + x_shift)
    ax_map.set_ylim(ymin_plot + y_shift, ymax_plot + y_shift)
    
    # ax_map.set_xlim(xmin_sq, xmax_sq)
    # ax_map.set_ylim(ymin_sq, ymax_sq)
    ax_map.set_aspect("equal", adjustable="box")
    ax_map.set_anchor("C")
    ax_map.axis("off")

    ax_map.text(
        0.01, 0.98, CITY_LABEL,
        transform=ax_map.transAxes,
        ha="left", va="top",
        fontsize=18.5, fontweight="bold", color=c_text
    )

    # -----------------------------
    # Donut chart: new-flow composition
    # -----------------------------
    qd = quadrant_df.set_index("quadrant")
    donut_labels = list(tradeoff_colors.keys())
    donut_values = [
        float(qd.loc[label, "share_pct"]) if label in qd.index and np.isfinite(qd.loc[label, "share_pct"]) else 0.0
        for label in donut_labels
    ]
    donut_colors = [tradeoff_colors[label] for label in donut_labels]

    # If all values are zero, draw a neutral empty donut instead of failing.
    if np.nansum(donut_values) <= 0:
        donut_values = [1.0]
        donut_colors = ["#BDBDBD"]

    def _autopct(pct):
        return f"{pct:.1f}" if pct >= 7 else ""

    # Hatched donut: white ring sectors with color-specific diagonal hatching.
    # In Matplotlib, hatch color follows patch edgecolor, so each sector is
    # drawn with its corresponding trade-off color as the hatch color.
    wedges, _, autotexts = ax_donut.pie(
        donut_values,
        colors=["white"] * len(donut_values),
        startangle=90,
        counterclock=False,
        autopct=_autopct,
        pctdistance=0.78,
        textprops={
            "fontsize": 12,
            "color": "#1F2937",
        },
        wedgeprops={
            "width": DONUT_WIDTH,
            "linewidth": 0.95,
            "antialiased": True,
        },
    )

    for w, color in zip(wedges, donut_colors):
        w.set_facecolor("white")
        w.set_edgecolor(color)
        w.set_hatch(DONUT_HATCH)
        w.set_linewidth(0.95)
        w.set_alpha(0.98)

    for t in autotexts:
        t.set_fontsize(12)
        t.set_color("#1F2937")

    ax_donut.text(
        0.0, 0.0,
        "New-flow\ncomposition",
        ha="center", va="center",
        fontsize=12,
        color=c_text
    )
    ax_donut.set_aspect("equal")
    ax_donut.axis("off")

    # -----------------------------
    # Legend: lower-left corner
    # -----------------------------
    legend_handles = [
        Line2D([0], [0], color=tradeoff_colors[k], lw=2.6, alpha=0.88, label=compact_labels[k])
        for k in tradeoff_colors
    ]
    legend_handles += [
        Line2D([0], [0], marker='o', linestyle='none',
               markerfacecolor=c_origin, markeredgecolor="white",
               markersize=7.0, label="CBG"),
        Line2D([0], [0], marker='o', linestyle='none',
               markerfacecolor=c_poi, markeredgecolor="none",
               markersize=7.0, label="POI"),
    ]

    ax_leg.legend(
        handles=legend_handles,
        loc="lower left",
        frameon=True,
        facecolor="white",
        edgecolor="none",
        framealpha=0.82,
        bbox_to_anchor=(0.00, 0.00),
        handlelength=1.30,
        handletextpad=0.50,
        labelspacing=0.58,
        fontsize=14,
        ncol=1,
        borderaxespad=0.0,
        borderpad=0.55
    )
    fig.savefig('figure2c.pdf',
                format='pdf',
                dpi=300,             # 仅影响位图元素
                bbox_inches='tight',
                transparent=False,   # 关闭透明，兼容性最好
                backend='pdf')

    if SHOW_FIGURES:
        plt.show()
    else:
        plt.close(fig)

    return fig

# ============================================================
# 9. Main
# ============================================================

def main():
    case_output = build_case_metrics()

    print("\n========== FIG.2C CASE SUMMARY ==========")
    for k, v in case_output["summary"].items():
        print(f"{k}: {v}")

    print("\n========== FIG.2C NEW-FLOW TRADE-OFF DECOMPOSITION ==========")
    print(case_output["quadrant_df"].round(3).to_string(index=False))

    cbg_poly, cbg_cent, poi_gdf, msa_gdf = prepare_spatial_layers(case_output)
    plot_fig2c_realized_corridors(case_output, cbg_poly, cbg_cent, poi_gdf, msa_gdf)


if __name__ == "__main__":
    main()