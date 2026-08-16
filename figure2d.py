# -*- coding: utf-8 -*-
"""
Created on Sat Jun 27 17:05:45 2026

@author: JZS
"""

# -*- coding: utf-8 -*-
"""
Standalone Fig.2d: destination-level SPSE contribution scatter.

Purpose
-------
Draw one integrated Fig.2d panel showing destination-level reallocation responses.

Each point = one POI.

x-axis:
    Δ POI visits
    = sum_i H*_ij - sum_i F_ij

y-axis:
    Δ structural potential social exposure at POI
    = sum_i 1(H*_ij > 0) S*_ij - sum_i 1(F_ij > 0) S0_ij

point size:
    absolute POI visit change
    = |sum_i H*_ij - sum_i F_ij|

point color:
    Δ weighted exposure contribution at POI
    = sum_i H*_ij S*_ij - sum_i F_ij S0_ij

point marker:
    baseline-flow POI tier
    - Top 10%
    - Middle 40%
    - Bottom 50%

Interpretation
--------------
This figure decomposes system-level SPSE change by destination:

    sum_j ΔSPSE_j = SPSE(H*) - SPSE(F)

It separates destination volume change from destination-level contribution to
structural potential social exposure.
"""

import os
import glob
import pickle
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

PROJECT_ROOT = r"d:\mobility_poi_core_place"
MATRIX_ROOT = os.path.join(PROJECT_ROOT, "matrices_A_D_S_Distribution")

SELECTED_POI_CODE = "624190"
CITY_LABEL = "Boston MSA"

DMAX_KM = 50
DISTANCE_SCALE = 1.0

SHOW_FIGURES = True
SAVE_FIGURES = False
SAVE_TABLES = False

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "fig2d_spse_destination_outputs")

DESTINATION_TIER_BASIS = "baseline_total_visits"

# Set to "" for final composite figure if you add panel label externally.
PANEL_TITLE = "Destination-level social exposure contributions"

EPS = 1e-9


# ============================================================
# 1. Metadata
# ============================================================

income_levels = [
    "low_income_pct",
    "lower_middle_income_pct",
    "upper_middle_income_pct",
    "high_income_pct",
]

income_score_weights = {
    "low_income_pct": 1.0,
    "lower_middle_income_pct": 2.0,
    "upper_middle_income_pct": 3.0,
    "high_income_pct": 4.0,
}

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


def safe_div(a, b):
    if b is None or not np.isfinite(b) or abs(b) < EPS:
        return np.nan
    return float(a) / float(b)


def weighted_mean(x, w):
    x = np.asarray(x, dtype=float)
    w = np.asarray(w, dtype=float)

    mask = np.isfinite(x) & np.isfinite(w) & (w > 0)

    if mask.sum() == 0:
        return np.nan

    return float(np.sum(x[mask] * w[mask]) / np.sum(w[mask]))


def gini_coefficient(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    x = x[x >= 0]

    if len(x) == 0:
        return np.nan

    if np.allclose(x.sum(), 0):
        return 0.0

    x = np.sort(x)
    n = len(x)

    return float(
        (2 * np.sum(np.arange(1, n + 1) * x)) / (n * x.sum())
        - (n + 1) / n
    )


def fmt_pct(x, nd=1):
    if pd.isna(x):
        return "NA"
    return f"{x:+.{nd}f}%" if x > 0 else f"{x:.{nd}f}%"


def set_nature_style():
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman", "Times", "DejaVu Serif"]
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
        np.linspace(0.20, 0.65, cmap_social.N // 2)
    )

    cmap_social_half = mcolors.LinearSegmentedColormap.from_list(
        "weighted_contribution_purples_half",
        social_half_range,
    )

    return cmap_social_half


def scatter_size_from_abs_delta_visits(delta_visits, vmax):
    """
    Marker size is proportional to the absolute magnitude of POI visit change.

    This treats large visit gains and large visit losses symmetrically:
        size_j ~ |optimized visits_j - baseline visits_j|
    """
    delta_visits = np.asarray(delta_visits, dtype=float)
    abs_change = np.abs(delta_visits)

    vmax = max(1.0, float(vmax))

    return 100 + 390 * np.sqrt(abs_change / vmax)


def compute_top10_outflow_reassignment(F_dom, H_dom, poi_tiers):
    """
    Estimate where flow removed from baseline Top-10% POIs is reassigned.

    The optimization preserves each CBG's total category-specific visits, but it
    does not explicitly store a trip-by-trip origin POI -> destination POI
    reassignment path. Therefore, for each CBG, flow removed from Top-10% POIs
    is attributed to the POI tiers that receive positive increments from the
    same CBG, proportional to those positive increments.

    Main inset percentages are shares of total Top-10% outflow:
        Top -> Middle share = inferred Top-to-Middle reassigned visits /
                              all visits removed from Top-10% POIs
        Top -> Bottom share = inferred Top-to-Bottom reassigned visits /
                              all visits removed from Top-10% POIs
    """
    F = F_dom.copy().astype(float)
    H = H_dom.copy().astype(float)
    diff = H.values - F.values

    add = np.maximum(diff, 0.0)
    loss = np.maximum(-diff, 0.0)

    cols = list(F.columns)
    tier_s = poi_tiers.set_index("poi_id")["poi_tier"].reindex(cols).fillna("")

    top_mask = tier_s.str.contains("Top 10%", case=False, regex=False).values
    mid_mask = tier_s.str.contains("Middle", case=False, regex=False).values
    bottom_mask = tier_s.str.contains("Bottom", case=False, regex=False).values

    if top_mask.sum() == 0 or mid_mask.sum() == 0 or bottom_mask.sum() == 0:
        return {
            "available": False,
            "reason": "Cannot identify Top/Middle/Bottom POI tiers.",
        }

    top_loss_by_origin = loss[:, top_mask].sum(axis=1)
    total_add_by_origin = add.sum(axis=1)

    add_top_by_origin = add[:, top_mask].sum(axis=1)
    add_mid_by_origin = add[:, mid_mask].sum(axis=1)
    add_bottom_by_origin = add[:, bottom_mask].sum(axis=1)

    valid = (top_loss_by_origin > EPS) & (total_add_by_origin > EPS)

    top_to_top = float(np.nansum(
        top_loss_by_origin[valid] * add_top_by_origin[valid] / total_add_by_origin[valid]
    ))
    top_to_middle = float(np.nansum(
        top_loss_by_origin[valid] * add_mid_by_origin[valid] / total_add_by_origin[valid]
    ))
    top_to_bottom = float(np.nansum(
        top_loss_by_origin[valid] * add_bottom_by_origin[valid] / total_add_by_origin[valid]
    ))

    top_outflow = float(np.nansum(top_loss_by_origin))
    baseline_top_flow = float(np.nansum(F.values[:, top_mask]))

    return {
        "available": bool(top_outflow > EPS),
        "top_outflow": top_outflow,
        "baseline_top_flow": baseline_top_flow,
        "top_to_top": top_to_top,
        "top_to_middle": top_to_middle,
        "top_to_bottom": top_to_bottom,
        "top_to_middle_share_of_top_outflow_pct": safe_div(top_to_middle, top_outflow) * 100,
        "top_to_bottom_share_of_top_outflow_pct": safe_div(top_to_bottom, top_outflow) * 100,
        "top_to_top_share_of_top_outflow_pct": safe_div(top_to_top, top_outflow) * 100,
        "top_to_middle_share_of_baseline_top_flow_pct": safe_div(top_to_middle, baseline_top_flow) * 100,
        "top_to_bottom_share_of_baseline_top_flow_pct": safe_div(top_to_bottom, baseline_top_flow) * 100,
        "n_top_pois": int(top_mask.sum()),
        "n_middle_pois": int(mid_mask.sum()),
        "n_bottom_pois": int(bottom_mask.sum()),
    }


def choose_size_legend_values(size_var, max_n=3):
    """Return readable representative values for the |Δ visits| legend."""
    x = np.asarray(size_var, dtype=float)
    x = x[np.isfinite(x) & (x > EPS)]

    if len(x) == 0:
        return [1.0]

    qs = np.nanpercentile(x, [35, 65, 90])

    vals = []
    for q in qs:
        if q <= 10:
            rounded = max(1.0, round(q))
        elif q <= 100:
            rounded = max(5.0, round(q / 5) * 5)
        else:
            rounded = max(10.0, round(q / 10) * 10)
        vals.append(float(rounded))

    vals_unique = []
    for v in vals:
        if v not in vals_unique:
            vals_unique.append(v)

    if len(vals_unique) < 2 and len(x) >= 2:
        fallback = [float(np.nanmin(x)), float(np.nanmedian(x)), float(np.nanmax(x))]
        vals_unique = []
        for v in fallback:
            if v <= 10:
                vv = max(1.0, round(v))
            elif v <= 100:
                vv = max(5.0, round(v / 5) * 5)
            else:
                vv = max(10.0, round(v / 10) * 10)
            if vv not in vals_unique:
                vals_unique.append(float(vv))

    return vals_unique[:max_n]



def add_top10_reassignment_inset(ax, top10_summary):
    """
    Add an upper-left inset showing inferred Top-10% -> Middle/Bottom flow.
    """
    if not top10_summary or not top10_summary.get("available", False):
        return None

    mid_pct = top10_summary.get("top_to_middle_share_of_top_outflow_pct", np.nan)
    bot_pct = top10_summary.get("top_to_bottom_share_of_top_outflow_pct", np.nan)
    top_outflow = top10_summary.get("top_outflow", np.nan)

    inax = ax.inset_axes([0.045, 0.710, 0.445, 0.245], zorder=8)
    inax.set_xlim(0, 1)
    inax.set_ylim(0, 1)
    inax.axis("off")

    bg = FancyBboxPatch(
        (0.00, 0.00),
        1.00,
        1.00,
        boxstyle="round,pad=0.015,rounding_size=0.035",
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
        0.03, 1.05,
        "Reassigned Top-10% outflow",
        ha="left", va="top",
        fontsize=10.5,
        color=c_text,
        transform=inax.transAxes,
        zorder=5,
    )
    inax.text(
        0.045, 0.935,
        f"denom. = {top_outflow:.0f} visits",
        ha="left", va="top",
        fontsize=10.5,
        color=c_muted,
        transform=inax.transAxes,
        zorder=5,
    )

    def node(xy, label):
        inax.text(
            xy[0], xy[1], label,
            ha="center", va="center",
            fontsize=7.8,
            color=c_text,
            bbox=dict(
                boxstyle="round,pad=0.24,rounding_size=0.10",
                facecolor="#F8FAFC",
                edgecolor="#D7DCE5",
                linewidth=0.70,
                alpha=0.98,
            ),
            transform=inax.transAxes,
            zorder=6,
        )

    p_top = (0.10, 0.460)
    p_mid = (0.410, 0.460)
    p_bot = (0.810, 0.460)

    node(p_top, "Top 10%")
    node(p_mid, "Middle 40%")
    node(p_bot, "Bottom 50%")

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

    inax.add_patch(arrow_mid)
    inax.add_patch(arrow_bot)


    pct_bbox = dict(
        boxstyle="round,pad=0.15,rounding_size=0.08",
        facecolor="white",
        edgecolor="none",
        alpha=0.88,
    )

    inax.text(
        0.26, 0.22,
        f"{mid_pct:.2f}%",
        ha="center", va="center",
        fontsize=8.2,
        color=c_arrow,
        fontweight="bold",
        bbox=pct_bbox,
        transform=inax.transAxes,
        zorder=7,
    )
    inax.text(
        0.50, 0.77,
        f"{bot_pct:.2f}%",
        ha="center", va="center",
        fontsize=8.2,
        color=c_arrow,
        fontweight="bold",
        bbox=pct_bbox,
        transform=inax.transAxes,
        zorder=7,
    )

    return inax


# ============================================================
# 3. Locate files
# ============================================================

def find_case_dir_by_poi_code(poi_code):
    candidates = sorted(glob.glob(
        os.path.join(MATRIX_ROOT, "**", f"H_opt_df_dynamic_{poi_code}.pkl"),
        recursive=True,
    ))

    if len(candidates) == 0:
        raise FileNotFoundError(
            f"Cannot find H_opt_df_dynamic_{poi_code}.pkl under {MATRIX_ROOT}"
        )

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

    matches = glob.glob(
        os.path.join(PROJECT_ROOT, "**", "cbg_income_level_distribution*.csv"),
        recursive=True,
    )

    if len(matches) == 0:
        raise FileNotFoundError("Cannot find cbg_income_level_distribution*.csv")

    return matches[0]


# ============================================================
# 4. Load income distribution
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


# ============================================================
# 5. Exposure computation
# ============================================================

def compute_all_pair_unmasked_exposure(flow_df, P_df):
    """
    Compute all-pair S_ij in the baseline POI domain.

    S_ij = 1 - dot(P_i, Q_j)

    P_i = income distribution of origin CBG i.
    Q_j = visitor-income distribution of POI j, estimated from flow_df.
    """
    F = flow_df.copy()
    F.index = F.index.astype(str).map(normalize_geoid)
    F.columns = F.columns.astype(str).str.strip()
    F = F.apply(pd.to_numeric, errors="coerce").fillna(0)

    common_cbgs = [g for g in F.index if g in P_df.index]

    if len(common_cbgs) == 0:
        raise ValueError("No common CBGs between flow matrix and income distribution.")

    F = F.loc[common_cbgs].copy()
    P = P_df.loc[common_cbgs, income_levels].copy()

    poi_total_flow = F.sum(axis=0)
    valid_pois = poi_total_flow[poi_total_flow > 0].index.tolist()

    if len(valid_pois) == 0:
        raise ValueError("No POI has positive baseline flow.")

    F = F[valid_pois].copy()

    F_values = F.values.astype(float)
    P_values = P.values.astype(float)

    poi_total_flow = F_values.sum(axis=0)
    Q_values = (F_values.T @ P_values) / poi_total_flow[:, None]

    Q_sum = Q_values.sum(axis=1, keepdims=True)
    Q_values = np.divide(
        Q_values,
        Q_sum,
        out=np.zeros_like(Q_values),
        where=Q_sum > 0,
    )

    S_values = 1.0 - (P_values @ Q_values.T)

    S = pd.DataFrame(S_values, index=F.index, columns=F.columns)
    Q = pd.DataFrame(Q_values, index=F.columns, columns=income_levels)

    return S, F, Q


def compute_exposure_on_fixed_domain(flow_df, P_df, fixed_columns):
    """
    Compute S_ij on a fixed POI domain.

    Used for post-optimization exposure so baseline and optimized exposure are
    evaluated on the same POI set.
    """
    F = flow_df.copy()
    F.index = F.index.astype(str).map(normalize_geoid)
    F.columns = F.columns.astype(str).str.strip()
    F = F.apply(pd.to_numeric, errors="coerce").fillna(0)

    fixed_columns = [str(c).strip() for c in fixed_columns]

    common_cbgs = [g for g in F.index if g in P_df.index]
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
    Q_values = np.divide(
        Q_values,
        Q_sum,
        out=np.zeros_like(Q_values),
        where=Q_sum > 0,
    )

    S_values = 1.0 - (P_values @ Q_values.T)

    S = pd.DataFrame(S_values, index=F.index, columns=F.columns)
    Q = pd.DataFrame(Q_values, index=F.columns, columns=income_levels)

    return S, Q


# ============================================================
# 6. Grouping and score helpers
# ============================================================

def compute_income_composition_score(P_model):
    """
    income_score_i =
        1 * low_income_pct
      + 2 * lower_middle_income_pct
      + 3 * upper_middle_income_pct
      + 4 * high_income_pct
    """
    P = P_model[income_levels].copy()
    weights = np.array([income_score_weights[c] for c in income_levels], dtype=float)
    score = pd.Series(P.values @ weights, index=P.index, name="income_composition_score")
    return score


def make_destination_tiers(poi_metrics, basis="baseline_total_visits"):
    """
    Group POIs into:
        Top 10%, Middle 40%, Bottom 50%.
    """
    if basis not in ["baseline_total_visits", "optimized_total_visits"]:
        raise ValueError(
            "basis must be 'baseline_total_visits' or 'optimized_total_visits'."
        )

    df = poi_metrics[[
        "poi_id",
        "baseline_total_visits",
        "optimized_total_visits",
    ]].copy()

    df = df.sort_values(basis, ascending=False).reset_index(drop=True)

    n = len(df)
    if n == 0:
        raise ValueError("No POIs available for destination tiering.")

    n_top = max(1, int(np.ceil(n * 0.10)))
    n_mid = max(1, int(np.ceil(n * 0.40)))
    n_mid = min(n_mid, max(0, n - n_top))

    if basis == "baseline_total_visits":
        tier_order = [
            "Top 10% baseline-flow POIs",
            "Middle 40% baseline-flow POIs",
            "Bottom 50% baseline-flow POIs",
        ]
        rank_col = "poi_rank_baseline_flow"
    else:
        tier_order = [
            "Top 10% final-flow POIs",
            "Middle 40% final-flow POIs",
            "Bottom 50% final-flow POIs",
        ]
        rank_col = "poi_rank_final_flow"

    tiers = []
    for pos in range(n):
        if pos < n_top:
            tiers.append(tier_order[0])
        elif pos < n_top + n_mid:
            tiers.append(tier_order[1])
        else:
            tiers.append(tier_order[2])

    df["poi_tier"] = tiers
    df[rank_col] = np.arange(1, n + 1)

    return df, tier_order


# ============================================================
# 7. Build destination-level SPSE summary
# ============================================================

def build_spse_destination_summary(
    F_dom,
    H_dom,
    S0_dom,
    S1_dom,
    D_dom,
    P_model,
    poi_tiers,
):
    """
    Build one-row-per-POI table for Fig.2d.

    Main plotted quantities:

    x:
        delta_visits
        = sum_i H*_ij - sum_i F_ij

    y:
        delta_spse_contribution
        = sum_i 1(H*_ij > 0) S*_ij - sum_i 1(F_ij > 0) S0_ij

    color:
        delta_weighted_contribution
        = sum_i H*_ij S*_ij - sum_i F_ij S0_ij

    size:
        positive_increment_received
        = sum_i max(H*_ij - F_ij, 0)
    """
    rows = F_dom.index.tolist()
    cols = F_dom.columns.tolist()

    F_v = F_dom.values.astype(float)
    H_v = H_dom.values.astype(float)
    S0_v = S0_dom.values.astype(float)
    S1_v = S1_dom.values.astype(float)
    D_v = D_dom.values.astype(float)

    diff_v = H_v - F_v
    add_v = np.maximum(diff_v, 0)
    remove_v = np.maximum(-diff_v, 0)

    income_score = compute_income_composition_score(P_model).reindex(rows)
    income_score_v = income_score.values.astype(float)

    poi_tier_map = poi_tiers.set_index("poi_id")["poi_tier"].to_dict()

    records = []

    for j, poi_id in enumerate(cols):
        f = F_v[:, j]
        h = H_v[:, j]
        s0 = S0_v[:, j]
        s1 = S1_v[:, j]
        d = D_v[:, j]

        add = add_v[:, j]
        remove = remove_v[:, j]

        baseline_visits = float(np.nansum(f))
        optimized_visits = float(np.nansum(h))
        delta_visits = optimized_visits - baseline_visits
        delta_visits_pct = safe_div(delta_visits, baseline_visits) * 100

        positive_increment_received = float(np.nansum(add))
        removed_flow = float(np.nansum(remove))
        net_flow_change = optimized_visits - baseline_visits

        active_before = f > EPS
        active_after = h > EPS
        active_union = active_before | active_after

        new_positive_links = (~active_before) & active_after
        removed_positive_links = active_before & (~active_after)
        retained_positive_links = active_before & active_after

        n_active_links_before = int(active_before.sum())
        n_active_links_after = int(active_after.sum())
        delta_active_links = n_active_links_after - n_active_links_before

        n_new_positive_links = int(new_positive_links.sum())
        n_removed_positive_links = int(removed_positive_links.sum())
        n_retained_positive_links = int(retained_positive_links.sum())

        # POI-level SPSE decomposition.
        spse_contribution_before = float(np.nansum(s0[active_before]))
        spse_contribution_after = float(np.nansum(s1[active_after]))
        delta_spse_contribution = spse_contribution_after - spse_contribution_before

        spse_from_new_links = float(np.nansum(s1[new_positive_links]))
        spse_lost_from_removed_links = float(np.nansum(s0[removed_positive_links]))

        spse_change_on_retained_links = float(
            np.nansum(s1[retained_positive_links] - s0[retained_positive_links])
        )
        # Average destination-level structural social exposure gain per visiting CBG.
        # Recommended denominator: number of CBGs visiting this POI after optimization.
        mean_delta_structural_social_exposure_per_visiting_cbg = safe_div(
            delta_spse_contribution,
            n_active_links_after,
            )
        
        # Alternative diagnostics, kept for robustness.
        mean_delta_structural_social_exposure_per_baseline_visiting_cbg = safe_div(
            delta_spse_contribution,
            n_active_links_before,
            )
        
        mean_delta_structural_social_exposure_per_union_visiting_cbg = safe_div(
            delta_spse_contribution,
            int(active_union.sum()),
            )

        # Weighted exposure contribution.
        weighted_contribution_before = float(np.nansum(f * s0))
        weighted_contribution_after = float(np.nansum(h * s1))
        delta_weighted_contribution = (
            weighted_contribution_after - weighted_contribution_before
        )

        weighted_added_contribution = float(np.nansum(add * s1))
        weighted_removed_contribution = float(np.nansum(remove * s0))

        # Visit-weighted exposure diagnostics.
        flow_weighted_exposure_before = weighted_mean(s0, f)
        flow_weighted_exposure_after = weighted_mean(s1, h)

        if (
            np.isfinite(flow_weighted_exposure_before)
            and np.isfinite(flow_weighted_exposure_after)
        ):
            delta_flow_weighted_exposure = (
                flow_weighted_exposure_after - flow_weighted_exposure_before
            )
        else:
            delta_flow_weighted_exposure = np.nan

        # Origin income-composition diagnostics.
        added_origin_income_score = weighted_mean(income_score_v, add)
        baseline_origin_income_score = weighted_mean(income_score_v, f)
        optimized_origin_income_score = weighted_mean(income_score_v, h)

        if (
            np.isfinite(baseline_origin_income_score)
            and np.isfinite(optimized_origin_income_score)
        ):
            delta_origin_income_score = (
                optimized_origin_income_score - baseline_origin_income_score
            )
        else:
            delta_origin_income_score = np.nan

        # Distance diagnostics.
        flow_weighted_distance_before = weighted_mean(d, f)
        flow_weighted_distance_after = weighted_mean(d, h)

        if (
            np.isfinite(flow_weighted_distance_before)
            and np.isfinite(flow_weighted_distance_after)
        ):
            delta_flow_weighted_distance_km = (
                flow_weighted_distance_after - flow_weighted_distance_before
            )
        else:
            delta_flow_weighted_distance_km = np.nan

        # Exposure-field diagnostics retained but not plotted.
        delta_s = s1 - s0
        finite_delta_s = np.isfinite(delta_s)

        delta_exposure_field_all_cbgs = float(np.nansum(delta_s[finite_delta_s]))
        n_exposure_field_all_cbgs = int(finite_delta_s.sum())
        mean_delta_exposure_field_all_cbgs = safe_div(
            delta_exposure_field_all_cbgs,
            n_exposure_field_all_cbgs,
        )

        delta_exposure_field_active_union = float(
            np.nansum(delta_s[active_union & finite_delta_s])
        )
        n_exposure_field_active_union = int((active_union & finite_delta_s).sum())
        mean_delta_exposure_field_active_union = safe_div(
            delta_exposure_field_active_union,
            n_exposure_field_active_union,
        )

        records.append({
            "poi_id": poi_id,
            "poi_tier": poi_tier_map.get(poi_id, "Unclassified"),

            "baseline_total_visits": baseline_visits,
            "optimized_total_visits": optimized_visits,
            "delta_visits": delta_visits,
            "delta_visits_pct": delta_visits_pct,

            "positive_increment_received": positive_increment_received,
            "removed_flow": removed_flow,
            "net_flow_change": net_flow_change,

            "n_active_links_before": n_active_links_before,
            "n_active_links_after": n_active_links_after,
            "delta_active_links": delta_active_links,
            "n_new_positive_links": n_new_positive_links,
            "n_removed_positive_links": n_removed_positive_links,
            "n_retained_positive_links": n_retained_positive_links,

            "spse_contribution_before": spse_contribution_before,
            "spse_contribution_after": spse_contribution_after,
            "delta_spse_contribution": delta_spse_contribution,
            "spse_from_new_links": spse_from_new_links,
            "spse_lost_from_removed_links": spse_lost_from_removed_links,
            "spse_change_on_retained_links": spse_change_on_retained_links,
            
            "mean_delta_structural_social_exposure_per_visiting_cbg": (
                mean_delta_structural_social_exposure_per_visiting_cbg
                ),
            "mean_delta_structural_social_exposure_per_baseline_visiting_cbg": (
                mean_delta_structural_social_exposure_per_baseline_visiting_cbg
                ),
            "mean_delta_structural_social_exposure_per_union_visiting_cbg": (
                mean_delta_structural_social_exposure_per_union_visiting_cbg
                ),

            "weighted_contribution_before": weighted_contribution_before,
            "weighted_contribution_after": weighted_contribution_after,
            "delta_weighted_contribution": delta_weighted_contribution,
            "weighted_added_contribution": weighted_added_contribution,
            "weighted_removed_contribution": weighted_removed_contribution,

            "flow_weighted_exposure_before": flow_weighted_exposure_before,
            "flow_weighted_exposure_after": flow_weighted_exposure_after,
            "delta_flow_weighted_exposure": delta_flow_weighted_exposure,

            "added_origin_income_score": added_origin_income_score,
            "baseline_origin_income_score": baseline_origin_income_score,
            "optimized_origin_income_score": optimized_origin_income_score,
            "delta_origin_income_score": delta_origin_income_score,

            "flow_weighted_distance_before": flow_weighted_distance_before,
            "flow_weighted_distance_after": flow_weighted_distance_after,
            "delta_flow_weighted_distance_km": delta_flow_weighted_distance_km,

            "delta_exposure_field_all_cbgs": delta_exposure_field_all_cbgs,
            "mean_delta_exposure_field_all_cbgs": mean_delta_exposure_field_all_cbgs,
            "delta_exposure_field_active_union": delta_exposure_field_active_union,
            "mean_delta_exposure_field_active_union": mean_delta_exposure_field_active_union,
        })

    scatter_df = pd.DataFrame(records)

    return scatter_df


# ============================================================
# 8. Build full case output
# ============================================================

def build_case_output():
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

    # Baseline exposure on all baseline-positive POIs after income alignment.
    S0_full, F_income, Q0_full = compute_all_pair_unmasked_exposure(F_raw, P_df)

    common_rows = [g for g in F_income.index if g in D_raw.index and g in S0_full.index]
    common_cols = [p for p in F_income.columns if p in D_raw.columns and p in S0_full.columns]

    if len(common_rows) == 0 or len(common_cols) == 0:
        raise ValueError("No common rows/columns among F, D, and S0.")

    F_full = F_income.loc[common_rows, common_cols].copy()
    D_full = D_raw.loc[common_rows, common_cols].copy()
    S0_full = S0_full.loc[common_rows, common_cols].copy()

    # Embed optimized H into the full baseline matrix.
    H_eval = F_full.copy().astype(float)

    rows_h = [g for g in H_opt.index if g in H_eval.index and g in P_df.index]
    cols_h = [p for p in H_opt.columns if p in H_eval.columns]

    if len(rows_h) == 0 or len(cols_h) == 0:
        raise ValueError("No common rows/columns between H_opt and baseline flow matrix.")

    H_eval.loc[rows_h, cols_h] = H_opt.loc[rows_h, cols_h].values

    # Model domain.
    F_dom = F_full.loc[rows_h, cols_h].copy()
    H_dom = H_eval.loc[rows_h, cols_h].copy()
    D_dom = D_full.loc[rows_h, cols_h].copy()
    S0_dom = S0_full.loc[rows_h, cols_h].copy()
    P_model = P_df.loc[rows_h, income_levels].copy()

    # Active-link reference from full diagnostic domain.
    Fv_full = F_full.values.astype(float)
    Dv_full = D_full.values.astype(float)
    S0v_full = S0_full.values.astype(float)
    Hv_full = H_eval.values.astype(float)

    valid_full = (
        np.isfinite(Fv_full)
        & np.isfinite(Dv_full)
        & np.isfinite(S0v_full)
        & np.isfinite(Hv_full)
    )

    distance_feasible_full = valid_full & (Dv_full >= 0) & (Dv_full <= DMAX_KM)
    active_ref_full = distance_feasible_full & (Fv_full > 0)

    if active_ref_full.sum() == 0:
        raise ValueError("No active feasible reference links in the full diagnostic domain.")

    active_w_s = weighted_mean(S0v_full[active_ref_full], Fv_full[active_ref_full])
    active_w_d = weighted_mean(Dv_full[active_ref_full], Fv_full[active_ref_full])

    # Post-optimization exposure on fixed POI domain.
    S1_full, Q1_full = compute_exposure_on_fixed_domain(H_eval, P_df, F_full.columns)
    S1_dom = S1_full.loc[rows_h, cols_h].copy()

    F_v = F_dom.values.astype(float)
    H_v = H_dom.values.astype(float)
    D_v = D_dom.values.astype(float)
    S0_v = S0_dom.values.astype(float)
    S1_v = S1_dom.values.astype(float)

    baseline_total_flow = float(np.nansum(F_v))
    optimized_total_flow = float(np.nansum(H_v))

    diff_v = H_v - F_v
    add_v = np.maximum(diff_v, 0)
    remove_v = np.maximum(-diff_v, 0)

    reassigned_visit_equiv = 0.5 * float(np.nansum(np.abs(diff_v)))
    total_positive_increment = float(np.nansum(add_v))
    total_removed_flow = float(np.nansum(remove_v))

    distance_before = float(np.nansum(F_v * D_v))
    distance_after = float(np.nansum(H_v * D_v))
    distance_change_pct = safe_div(distance_after - distance_before, distance_before) * 100

    active_before_mask = F_v > EPS
    active_after_mask = H_v > EPS

    total_active_links_before = int(active_before_mask.sum())
    total_active_links_after = int(active_after_mask.sum())
    total_delta_active_links = total_active_links_after - total_active_links_before

    spse_before = float(np.nansum(S0_v[active_before_mask]))
    spse_after = float(np.nansum(S1_v[active_after_mask]))
    spse_delta = spse_after - spse_before
    spse_change_pct = safe_div(spse_delta, spse_before) * 100

    weighted_exposure_numerator_before = float(np.nansum(F_v * S0_v))
    weighted_exposure_numerator_after = float(np.nansum(H_v * S1_v))
    weighted_exposure_numerator_delta = (
        weighted_exposure_numerator_after - weighted_exposure_numerator_before
    )

    fw_before = safe_div(weighted_exposure_numerator_before, baseline_total_flow)
    fw_after = safe_div(weighted_exposure_numerator_after, optimized_total_flow)
    fw_change_pct = safe_div(fw_after - fw_before, fw_before) * 100

    new_link_mask = (F_v <= EPS) & (H_v > EPS)
    removed_link_mask = (F_v > EPS) & (H_v <= EPS)
    retained_link_mask = (F_v > EPS) & (H_v > EPS)
    positive_increment_mask = diff_v > EPS

    add_mask = add_v > EPS

    if add_mask.any():
        w = add_v[add_mask]
        realized_adv = S1_v[add_mask] - active_w_s
        distance_adv = D_v[add_mask] - active_w_d

        positive_increment_higher_exposure_share_pct = safe_div(
            np.nansum(w[realized_adv > 0]),
            np.nansum(w),
        ) * 100

        positive_increment_no_farther_share_pct = safe_div(
            np.nansum(w[distance_adv <= 0]),
            np.nansum(w),
        ) * 100

        positive_increment_higher_and_no_farther_share_pct = safe_div(
            np.nansum(w[(realized_adv > 0) & (distance_adv <= 0)]),
            np.nansum(w),
        ) * 100
    else:
        positive_increment_higher_exposure_share_pct = np.nan
        positive_increment_no_farther_share_pct = np.nan
        positive_increment_higher_and_no_farther_share_pct = np.nan

    poi_metrics = pd.DataFrame({
        "poi_id": cols_h,
        "baseline_total_visits": F_dom.sum(axis=0).reindex(cols_h).values.astype(float),
        "optimized_total_visits": H_dom.sum(axis=0).reindex(cols_h).values.astype(float),
    })

    poi_tiers, tier_order = make_destination_tiers(
        poi_metrics=poi_metrics,
        basis=DESTINATION_TIER_BASIS,
    )

    top10_reassignment_summary = compute_top10_outflow_reassignment(
        F_dom=F_dom,
        H_dom=H_dom,
        poi_tiers=poi_tiers,
    )

    scatter_df = build_spse_destination_summary(
        F_dom=F_dom,
        H_dom=H_dom,
        S0_dom=S0_dom,
        S1_dom=S1_dom,
        D_dom=D_dom,
        P_model=P_model,
        poi_tiers=poi_tiers,
    )

    sum_delta_spse_from_pois = float(scatter_df["delta_spse_contribution"].sum())
    sum_delta_weighted_contribution_from_pois = float(
        scatter_df["delta_weighted_contribution"].sum()
    )
    sum_delta_active_links_from_pois = int(scatter_df["delta_active_links"].sum())

    spse_decomposition_error = sum_delta_spse_from_pois - spse_delta
    weighted_decomposition_error = (
        sum_delta_weighted_contribution_from_pois - weighted_exposure_numerator_delta
    )
    active_link_decomposition_error = (
        sum_delta_active_links_from_pois - total_delta_active_links
    )

    origin_reallocation = 0.5 * np.abs(diff_v).sum(axis=1)
    poi_positive_increment = add_v.sum(axis=0)

    case_summary = {
        "poi_code": SELECTED_POI_CODE,
        "poi_full_label": poi_code_to_full_label.get(SELECTED_POI_CODE, SELECTED_POI_CODE),
        "city_label": CITY_LABEL,
        "case_dir": case_dir,
        "diagnostic_domain": "full_baseline_matrix_after_alignment",

        "n_model_cbgs": len(rows_h),
        "n_model_pois": len(cols_h),

        "active_weighted_exposure_ref_full": active_w_s,
        "active_weighted_distance_ref_full": active_w_d,

        "baseline_total_flow": baseline_total_flow,
        "optimized_total_flow": optimized_total_flow,
        "total_positive_increment": total_positive_increment,
        "total_removed_flow": total_removed_flow,
        "reassigned_visit_equiv": reassigned_visit_equiv,

        "distance_before": distance_before,
        "distance_after": distance_after,
        "distance_change_pct": distance_change_pct,

        "total_active_links_before": total_active_links_before,
        "total_active_links_after": total_active_links_after,
        "total_delta_active_links": total_delta_active_links,

        "spse_before": spse_before,
        "spse_after": spse_after,
        "spse_delta": spse_delta,
        "spse_change_pct": spse_change_pct,

        "sum_delta_spse_from_pois": sum_delta_spse_from_pois,
        "spse_decomposition_error": spse_decomposition_error,

        "flow_weighted_exposure_before": fw_before,
        "flow_weighted_exposure_after": fw_after,
        "fw_change_pct": fw_change_pct,

        "weighted_exposure_numerator_before": weighted_exposure_numerator_before,
        "weighted_exposure_numerator_after": weighted_exposure_numerator_after,
        "weighted_exposure_numerator_delta": weighted_exposure_numerator_delta,

        "sum_delta_weighted_contribution_from_pois": sum_delta_weighted_contribution_from_pois,
        "weighted_decomposition_error": weighted_decomposition_error,

        "sum_delta_active_links_from_pois": sum_delta_active_links_from_pois,
        "active_link_decomposition_error": active_link_decomposition_error,

        "n_new_links": int(new_link_mask.sum()),
        "n_removed_links": int(removed_link_mask.sum()),
        "n_retained_links": int(retained_link_mask.sum()),
        "n_positive_increment_cells": int(positive_increment_mask.sum()),

        "positive_increment_higher_exposure_share_pct": positive_increment_higher_exposure_share_pct,
        "positive_increment_no_farther_share_pct": positive_increment_no_farther_share_pct,
        "positive_increment_higher_and_no_farther_share_pct": positive_increment_higher_and_no_farther_share_pct,

        "origin_reallocation_gini": gini_coefficient(origin_reallocation),
        "destination_positive_increment_gini": gini_coefficient(poi_positive_increment),
        "row_balance_error_max_abs": float(np.nanmax(np.abs(np.nansum(diff_v, axis=1)))),

        "top10_outflow": top10_reassignment_summary.get("top_outflow", np.nan),
        "top10_to_middle_share_of_top_outflow_pct": top10_reassignment_summary.get(
            "top_to_middle_share_of_top_outflow_pct", np.nan
        ),
        "top10_to_bottom_share_of_top_outflow_pct": top10_reassignment_summary.get(
            "top_to_bottom_share_of_top_outflow_pct", np.nan
        ),

        "upper_left_outcome": bool((distance_change_pct < 0) and (spse_change_pct > 0)),
        "fw_positive": bool(fw_change_pct > 0),
    }

    return {
        "case_summary": case_summary,
        "scatter_df": scatter_df,
        "poi_tiers": poi_tiers,
        "poi_metrics": poi_metrics,
        "top10_reassignment_summary": top10_reassignment_summary,
        "case_dir": case_dir,
    }


# ============================================================
# 9. Plot Fig.2d
# ============================================================

def add_quadrant_background(ax, xlim, ylim):
    """
    Very subtle quadrant background for publication-style plotting.
    """
    x0, x1 = xlim
    y0, y1 = ylim

    zero_y = (0 - y0) / (y1 - y0)
    zero_y = np.clip(zero_y, 0, 1)

    # More visits + SPSE gain
    ax.axvspan(
        0, x1,
        ymin=zero_y,
        ymax=1,
        facecolor="#EAF7F3",
        alpha=0.24,
        zorder=0,
    )

    # Fewer visits + SPSE gain
    ax.axvspan(
        x0, 0,
        ymin=zero_y,
        ymax=1,
        facecolor="#EEF3FC",
        alpha=0.22,
        zorder=0,
    )

    # More visits + SPSE loss
    ax.axvspan(
        0, x1,
        ymin=0,
        ymax=zero_y,
        facecolor="#FBF1F1",
        alpha=0.18,
        zorder=0,
    )

    # Fewer visits + SPSE loss
    ax.axvspan(
        x0, 0,
        ymin=0,
        ymax=zero_y,
        facecolor="#F5F6FA",
        alpha=0.20,
        zorder=0,
    )


def make_symmetric_xlim(x):
    finite_x = x[np.isfinite(x)]

    if len(finite_x) == 0:
        return (-1, 1)

    xmax = float(np.nanmax(np.abs(finite_x)))
    xmax = max(5, np.ceil((xmax + 5) / 10) * 10)

    return (-xmax, xmax)


def make_zero_including_ylim(y):
    finite_y = y[np.isfinite(y)]

    if len(finite_y) == 0:
        return (-1, 1)

    ymin = float(np.nanmin(finite_y))
    ymax = float(np.nanmax(finite_y))

    if np.isclose(ymin, ymax):
        ymin -= 1
        ymax += 1

    yrange = ymax - ymin
    ymin = ymin - 0.12 * yrange
    ymax = ymax + 0.12 * yrange

    ymin = min(ymin, -0.2)
    ymax = max(ymax, 0.5)

    return (ymin, ymax)


def make_contribution_norm(color_var):
    """
    Sequential normalization for the purple colorbar.

    Low values are light purple; high values are darker purple.
    This no longer treats zero as a diverging center.
    """
    finite_color = color_var[np.isfinite(color_var)]

    if len(finite_color) == 0:
        return mcolors.Normalize(vmin=0.0, vmax=1.0, clip=True)

    vmin = float(np.nanpercentile(finite_color, 5))
    vmax = float(np.nanpercentile(finite_color, 95))

    if (not np.isfinite(vmin)) or (not np.isfinite(vmax)) or np.isclose(vmin, vmax):
        vmin = float(np.nanmin(finite_color))
        vmax = float(np.nanmax(finite_color))

    if (not np.isfinite(vmin)) or (not np.isfinite(vmax)) or np.isclose(vmin, vmax):
        vmin, vmax = 0.0, 1.0

    return mcolors.Normalize(vmin=vmin, vmax=vmax, clip=True)


def plot_fused_fig2d(case_output):
    set_nature_style()

    df = case_output["scatter_df"].copy()
    top10_summary = case_output.get("top10_reassignment_summary", {})

    c_grid = "#E1E7F0"
    c_axis = "#9CA3AF"
    c_text = "#243447"

    cmap = make_weighted_contribution_cmap()

    # Whether to show quadrant-level summed structural social exposure contribution.
    # If False, only POI counts are displayed.
    SHOW_QUADRANT_SUMS = False

    # --------------------------------------------------------
    # Plotted variables.
    # --------------------------------------------------------
    x = df["delta_visits"].astype(float).values
    # Average structural social exposure gain per CBG visiting the POI.
    y = df["mean_delta_structural_social_exposure_per_visiting_cbg"].astype(float).values

    # Marker size represents absolute POI visit change.
    size_var = np.abs(df["delta_visits"].astype(float).values)

    # Marker color represents POI-level weighted exposure contribution.
    color_var = df["delta_weighted_contribution"].astype(float).values

    xlim = make_symmetric_xlim(x)
    ylim = make_zero_including_ylim(y)
    norm = make_contribution_norm(color_var)

    size_vmax = max(1.0, float(np.nanmax(size_var))) if len(size_var) else 1.0

    # --------------------------------------------------------
    # Figure layout.
    # Colorbar height is aligned with the main axes height.
    # --------------------------------------------------------
    fig = plt.figure(figsize=(8, 8), dpi=300)
    fig.patch.set_facecolor("white")

    ax_left = 0.120
    ax_bottom = 0.155
    ax_width = 0.675
    ax_height = 0.720

    ax = fig.add_axes([ax_left, ax_bottom, ax_width, ax_height])

    cax = fig.add_axes([
        ax_left + ax_width + 0.045,
        ax_bottom,
        0.020,
        ax_height,
    ])

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)

    add_quadrant_background(ax, xlim, ylim)

    ax.axhline(0, color=c_axis, linewidth=0.85, linestyle="--", zorder=1)
    ax.axvline(0, color=c_axis, linewidth=0.85, linestyle="--", zorder=1)

    ax.grid(color=c_grid, linewidth=0.65, zorder=0.5)

    tier_order_short = [
        ("Top 10% baseline-flow POIs", "Top 10%", "^"),
        ("Middle 40% baseline-flow POIs", "Middle 40%", "o"),
        ("Bottom 50% baseline-flow POIs", "Bottom 50%", "s"),
    ]

    for tier, _, marker in tier_order_short:
        sub = df[df["poi_tier"] == tier].copy()

        if len(sub) == 0:
            continue

        sub_x = sub["delta_visits"].astype(float).values
        #sub_y = sub["delta_spse_contribution"].astype(float).values
        sub_y = sub["mean_delta_structural_social_exposure_per_visiting_cbg"].astype(float).values

        sub_s = scatter_size_from_abs_delta_visits(
            sub["delta_visits"].astype(float).values,
            size_vmax,
        )

        sub_c = sub["delta_weighted_contribution"].astype(float).values

        color_mask = np.isfinite(sub_c)

        if color_mask.any():
            ax.scatter(
                sub_x[color_mask],
                sub_y[color_mask],
                s=sub_s[color_mask],
                c=sub_c[color_mask],
                cmap=cmap,
                norm=norm,
                marker=marker,
                edgecolor="white",
                linewidth=0.70,
                alpha=0.90,
                zorder=3,
            )

        if (~color_mask).any():
            ax.scatter(
                sub_x[~color_mask],
                sub_y[~color_mask],
                s=sub_s[~color_mask],
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
    m_pp = (x > 0) & (y > 0)
    m_mp = (x < 0) & (y > 0)
    m_mm = (x < 0) & (y < 0)
    m_pm = (x > 0) & (y < 0)

    q_plus_plus = int(m_pp.sum())
    q_minus_plus = int(m_mp.sum())
    q_minus_minus = int(m_mm.sum())
    q_plus_minus = int(m_pm.sum())

    sum_plus_plus = float(np.nansum(y[m_pp]))
    sum_minus_plus = float(np.nansum(y[m_mp]))
    sum_minus_minus = float(np.nansum(y[m_mm]))
    sum_plus_minus = float(np.nansum(y[m_pm]))

    print("\n========== FIG.2D QUADRANT COUNTS ==========")
    print(f"More visits + structural social exposure gain: {q_plus_plus}; sum Δ = {sum_plus_plus:.3f}")
    print(f"Fewer visits + structural social exposure gain: {q_minus_plus}; sum Δ = {sum_minus_plus:.3f}")
    print(f"Fewer visits + structural social exposure loss: {q_minus_minus}; sum Δ = {sum_minus_minus:.3f}")
    print(f"More visits + structural social exposure loss: {q_plus_minus}; sum Δ = {sum_plus_minus:.3f}")

    def quadrant_label(n, s):
        if SHOW_QUADRANT_SUMS:
            return f"n = {n}\nΣΔ = {s:+.1f}"
        return f"n = {n}"

    quad_bbox = dict(
        facecolor="white",
        edgecolor="none",
        alpha=0.78,
        pad=1.35,
    )

    # Upper-left: fewer visits, structural social exposure gain.
    # Moved downward to leave room for the Top-10% reassignment inset.
    ax.text(
        0.025, 0.675,
        quadrant_label(q_minus_plus, sum_minus_plus),
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
        0.975, 0.965,
        quadrant_label(q_plus_plus, sum_plus_plus),
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
        0.025, 0.040,
        quadrant_label(q_minus_minus, sum_minus_minus),
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
        0.975, 0.040,
        quadrant_label(q_plus_minus, sum_plus_minus),
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
    add_top10_reassignment_inset(ax, top10_summary)

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

    ax.set_xlabel("Δ POI visits", labelpad=6,fontsize=14)
    ax.set_ylabel(
    "Average Δ structural social exposure per visiting CBG",
    labelpad=7,fontsize=14
    )

    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="both", length=2.8, color="#6B7280")

    # --------------------------------------------------------
    # Purple colorbar: same height as main axes.
    # --------------------------------------------------------
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])

    cbar = fig.colorbar(sm, cax=cax, orientation="vertical")
    cbar.set_label(
        "Δ weighted contribution",
        fontsize=14,
        labelpad=7,
    )
    cbar.ax.tick_params(labelsize=12, length=2)

    # --------------------------------------------------------
    # Marker-shape legend inside the plotting area.
    # --------------------------------------------------------
    legend_handles = [
        Line2D(
            [0], [0],
            marker="^",
            linestyle="none",
            markerfacecolor="#C7B9E8",
            markeredgecolor="white",
            markeredgewidth=0.7,
            markersize=6.4,
            label="Top 10%",
        ),
        Line2D(
            [0], [0],
            marker="o",
            linestyle="none",
            markerfacecolor="#C7B9E8",
            markeredgecolor="white",
            markeredgewidth=0.7,
            markersize=6.4,
            label="Middle 40%",
        ),
        Line2D(
            [0], [0],
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
        bbox_to_anchor=(0.975, 0.045),
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
    ax.add_artist(tier_legend)

    # --------------------------------------------------------
    # Marker-size legend: absolute POI visit change.
    # --------------------------------------------------------
    size_legend_values = choose_size_legend_values(size_var, max_n=3)
    size_handles = []
    for v in size_legend_values:
        s_legend = scatter_size_from_abs_delta_visits(np.array([v]), size_vmax)[0]
        size_handles.append(
            Line2D(
                [0], [0],
                marker="o",
                linestyle="none",
                markerfacecolor="#C7B9E8",
                markeredgecolor="white",
                markeredgewidth=0.7,
                markersize=np.sqrt(s_legend) / 1.65,
                alpha=0.90,
                label=f"{v:.0f}",
            )
        )

    ax.legend(
        handles=size_handles,
        title="|Δ POI visits|",
        loc="upper right",
        bbox_to_anchor=(0.855, 0.38),
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

    fig.savefig('figure2d.pdf',
                format='pdf',
                dpi=300,             # 仅影响位图元素
                bbox_inches='tight',
                transparent=False,   # 关闭透明，兼容性最好
                backend='pdf')

    plt.show()


    return fig


# ============================================================
# 11. Main
# ============================================================

def main():
    case_output = build_case_output()

    print("\n========== FIG.2D SPSE DESTINATION CASE SUMMARY ==========")
    for k, v in case_output["case_summary"].items():
        print(f"{k}: {v}")

    top10_summary = case_output.get("top10_reassignment_summary", {})
    if top10_summary.get("available", False):
        print("\n========== TOP-10% BASELINE-FLOW POI OUTFLOW REASSIGNMENT ==========")
        print(
            "Top-10% outflow: "
            f"{top10_summary['top_outflow']:.3f} visits; "
            "Top -> Middle: "
            f"{top10_summary['top_to_middle']:.3f} visits "
            f"({top10_summary['top_to_middle_share_of_top_outflow_pct']:.2f}% of Top outflow); "
            "Top -> Bottom: "
            f"{top10_summary['top_to_bottom']:.3f} visits "
            f"({top10_summary['top_to_bottom_share_of_top_outflow_pct']:.2f}% of Top outflow)"
        )

    print("\n========== SPSE DESTINATION DATA ==========")

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
        case_output["scatter_df"][cols_show]
        .sort_values("delta_visits", ascending=False)
        .round(4)
        .to_string(index=False)
    )

    print("\n========== DECOMPOSITION CHECK ==========")
    s = case_output["case_summary"]

    print(
        "SPSE decomposition: "
        f"sum_j ΔSPSE_j = {s['sum_delta_spse_from_pois']:.10f}; "
        f"system ΔSPSE = {s['spse_delta']:.10f}; "
        f"error = {s['spse_decomposition_error']:.10e}"
    )

    print(
        "Weighted exposure contribution decomposition: "
        f"sum_j ΔC_j = {s['sum_delta_weighted_contribution_from_pois']:.10f}; "
        f"system ΔC = {s['weighted_exposure_numerator_delta']:.10f}; "
        f"error = {s['weighted_decomposition_error']:.10e}"
    )

    print(
        "Positive-link support decomposition: "
        f"sum_j Δlinks_j = {s['sum_delta_active_links_from_pois']}; "
        f"system Δlinks = {s['total_delta_active_links']}; "
        f"error = {s['active_link_decomposition_error']}"
    )

    

    plot_fused_fig2d(case_output)


if __name__ == "__main__":
    main()