# -*- coding: utf-8 -*-
"""
Fig.3a second plotting code.

This version plots 101 values:
    b = 0%, 1%, 2%, ..., 100%

All three panels:
    left y-axis  = raw value
    right y-axis = percentage change from observed baseline

Panels:
    1) Left:  SPSE
       Right: SPSE change from baseline (%)

    2) Left:  Total travel distance
       Right: Total distance change from baseline (%)

    3) Left:  Flow-weighted exposure
       Right: Flow-weighted exposure change from baseline (%)

Required input:
    k_matrices_boston_family_budget.csv

Required columns:
    budget
    f_last
    distances_last
    social_last
    social_weight_last
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# User parameters
# ============================================================

city = "boston"
category = "Other Individual and Family Services"
cat_dir = f"matrices_A_D_S_Distribution/{category.replace(' ', '_')}"

CSV_PATH = "k_matrices_boston_family_budget.csv"

flow_path = f"{cat_dir}/flow_matrix.csv"
distance_path = f"{cat_dir}/distance_matrix.csv"
income_path = r"matrices_A_D_S_Distribution/cbg_income_level_distribution_boston_msa.csv"

OUT_PDF = "figure3a_101_values_dual_axis.pdf"
OUT_PNG = "figure3a_101_values_dual_axis.png"
OUT_CSV = "figure3a_101_values_dual_axis.csv"

EPS = 1e-10

income_levels = [
    "low_income_pct",
    "lower_middle_income_pct",
    "upper_middle_income_pct",
    "high_income_pct"
]


# ============================================================
# Helper functions
# ============================================================

def normalize_key(x):
    if pd.isna(x):
        return ""
    s = str(x).strip()
    if s.endswith(".0") and s[:-2].isdigit():
        s = s[:-2]
    return s


def normalize_df_index_columns(df):
    df = df.copy()
    df.index = [normalize_key(x) for x in df.index]
    df.columns = [normalize_key(x) for x in df.columns]
    return df


def sparse_marker(arr, step=5):
    """Return an array with values only every `step` points."""
    arr = np.asarray(arr, dtype=float)
    out = np.full(arr.shape, np.nan, dtype=float)
    out[::step] = arr[::step]
    return out


def load_cbg_income_distribution(income_path, cbg_ids, income_levels):
    if not os.path.isfile(income_path):
        raise FileNotFoundError(f"Income file not found: {income_path}")

    income_df = pd.read_csv(income_path)

    if "GEOID" not in income_df.columns:
        raise ValueError(f"`GEOID` column not found. Columns: {list(income_df.columns)}")

    income_df["GEOID"] = income_df["GEOID"].map(normalize_key)
    income_df = income_df.set_index("GEOID")

    missing_cols = [c for c in income_levels if c not in income_df.columns]
    if missing_cols:
        raise ValueError(f"Missing income columns: {missing_cols}")

    cbg_ids = [normalize_key(x) for x in cbg_ids]

    P_df = income_df.reindex(cbg_ids)[income_levels].copy()
    P_df = P_df.apply(pd.to_numeric, errors="coerce")

    if P_df.isna().any().any():
        missing_ids = P_df[P_df.isna().any(axis=1)].index.tolist()[:10]
        raise ValueError(f"Some CBGs have missing income distribution, e.g. {missing_ids}")

    P = P_df.values.astype(float)
    row_sum = P.sum(axis=1, keepdims=True)

    # If income shares are percentages, convert to proportions.
    if np.nanmedian(row_sum) > 1.5:
        P = P / 100.0
        row_sum = P.sum(axis=1, keepdims=True)

    row_sum[row_sum == 0] = np.nan
    P = P / row_sum
    P = np.nan_to_num(P, nan=0.0)

    P_df = pd.DataFrame(P, index=cbg_ids, columns=income_levels)

    print("[INCOME] loaded and normalized")
    print(f"[INCOME] n_CBGs = {P_df.shape[0]}")

    return P_df


def calculate_poi_income_distribution(H_df, P_df):
    """
    Q_j(H) = sum_i H_ij P_i / sum_i H_ij
    """
    H = H_df.values.astype(float)
    P = P_df.values.astype(float)

    poi_flow = H.sum(axis=0)

    Q = np.full((H.shape[1], P.shape[1]), np.nan, dtype=float)
    positive_poi = poi_flow > EPS

    Q[positive_poi, :] = (H[:, positive_poi].T @ P) / poi_flow[positive_poi, None]

    Q_df = pd.DataFrame(Q, index=H_df.columns, columns=P_df.columns)
    return Q_df


def calculate_social_exposure_matrix(H_df, P_df):
    """
    S_ij(H) = 1 - P_i dot Q_j(H)
    """
    Q_df = calculate_poi_income_distribution(H_df, P_df)

    P = P_df.values.astype(float)
    Q = Q_df.values.astype(float)

    S = 1.0 - (P @ Q.T)

    S_df = pd.DataFrame(S, index=H_df.index, columns=H_df.columns)
    return S_df, Q_df


def calculate_structural_potential_social_exposure(H_df, S_df):
    """
    SPSE(H) = sum_{(i,j): H_ij > 0} S_ij(H)
    """
    H = H_df.values.astype(float)
    S = S_df.values.astype(float)

    positive_mask = H > EPS

    if positive_mask.sum() == 0:
        return np.nan

    return float(np.nansum(S[positive_mask]))


def calculate_flow_weighted_exposure(H_df, S_df):
    """
    E_w(H) = sum_ij H_ij S_ij(H) / sum_ij H_ij
    """
    H = H_df.values.astype(float)
    S = S_df.values.astype(float)

    total_flow = np.nansum(H)

    if total_flow <= EPS:
        return np.nan

    return float(np.nansum(H * S) / total_flow)


def calculate_total_distance(H_df, D_df):
    """
    Total travel distance = sum_ij H_ij D_ij
    """
    H = H_df.values.astype(float)
    D = D_df.values.astype(float)
    return float(np.nansum(H * D))


def make_secondary_pct_axis(ax, baseline_value, ylabel):
    """
    Add a secondary y-axis showing percentage change from baseline.
    Left axis uses raw value.
    Right axis uses:
        (value - baseline) / baseline * 100
    """
    if abs(baseline_value) <= EPS:
        raise ValueError("Baseline value is zero or too small for percentage-axis conversion.")

    def value_to_pct(y):
        return (np.asarray(y) - baseline_value) / baseline_value * 100.0

    def pct_to_value(p):
        return baseline_value * (1.0 + np.asarray(p) / 100.0)

    secax = ax.secondary_yaxis("right", functions=(value_to_pct, pct_to_value))
    secax.set_ylabel(ylabel, labelpad=2)
    secax.tick_params(axis="y", which="major", labelsize=9, width=0.6, direction="in")

    return secax


# ============================================================
# Load budget results
# ============================================================

if not os.path.isfile(CSV_PATH):
    raise FileNotFoundError(f"CSV file not found: {CSV_PATH}")

budget_df = pd.read_csv(CSV_PATH)
budget_df = budget_df.sort_values("budget").reset_index(drop=True)

required_cols = [
    "budget",
    "f_last",
    "distances_last",
    "social_last",
    "social_weight_last"
]

missing_cols = [c for c in required_cols if c not in budget_df.columns]
if missing_cols:
    raise ValueError(
        f"Missing required columns in {CSV_PATH}: {missing_cols}\n"
        f"Available columns: {list(budget_df.columns)}"
    )

print("[LOAD] budget CSV:", CSV_PATH)
print("[LOAD] n budget rows:", len(budget_df))
print("[LOAD] columns:", list(budget_df.columns))


# ============================================================
# Load baseline matrices and compute original baseline
# ============================================================

flow_matrix = pd.read_csv(flow_path, index_col=0)
distance_matrix = pd.read_csv(distance_path, index_col=0)

flow_matrix = normalize_df_index_columns(flow_matrix)
distance_matrix = normalize_df_index_columns(distance_matrix)

distance_matrix = distance_matrix.reindex(index=flow_matrix.index, columns=flow_matrix.columns)

if distance_matrix.isna().any().any():
    raise ValueError("Distance matrix has missing values after alignment with flow matrix.")

# Same selection logic as the budget experiment:
# all POIs and CBGs with at least one observed visit.
poi_total_flow = flow_matrix.sum(axis=0)
selected_pois = poi_total_flow.sort_values(ascending=False).index.tolist()

selected_cbgs = set()
for poi in selected_pois:
    cbgs_with_flow = flow_matrix.index[flow_matrix[poi] > 0].tolist()
    selected_cbgs.update(cbgs_with_flow)

selected_cbgs = list(selected_cbgs)

F_df = flow_matrix.loc[selected_cbgs, selected_pois].astype(float)
D_df = distance_matrix.loc[selected_cbgs, selected_pois].astype(float)

P_df = load_cbg_income_distribution(income_path, selected_cbgs, income_levels)
P_df = P_df.reindex(F_df.index)

S_base_df, _ = calculate_social_exposure_matrix(F_df, P_df)

baseline_spse = calculate_structural_potential_social_exposure(F_df, S_base_df)
baseline_flow_weighted_exposure = calculate_flow_weighted_exposure(F_df, S_base_df)
baseline_total_distance = calculate_total_distance(F_df, D_df)
baseline_total_flow = float(F_df.values.sum())

print("\n========== ORIGINAL BASELINE ==========")
print(f"n_CBGs: {F_df.shape[0]}")
print(f"n_POIs: {F_df.shape[1]}")
print(f"baseline total flow: {baseline_total_flow:.6f}")
print(f"baseline SPSE: {baseline_spse:.6f}")
print(f"baseline flow-weighted exposure: {baseline_flow_weighted_exposure:.6f}")
print(f"baseline total travel distance: {baseline_total_distance:.6f}")


# ============================================================
# Build plotting table: 101 values
# ============================================================

budget_part = budget_df[[
    "budget",
    "f_last",
    "distances_last",
    "social_last",
    "social_weight_last"
]].copy()

budget_part = budget_part.rename(columns={
    "f_last": "objective",
    "distances_last": "total_distance",
    "social_last": "spse",
    "social_weight_last": "flow_weighted_exposure"
})

budget_part["budget_pct"] = budget_part["budget"].astype(float) * 100.0

# Add b = 0 observed baseline row.
baseline_row = pd.DataFrame([{
    "budget": 0.0,
    "budget_pct": 0.0,
    "objective": np.nan,
    "total_distance": baseline_total_distance,
    "spse": baseline_spse,
    "flow_weighted_exposure": baseline_flow_weighted_exposure
}])

fig_df = pd.concat([baseline_row, budget_part], ignore_index=True)
fig_df = fig_df.sort_values("budget_pct").reset_index(drop=True)

# Right-axis percentage changes.
fig_df["spse_change_pct"] = (
    (fig_df["spse"] - baseline_spse) / baseline_spse * 100.0
)

fig_df["total_distance_change_pct"] = (
    (fig_df["total_distance"] - baseline_total_distance)
    / baseline_total_distance
    * 100.0
)

fig_df["flow_weighted_exposure_change_pct"] = (
    (fig_df["flow_weighted_exposure"] - baseline_flow_weighted_exposure)
    / baseline_flow_weighted_exposure
    * 100.0
)

if len(fig_df) != 101:
    print(f"[WARN] Expected 101 rows, but got {len(fig_df)} rows.")


# ============================================================
# Print Figure 3a key values
# ============================================================

key_budgets = [
    0.01, 0.05, 0.10, 0.15, 0.20,
    0.30, 0.40, 0.60, 0.80, 1.00
]

key_rows = []
for b in key_budgets:
    diff = (fig_df["budget"].astype(float) - b).abs()
    idx = diff.idxmin()

    if diff.loc[idx] > 1e-8:
        print(
            f"[WARN] budget={b:.2f} was not found exactly; "
            f"using nearest budget={fig_df.loc[idx, 'budget']:.6f}"
        )

    key_rows.append(fig_df.loc[idx])

key_values_df = pd.DataFrame(key_rows)[[
    "budget",
    "objective",
    "total_distance",
    "spse",
    "flow_weighted_exposure"
]].copy()

key_values_df = key_values_df.rename(columns={
    "total_distance": "distance",
    "spse": "social_exposure"
})

print("\n=== Figure 3a key values ===")
print(
    key_values_df.to_string(
        index=False,
        formatters={
            "budget": lambda x: f"{x:.2f}",
            "objective": lambda x: f"{x:.6f}",
            "distance": lambda x: f"{x:.6f}",
            "social_exposure": lambda x: f"{x:.6f}",
            "flow_weighted_exposure": lambda x: f"{x:.6f}",
        }
    )
)


# ============================================================
# Extract arrays
# ============================================================

x_budget = fig_df["budget_pct"].values.astype(float)

spse_value = fig_df["spse"].values.astype(float)
total_distance_value = fig_df["total_distance"].values.astype(float)
flow_weighted_exposure_value = fig_df["flow_weighted_exposure"].values.astype(float)


# ============================================================
# Plot
# ============================================================

color_static = "#4C78A8"
color_dynamic = "#8E5EA2"

colors = ["#3498db", "#7c5bb8", "#e74c3c"]   # blue / purple / red
colors = ["#4C78A8", "#8E5EA2", "#C76B6B"]   # blue / purple / red
markers = ["o", "s", "^"]

plt.rcParams.update({
    "font.size": 10,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "grid.linewidth": 0.4,
    "grid.alpha": 0.5,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

fig, axs = plt.subplots(1, 3, figsize=(15, 5), dpi=300)

# ------------------------------------------------------------
# Panel 2: Total travel distance
# Left axis: total travel distance
# Right axis: distance change from baseline (%)
# ------------------------------------------------------------

ax = axs[0]

ax.plot(
    x_budget,
    total_distance_value,
    color=colors[0],
    linewidth=3
)

ax.plot(
    x_budget,
    sparse_marker(total_distance_value),
    color=colors[0],
    marker=markers[0],
    markersize=8,
    markeredgewidth=1,
    markerfacecolor="w",
    linestyle="None",
    clip_on=False
)

ax.set_xlabel("Reallocation budget, $\gamma$", labelpad=2)
ax.set_ylabel("Total travel distance (km)", labelpad=2)
ax.set_title("Travel cost")
ax.grid(True, linestyle="--", alpha=0.4)
ax.set_xlim(x_budget.min(), x_budget.max())

make_secondary_pct_axis(
    ax,
    baseline_total_distance,
    "Distance change from baseline (%)"
)


# ------------------------------------------------------------
# Panel 2: Structural potential social exposure
# Left axis: SPSE
# Right axis: SPSE change from baseline (%)
# ------------------------------------------------------------

ax = axs[1]

ax.plot(
    x_budget,
    spse_value,
    color=colors[1],
    linewidth=3
)

ax.plot(
    x_budget,
    sparse_marker(spse_value),
    color=colors[1],
    marker=markers[1],
    markersize=8,
    markeredgewidth=1,
    markerfacecolor="w",
    linestyle="None",
    clip_on=False
)

ax.set_xlabel("Reallocation budget, $\gamma$", labelpad=2)
ax.set_ylabel("Structural potential social exposure", labelpad=2)
ax.set_title("Structural exposure")
ax.grid(True, linestyle="--", alpha=0.4)
ax.set_xlim(x_budget.min(), x_budget.max())

make_secondary_pct_axis(
    ax,
    baseline_spse,
    "SPSE change from baseline (%)"
)



# ------------------------------------------------------------
# Panel 3: Flow-weighted exposure
# Left axis: flow-weighted exposure
# Right axis: flow-weighted exposure change from baseline (%)
# ------------------------------------------------------------

ax = axs[2]

ax.plot(
    x_budget,
    flow_weighted_exposure_value,
    color=colors[2],
    linewidth=3
)

ax.plot(
    x_budget,
    sparse_marker(flow_weighted_exposure_value),
    color=colors[2],
    marker=markers[2],
    markersize=8,
    markeredgewidth=1,
    markerfacecolor="w",
    linestyle="None",
    clip_on=False
)

ax.set_xlabel("Reallocation budget, $\gamma$", labelpad=2)
ax.set_ylabel("Flow-weighted exposure", labelpad=2)
ax.set_title("Flow-weighted exposure")
ax.grid(True, linestyle="--", alpha=0.4)
ax.set_xlim(x_budget.min(), x_budget.max())

make_secondary_pct_axis(
    ax,
    baseline_flow_weighted_exposure,
    "Flow-weighted exposure change (%)"
)


# ------------------------------------------------------------
# Common formatting
# ------------------------------------------------------------

for ax in axs:
    ax.tick_params(axis="both", which="major", labelsize=9)
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.set_xticklabels(["0", "20", "40", "60", "80", "100"])

plt.tight_layout()

plt.savefig('figure3a.pdf',
            format='pdf',
            dpi=300,
            bbox_inches='tight',
            transparent=False,
            backend='pdf')
plt.show()


# ============================================================
# 输出 Fig.3a 正文对应的关键数值
# ============================================================

def get_budget_row(budget):
    idx = (fig_df["budget"].astype(float) - budget).abs().idxmin()
    return fig_df.loc[idx]


row_001 = get_budget_row(0.01)
row_020 = get_budget_row(0.20)

positive_df = fig_df[fig_df["budget"] > 0].copy()
max_spse_row = positive_df.loc[positive_df["spse"].idxmax()]
last_spse = spse_value[-1]

spse_recovered_pct = (
    (row_020["spse"] - row_001["spse"])
    / (last_spse - row_001["spse"])
    * 100
)

large_budget_df = fig_df[fig_df["budget"] >= 0.20]
plateau_df = fig_df[fig_df["budget"] >= 0.40]

print("\n========== Fig.3a manuscript values ==========")

print(
    f"Baseline: SPSE={baseline_spse:.4f}, "
    f"flow-weighted exposure={baseline_flow_weighted_exposure:.6f}, "
    f"distance={baseline_total_distance:.2f} km"
)

print(
    f"gamma=0.01: SPSE={row_001['spse']:.4f} "
    f"({row_001['spse_change_pct']:+.2f}%), "
    f"flow-weighted exposure={row_001['flow_weighted_exposure']:.6f} "
    f"({row_001['flow_weighted_exposure_change_pct']:+.2f}%), "
    f"distance={row_001['total_distance']:.2f} km "
    f"({row_001['total_distance_change_pct']:+.2f}%)"
)

print(
    f"gamma=0.20: SPSE={row_020['spse']:.4f} "
    f"({row_020['spse_change_pct']:+.2f}%), "
    f"flow-weighted exposure={row_020['flow_weighted_exposure']:.6f} "
    f"({row_020['flow_weighted_exposure_change_pct']:+.2f}%), "
    f"distance={row_020['total_distance']:.2f} km "
    f"({row_020['total_distance_change_pct']:+.2f}%)"
)


print(
    f"Share of the gamma=0.01-to-last SPSE gain "
    f"recovered by gamma=0.20: {spse_recovered_pct:.2f}%"
)

print(
    f"Flow-weighted exposure for gamma>=0.20: "
    f"mean={large_budget_df['flow_weighted_exposure'].mean():.6f}, "
    f"range=[{large_budget_df['flow_weighted_exposure'].min():.6f}, "
    f"{large_budget_df['flow_weighted_exposure'].max():.6f}]"
)

print(
    f"For gamma>=0.40: "
    f"SPSE range={plateau_df['spse'].max() - plateau_df['spse'].min():.4f}, "
    f"distance range="
    f"{plateau_df['total_distance'].max() - plateau_df['total_distance'].min():.2f} km, "
    f"flow-weighted exposure range="
    f"{plateau_df['flow_weighted_exposure'].max() - plateau_df['flow_weighted_exposure'].min():.6f}"
)

print(
    "Distance below baseline at every positive budget:",
    bool((positive_df["total_distance"] < baseline_total_distance).all())
)
