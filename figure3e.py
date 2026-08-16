# -*- coding: utf-8 -*-
"""
Fig.3e: Distribution of origin-level reassigned visits across budgets

For each active origin i and budget k:
    A_i(k) = 0.5 * sum_j |H_ij(k) - F_ij|

The figure plots the empirical CDF:
    x-axis: reassigned visits per origin
    y-axis: cumulative share of active origins
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import pickle
import os


# ============================================================
# Basic settings
# ============================================================
city = "boston"
category = "Other Individual and Family Services"

cat_dir = f"matrices_A_D_S_Distribution/{category.replace(' ', '_')}"
k_matrix_path = "k_matrices_boston_family_budget.pkl"

plot_ks = [0.01, 0.05, 0.20]

outdir = "k_flow_change_outputs"
os.makedirs(outdir, exist_ok=True)

fig_path = "figure3e.pdf"
dpi = 300


# ============================================================
# Load data
# ============================================================
flow_matrix = pd.read_csv(f"{cat_dir}/flow_matrix.csv", index_col=0)

boston_msa_cbg = gpd.read_file("geo_data/tl_2021_boston_msa_bg/tl_2021_boston_msa_bg.shp")
boston_msa_cbg["GEOID"] = boston_msa_cbg["GEOID"].astype(str)

pad_len = int(boston_msa_cbg["GEOID"].str.len().max())

with open(k_matrix_path, "rb") as f:
    k_matrices = pickle.load(f)


# ============================================================
# Build baseline matrix
# ============================================================
poi_total_flow = flow_matrix.sum(axis=0)
poi_num = flow_matrix.shape[1]
selected_pois = poi_total_flow.sort_values(ascending=False).head(poi_num).index.tolist()

selected_cbgs = set()
for poi in selected_pois:
    cbgs_with_flow = flow_matrix.index[flow_matrix[poi] > 0].tolist()
    selected_cbgs.update(cbgs_with_flow)

selected_cbgs = list(selected_cbgs)

baseline = flow_matrix.loc[selected_cbgs, selected_pois].copy()

baseline.index = [str(x).zfill(pad_len) for x in baseline.index]
baseline.columns = [str(x) for x in baseline.columns]

print("Baseline shape:", baseline.shape)
print("Baseline total flow:", baseline.values.sum())


# ============================================================
# Helper functions
# ============================================================
def find_best_key_in_kmat(k_val, k_matrices):
    """
    Find the key in k_matrices closest to k_val.
    """
    best = None
    bestd = 1e9

    for kk in k_matrices.keys():
        try:
            d = abs(float(kk) - float(k_val))
            if d < bestd:
                bestd = d
                best = kk
        except Exception:
            pass

    return best


def compute_reassigned_visits_per_origin(F_df, H_df):
    """
    Compute origin-level reassigned visits:

        A_i(k) = 0.5 * sum_j |H_ij(k) - F_ij|

    The 0.5 factor avoids double-counting one reassignment as both
    a decrease at one destination and an increase at another destination.
    """
    F = F_df.copy()
    H = H_df.copy()

    H_aligned = H.reindex(index=F.index, columns=F.columns, fill_value=0.0)
    F_aligned = F.reindex(index=H_aligned.index, columns=H_aligned.columns, fill_value=0.0)

    diff = (H_aligned - F_aligned).abs()

    per_origin = 0.5 * diff.sum(axis=1)

    return per_origin.sort_values(ascending=False)


def get_H_for_k(k_val, k_matrices, baseline, pad_len):
    """
    Load and align optimized matrix for a given k.
    """
    H = None

    if k_val in k_matrices:
        H = k_matrices[k_val]
    else:
        best = find_best_key_in_kmat(k_val, k_matrices)
        if best is not None:
            H = k_matrices[best]

    if H is None:
        return None

    H_local = H.copy()

    try:
        H_local.index = [str(x).zfill(pad_len) for x in H_local.index]
    except Exception:
        H_local.index = H_local.index.astype(str)

    H_local.columns = [str(x) for x in H_local.columns]

    H_local = H_local.reindex(
        index=baseline.index,
        columns=baseline.columns,
        fill_value=0.0
    )

    return H_local


# ============================================================
# Plot Fig.3e
# ============================================================
plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10.5,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 8.5
})

line_styles = ["-", "--", ":"]
colors = ["#3498db", "#7c5bb8", "#e74c3c"]

fig, ax = plt.subplots(figsize=(5, 5), dpi=dpi)

summary_rows = []

for i, k in enumerate(plot_ks):
    H_local = get_H_for_k(
        k_val=k,
        k_matrices=k_matrices,
        baseline=baseline,
        pad_len=pad_len
    )

    if H_local is None:
        print(f"[WARN] skip k={k}, matrix not found")
        continue

    per_origin = compute_reassigned_visits_per_origin(baseline, H_local)

    vals = per_origin.values
    sorted_vals = np.sort(vals).ravel()

    n = len(sorted_vals)
    y = np.arange(1, n + 1) / n

    n_zero = int((sorted_vals == 0).sum())
    n_changed = n - n_zero

    budget_label = f"{int(round(k * 100))}% budget"

    ax.step(
        sorted_vals,
        y,
        where="post",
        linestyle=line_styles[i % len(line_styles)],
        color=colors[i % len(colors)],
        linewidth=2.4,
        label=f"{budget_label}, unchanged {n_zero}/{n}"
    )

    summary_rows.append({
        "k": k,
        "budget_label": budget_label,
        "n_origins": n,
        "n_changed": n_changed,
        "n_unchanged": n_zero,
        "unchanged_share": n_zero / n if n > 0 else np.nan,
        "total_reassigned_visits": vals.sum(),
        "median_reassigned_visits": np.median(vals),
        "p90_reassigned_visits": np.percentile(vals, 90),
        "p99_reassigned_visits": np.percentile(vals, 99),
        "max_reassigned_visits": np.max(vals)
    })

# Save summary table
summary_df = pd.DataFrame(summary_rows)
summary_path = os.path.join(outdir, "figure3e_reassigned_visits_cdf_summary.csv")
summary_df.to_csv(summary_path, index=False)

print("\nFig.3e summary:")
print(summary_df)
print(f"\nSaved summary to: {summary_path}")


# ============================================================
# Figure styling
# ============================================================
ax.set_xlabel("Reassigned visits per origin")
ax.set_ylabel("Cumulative share of active origins")
ax.set_title("Distribution of origin-level reassigned visits", pad=7)

ax.set_ylim(0, 1.02)


ax.legend(
    loc="lower right",
    fontsize="small",
    frameon=True,
    framealpha=0.95,
    edgecolor="0.8"
)

fig.subplots_adjust(
    left=0.14,
    right=0.98,
    bottom=0.14,
    top=0.90
)

plt.savefig(
    fig_path,
    format="pdf",
    dpi=dpi,
    bbox_inches="tight",
    pad_inches=0.01,
    transparent=False,
    backend="pdf"
)

print(f"\nSaved figure to: {fig_path}")

plt.show()