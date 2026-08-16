# -*- coding: utf-8 -*-
"""
Figure 4d: Behavioural anchoring and retained performance

Final design:
    Panel a:
        Main: origin-level behavioural pullback toward observed matrix F.
              CBGs are ranked by c_i^F = d_i(H_no,F) - d_i(H_reg,F).
              The two curves show d_i(H_no,F) and d_i(H_reg,F).
        Inset: matrix-level pullback after L1 regularization, excluding D(F,R).

    Panel b:
        Total travel distance trajectory.

    Panel c:
        Structural potential social exposure trajectory.
        Left axis: raw SPSE.
        Right axis: percentage change relative to observed baseline.

Input files:
    results_regu_boston_624190.csv
    results_boston_624190.csv
    H_opt_df_no_regu_boston_624190.pkl
    H_opt_df_regu_boston_624190.pkl
    pred_rownorm_int_preserve.csv
    flow_matrix.csv
    distance_matrix.csv
    social_exposure_matrix.csv

Output:
    figure4d.pdf
    figure4d.png
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


# ============================================================
# Style
# ============================================================

try:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
except Exception:
    pass

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12.5,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9.5,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


# ============================================================
# Configuration
# ============================================================

city = "boston"
category = "Other Individual and Family Services"
poi_code = "624190"

cat_dir = f"matrices_A_D_S_Distribution/{category.replace(' ', '_')}"

regu_csv_path = os.path.join(cat_dir, f"results_regu_{city}_{poi_code}.csv")
dynamic_csv_path = os.path.join(cat_dir, f"results_{city}_{poi_code}.csv")

no_regu_file = os.path.join(cat_dir, f"H_opt_df_no_regu_{city}_{poi_code}.pkl")
regu_file = os.path.join(cat_dir, f"H_opt_df_regu_{city}_{poi_code}.pkl")

flow_matrix_path = os.path.join(cat_dir, "flow_matrix.csv")
distance_matrix_path = os.path.join(cat_dir, "distance_matrix.csv")
social_matrix_path = os.path.join(cat_dir, "social_exposure_matrix.csv")
R_path = os.path.join(cat_dir, "pred_rownorm_int_preserve.csv")

out_pdf = "figure4d.pdf"
out_png = "figure4d.png"

dpi = 300
figsize = (15.5, 5.0)

dynamic_color = "#3498db"
regu_color = "#7C5BB8"
baseline_color = "0.45"

curve_F_color = "#5C5C5C"
curve_R_color = "#7E57C2"


# ============================================================
# Helper functions
# ============================================================

def check_file(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Cannot find file: {path}")


def to_numeric_array(df, col):
    return pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)


def pct_change(new, base):
    if base == 0 or np.isnan(base) or np.isnan(new):
        return np.nan
    return (new - base) / abs(base) * 100.0


def pct_axis_formatter(x, pos):
    return f"{x:.0f}%"


def add_percent_secondary_axis(ax, baseline, label="Change from baseline (%)"):
    """
    Add a right y-axis tied to the raw-value left y-axis.
    """
    if baseline == 0 or np.isnan(baseline):
        return None

    def value_to_pct(y):
        y = np.asarray(y, dtype=float)
        return (y - baseline) / abs(baseline) * 100.0

    def pct_to_value(p):
        p = np.asarray(p, dtype=float)
        return baseline + p / 100.0 * abs(baseline)

    secax = ax.secondary_yaxis("right", functions=(value_to_pct, pct_to_value))
    secax.set_ylabel(label, labelpad=9)
    secax.yaxis.set_major_formatter(FuncFormatter(pct_axis_formatter))
    secax.tick_params(axis="y", labelsize=10)
    return secax


def fmt_num(x):
    if pd.isna(x):
        return "NA"
    if abs(x) >= 100:
        return f"{x:.2f}"
    elif abs(x) >= 1:
        return f"{x:.4f}"
    else:
        return f"{x:.6f}"


def fmt_pct(x):
    if pd.isna(x):
        return "NA"
    return f"{x:.2f}\\%"


def infer_cbg_pad_len(index_values):
    """
    CBG GEOID is usually 12 digits. Use 12 as a safe lower bound.
    """
    lengths = [len(str(x)) for x in index_values]
    return max(12, max(lengths))


def load_H(path, baseline_df, pad_len):
    """
    Load optimized H matrix and make index/columns string-compatible.
    """
    check_file(path)

    with open(path, "rb") as f:
        H = pickle.load(f)

    if isinstance(H, pd.DataFrame):
        H = H.copy()
        H.index = [str(x).zfill(pad_len) for x in H.index]
        H.columns = [str(x) for x in H.columns]
        return H
    else:
        return pd.DataFrame(H, index=baseline_df.index, columns=baseline_df.columns)


def align_to_baseline(df, baseline_df, pad_len, fill_value=0.0):
    """
    Align any matrix to baseline rows and columns.
    Rows are zero-padded CBG GEOIDs; columns are POI ids as strings.
    """
    X = df.copy()
    X.index = X.index.map(lambda x: str(x))
    X.columns = X.columns.map(lambda x: str(x))

    X_try = X.reindex(index=baseline_df.index, columns=baseline_df.columns)

    if X_try.notna().values.any():
        return X_try.fillna(fill_value)

    X2 = X.copy()
    try:
        X2.index = X2.index.map(lambda x: str(x).zfill(pad_len))
    except Exception:
        pass

    X_try2 = X2.reindex(index=baseline_df.index, columns=baseline_df.columns)
    return X_try2.fillna(fill_value)


def l1_flow_distance_share(H, R, total_flow):
    """
    Normalized L1 flow distance:
        0.5 * sum |H - R| / total_flow * 100

    Interpretation:
        percentage of total visits that would need to be reassigned
        to transform one matrix into the other.
    """
    if total_flow == 0 or np.isnan(total_flow):
        return np.nan

    H_arr = H.to_numpy(dtype=float)
    R_arr = R.to_numpy(dtype=float)

    return 0.5 * np.abs(H_arr - R_arr).sum() / total_flow * 100.0


def origin_l1_raw(A, B):
    """
    Origin-level raw L1 reassignment volume:
        0.5 * sum_j |A_ij - B_ij|

    Interpretation:
        for each origin i, how many visits would need to be reassigned
        to transform row A_i into row B_i.
    """
    return 0.5 * (A.astype(float) - B.astype(float)).abs().sum(axis=1)


def gain_retention(lower_is_better, baseline, dynamic_final, regu_final):
    """
    Retained share of the unregularized gain after regularization.
    """
    if lower_is_better:
        dynamic_gain = baseline - dynamic_final
        regu_gain = baseline - regu_final
    else:
        dynamic_gain = dynamic_final - baseline
        regu_gain = regu_final - baseline

    if dynamic_gain == 0 or np.isnan(dynamic_gain) or np.isnan(regu_gain):
        return np.nan, dynamic_gain, regu_gain

    retention = regu_gain / dynamic_gain * 100.0
    return retention, dynamic_gain, regu_gain


def top_share(sorted_values, top_pct):
    """
    Share of total positive contribution accounted for by top_pct origins.
    sorted_values must be sorted in descending order.
    """
    values = np.asarray(sorted_values, dtype=float)
    n = len(values)
    if n == 0 or values.sum() <= 0:
        return np.nan

    k = int(np.ceil(n * top_pct / 100.0))
    k = min(max(k, 1), n)
    return values[:k].sum() / values.sum() * 100.0


def smooth_array(values, smooth_fraction=0.06):
    """
    Mild centered rolling mean for cleaner line plots.
    Set smooth_fraction=0 to disable smoothing.
    """
    values = np.asarray(values, dtype=float)

    if len(values) == 0:
        return values

    if smooth_fraction <= 0:
        return values

    window = max(3, int(round(len(values) * smooth_fraction)))

    return (
        pd.Series(values)
        .rolling(window=window, center=True, min_periods=1)
        .mean()
        .to_numpy(dtype=float)
    )


def build_ranked_departure_curves(
    dev_no,
    dev_reg,
    contrib,
    changed,
    tol=1e-9,
    smooth_fraction=0.06
):
    """
    Build ranked origin-level departure curves for Panel a.

    Selection:
        keep all origins whose allocation changes between H_no and H_reg.

    Ranking:
        CBGs are ranked by c_i^F = d_i(H_no,F) - d_i(H_reg,F), descending.

    Interpretation:
        Left side: CBGs with largest positive pullback toward F.
        Right side: CBGs with small, zero, or negative pullback toward F.

    Returns:
        x_rank: rank position among changed CBGs, 0-100
        dev_no_plot: smoothed d_i(H_no,F)
        dev_reg_plot: smoothed d_i(H_reg,F)
        dev_no_raw: raw ranked d_i(H_no,F)
        dev_reg_raw: raw ranked d_i(H_reg,F)
        contrib_raw: raw ranked c_i^F
        changed_raw: raw ranked change volume between H_no and H_reg
    """
    mask_changed = changed > tol

    dev_no_sel = dev_no[mask_changed].copy()
    dev_reg_sel = dev_reg[mask_changed].copy()
    contrib_sel = contrib[mask_changed].copy()
    changed_sel = changed[mask_changed].copy()

    if len(contrib_sel) == 0:
        return (
            np.array([]),
            np.array([]),
            np.array([]),
            np.array([]),
            np.array([]),
            np.array([]),
            np.array([])
        )

    # Rank all changed CBGs by anchoring contribution toward F
    rank_index = contrib_sel.sort_values(ascending=False).index

    dev_no_raw = dev_no_sel.loc[rank_index].to_numpy(dtype=float)
    dev_reg_raw = dev_reg_sel.loc[rank_index].to_numpy(dtype=float)
    contrib_raw = contrib_sel.loc[rank_index].to_numpy(dtype=float)
    changed_raw = changed_sel.loc[rank_index].to_numpy(dtype=float)

    x_rank = np.arange(1, len(dev_no_raw) + 1) / len(dev_no_raw) * 100.0

    dev_no_plot = smooth_array(dev_no_raw, smooth_fraction=smooth_fraction)
    dev_reg_plot = smooth_array(dev_reg_raw, smooth_fraction=smooth_fraction)

    return x_rank, dev_no_plot, dev_reg_plot, dev_no_raw, dev_reg_raw, contrib_raw, changed_raw


# ============================================================
# Check inputs
# ============================================================

for path in [
    regu_csv_path,
    dynamic_csv_path,
    no_regu_file,
    regu_file,
    flow_matrix_path,
    distance_matrix_path,
    social_matrix_path,
    R_path,
]:
    check_file(path)


# ============================================================
# Load baseline matrices
# ============================================================

flow_matrix = pd.read_csv(flow_matrix_path, index_col=0)
distance_matrix = pd.read_csv(distance_matrix_path, index_col=0)
social_exposure_matrix = pd.read_csv(social_matrix_path, index_col=0)
R_pre = pd.read_csv(R_path, index_col=0)

pad_len = infer_cbg_pad_len(flow_matrix.index)

# Select POIs used in the optimization domain
poi_total_flow = flow_matrix.sum(axis=0)
selected_pois = poi_total_flow.sort_values(ascending=False).head(53).index.tolist()

# Select all CBGs with positive flow to selected POIs
selected_cbgs = set()
for poi in selected_pois:
    cbgs_with_flow = flow_matrix.index[flow_matrix[poi] > 0].tolist()
    selected_cbgs.update(cbgs_with_flow)
selected_cbgs = list(selected_cbgs)

baseline = flow_matrix.loc[selected_cbgs, selected_pois].copy()
baseline.index = [str(x).zfill(pad_len) for x in baseline.index]
baseline.columns = [str(x) for x in baseline.columns]
baseline = baseline.apply(pd.to_numeric, errors="coerce").fillna(0.0)

total_flow = baseline.to_numpy(dtype=float).sum()

print("\n========== Matrix domain ==========")
print(f"n_CBGs: {baseline.shape[0]}")
print(f"n_POIs: {baseline.shape[1]}")
print(f"baseline total flow: {total_flow:.6f}")


# ============================================================
# Load and align H_no, H_reg, R, distance, social matrices
# ============================================================

H_no_pre = load_H(no_regu_file, baseline, pad_len)
H_reg_pre = load_H(regu_file, baseline, pad_len)

H_no = align_to_baseline(H_no_pre, baseline, pad_len, fill_value=0.0)
H_reg = align_to_baseline(H_reg_pre, baseline, pad_len, fill_value=0.0)
R = align_to_baseline(R_pre, baseline, pad_len, fill_value=0.0)

D = align_to_baseline(distance_matrix, baseline, pad_len, fill_value=np.nan)
S0 = align_to_baseline(social_exposure_matrix, baseline, pad_len, fill_value=np.nan)

H_no = H_no.apply(pd.to_numeric, errors="coerce").fillna(0.0)
H_reg = H_reg.apply(pd.to_numeric, errors="coerce").fillna(0.0)
R = R.apply(pd.to_numeric, errors="coerce").fillna(0.0)
D = D.apply(pd.to_numeric, errors="coerce")
S0 = S0.apply(pd.to_numeric, errors="coerce")


# ============================================================
# Observed baseline values from matrices
# ============================================================

baseline_distance = float((baseline.to_numpy(dtype=float) * D.to_numpy(dtype=float)).sum())

baseline_social = float(
    np.nansum((baseline.to_numpy(dtype=float) > 0).astype(float) * S0.to_numpy(dtype=float))
)

print("\n========== Observed baseline values from matrices ==========")
print(f"baseline total distance: {baseline_distance:.6f}")
print(f"baseline structural social exposure: {baseline_social:.6f}")


# ============================================================
# Matrix-level behavioural anchoring diagnostics
# ============================================================

no_regu_ref_dev_pct = l1_flow_distance_share(H_no, R, total_flow)
regu_ref_dev_pct = l1_flow_distance_share(H_reg, R, total_flow)

if no_regu_ref_dev_pct == 0 or np.isnan(no_regu_ref_dev_pct) or np.isnan(regu_ref_dev_pct):
    ref_dev_reduction_pct = np.nan
else:
    ref_dev_reduction_pct = (
        (no_regu_ref_dev_pct - regu_ref_dev_pct) / no_regu_ref_dev_pct * 100.0
    )

no_regu_from_F_pct = l1_flow_distance_share(H_no, baseline, total_flow)
regu_from_F_pct = l1_flow_distance_share(H_reg, baseline, total_flow)
R_from_F_pct = l1_flow_distance_share(R, baseline, total_flow)

if no_regu_from_F_pct == 0 or np.isnan(no_regu_from_F_pct) or np.isnan(regu_from_F_pct):
    from_F_reduction_pct = np.nan
else:
    from_F_reduction_pct = (
        (no_regu_from_F_pct - regu_from_F_pct) / no_regu_from_F_pct * 100.0
    )

print("\n========== Behavioural anchoring diagnostics ==========")
print(f"Deviation from R, unregularized H_no:  {no_regu_ref_dev_pct:.6f}%")
print(f"Deviation from R, regularized H_reg:   {regu_ref_dev_pct:.6f}%")
print(f"Reduction in deviation from R:         {ref_dev_reduction_pct:.6f}%")

print(f"\nDeviation from observed F, unregularized H_no: {no_regu_from_F_pct:.6f}%")
print(f"Deviation from observed F, regularized H_reg:  {regu_from_F_pct:.6f}%")
print(f"Reduction in deviation from observed F:        {from_F_reduction_pct:.6f}%")
print(f"Diagnostic only, not shown in figure: D(F,R) = {R_from_F_pct:.6f}%")


# ============================================================
# Origin-level anchoring contribution for Panel a
# ============================================================

# Raw origin-level departure from F
dev_no_from_F_raw = origin_l1_raw(H_no, baseline)
dev_reg_from_F_raw = origin_l1_raw(H_reg, baseline)
contrib_from_F_raw = dev_no_from_F_raw - dev_reg_from_F_raw

# Raw origin-level departure from R, used for diagnostics only
dev_no_from_R_raw = origin_l1_raw(H_no, R)
dev_reg_from_R_raw = origin_l1_raw(H_reg, R)
contrib_from_R_raw = dev_no_from_R_raw - dev_reg_from_R_raw

# Positive contributions
tol = 1e-9
contrib_from_F_pos = contrib_from_F_raw.clip(lower=0.0)
contrib_from_R_pos = contrib_from_R_raw.clip(lower=0.0)

# Negative offsets
contrib_from_F_neg = (-contrib_from_F_raw.clip(upper=0.0))
contrib_from_R_neg = (-contrib_from_R_raw.clip(upper=0.0))

# Convert net raw reduction to percentage points of total flow
net_reduction_from_F_pp = contrib_from_F_raw.sum() / total_flow * 100.0
net_reduction_from_R_pp = contrib_from_R_raw.sum() / total_flow * 100.0

positive_reduction_from_F_pp = contrib_from_F_pos.sum() / total_flow * 100.0
positive_reduction_from_R_pp = contrib_from_R_pos.sum() / total_flow * 100.0

negative_offset_from_F_pp = contrib_from_F_neg.sum() / total_flow * 100.0
negative_offset_from_R_pp = contrib_from_R_neg.sum() / total_flow * 100.0

# Sorted positive contributions for summary diagnostics
sorted_F_pos = np.sort(contrib_from_F_pos[contrib_from_F_pos > tol].to_numpy(dtype=float))[::-1]
sorted_R_pos = np.sort(contrib_from_R_pos[contrib_from_R_pos > tol].to_numpy(dtype=float))[::-1]

top10_F = top_share(sorted_F_pos, 10)
top20_F = top_share(sorted_F_pos, 20)
top10_R = top_share(sorted_R_pos, 10)
top20_R = top_share(sorted_R_pos, 20)

share_positive_F = (contrib_from_F_pos > tol).mean() * 100.0
share_positive_R = (contrib_from_R_pos > tol).mean() * 100.0
share_positive_both = ((contrib_from_F_pos > tol) & (contrib_from_R_pos > tol)).mean() * 100.0

change_no_to_reg_raw = origin_l1_raw(H_no, H_reg)

# Ranked departure curves for Panel a
(
    x_rank_F,
    dev_no_F_plot,
    dev_reg_F_plot,
    dev_no_F_rank_raw,
    dev_reg_F_rank_raw,
    contrib_F_rank_raw,
    changed_F_rank_raw
) = build_ranked_departure_curves(
    dev_no=dev_no_from_F_raw,
    dev_reg=dev_reg_from_F_raw,
    contrib=contrib_from_F_raw,
    changed=change_no_to_reg_raw,
    tol=tol,
    smooth_fraction=0.06
)

if len(x_rank_F) == 0:
    raise ValueError("No positive-contribution CBGs found for c_i^F.")

print("\n========== Panel a anchoring contribution diagnostics ==========")
print(f"Net reduction from F:       {net_reduction_from_F_pp:.6f} pp")
print(f"Positive reduction from F:  {positive_reduction_from_F_pp:.6f} pp")
print(f"Negative offset from F:     {negative_offset_from_F_pp:.6f} pp")
print(f"Origins with positive F contribution: {share_positive_F:.2f}%")
print(f"Top 10% positive F origins account for F reduction: {top10_F:.2f}%")
print(f"Top 20% positive F origins account for F reduction: {top20_F:.2f}%")

print(f"\nNet reduction from R:       {net_reduction_from_R_pp:.6f} pp")
print(f"Positive reduction from R:  {positive_reduction_from_R_pp:.6f} pp")
print(f"Negative offset from R:     {negative_offset_from_R_pp:.6f} pp")
print(f"Origins with positive R contribution: {share_positive_R:.2f}%")
print(f"Top 10% positive R origins account for R reduction: {top10_R:.2f}%")
print(f"Top 20% positive R origins account for R reduction: {top20_R:.2f}%")
print(f"\nOrigins with positive contribution to both F and R: {share_positive_both:.2f}%")


# ============================================================
# Load trajectory CSVs
# ============================================================

df_regu = pd.read_csv(regu_csv_path)
df_dynamic = pd.read_csv(dynamic_csv_path)

required_cols = ["f_values_iter", "distances_iter", "social_iter"]

for name, df in [("Regularized", df_regu), ("Unregularized continuous", df_dynamic)]:
    miss = [c for c in required_cols if c not in df.columns]
    if miss:
        print(f"[ERROR] {name} file missing columns: {miss}")
        print(f"Current columns: {list(df.columns)}")
        sys.exit(1)

regu_f = to_numeric_array(df_regu, "f_values_iter")
regu_d = to_numeric_array(df_regu, "distances_iter")
regu_s = to_numeric_array(df_regu, "social_iter")

dyn_f = to_numeric_array(df_dynamic, "f_values_iter")
dyn_d = to_numeric_array(df_dynamic, "distances_iter")
dyn_s = to_numeric_array(df_dynamic, "social_iter")

valid_regu = ~(np.isnan(regu_d) & np.isnan(regu_s))
valid_dyn = ~(np.isnan(dyn_d) & np.isnan(dyn_s))

regu_f = regu_f[valid_regu]
regu_d = regu_d[valid_regu]
regu_s = regu_s[valid_regu]

dyn_f = dyn_f[valid_dyn]
dyn_d = dyn_d[valid_dyn]
dyn_s = dyn_s[valid_dyn]

if len(regu_d) == 0 or len(dyn_d) == 0:
    print("[ERROR] One trajectory file has no valid data rows.")
    sys.exit(1)

x_dynamic = np.arange(1, len(dyn_d) + 1)

if len(regu_d) == 100:
    x_regu = np.arange(1, 101)
else:
    x_regu = np.linspace(1, 100, len(regu_d))


# ============================================================
# Final values and gain retention
# ============================================================

dynamic_final_f = float(dyn_f[-1])
dynamic_final_d = float(dyn_d[-1])
dynamic_final_s = float(dyn_s[-1])

regu_final_f = float(regu_f[-1])
regu_final_d = float(regu_d[-1])
regu_final_s = float(regu_s[-1])

distance_retention_pct, dynamic_d_gain, regu_d_gain = gain_retention(
    lower_is_better=True,
    baseline=baseline_distance,
    dynamic_final=dynamic_final_d,
    regu_final=regu_final_d
)

social_retention_pct, dynamic_s_gain, regu_s_gain = gain_retention(
    lower_is_better=False,
    baseline=baseline_social,
    dynamic_final=dynamic_final_s,
    regu_final=regu_final_s
)

print("\n========== Benefit retention relative to observed baseline ==========")
print(f"Dynamic distance gain:       {dynamic_d_gain:.6f}")
print(f"Regularized distance gain:   {regu_d_gain:.6f}")
print(f"Distance retained:           {distance_retention_pct:.6f}%")

print(f"\nDynamic social gain:         {dynamic_s_gain:.6f}")
print(f"Regularized social gain:     {regu_s_gain:.6f}")
print(f"Social exposure retained:    {social_retention_pct:.6f}%")


# ============================================================
# Plot Figure 4d
# ============================================================

fig, axes = plt.subplots(1, 3, figsize=figsize, dpi=dpi)


# ------------------------------------------------------------
# Panel a: origin-level behavioural pullback toward F
# ------------------------------------------------------------

ax = axes[0]

ax.fill_between(
    x_rank_F,
    dev_reg_F_plot,
    dev_no_F_plot,
    where=(dev_no_F_plot >= dev_reg_F_plot),
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

ax.set_title(r"Origin-level behavioral pullback toward $F$")
ax.set_xlabel(r"Changed CBGs ranked by pullback $c_i^F$ (%)")
ax.set_ylabel(r"Origin-level departure $d_i(H,F)$")

ax.set_xlim(0, 100)

ymax_a = max(np.nanmax(dev_no_F_plot), np.nanmax(dev_reg_F_plot))
ax.set_ylim(0, ymax_a * 1.12)

ax.grid(True, linestyle=":", linewidth=0.55, alpha=0.75)

# Main legend
ax.legend(
    loc="upper right",
    fontsize=7.8,
    frameon=False,
    handlelength=2.1
)

share_changed_no_to_reg = (change_no_to_reg_raw > tol).mean() * 100.0

# Inset: matrix-level pullback, excluding D(F,R)
inset = ax.inset_axes([0.24, 0.48, 0.47, 0.35])

metrics = [r"from $F$", r"from $R$"]
unreg_vals = np.array([no_regu_from_F_pct, no_regu_ref_dev_pct], dtype=float)
regu_vals = np.array([regu_from_F_pct, regu_ref_dev_pct], dtype=float)
reduction_vals = np.array([from_F_reduction_pct, ref_dev_reduction_pct], dtype=float)

y = np.arange(len(metrics))
bar_h = 0.32

inset.barh(
    y + bar_h / 2,
    unreg_vals,
    height=bar_h,
    color=dynamic_color,
    alpha=0.85,
    label="Unregularized",
    zorder=2
)

inset.barh(
    y - bar_h / 2,
    regu_vals,
    height=bar_h,
    color=regu_color,
    alpha=0.85,
    label=r"$L_1$-regularized",
    zorder=3
)

for i, (u, r, red) in enumerate(zip(unreg_vals, regu_vals, reduction_vals)):
    inset.text(
        u + 1.0,
        y[i] + bar_h / 2,
        f"{u:.1f}",
        ha="left",
        va="center",
        fontsize=7.1,
        color=dynamic_color
    )

    inset.text(
        r + 1.0,
        y[i] - bar_h / 2,
        f"{r:.1f}",
        ha="left",
        va="center",
        fontsize=7.1,
        color=regu_color
    )

inset.set_yticks(y)
inset.set_yticklabels(metrics, fontsize=7.8)
inset.invert_yaxis()

inset.set_xlim(0, 70)
inset.set_ylim(-0.45, 1.45)

inset.set_xlabel("Matrix departure (% visits)", fontsize=7.7, labelpad=1)
inset.set_title("Global pullback", fontsize=8.4, pad=3)

inset.tick_params(axis="x", labelsize=7.2, length=2)
inset.tick_params(axis="y", labelsize=7.6, length=0)

inset.grid(True, axis="x", linestyle=":", linewidth=0.45, alpha=0.7)
inset.grid(False, axis="y")

inset.legend(
    loc="lower right",
    fontsize=6.6,
    frameon=False,
    handlelength=1.1,
    borderaxespad=0.2
)

for spine in inset.spines.values():
    spine.set_linewidth(0.7)
    spine.set_edgecolor("0.55")

try:
    ax.set_box_aspect(1)
except Exception:
    pass


# ------------------------------------------------------------
# Panel b: distance trajectory
# ------------------------------------------------------------

ax = axes[1]

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
    [x_dynamic[-1]],
    [dynamic_final_d],
    color=dynamic_color,
    s=38,
    zorder=4
)

ax.scatter(
    [x_regu[-1]],
    [regu_final_d],
    color=regu_color,
    s=38,
    zorder=4
)

ax.set_title("Travel-distance trajectory")
ax.set_xlabel("Iteration")
ax.set_ylabel("Total travel distance")
ax.set_xlim(1, 100)
ax.grid(True, linestyle=":", linewidth=0.55, alpha=0.8)


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
    ax.set_box_aspect(1)
except Exception:
    pass


# ------------------------------------------------------------
# Panel c: structural social exposure trajectory
# ------------------------------------------------------------

ax = axes[2]

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
    [x_dynamic[-1]],
    [dynamic_final_s],
    color=dynamic_color,
    s=38,
    zorder=4
)

ax.scatter(
    [x_regu[-1]],
    [regu_final_s],
    color=regu_color,
    s=38,
    zorder=4
)

ax.set_title("Structural-exposure trajectory")
ax.set_xlabel("Iteration")
ax.set_ylabel("Structural potential social exposure")
ax.set_xlim(1, 100)
ax.grid(True, linestyle=":", linewidth=0.55, alpha=0.8)

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
    ax.set_box_aspect(1)
except Exception:
    pass


# ============================================================
# Layout and output
# ============================================================

plt.tight_layout()
plt.subplots_adjust(wspace=0.25, bottom=0.02)

plt.savefig('figure4d.pdf',
            format='pdf',
            dpi=300,
            bbox_inches='tight',
            transparent=False,
            backend='pdf')

plt.show()


# ============================================================
# Tables and LaTeX-ready outputs
# ============================================================

final_compare = {
    "Travel distance": {
        "Observed baseline": baseline_distance,
        "Dynamic final": dynamic_final_d,
        "Regularized final": regu_final_d,
        "Regu_vs_Dynamic_percent": pct_change(regu_final_d, dynamic_final_d),
        "Dynamic_vs_baseline_percent": pct_change(dynamic_final_d, baseline_distance),
        "Regularized_vs_baseline_percent": pct_change(regu_final_d, baseline_distance),
        "Gain_retention_percent": distance_retention_pct,
    },
    "Structural social exposure": {
        "Observed baseline": baseline_social,
        "Dynamic final": dynamic_final_s,
        "Regularized final": regu_final_s,
        "Regu_vs_Dynamic_percent": pct_change(regu_final_s, dynamic_final_s),
        "Dynamic_vs_baseline_percent": pct_change(dynamic_final_s, baseline_social),
        "Regularized_vs_baseline_percent": pct_change(regu_final_s, baseline_social),
        "Gain_retention_percent": social_retention_pct,
    }
}

anchoring_compare = {
    "Deviation from R": {
        "Unregularized final": no_regu_ref_dev_pct,
        "Regularized final": regu_ref_dev_pct,
        "Reduction_percent": ref_dev_reduction_pct,
    },
    "Deviation from observed F": {
        "Unregularized final": no_regu_from_F_pct,
        "Regularized final": regu_from_F_pct,
        "Reduction_percent": from_F_reduction_pct,
    },
    "Diagnostic D(F,R)": {
        "Unregularized final": np.nan,
        "Regularized final": np.nan,
        "Reduction_percent": R_from_F_pct,
    },
    "Anchoring concentration from F": {
        "Top10_positive_origin_percent": top10_F,
        "Top20_positive_origin_percent": top20_F,
        "Positive_origin_share": share_positive_F,
    },
    "Anchoring concentration from R": {
        "Top10_positive_origin_percent": top10_R,
        "Top20_positive_origin_percent": top20_R,
        "Positive_origin_share": share_positive_R,
    }
}

final_compare_df = pd.DataFrame(final_compare).T
anchoring_compare_df = pd.DataFrame(anchoring_compare).T

print("\n=== Figure 4d final performance comparison ===")
print(final_compare_df.to_string(float_format=lambda x: f"{x:.6f}"))

print("\n=== Figure 4d behavioural anchoring and concentration comparison ===")
print(anchoring_compare_df.to_string(float_format=lambda x: f"{x:.6f}"))


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

print("\n=== LaTeX-ready sentence for Results ===")
print(latex_sentence)