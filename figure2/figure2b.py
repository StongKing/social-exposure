# -*- coding: utf-8 -*-
"""
Fig. 2b — public plotting/reproduction script

The private flow_matrix.csv is NOT required.

Required baseline-derived files
-------------------------------
figure2b_active_pois.csv
figure2b_baseline_system.csv
figure2b_baseline_cbg_contribution.csv

Other required public files
---------------------------
distance_matrix.csv
H_opt_df_static_624190.pkl
H_opt_df_dynamic_624190.pkl
cbg_income_level_distribution_boston_msa.csv
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 0. Configuration
# ============================================================

naics_code = "624190"

cat_dir = (
    r"matrices_A_D_S_Distribution/"
    r"Other_Individual_and_Family_Services"
)

distance_path = os.path.join(
    cat_dir,
    "distance_matrix.csv"
)

static_h_path = os.path.join(
    cat_dir,
    f"H_opt_df_static_{naics_code}.pkl"
)

dynamic_h_path = os.path.join(
    cat_dir,
    f"H_opt_df_dynamic_{naics_code}.pkl"
)

income_path = (
    r"matrices_A_D_S_Distribution/"
    r"cbg_income_level_distribution_boston_msa.csv"
)


ACTIVE_POI_FILE = (
    "figure2b_active_pois.csv"
)

BASELINE_SYSTEM_FILE = (
    "figure2b_baseline_system.csv"
)

BASELINE_CBG_FILE = (
    "figure2b_baseline_cbg_contribution.csv"
)


income_levels = [
    "low_income_pct",
    "lower_middle_income_pct",
    "upper_middle_income_pct",
    "high_income_pct",
]


EPS = 1e-4

OUTPUT_PDF = "figure2b.pdf"


# ============================================================
# 1. Helpers
# ============================================================

def normalize_key(x):

    s = str(x).strip()

    if s.endswith(".0"):
        s = s[:-2]

    return s


def load_csv_matrix(path):

    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"File not found: {path}"
        )

    df = pd.read_csv(
        path,
        index_col=0
    )

    df.index = [
        normalize_key(x)
        for x in df.index
    ]

    df.columns = [
        normalize_key(x)
        for x in df.columns
    ]

    df = (
        df
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0.0)
    )

    return df


def load_pkl_matrix(path):

    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"File not found: {path}"
        )

    obj = pd.read_pickle(path)

    if isinstance(obj, pd.DataFrame):
        df = obj.copy()

    elif isinstance(obj, np.ndarray):
        df = pd.DataFrame(obj)

    else:
        raise TypeError(
            f"Unsupported pkl object type: {type(obj)}"
        )

    df.index = [
        normalize_key(x)
        for x in df.index
    ]

    df.columns = [
        normalize_key(x)
        for x in df.columns
    ]

    df = (
        df
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0.0)
    )

    return df


def load_cbg_income_distribution(
        income_path,
        cbg_ids,
        income_levels):

    if not os.path.isfile(income_path):

        raise FileNotFoundError(
            f"Income file not found: {income_path}"
        )

    income_df = pd.read_csv(
        income_path
    )

    if "GEOID" not in income_df.columns:

        raise ValueError(
            "`GEOID` column not found. "
            f"Columns: {list(income_df.columns)}"
        )

    income_df["GEOID"] = (
        income_df["GEOID"]
        .map(normalize_key)
    )

    income_df = (
        income_df
        .set_index("GEOID")
    )

    missing_cols = [
        c
        for c in income_levels
        if c not in income_df.columns
    ]

    if missing_cols:

        raise ValueError(
            f"Missing income columns: {missing_cols}"
        )

    P_df = (
        income_df
        .reindex(cbg_ids)[income_levels]
        .copy()
    )

    P_df = P_df.apply(
        pd.to_numeric,
        errors="coerce"
    )

    if P_df.isna().any().any():

        missing_ids = (
            P_df[
                P_df.isna().any(axis=1)
            ]
            .index
            .tolist()[:10]
        )

        raise ValueError(
            "Some CBGs have missing income "
            f"distribution, e.g. {missing_ids}"
        )

    P = P_df.values.astype(float)

    row_sum = P.sum(
        axis=1,
        keepdims=True
    )

    if np.nanmedian(row_sum) > 1.5:

        P = P / 100.0

        row_sum = P.sum(
            axis=1,
            keepdims=True
        )

    row_sum[
        row_sum == 0
    ] = np.nan

    P = P / row_sum

    P = np.nan_to_num(
        P,
        nan=0.0
    )

    P_df = pd.DataFrame(
        P,
        index=cbg_ids,
        columns=income_levels
    )

    print("[INCOME] loaded and normalized")
    print(
        f"[INCOME] n_CBGs = "
        f"{P_df.shape[0]}"
    )
    print(
        f"[INCOME] columns = "
        f"{income_levels}"
    )

    return P_df


# ============================================================
# 2. Exposure calculation
# ============================================================

def calculate_poi_income_distribution(
        H_df,
        P_df):

    H = H_df.values.astype(float)
    P = P_df.values.astype(float)

    poi_flow = H.sum(axis=0)

    Q = np.full(
        (
            H.shape[1],
            P.shape[1]
        ),
        np.nan,
        dtype=float
    )

    positive_poi = (
        poi_flow > EPS
    )

    Q[
        positive_poi,
        :
    ] = (
        H[:, positive_poi].T @ P
    ) / poi_flow[
        positive_poi,
        None
    ]

    return pd.DataFrame(
        Q,
        index=H_df.columns,
        columns=P_df.columns
    )


def calculate_social_exposure_matrix(
        H_df,
        P_df):

    Q_df = (
        calculate_poi_income_distribution(
            H_df,
            P_df
        )
    )

    P = P_df.values.astype(float)
    Q = Q_df.values.astype(float)

    S = 1.0 - (
        P @ Q.T
    )

    S_df = pd.DataFrame(
        S,
        index=H_df.index,
        columns=H_df.columns
    )

    return S_df, Q_df


# ============================================================
# 3. Evaluation metrics
# ============================================================

def calculate_total_structural_exposure(
        H_df,
        S_df):

    H = H_df.values.astype(float)
    S = S_df.values.astype(float)

    positive = H > EPS

    return float(
        np.nansum(
            S[positive]
        )
    )


def calculate_cbg_structural_contribution(
        H_df,
        S_df):

    H = H_df.values.astype(float)
    S = S_df.values.astype(float)

    positive = H > EPS

    contrib = np.nansum(
        np.where(
            positive,
            S,
            0.0
        ),
        axis=1
    )

    return pd.Series(
        contrib,
        index=H_df.index,
        name="cbg_structural_contribution",
    )


def calculate_total_distance(
        H_df,
        D_df):

    return float(
        np.nansum(
            H_df.values.astype(float)
            *
            D_df.values.astype(float)
        )
    )


def calculate_avg_distance_per_visit(
        H_df,
        D_df):

    total_flow = float(
        np.nansum(
            H_df.values.astype(float)
        )
    )

    total_distance = (
        calculate_total_distance(
            H_df,
            D_df
        )
    )

    if total_flow <= EPS:
        return np.nan

    return (
        total_distance
        /
        total_flow
    )


def calculate_flow_weighted_exposure(
        H_df,
        S_df):

    H = H_df.values.astype(float)
    S = S_df.values.astype(float)

    total_flow = np.nansum(H)

    if total_flow <= EPS:
        return np.nan

    return float(
        np.nansum(H * S)
        /
        total_flow
    )


# ============================================================
# 4. Load public model outputs
# ============================================================

distance_matrix = load_csv_matrix(
    distance_path
)

H_static = load_pkl_matrix(
    static_h_path
)

H_dynamic = load_pkl_matrix(
    dynamic_h_path
)


# ============================================================
# 5. Load baseline-derived data
# ============================================================

active_poi_df = pd.read_csv(
    ACTIVE_POI_FILE,
    dtype=str
)

active_poi_ids = [
    normalize_key(x)
    for x in active_poi_df["poi"]
]


baseline_system_df = pd.read_csv(
    BASELINE_SYSTEM_FILE
)


baseline_cbg_df = pd.read_csv(
    BASELINE_CBG_FILE,
    dtype={"cbg": str}
)

baseline_cbg_df["cbg"] = (
    baseline_cbg_df["cbg"]
    .map(normalize_key)
)


# ============================================================
# 6. Determine common public domain
# ============================================================

cbg_ids = (
    baseline_cbg_df["cbg"]
    .tolist()
)

poi_ids = active_poi_ids


# Check required rows and columns.
for name, df in [
    ("distance_matrix", distance_matrix),
    ("H_static", H_static),
    ("H_dynamic", H_dynamic),
]:

    missing_cbgs = (
        set(cbg_ids)
        -
        set(df.index)
    )

    missing_pois = (
        set(poi_ids)
        -
        set(df.columns)
    )

    if missing_cbgs:

        raise ValueError(
            f"{name}: missing CBGs, e.g. "
            f"{list(missing_cbgs)[:10]}"
        )

    if missing_pois:

        raise ValueError(
            f"{name}: missing POIs, e.g. "
            f"{list(missing_pois)[:10]}"
        )


distance_matrix = (
    distance_matrix.loc[
        cbg_ids,
        poi_ids
    ].copy()
)

H_static = (
    H_static.loc[
        cbg_ids,
        poi_ids
    ].copy()
)

H_dynamic = (
    H_dynamic.loc[
        cbg_ids,
        poi_ids
    ].copy()
)


print(
    "\n========== ANALYSIS DOMAIN =========="
)

print(
    f"n_CBGs: {len(cbg_ids)}"
)

print(
    f"Analysis POIs: {len(poi_ids)}"
)


assert len(poi_ids) == 44, (
    "Expected 44 baseline-positive POIs, "
    f"but found {len(poi_ids)}."
)


# ============================================================
# 7. Load CBG income distributions
# ============================================================

P_df = load_cbg_income_distribution(
    income_path=income_path,
    cbg_ids=cbg_ids,
    income_levels=income_levels,
)


# ============================================================
# 8. Recompute Static and Dynamic exposure matrices
# ============================================================

S_static, Q_static = (
    calculate_social_exposure_matrix(
        H_static,
        P_df
    )
)

S_dynamic, Q_dynamic = (
    calculate_social_exposure_matrix(
        H_dynamic,
        P_df
    )
)


print(
    "\n========== SOCIAL EXPOSURE MATRICES =========="
)

print(
    f"S_static shape: "
    f"{S_static.shape}"
)

print(
    f"S_dynamic shape: "
    f"{S_dynamic.shape}"
)


# ============================================================
# 9. System-level outcomes
# ============================================================

# Baseline metrics generated from private flow_matrix.csv
base_avg_dist = float(
    baseline_system_df.loc[
        baseline_system_df["scenario"]
        ==
        "Baseline",
        "avg_distance_per_visit"
    ].iloc[0]
)

base_structural = float(
    baseline_system_df.loc[
        baseline_system_df["scenario"]
        ==
        "Baseline",
        "structural_exposure"
    ].iloc[0]
)

base_fw_exp = float(
    baseline_system_df.loc[
        baseline_system_df["scenario"]
        ==
        "Baseline",
        "flow_weighted_exposure"
    ].iloc[0]
)

base_n_positive_links = int(
    baseline_system_df.loc[
        baseline_system_df["scenario"]
        ==
        "Baseline",
        "n_positive_links"
    ].iloc[0]
)


# Static/Dynamic values recomputed from public data.
static_avg_dist = (
    calculate_avg_distance_per_visit(
        H_static,
        distance_matrix
    )
)

dynamic_avg_dist = (
    calculate_avg_distance_per_visit(
        H_dynamic,
        distance_matrix
    )
)


static_structural = (
    calculate_total_structural_exposure(
        H_static,
        S_static
    )
)

dynamic_structural = (
    calculate_total_structural_exposure(
        H_dynamic,
        S_dynamic
    )
)


static_fw_exp = (
    calculate_flow_weighted_exposure(
        H_static,
        S_static
    )
)

dynamic_fw_exp = (
    calculate_flow_weighted_exposure(
        H_dynamic,
        S_dynamic
    )
)


system_df = pd.DataFrame(
    [
        {
            "scenario": "Baseline",
            "avg_distance_per_visit":
                base_avg_dist,
            "structural_exposure":
                base_structural,
            "flow_weighted_exposure":
                base_fw_exp,
            "n_positive_links":
                base_n_positive_links,
        },

        {
            "scenario": "Static",
            "avg_distance_per_visit":
                static_avg_dist,
            "structural_exposure":
                static_structural,
            "flow_weighted_exposure":
                static_fw_exp,
            "n_positive_links":
                int(
                    (
                        H_static.values > EPS
                    ).sum()
                ),
        },

        {
            "scenario": "Dynamic",
            "avg_distance_per_visit":
                dynamic_avg_dist,
            "structural_exposure":
                dynamic_structural,
            "flow_weighted_exposure":
                dynamic_fw_exp,
            "n_positive_links":
                int(
                    (
                        H_dynamic.values > EPS
                    ).sum()
                ),
        },
    ]
)


system_df["distance_change_pct"] = (
    system_df[
        "avg_distance_per_visit"
    ]
    /
    base_avg_dist
    -
    1.0
) * 100.0


system_df[
    "structural_exposure_change_pct"
] = (
    system_df[
        "structural_exposure"
    ]
    /
    base_structural
    -
    1.0
) * 100.0


system_df[
    "flow_weighted_exposure_change_pct"
] = (
    system_df[
        "flow_weighted_exposure"
    ]
    /
    base_fw_exp
    -
    1.0
) * 100.0


print(
    "\n========== SYSTEM-LEVEL OUTCOMES =========="
)

print(
    system_df
    .round(6)
    .to_string(index=False)
)


# ============================================================
# 10. CBG-level structural exposure gains
# ============================================================

# Baseline contribution comes from private-flow preprocessing.
cbg_base_contrib = (
    baseline_cbg_df
    .set_index("cbg")[
        "baseline_structural_contrib"
    ]
    .reindex(cbg_ids)
)


if cbg_base_contrib.isna().any():

    raise ValueError(
        "Missing baseline structural "
        "contribution for some CBGs."
    )


# Static and Dynamic contributions are recomputed.
cbg_static_contrib = (
    calculate_cbg_structural_contribution(
        H_static,
        S_static
    )
)

cbg_dynamic_contrib = (
    calculate_cbg_structural_contribution(
        H_dynamic,
        S_dynamic
    )
)


cbg_gain_df = pd.DataFrame(
    {
        "cbg": cbg_ids,

        "baseline_structural_contrib":
            cbg_base_contrib.values,

        "static_structural_contrib":
            cbg_static_contrib.values,

        "dynamic_structural_contrib":
            cbg_dynamic_contrib.values,
    }
)


cbg_gain_df["static_gain"] = (
    cbg_gain_df[
        "static_structural_contrib"
    ]
    -
    cbg_gain_df[
        "baseline_structural_contrib"
    ]
)


cbg_gain_df["dynamic_gain"] = (
    cbg_gain_df[
        "dynamic_structural_contrib"
    ]
    -
    cbg_gain_df[
        "baseline_structural_contrib"
    ]
)


cbg_gain_df[
    "dynamic_minus_static"
] = (
    cbg_gain_df[
        "dynamic_gain"
    ]
    -
    cbg_gain_df[
        "static_gain"
    ]
)


print(
    "\n========== CBG-LEVEL STRUCTURAL EXPOSURE GAINS =========="
)


for method in [
    "static",
    "dynamic"
]:

    x = (
        cbg_gain_df[
            f"{method}_gain"
        ]
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
        .dropna()
    )

    print(
        f"{method.capitalize():>8s}: "
        f"sum={x.sum():.6f}, "
        f"mean={x.mean():.6f}, "
        f"median={x.median():.6f}, "
        f"p25={x.quantile(0.25):.6f}, "
        f"p75={x.quantile(0.75):.6f}, "
        f"share_positive="
        f"{(x > 0).mean() * 100:.2f}%"
    )


print(
    "\n[CHECK] Sum of CBG gains should "
    "equal total structural change:"
)

print(
    "Static  CBG gain sum: "
    f"{cbg_gain_df['static_gain'].sum():.6f}"
)

print(
    "Static  structural change: "
    f"{static_structural - base_structural:.6f}"
)

print(
    "Dynamic CBG gain sum: "
    f"{cbg_gain_df['dynamic_gain'].sum():.6f}"
)

print(
    "Dynamic structural change: "
    f"{dynamic_structural - base_structural:.6f}"
)


# ============================================================
# 11. Plot Fig. 2b
# ============================================================

plt.rcParams["font.family"] = "Arial"
plt.rcParams["axes.unicode_minus"] = False

color_static = "#4C78A8"
color_dynamic = "#8E5EA2"
color_base = "#9E9E9E"
grid_color = "#D8D8D8"

fig, axes = plt.subplots(1, 2, figsize=(16, 8), dpi=300)


# ------------------------------------------------------------
# Panel A: System-level aggregate outcome
# ------------------------------------------------------------

ax = axes[0]

plot_system = system_df[
    system_df["scenario"].isin(["Static", "Dynamic"])
].copy()

ax.axvline(0, color=color_base, linestyle="--", linewidth=1.0, zorder=1)
ax.axhline(0, color=color_base, linestyle="--", linewidth=1.0, zorder=1)

for _, row in plot_system.iterrows():
    scenario = row["scenario"]
    c = color_static if scenario == "Static" else color_dynamic

    x = row["distance_change_pct"]
    y = row["structural_exposure_change_pct"]

    ax.plot(
        [0, x],
        [0, y],
        color=c,
        linewidth=2.0,
        alpha=0.72,
        zorder=2,
    )

    ax.scatter(
        x,
        y,
        s=135,
        color=c,
        edgecolor="white",
        linewidth=0.9,
        zorder=4,
    )

static_row = plot_system[plot_system["scenario"] == "Static"].iloc[0]
dynamic_row = plot_system[plot_system["scenario"] == "Dynamic"].iloc[0]

ax.annotate(
    (
        f"Static\n"
        f"Distance {static_row['distance_change_pct']:+.1f}%\n"
        f"Structural exposure "
        f"{static_row['structural_exposure_change_pct']:+.1f}%"
    ),
    xy=(
        static_row["distance_change_pct"],
        static_row["structural_exposure_change_pct"],
    ),
    xytext=(-8, 13),
    textcoords="offset points",
    fontsize=13.5,
    color=color_static,
    ha="center",
    va="bottom",
)

ax.annotate(
    (
        f"Dynamic\n"
        f"Distance {dynamic_row['distance_change_pct']:+.1f}%\n"
        f"Structural exposure "
        f"{dynamic_row['structural_exposure_change_pct']:+.1f}%"
    ),
    xy=(
        dynamic_row["distance_change_pct"],
        dynamic_row["structural_exposure_change_pct"],
    ),
    xytext=(0, 13),
    textcoords="offset points",
    fontsize=13.5,
    color=color_dynamic,
    ha="center",
    va="bottom",
)

ax.set_xlabel("Change in mean travel distance per visit (%)", fontsize=14)
ax.set_ylabel(r"Change in structural exposure, $E(H)$ (%)", fontsize=14)
ax.set_title("System-level outcomes", fontsize=15, pad=8)

ax.grid(True, linestyle=":", linewidth=0.6, color=grid_color)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

x_min = min(plot_system["distance_change_pct"].min(), 0) - 5
x_max = max(plot_system["distance_change_pct"].max(), 0) + 4
y_min = min(plot_system["structural_exposure_change_pct"].min(), 0) - 5
y_max = max(plot_system["structural_exposure_change_pct"].max(), 0) + 10

ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)


# ------------------------------------------------------------
# Inset in Panel A: flow-weighted mean exposure comparison
# ------------------------------------------------------------

axins_fw = ax.inset_axes([0.08, 0.08, 0.38, 0.34])

fw_static = static_row["flow_weighted_exposure_change_pct"]
fw_dynamic = dynamic_row["flow_weighted_exposure_change_pct"]

fw_values = [fw_static, fw_dynamic]
fw_positions = [0, 1]

bars = axins_fw.bar(
    fw_positions,
    fw_values,
    width=0.58,
    color=[color_static, color_dynamic],
    alpha=0.82,
    edgecolor="none",
    zorder=2,
)

axins_fw.axhline(
    0,
    color=color_base,
    linestyle="--",
    linewidth=0.8,
    zorder=1,
)

fw_span = max(abs(fw_static), abs(fw_dynamic), 0.1)
label_pad = fw_span * 0.06

for bar, value in zip(bars, fw_values):
    if value >= 0:
        y_text = value + label_pad
        va = "bottom"
    else:
        y_text = value - label_pad
        va = "top"

    axins_fw.text(
        bar.get_x() + bar.get_width() / 2,
        y_text,
        f"{value:+.1f}%",
        ha="center",
        va=va,
        fontsize=10.5,
        color="#333333",
    )

axins_fw.set_xticks(fw_positions)
axins_fw.set_xticklabels(
    ["Static", "Dynamic"],
    fontsize=10.5,
)

axins_fw.set_ylabel(
    "Change (%)",
    fontsize=10.5,
)

axins_fw.set_title(
    "Flow-weighted mean exposure",
    fontsize=11.5,
    pad=3,
)

fw_min = min(0, min(fw_values))
fw_max = max(0, max(fw_values))
fw_range = fw_max - fw_min

if fw_range < 0.2:
    fw_range = 0.2

axins_fw.set_ylim(
    fw_min - 0.18 * fw_range,
    fw_max + 0.28 * fw_range,
)

axins_fw.tick_params(
    axis="y",
    labelsize=9.5,
    length=2,
)

axins_fw.grid(
    axis="y",
    linestyle=":",
    linewidth=0.45,
    color=grid_color,
)

axins_fw.spines["top"].set_visible(False)
axins_fw.spines["right"].set_visible(False)


# ------------------------------------------------------------
# Panel B: CBG-level structural exposure gain violin plot
# ------------------------------------------------------------

ax = axes[1]

data_cbg_static = (
    cbg_gain_df["static_gain"]
    .replace([np.inf, -np.inf], np.nan)
    .dropna()
    .values
)

data_cbg_dynamic = (
    cbg_gain_df["dynamic_gain"]
    .replace([np.inf, -np.inf], np.nan)
    .dropna()
    .values
)

parts = ax.violinplot(
    [data_cbg_static, data_cbg_dynamic],
    positions=[1, 2],
    widths=0.68,
    showmeans=True,
    showmedians=False,
    showextrema=False,
)

for i, body in enumerate(parts["bodies"]):
    body.set_facecolor(color_static if i == 0 else color_dynamic)
    body.set_edgecolor("none")
    body.set_alpha(0.35)

parts["cmeans"].set_color("#333333")
parts["cmeans"].set_linewidth(1.5)

# Jittered CBG points
rng = np.random.default_rng(42)

for pos, values, c in [
    (1, data_cbg_static, color_static),
    (2, data_cbg_dynamic, color_dynamic),
]:
    if len(values) > 700:
        values_plot = rng.choice(values, size=700, replace=False)
    else:
        values_plot = values

    x_jitter = rng.normal(loc=pos, scale=0.045, size=len(values_plot))

    ax.scatter(
        x_jitter,
        values_plot,
        s=8,
        color=c,
        alpha=0.8,
        edgecolor="none",
        zorder=2,
    )


# ------------------------------------------------------------
# Inset in Panel B: union of top 10% gain CBGs
# ------------------------------------------------------------

top_share = 0.10

pair_df = cbg_gain_df[
    ["cbg", "static_gain", "dynamic_gain", "dynamic_minus_static"]
].copy()

pair_df = (
    pair_df
    .replace([np.inf, -np.inf], np.nan)
    .dropna(subset=["static_gain", "dynamic_gain"])
)

n_cbg = len(pair_df)
n_top = max(1, int(np.ceil(n_cbg * top_share)))

static_top_set = set(
    pair_df
    .sort_values("static_gain", ascending=False)
    .head(n_top)["cbg"]
)

dynamic_top_set = set(
    pair_df
    .sort_values("dynamic_gain", ascending=False)
    .head(n_top)["cbg"]
)

top_union_set = static_top_set | dynamic_top_set
top_intersection_set = static_top_set & dynamic_top_set

top_pair_df = pair_df[pair_df["cbg"].isin(top_union_set)].copy()

top_pair_df["top_group"] = "Static-only top"
top_pair_df.loc[
    top_pair_df["cbg"].isin(dynamic_top_set)
    & ~top_pair_df["cbg"].isin(static_top_set),
    "top_group",
] = "Dynamic-only top"

top_pair_df.loc[
    top_pair_df["cbg"].isin(top_intersection_set),
    "top_group",
] = "Both top"

axins_b = ax.inset_axes([0.33, 0.58, 0.35, 0.35])

xy_values = top_pair_df[["static_gain", "dynamic_gain"]].values
xy_min = np.nanmin(xy_values)
xy_max = np.nanmax(xy_values)

pad = (xy_max - xy_min) * 0.08 if xy_max > xy_min else 0.1
xy_min -= pad
xy_max += pad

axins_b.plot(
    [xy_min, xy_max],
    [xy_min, xy_max],
    color=color_base,
    linestyle="--",
    linewidth=1.0,
    zorder=1,
)

axins_b.axvline(
    0,
    color="#CFCFCF",
    linestyle=":",
    linewidth=0.7,
    zorder=1,
)

axins_b.axhline(
    0,
    color="#CFCFCF",
    linestyle=":",
    linewidth=0.7,
    zorder=1,
)

plot_specs = [
    ("Static-only top", color_static, "o", 0.72, 20),
    ("Dynamic-only top", color_dynamic, "o", 0.78, 20),
    ("Both top", "#333333", "D", 0.86, 20),
]

for group, c, marker, alpha, size in plot_specs:
    sub = top_pair_df[top_pair_df["top_group"] == group]

    if len(sub) == 0:
        continue

    axins_b.scatter(
        sub["static_gain"],
        sub["dynamic_gain"],
        s=size,
        color=c,
        marker=marker,
        alpha=alpha,
        edgecolor="white",
        linewidth=0.25,
        zorder=3,
        label=group,
    )

axins_b.set_xlim(xy_min, xy_max)
axins_b.set_ylim(xy_min, xy_max)
axins_b.set_title("Union of top 10% gain CBGs", fontsize=10.5, pad=2)
axins_b.set_xlabel("Static gain", fontsize=10.5)
axins_b.set_ylabel("Dynamic gain", fontsize=10.5)

axins_b.tick_params(axis="both", labelsize=7.5, length=2)
axins_b.grid(True, linestyle=":", linewidth=0.4, color=grid_color)
axins_b.spines["top"].set_visible(False)
axins_b.spines["right"].set_visible(False)

axins_b.legend(
    frameon=False,
    fontsize=7.5,
    loc="lower right",
    handlelength=1.0,
    borderpad=0.2,
    labelspacing=0.25,
)

# Main Panel B formatting
ax.axhline(0, color=color_base, linestyle="--", linewidth=1.0)

ax.set_xticks([1, 2])
ax.set_xticklabels(["Static", "Dynamic"], fontsize=14)
ax.set_ylabel("CBG-level structural exposure gain", fontsize=14)
ax.set_title("Origin-level gain distribution", fontsize=14, pad=8)

ax.grid(axis="y", linestyle=":", linewidth=0.6, color=grid_color)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

all_cbg_values = np.concatenate([data_cbg_static, data_cbg_dynamic])

y_low = min(np.nanpercentile(all_cbg_values, 1), -0.05)
y_high = max(np.nanpercentile(all_cbg_values, 99), 1.5)

ax.set_ylim(y_low - 0.05, y_high + 0.25)


# ------------------------------------------------------------
# Global layout
# ------------------------------------------------------------

plt.subplots_adjust(
    left=0.075,
    right=0.985,
    bottom=0.20,
    top=0.82,
    wspace=0.10,
)

fig.savefig(
    OUTPUT_PDF,
    format="pdf",
    dpi=300,
    bbox_inches="tight",
    transparent=False,
    backend="pdf",
)

plt.show()


# ============================================================
# 12. Final summary
# ============================================================

print("\n========== FINAL FIG. 2b SUMMARY ==========")

print(
    f"Analysis POIs: {len(poi_ids)}"
)

print(
    "Static: "
    f"distance {static_row['distance_change_pct']:+.4f}%, "
    f"structural exposure "
    f"{static_row['structural_exposure_change_pct']:+.4f}%, "
    f"flow-weighted exposure "
    f"{static_row['flow_weighted_exposure_change_pct']:+.4f}%"
)

print(
    "Dynamic: "
    f"distance {dynamic_row['distance_change_pct']:+.4f}%, "
    f"structural exposure "
    f"{dynamic_row['structural_exposure_change_pct']:+.4f}%, "
    f"flow-weighted exposure "
    f"{dynamic_row['flow_weighted_exposure_change_pct']:+.4f}%"
)

print(
    f"\nSaved figure: {OUTPUT_PDF}"
)