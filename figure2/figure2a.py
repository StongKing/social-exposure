
"""
Figure 2a: show CBG-level POI visit-share redistribution.

Core change:
1. The optimization keeps each CBG's total visits conserved.
2. The final heatmap no longer displays absolute flow values.
   Instead, each CBG row is normalized to sum to 1, so the figure shows
   how each CBG's visit share is redistributed across POIs.
"""

import os
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import gridspec
from matplotlib.ticker import PercentFormatter
from matplotlib.colors import PowerNorm, LinearSegmentedColormap
from ortools.linear_solver import pywraplp
from scipy.stats import entropy


# =========================================================
# 0. Basic settings
# =========================================================
city = 'boston'
category = 'Museums'
cat_dir = f'matrices_A_D_S_Distribution_{city}_core/{category.replace(" ", "_")}'

# Optimization parameters
alpha = 0.5
beta = 2
F_ratio = 0.01
epsilon = 1e-4
max_iterations = 100

# Display settings
poi_num = 10
cbg_num = 10
save_fig = True
output_fig = 'figure2a.pdf'

# Heatmap color-enhancement settings.
# VISUAL_GAMMA < 1 compresses the high-share range and enlarges low-share differences.
# CMAP_MIN_CUT removes the nearly-white lower end of Blues, so low nonzero shares are more visible.
# ZERO_TOL controls which cells are treated as zero and drawn in pure white.
VISUAL_GAMMA = 0.45
CMAP_MIN_CUT = 0.18
CMAP_MAX_CUT = 1.00
ZERO_TOL = 1e-12  # values with abs(value) <= ZERO_TOL are displayed as pure white


# =========================================================
# 1. Helper functions
# =========================================================
def calculate_total_distance(flow_df, distance_df):
    """Calculate total travel distance weighted by flow."""
    return np.sum(flow_df.values * distance_df.values)


def calculate_poi_income_distribution(A_sub, cbgs, pois, cbg_income_dist_dict, income_levels):
    """Calculate visitor income composition for each POI."""
    num_levels = len(income_levels)
    poi_income_dist = {}

    for poi in pois:
        flows = A_sub[poi].values
        total_flow = flows.sum()

        if total_flow == 0:
            poi_income_dist[poi] = np.zeros(num_levels)
            continue

        Q_j = np.zeros(num_levels)
        for i, cbg in enumerate(cbgs):
            if cbg in cbg_income_dist_dict:
                P_i = np.array([cbg_income_dist_dict[cbg][level] for level in income_levels])
                Q_j += (flows[i] / total_flow) * P_i

        if Q_j.sum() > 0:
            Q_j /= Q_j.sum()

        poi_income_dist[poi] = Q_j

    return poi_income_dist


def js_divergence(P, Q):
    """Calculate Jensen-Shannon divergence."""
    P = P + 1e-10
    Q = Q + 1e-10
    M = 0.5 * (P + Q)
    kl_p_m = entropy(P, M, base=2)
    kl_q_m = entropy(Q, M, base=2)
    return 0.5 * (kl_p_m + kl_q_m)


def calculate_social_exposure(P_i, Q_j):
    """Calculate social exposure between CBG income distribution and POI visitor distribution."""
    if len(P_i) != 4 or len(Q_j) != 4:
        raise ValueError("收入分布向量必须为4维")
    return np.sum(P_i * (1 - Q_j))


def normalize_rows_to_share(df):
    """
    Normalize each row to sum to 1.

    This converts absolute CBG-to-POI flows into within-CBG visit shares.
    If a row sum is 0, the normalized row is set to 0.
    """
    row_sums = df.sum(axis=1).replace(0, np.nan)
    share_df = df.div(row_sums, axis=0).fillna(0.0)
    return share_df


def truncate_colormap(base_cmap_name='Blues', min_cut=0.18, max_cut=1.00, n=256):
    """
    Remove the nearly-white part of a matplotlib colormap.

    This keeps zero and very small shares visually distinguishable while preserving
    the same blue color family.
    """
    base_cmap = plt.get_cmap(base_cmap_name)
    colors = base_cmap(np.linspace(min_cut, max_cut, n))
    return LinearSegmentedColormap.from_list(
        f'{base_cmap_name}_truncated_{min_cut:.2f}_{max_cut:.2f}',
        colors
    )
def make_visit_share_blue_cmap(n=256):
    """
    Sequential blue colormap:
    - darkest color: #4C78A8
    - middle color:  #C0D0E1
    - low values:    fade to near-white
    """
    colors = [
        "#FFFFFF",  # zero / near-zero neighborhood
        "#F4F7FA",
        "#E7EEF5",
        "#D5E1EC",
        "#C0D0E1",  # middle tone
        "#A9BDD4",
        "#89A8C8",
        "#6C90B9",
        "#4C78A8"   # darkest tone
    ]

    return LinearSegmentedColormap.from_list(
        "visit_share_soft_blue_balanced",
        colors,
        N=n
    )
def update_social_exposure_matrix(flow_df, S_sub, selected_cbgs, selected_pois,
                                  cbg_income_dist_dict, income_levels):
    """Update social exposure matrix according to current optimized flows."""
    poi_income_dist = calculate_poi_income_distribution(
        flow_df,
        selected_cbgs,
        selected_pois,
        cbg_income_dist_dict,
        income_levels
    )

    for poi in selected_pois:
        Q_j = poi_income_dist[poi]
        for cbg in selected_cbgs:
            if cbg not in cbg_income_dist_dict:
                S_sub.at[cbg, poi] = 0
                continue

            P_i = np.array([cbg_income_dist_dict[cbg][level] for level in income_levels])
            S_sub.at[cbg, poi] = calculate_social_exposure(P_i, Q_j)

    return S_sub


def plot_iteration_diagnostics(f_values_iter, diff_iter, distances_iter, social_iter):
    """Plot optimization diagnostics."""
    fig, axs = plt.subplots(2, 2, figsize=(14, 10), dpi=300)
    colors = sns.color_palette("husl", 4)

    axs[0, 0].plot(f_values_iter, color=colors[0], linewidth=2, marker='^', markersize=4)
    axs[0, 0].set_title('Function Values', pad=10)
    axs[0, 0].set_xlabel('Iteration')
    axs[0, 0].set_ylabel('F-value')
    axs[0, 0].grid(True, linestyle='--', alpha=0.7)

    axs[0, 1].plot(diff_iter, color=colors[1], linewidth=2, marker='o', markersize=4)
    axs[0, 1].set_title(r'Difference between traffic flow $\|H^{k+1}-H^{k}\|_2$', pad=10)
    axs[0, 1].set_xlabel('Iteration')
    axs[0, 1].set_ylabel('Difference')
    axs[0, 1].grid(True, linestyle='--', alpha=0.7)

    axs[1, 0].plot(distances_iter, color=colors[2], linewidth=2, marker='d', markersize=4)
    axs[1, 0].set_title('Distance', pad=10)
    axs[1, 0].set_xlabel('Iteration')
    axs[1, 0].set_ylabel('Distance')
    axs[1, 0].grid(True, linestyle='--', alpha=0.7)

    axs[1, 1].plot(social_iter, color=colors[3], linewidth=2, marker='s', markersize=4)
    axs[1, 1].set_title('Social Exposure', pad=10)
    axs[1, 1].set_xlabel('Iteration')
    axs[1, 1].set_ylabel('Sum of Social Exposure Matrix')
    axs[1, 1].grid(True, linestyle='--', alpha=0.7)

    # Avoid tight_layout here because the colorbar is manually positioned.
    # If you need more left/right margin, adjust these values instead.
    plt.subplots_adjust(left=0.08, right=0.90, bottom=0.16, top=0.88)
    plt.show()


def plot_cbg_visit_share_heatmaps(A_original, H_optimized, selected_cbgs, selected_pois,
                                  output_fig='figure2a_cbg_visit_share_deep_zero_white.pdf',
                                  save_fig=True,
                                  visual_gamma=0.45,
                                  cmap_min_cut=0.18,
                                  cmap_max_cut=1.00,
                                  zero_tol=1e-12):
    """
    Plot original and optimized CBG-level visit shares.

    Each row is normalized to 1, so colors represent the share of a CBG's
    visits allocated to each POI, not absolute visit counts.

    The heatmap intentionally uses PowerNorm(gamma < 1) plus a truncated Blues
    colormap. This visually strengthens low nonzero visit shares while keeping
    the same 0%--100% scale on the colorbar.

    Zero cells are masked and shown as pure white, so white means exactly
    no CBG-to-POI visits after row normalization, not merely a small share.
    """
    A_share = normalize_rows_to_share(A_original)
    H_share = normalize_rows_to_share(H_optimized)

    nrows = len(selected_cbgs)
    ncols = len(selected_pois)

    # Shared normalization: visit share is always between 0 and 1.
    # Use nonlinear color normalization to make low visit shares more visible.
    # With gamma < 1, small positive values receive stronger color contrast.
    norm = PowerNorm(gamma=visual_gamma, vmin=0, vmax=1)

    # Remove the almost-white lower end of Blues to deepen low-share cells.
    # Then explicitly set masked cells to pure white.
    # This way, zero cells are white, while low-but-positive cells are still blue.
    cmap = make_visit_share_blue_cmap()
    cmap = cmap.copy()

    # Use the same very light background tone as the new blue palette.
    # Masked zero cells will be drawn with this color.
    zero_color = "#F6F8FC"
    cmap.set_bad(color=zero_color)

    # Mask exact or near-zero shares. These cells will be drawn as pure white.
    A_share_plot = np.ma.masked_where(A_share.values <= zero_tol, A_share.values)
    H_share_plot = np.ma.masked_where(H_share.values <= zero_tol, H_share.values)

    # Two heatmap panels only. The colorbar is placed manually relative to ax2.
    # This makes the colorbar height and the gap between the colorbar and heatmap easier to control.
    axis_aspect = ncols / nrows
    width_ratios = [axis_aspect, axis_aspect]

    fig = plt.figure(figsize=(16, 8), dpi=300)
    gs = gridspec.GridSpec(
        nrows=1,
        ncols=2,
        width_ratios=width_ratios,
        wspace=0.10   # 控制两个热图之间的距离；越小越近
    )

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    im1 = ax1.imshow(
        A_share_plot,
        cmap=cmap,
        norm=norm,
        interpolation='nearest',
        origin='upper',
        aspect='equal'
    )
    im2 = ax2.imshow(
        H_share_plot,
        cmap=cmap,
        norm=norm,
        interpolation='nearest',
        origin='upper',
        aspect='equal'
    )

    # -----------------------------------------------------
    # Manual colorbar placement.
    # Main controls:
    #   cbar_gap          : distance from the optimized heatmap to the colorbar
    #   cbar_width        : width of the colorbar
    #   cbar_height_ratio : height relative to the optimized heatmap
    #   cbar_y_shift      : vertical shift; positive moves upward, negative moves downward
    # -----------------------------------------------------
    pos2 = ax2.get_position()

    cbar_gap = 0.006          # 越小，cbar 离右侧热图越近；可试 0.003, 0.006, 0.012
    cbar_width = 0.014        # cbar 宽度；可试 0.010--0.020
    cbar_height_ratio = 0.97  # cbar 高度；1.00 与热图等高，0.85 为 85%
    cbar_y_shift = 0.03      # 向上移动用正数，如 0.02；向下移动用负数

    cbar_height = pos2.height * cbar_height_ratio
    cbar_x = pos2.x1 + cbar_gap
    cbar_y = pos2.y0 + (pos2.height - cbar_height) / 2 + cbar_y_shift

    cax = fig.add_axes([cbar_x, cbar_y, cbar_width, cbar_height])

    poi_labels = [f'POI{i + 1}' for i in range(ncols)]
    cbg_labels = [f'CBG{i + 1}' for i in range(nrows)]

    for ax in (ax1, ax2):
        ax.set_xticks(np.arange(ncols))
        ax.set_xticklabels(poi_labels, rotation=0, ha='center', fontsize=14)
        ax.set_yticks(np.arange(nrows))
        ax.set_yticklabels(cbg_labels, fontsize=14)

        # Add light grid lines to make row-wise redistribution clearer.
        ax.set_xticks(np.arange(-0.5, ncols, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, nrows, 1), minor=True)
        ax.grid(which='minor', color='white', linestyle='-', linewidth=0.6)
        ax.tick_params(which='minor', bottom=False, left=False)

    # Only keep y labels on the left panel.
    ax2.tick_params(left=False, labelleft=False)

    ax1.set_title('Original Visit Share', pad=8, fontsize=14.5)
    ax2.set_title('Reassigned Visit Share', pad=8, fontsize=14.5)

    cb = fig.colorbar(im2, cax=cax, orientation='vertical')
    cb.set_label('Within-CBG Visit Share', rotation=90, labelpad=6, fontsize=12)
    cb.ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    
    # 调整刻度数字大小
    cb.ax.tick_params(axis='y', labelsize=12)

    # Avoid tight_layout here because the colorbar is manually positioned.
    # If you need more left/right margin, adjust these values instead.
    plt.subplots_adjust(left=0.08, right=0.90, bottom=0.16, top=0.88)

    plt.savefig('figure2a.pdf',
                format='pdf',
                dpi=300,
                bbox_inches='tight',
                transparent=False,
                backend='pdf')

    plt.show()

    return A_share, H_share


# =========================================================
# 2. Load matrices
# =========================================================
flow_matrix = pd.read_csv(f'{cat_dir}/flow_matrix.csv', index_col=0)
distance_matrix = pd.read_csv(f'{cat_dir}/distance_matrix.csv', index_col=0)
social_exposure_matrix_js = pd.read_csv(f'{cat_dir}/social_exposure_matrix.csv', index_col=0)

cbg_income_dist_df = pd.read_csv(
    f'matrices_A_D_S_Distribution_{city}_core/cbg_income_level_distribution_{city}_msa_core.csv',
    dtype={'GEOID': np.int64}
)
cbg_income_dist_dict = cbg_income_dist_df.set_index('GEOID').to_dict(orient='index')

income_levels = [
    'low_income_pct',
    'lower_middle_income_pct',
    'upper_middle_income_pct',
    'high_income_pct'
]

# Make sure CBG ids have the same type as the income dictionary keys.
try:
    flow_matrix.index = flow_matrix.index.astype(np.int64)
    distance_matrix.index = distance_matrix.index.astype(np.int64)
    social_exposure_matrix_js.index = social_exposure_matrix_js.index.astype(np.int64)
except ValueError:
    pass


# =========================================================
# 3. Select POIs and CBGs
# =========================================================
# Select POIs using the same logic as the original code: middle 10 among high-flow POIs.
poi_total_flow = flow_matrix.sum(axis=0)
sorted_pois = poi_total_flow.sort_values(ascending=False)
mid_index = 7
mid_start = max(0, mid_index - 5)
mid_end = mid_index + 5
selected_pois = sorted_pois[mid_start:mid_end].index.tolist()

# Select top-flow CBGs.
cbg_total_flow = flow_matrix.sum(axis=1)
selected_cbgs = cbg_total_flow.sort_values(ascending=False).head(cbg_num).index.tolist()

# Extract original submatrices.
A_original = flow_matrix.loc[selected_cbgs, selected_pois].copy()
D_sub = distance_matrix.loc[selected_cbgs, selected_pois].copy()
S_sub = social_exposure_matrix_js.loc[selected_cbgs, selected_pois].copy()

# This is the fixed CBG-level total-visit amount.
# The optimization will conserve each row sum relative to this original matrix.
initial_cbg_total_visits = A_original.sum(axis=1).copy()


# =========================================================
# 4. Initialize optimization
# =========================================================
total_flow = A_original.sum().sum()
F = F_ratio * total_flow

H_prev = A_original.copy()
H_opt_df = H_prev.copy()
F_sub = A_original.copy()
iteration = 0

# Diagnostics
diff_iter = []
social_iter = []
f_values_iter = []
distances_iter = []

distances_iter.append(calculate_total_distance(A_original, D_sub))
social_iter.append(S_sub.sum().sum())

A_flat = A_original.values.flatten()
D_flat = D_sub.values.flatten()
S_flat = S_sub.values.flatten() * 100
c = alpha * D_flat - (1 - alpha) * S_flat
current_objective_value = np.dot(c, A_flat)
f_values_iter.append(current_objective_value)

# Initial social exposure update.
S_sub = update_social_exposure_matrix(
    A_original,
    S_sub,
    selected_cbgs,
    selected_pois,
    cbg_income_dist_dict,
    income_levels
)


# =========================================================
# 5. Iterative optimization
# =========================================================
start_time = time.time()

while iteration < max_iterations:
    A_current = H_prev.copy()

    # Restricted OD pairs: if current flow is 0 and distance is too large, keep H_ij = 0.
    restricted_mask = (A_current == 0) & (D_sub >= 50)

    # Existing positive OD pairs keep a lower bound of 1.
    A_current_greater_than_1 = (H_prev >= 1)

    A_flat = A_current.values.flatten()
    D_flat = D_sub.values.flatten()
    S_flat = S_sub.values.flatten() * 100

    c = alpha * D_flat - (1 - alpha) * S_flat

    solver = pywraplp.Solver.CreateSolver('SCIP')
    if solver is None:
        raise RuntimeError("SCIP solver is not available. Please check your OR-Tools installation.")

    H_vars = {}
    Z_vars = {}

    for i, cbg in enumerate(selected_cbgs):
        for j, poi in enumerate(selected_pois):
            k = i * len(selected_pois) + j

            if restricted_mask.values.flatten()[k]:
                H_vars[(i, j)] = solver.IntVar(0, 0, f"H_{i}_{j}")
            else:
                lb = 1 if A_current_greater_than_1.values.flatten()[k] else 0
                H_vars[(i, j)] = solver.IntVar(lb, solver.infinity(), f"H_{i}_{j}")

            Z_vars[(i, j)] = solver.NumVar(0, solver.infinity(), f"Z_{i}_{j}")

    # Objective function.
    obj = sum(
        c[i * len(selected_pois) + j] * H_vars[(i, j)]
        for i in range(len(selected_cbgs))
        for j in range(len(selected_pois))
    )
    solver.Minimize(obj)

    # Constraint 1: linearized absolute change |H_ij - A_ij| <= Z_ij.
    for i, cbg in enumerate(selected_cbgs):
        for j, poi in enumerate(selected_pois):
            k = i * len(selected_pois) + j
            solver.Add(H_vars[(i, j)] - Z_vars[(i, j)] <= A_flat[k])
            solver.Add(-H_vars[(i, j)] - Z_vars[(i, j)] <= -A_flat[k])

    # Constraint 2: total adjustment budget.
    solver.Add(
        sum(
            Z_vars[(i, j)]
            for i in range(len(selected_cbgs))
            for j in range(len(selected_pois))
        ) <= 2 * F
    )

    # Constraint 3: CBG-level total visits are conserved.
    # This is the key row-sum conservation constraint.
    for i, cbg in enumerate(selected_cbgs):
        solver.Add(
            sum(H_vars[(i, j)] for j in range(len(selected_pois)))
            == float(initial_cbg_total_visits.loc[cbg])
        )

    # Constraint 4: POI capacity constraint.
    for j, poi in enumerate(selected_pois):
        solver.Add(
            sum(H_vars[(i, j)] for i in range(len(selected_cbgs)))
            <= beta * F_sub[poi].sum()
        )

    status = solver.Solve()

    if status == pywraplp.Solver.OPTIMAL:
        H_opt = np.array([
            [H_vars[(i, j)].solution_value() for j in range(len(selected_pois))]
            for i in range(len(selected_cbgs))
        ])
        H_opt_df = pd.DataFrame(H_opt, index=selected_cbgs, columns=selected_pois)

        f_values_iter.append(solver.Objective().Value())

        diff = np.linalg.norm(H_opt_df.values - H_prev.values)
        diff_iter.append(diff)
        print(f"Iteration {iteration + 1}: Change in H = {diff}")

        H_prev = H_opt_df.copy()
        distances_iter.append(calculate_total_distance(H_opt_df, D_sub))

        S_sub = update_social_exposure_matrix(
            H_opt_df,
            S_sub,
            selected_cbgs,
            selected_pois,
            cbg_income_dist_dict,
            income_levels
        )

        total_social_exposure = S_sub.sum().sum()
        social_iter.append(total_social_exposure)

        iteration += 1
        print(f'迭代次数 {iteration}')
        print(f'S_sub 矩阵的所有元素之和为: {total_social_exposure}')

        if diff < epsilon:
            print(f'收敛：Change in H < epsilon ({epsilon})')
            break

    else:
        print("优化失败")
        break

end_time = time.time()
elapsed_time = end_time - start_time
print(f"程序运行时间：{elapsed_time} 秒")


# =========================================================
# 6. Conservation check
# =========================================================
row_conservation_error = (H_opt_df.sum(axis=1) - initial_cbg_total_visits).abs().max()
print(f"最大 CBG 行和守恒误差: {row_conservation_error}")

if row_conservation_error > 1e-6:
    print("警告：存在 CBG 行和不完全守恒，请检查输入流量是否为整数或约束是否可行。")


# =========================================================
# 7. Plot diagnostics and row-normalized Figure 2a
# =========================================================
# plot_iteration_diagnostics(
#     f_values_iter=f_values_iter,
#     diff_iter=diff_iter,
#     distances_iter=distances_iter,
#     social_iter=social_iter
# )

A_share, H_share = plot_cbg_visit_share_heatmaps(
    A_original=A_original,
    H_optimized=H_opt_df,
    selected_cbgs=selected_cbgs,
    selected_pois=selected_pois,
    output_fig=output_fig,
    save_fig=save_fig,
    visual_gamma=VISUAL_GAMMA,
    cmap_min_cut=CMAP_MIN_CUT,
    cmap_max_cut=CMAP_MAX_CUT,
    zero_tol=ZERO_TOL
)

# # Optional: save normalized matrices for checking or manuscript use.
# A_share.to_csv('figure2a_original_visit_share.csv')
# H_share.to_csv('figure2a_optimized_visit_share.csv')

print("原始访问份额矩阵每行和：")
print(A_share.sum(axis=1))
print("优化后访问份额矩阵每行和：")
print(H_share.sum(axis=1))
