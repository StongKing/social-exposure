# -*- coding: utf-8 -*-
"""
Preprocessing for Figure 2a.

Purpose
-------
1. Load the original CBG-POI flow, distance, social-exposure,
   and CBG income-distribution data.
2. Run the sequential reallocation model.
3. Convert the original and reassigned absolute flows into
   within-CBG visit shares.
4. Anonymize CBG and POI identifiers.
5. Save only the normalized plotting data used by Figure 2a.

Outputs
-------
figure2a_original_visit_share.csv
figure2a_reassigned_visit_share.csv

These two files are sufficient for reproducing Figure 2a
without releasing the original absolute mobility-flow data.
"""

import os
import time

import numpy as np
import pandas as pd
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

# Figure-2a sample size
poi_num = 10
cbg_num = 10

# Output plotting data
original_share_file = 'figure2a_original_visit_share.csv'
reassigned_share_file = 'figure2a_reassigned_visit_share.csv'


# =========================================================
# 1. Helper functions
# =========================================================
def calculate_total_distance(flow_df, distance_df):
    """Calculate total travel distance weighted by flow."""
    return np.sum(flow_df.values * distance_df.values)


def calculate_poi_income_distribution(
        A_sub,
        cbgs,
        pois,
        cbg_income_dist_dict,
        income_levels):
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
                P_i = np.array([
                    cbg_income_dist_dict[cbg][level]
                    for level in income_levels
                ])

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
    """
    Calculate social exposure between
    CBG income distribution and POI visitor distribution.
    """
    if len(P_i) != 4 or len(Q_j) != 4:
        raise ValueError("收入分布向量必须为4维")

    return np.sum(P_i * (1 - Q_j))


def normalize_rows_to_share(df):
    """
    Normalize each row to sum to 1.

    This converts absolute CBG-to-POI flows into within-CBG
    visit shares.

    If a row sum is 0, the normalized row is set to 0.
    """
    row_sums = df.sum(axis=1).replace(0, np.nan)

    share_df = df.div(
        row_sums,
        axis=0
    ).fillna(0.0)

    return share_df


def update_social_exposure_matrix(
        flow_df,
        S_sub,
        selected_cbgs,
        selected_pois,
        cbg_income_dist_dict,
        income_levels):
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

            P_i = np.array([
                cbg_income_dist_dict[cbg][level]
                for level in income_levels
            ])

            S_sub.at[cbg, poi] = calculate_social_exposure(
                P_i,
                Q_j
            )

    return S_sub


# =========================================================
# 2. Load matrices
# =========================================================
flow_matrix = pd.read_csv(
    f'{cat_dir}/flow_matrix.csv',
    index_col=0
)

distance_matrix = pd.read_csv(
    f'{cat_dir}/distance_matrix.csv',
    index_col=0
)

social_exposure_matrix_js = pd.read_csv(
    f'{cat_dir}/social_exposure_matrix.csv',
    index_col=0
)

cbg_income_dist_df = pd.read_csv(
    f'matrices_A_D_S_Distribution_{city}_core/'
    f'cbg_income_level_distribution_{city}_msa_core.csv',
    dtype={'GEOID': np.int64}
)

cbg_income_dist_dict = (
    cbg_income_dist_df
    .set_index('GEOID')
    .to_dict(orient='index')
)

income_levels = [
    'low_income_pct',
    'lower_middle_income_pct',
    'upper_middle_income_pct',
    'high_income_pct'
]


# Make sure CBG ids have the same type
# as the income dictionary keys.
try:
    flow_matrix.index = flow_matrix.index.astype(np.int64)
    distance_matrix.index = distance_matrix.index.astype(np.int64)
    social_exposure_matrix_js.index = (
        social_exposure_matrix_js.index.astype(np.int64)
    )

except ValueError:
    pass


# =========================================================
# 3. Select POIs and CBGs
# =========================================================

# Select POIs using the same logic as the original code:
# middle 10 among high-flow POIs.
poi_total_flow = flow_matrix.sum(axis=0)

sorted_pois = poi_total_flow.sort_values(
    ascending=False
)

mid_index = 7
mid_start = max(0, mid_index - 5)
mid_end = mid_index + 5

selected_pois = (
    sorted_pois[mid_start:mid_end]
    .index
    .tolist()
)


# Select top-flow CBGs.
cbg_total_flow = flow_matrix.sum(axis=1)

selected_cbgs = (
    cbg_total_flow
    .sort_values(ascending=False)
    .head(cbg_num)
    .index
    .tolist()
)


# Extract original submatrices.
A_original = flow_matrix.loc[
    selected_cbgs,
    selected_pois
].copy()

D_sub = distance_matrix.loc[
    selected_cbgs,
    selected_pois
].copy()

S_sub = social_exposure_matrix_js.loc[
    selected_cbgs,
    selected_pois
].copy()


# Fixed CBG-level total-visit amount.
initial_cbg_total_visits = (
    A_original.sum(axis=1).copy()
)


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


distances_iter.append(
    calculate_total_distance(
        A_original,
        D_sub
    )
)

social_iter.append(
    S_sub.sum().sum()
)


A_flat = A_original.values.flatten()
D_flat = D_sub.values.flatten()
S_flat = S_sub.values.flatten() * 100

c = (
    alpha * D_flat
    -
    (1 - alpha) * S_flat
)

current_objective_value = np.dot(
    c,
    A_flat
)

f_values_iter.append(
    current_objective_value
)


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


    # Restricted OD pairs:
    # if current flow is 0 and distance is too large,
    # keep H_ij = 0.
    restricted_mask = (
        (A_current == 0)
        &
        (D_sub >= 50)
    )


    # Existing positive OD pairs keep a lower bound of 1.
    A_current_greater_than_1 = (
        H_prev >= 1
    )


    A_flat = A_current.values.flatten()
    D_flat = D_sub.values.flatten()
    S_flat = S_sub.values.flatten() * 100


    c = (
        alpha * D_flat
        -
        (1 - alpha) * S_flat
    )


    solver = pywraplp.Solver.CreateSolver('SCIP')

    if solver is None:
        raise RuntimeError(
            "SCIP solver is not available. "
            "Please check your OR-Tools installation."
        )


    H_vars = {}
    Z_vars = {}


    for i, cbg in enumerate(selected_cbgs):

        for j, poi in enumerate(selected_pois):

            k = i * len(selected_pois) + j


            if restricted_mask.values.flatten()[k]:

                H_vars[(i, j)] = solver.IntVar(
                    0,
                    0,
                    f"H_{i}_{j}"
                )

            else:

                lb = (
                    1
                    if A_current_greater_than_1
                    .values
                    .flatten()[k]
                    else 0
                )

                H_vars[(i, j)] = solver.IntVar(
                    lb,
                    solver.infinity(),
                    f"H_{i}_{j}"
                )


            Z_vars[(i, j)] = solver.NumVar(
                0,
                solver.infinity(),
                f"Z_{i}_{j}"
            )


    # -----------------------------------------------------
    # Objective function
    # -----------------------------------------------------

    obj = sum(
        c[i * len(selected_pois) + j]
        *
        H_vars[(i, j)]

        for i in range(len(selected_cbgs))
        for j in range(len(selected_pois))
    )

    solver.Minimize(obj)


    # -----------------------------------------------------
    # Constraint 1:
    # linearized absolute change
    # |H_ij - A_ij| <= Z_ij
    # -----------------------------------------------------

    for i, cbg in enumerate(selected_cbgs):

        for j, poi in enumerate(selected_pois):

            k = i * len(selected_pois) + j

            solver.Add(
                H_vars[(i, j)]
                -
                Z_vars[(i, j)]
                <=
                A_flat[k]
            )

            solver.Add(
                -H_vars[(i, j)]
                -
                Z_vars[(i, j)]
                <=
                -A_flat[k]
            )


    # -----------------------------------------------------
    # Constraint 2:
    # total adjustment budget
    # -----------------------------------------------------

    solver.Add(
        sum(
            Z_vars[(i, j)]

            for i in range(len(selected_cbgs))
            for j in range(len(selected_pois))
        )

        <=

        2 * F
    )


    # -----------------------------------------------------
    # Constraint 3:
    # CBG-level total visits are conserved.
    # -----------------------------------------------------

    for i, cbg in enumerate(selected_cbgs):

        solver.Add(
            sum(
                H_vars[(i, j)]
                for j in range(len(selected_pois))
            )

            ==

            float(
                initial_cbg_total_visits.loc[cbg]
            )
        )


    # -----------------------------------------------------
    # Constraint 4:
    # POI capacity constraint
    # -----------------------------------------------------

    for j, poi in enumerate(selected_pois):

        solver.Add(
            sum(
                H_vars[(i, j)]
                for i in range(len(selected_cbgs))
            )

            <=

            beta * F_sub[poi].sum()
        )


    # -----------------------------------------------------
    # Solve
    # -----------------------------------------------------

    status = solver.Solve()


    if status == pywraplp.Solver.OPTIMAL:

        H_opt = np.array([
            [
                H_vars[(i, j)].solution_value()
                for j in range(len(selected_pois))
            ]

            for i in range(len(selected_cbgs))
        ])


        H_opt_df = pd.DataFrame(
            H_opt,
            index=selected_cbgs,
            columns=selected_pois
        )


        f_values_iter.append(
            solver.Objective().Value()
        )


        diff = np.linalg.norm(
            H_opt_df.values
            -
            H_prev.values
        )

        diff_iter.append(diff)

        print(
            f"Iteration {iteration + 1}: "
            f"Change in H = {diff}"
        )


        H_prev = H_opt_df.copy()


        distances_iter.append(
            calculate_total_distance(
                H_opt_df,
                D_sub
            )
        )


        S_sub = update_social_exposure_matrix(
            H_opt_df,
            S_sub,
            selected_cbgs,
            selected_pois,
            cbg_income_dist_dict,
            income_levels
        )


        total_social_exposure = (
            S_sub.sum().sum()
        )

        social_iter.append(
            total_social_exposure
        )


        iteration += 1

        print(
            f'迭代次数 {iteration}'
        )

        print(
            f'S_sub 矩阵的所有元素之和为: '
            f'{total_social_exposure}'
        )


        if diff < epsilon:

            print(
                f'收敛：Change in H < epsilon '
                f'({epsilon})'
            )

            break


    else:

        print("优化失败")

        break


end_time = time.time()

elapsed_time = end_time - start_time

print(
    f"程序运行时间：{elapsed_time} 秒"
)


# =========================================================
# 6. Conservation check
# =========================================================

row_conservation_error = (
    H_opt_df.sum(axis=1)
    -
    initial_cbg_total_visits
).abs().max()


print(
    f"最大 CBG 行和守恒误差: "
    f"{row_conservation_error}"
)


if row_conservation_error > 1e-6:

    print(
        "警告：存在 CBG 行和不完全守恒，"
        "请检查输入流量是否为整数或约束是否可行。"
    )


# =========================================================
# 7. Generate Figure 2a plotting data
# =========================================================

# Convert absolute flow to within-CBG visit share.
A_share = normalize_rows_to_share(
    A_original
)

H_share = normalize_rows_to_share(
    H_opt_df
)


# =========================================================
# 8. Anonymize identifiers
# =========================================================

anonymous_cbgs = [
    f'CBG{i + 1}'
    for i in range(len(A_share))
]

anonymous_pois = [
    f'POI{i + 1}'
    for i in range(len(A_share.columns))
]


A_share_public = A_share.copy()

A_share_public.index = anonymous_cbgs
A_share_public.columns = anonymous_pois


H_share_public = H_share.copy()

H_share_public.index = anonymous_cbgs
H_share_public.columns = anonymous_pois


# =========================================================
# 9. Save only plotting data
# =========================================================

A_share_public.to_csv(
    original_share_file
)

H_share_public.to_csv(
    reassigned_share_file
)


# =========================================================
# 10. Final checks
# =========================================================

print("\n============================================")
print("Figure 2a plotting data successfully saved")
print("============================================")

print(
    f"\nOriginal visit-share file:\n"
    f"{original_share_file}"
)

print(
    f"\nReassigned visit-share file:\n"
    f"{reassigned_share_file}"
)


print(
    "\nOriginal visit-share matrix row sums:"
)

print(
    A_share_public.sum(axis=1)
)


print(
    "\nReassigned visit-share matrix row sums:"
)

print(
    H_share_public.sum(axis=1)
)


print(
    "\nMaximum difference between original "
    "and reassigned row sums:"
)

print(
    (
        A_share_public.sum(axis=1)
        -
        H_share_public.sum(axis=1)
    ).abs().max()
)