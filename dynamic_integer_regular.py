# -*- coding: utf-8 -*-
"""
Created on Sun Aug 10 16:55:35 2025

@author: 天天向上
"""



import pandas as pd
import numpy as np
from ortools.linear_solver import pywraplp  # 导入 OR-Tools 的线性求解器
from scipy.stats import entropy
from collections import Counter
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import random
import time

'''
naics_map = {
    '624190':  'Other Individual and Family Services', 53
    '711310':  'Promoters of Performing Arts, Sports, and Similar Events with Facilities',58
    '712110':  'Museums',193
    '713940':  'Fitness and Recreational Sports Centers',
    '722410':  'Drinking Places (Alcoholic Beverages)',
    '813110':  'Religious Organizations',
    '813410':  'Civic and Social Organizations'
}
'''

# 设置类别和文件路径
city = "boston"
category = 'Other Individual and Family Services'
cat_dir = f'matrices_A_D_S_Distribution/{category.replace(" ", "_")}'

# 1. 加载矩阵
flow_matrix = pd.read_csv(f'{cat_dir}/flow_matrix.csv', index_col=0)
distance_matrix = pd.read_csv(f'{cat_dir}/distance_matrix.csv', index_col=0)
social_exposure_matrix_js = pd.read_csv(f'{cat_dir}/social_exposure_matrix.csv', index_col=0)

# 加载 CBG 收入水平数据
cbg_income_dist_df = pd.read_csv('matrices_A_D_S_Distribution/cbg_income_level_distribution_boston_msa.csv', dtype={'GEOID': np.int64})
cbg_income_dist_dict = cbg_income_dist_df.set_index('GEOID').to_dict(orient='index')

# 固定的收入水平顺序
income_levels = ['low_income_pct', 'lower_middle_income_pct', 'upper_middle_income_pct', 'high_income_pct']

# 计算出行总距离
def calculate_total_distance(flow_matrix, distance_matrix):
    return np.sum(flow_matrix.values * distance_matrix.values)

def calculate_poi_income_distribution(A_sub, cbgs, pois, cbg_income_dist_dict, income_levels):
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

# 计算 JS 散度
def js_divergence(P, Q):
    P = P + 1e-10
    Q = Q + 1e-10
    M = 0.5 * (P + Q)
    kl_p_m = entropy(P, M, base=2)
    kl_q_m = entropy(Q, M, base=2)
    return 0.5 * (kl_p_m + kl_q_m)

def calculate_social_exposure(P_i, Q_j):
    if len(P_i) != 4 or len(Q_j) != 4:
        raise ValueError("收入分布向量必须为4维")
    exposure = np.sum(P_i * (1 - Q_j))
    return exposure

# 2. 选择流量排名前 10 的 POI
poi_total_flow = flow_matrix.sum(axis=0)
poi_num = flow_matrix.shape[1]
selected_pois = (
    poi_total_flow.sort_values(ascending=False)
    .head(poi_num)
    .index.tolist()
)



# 3. 获取所有访问这些 POI 的 CBG
selected_cbgs = set()
for poi in selected_pois:
    cbgs_with_flow = flow_matrix.index[flow_matrix[poi] > 0].tolist()
    selected_cbgs.update(cbgs_with_flow)
selected_cbgs = list(selected_cbgs)

# selected_cbgs = sorted(selected_cbgs)

# 提取子矩阵
A_sub = flow_matrix.loc[selected_cbgs, selected_pois]
D_sub = distance_matrix.loc[selected_cbgs, selected_pois]
S_sub = social_exposure_matrix_js.loc[selected_cbgs, selected_pois]
R_s = pd.read_csv(f'{cat_dir}/pred_rownorm_int_preserve.csv', index_col=0)
R_sub = R_s.loc[selected_cbgs, selected_pois]

# 预先展平的一些参数
num_cbgs = len(selected_cbgs)
num_pois = len(selected_pois)
num_od = num_cbgs * num_pois


# 4. 设置优化参数
alpha = 0.5
beta = 2
total_flow = A_sub.sum().sum()
F = 0.01 * total_flow
epsilon = 1e-4
max_iterations = 100

alpha1 = 1/3
alpha2 = 1/3
alpha3 = 1/3

# 初始化
H_prev = A_sub.copy()
F_sub = A_sub.copy()
iteration = 0

diff_iter = []
social_iter = []
f_values_iter = []
distances_iter = []

distances_iter.append(calculate_total_distance(A_sub, D_sub))
social_iter.append(S_sub.sum().sum())

# 展平矩阵
A_flat = A_sub.values.flatten()
D_flat = D_sub.values.flatten()
S_flat = S_sub.values.flatten() * 100


# 计算初始目标值（可选）
c_base = alpha1 * D_flat - alpha2 * S_flat
current_objective_value = np.dot(c_base, A_flat)
current_objective_value = np.dot(c_base, A_flat) + alpha3 *np.sum(np.abs(A_flat - R_sub.values.flatten()))
f_values_iter.append(current_objective_value)


# 计算所有 POI 的收入分布
poi_income_dist = calculate_poi_income_distribution(A_sub, selected_cbgs, selected_pois, cbg_income_dist_dict, income_levels)

# 更新社交暴露矩阵
for poi in selected_pois:
    Q_j = poi_income_dist[poi]
    for cbg in selected_cbgs:
        if cbg in cbg_income_dist_dict:
            P_i = np.array([cbg_income_dist_dict[cbg][level] for level in income_levels])
            exposure = calculate_social_exposure(P_i, Q_j)
            S_sub.at[cbg, poi] = exposure


start_time = time.time()

while iteration < max_iterations:
    A_sub = H_prev
    # 识别受限的 OD 对：A_ij = 0 且 D_ij >= 50
    restricted_mask = (A_sub == 0) & (D_sub >= 50)
    # 识别 A_sub 中大于1的位置
    A_sub_greater_than_1 = (H_prev >= 1)
    # 展平矩阵
    A_flat = A_sub.values.flatten()
    D_flat = D_sub.values.flatten()
    S_flat = S_sub.values.flatten() * 100
    R_flat = R_sub.values.flatten().astype(float)

    # 目标函数系数
    c = alpha1 * D_flat - alpha2 * S_flat

    # 创建 OR-Tools 求解器
    solver = pywraplp.Solver.CreateSolver('SCIP')  # 使用 SCIP 求解器支持整数规划

    # 定义决策变量 H_ij（整数）和 Z_ij（连续）
    num_od = len(A_flat)
    H_vars = {}
    Z_vars = {}
    W_vars = {}  # 用于线性化 |H - R|
    for i, cbg in enumerate(selected_cbgs):
        for j, poi in enumerate(selected_pois):
            k = i * len(selected_pois) + j
            # H_ij 为整数变量
            if restricted_mask.values.flatten()[k]:
                H_vars[(i, j)] = solver.IntVar(0, 0, f"H_{i}_{j}")  # 受限 OD 对，H_ij = 0
            else:
                lb = 1 if A_sub_greater_than_1.values.flatten()[k] else 0
                H_vars[(i, j)] = solver.IntVar(lb, solver.infinity(), f"H_{i}_{j}")
            # Z_ij 为连续变量
            Z_vars[(i, j)] = solver.NumVar(0, solver.infinity(), f"Z_{i}_{j}")
            # W 用于 |H - R| 线性化（新引入的正则）
            W_vars[(i, j)] = solver.NumVar(0.0, solver.infinity(), f'W_{i}_{j}')

    # 设置目标函数
    
    # 目标函数： sum(c_k * H_k) + alpha3 * sum(W_k)
    objective = solver.Objective()
    for i in range(num_cbgs):
        for j in range(num_pois):
            k = i * num_pois + j
            objective.SetCoefficient(H_vars[(i, j)], c[k])
            objective.SetCoefficient(W_vars[(i, j)], alpha3)
    objective.SetMinimization()

    # 添加约束
    # 1. 变动预算线性化约束
    for i, cbg in enumerate(selected_cbgs):
        for j, poi in enumerate(selected_pois):
            k = i * len(selected_pois) + j
            solver.Add(H_vars[(i, j)] - Z_vars[(i, j)] <= A_flat[k])
            solver.Add(-H_vars[(i, j)] - Z_vars[(i, j)] <= -A_flat[k])

    # 2. sum Z_ij <= 2F
    solver.Add(sum(Z_vars[(i, j)] for i in range(len(selected_cbgs)) 
                   for j in range(len(selected_pois))) <= 2 * F)

    # 3. 流量守恒约束
    for i, cbg in enumerate(selected_cbgs):
        solver.Add(sum(H_vars[(i, j)] for j in range(len(selected_pois))) == H_prev.iloc[i, :].sum())

    # 4. 容量约束
    for j, poi in enumerate(selected_pois):
        solver.Add(sum(H_vars[(i, j)] for i in range(len(selected_cbgs))) <= beta * F_sub[poi].sum())
    
    # 约束 5：线性化 |H - R|，使 W_k >= |H_k - R_flat[k]|
    for i in range(num_cbgs):
        for j in range(num_pois):
            k = i * num_pois + j
            solver.Add(H_vars[(i, j)] - W_vars[(i, j)] <= R_flat[k])
            solver.Add(-H_vars[(i, j)] - W_vars[(i, j)] <= -R_flat[k])
    
    # 求解优化问题
    status = solver.Solve()

    if status == pywraplp.Solver.OPTIMAL:
        # 提取优化结果
        H_opt = np.array([[H_vars[(i, j)].solution_value() for j in range(len(selected_pois))] 
                          for i in range(len(selected_cbgs))])
        H_opt_df = pd.DataFrame(H_opt, index=selected_cbgs, columns=selected_pois)
        
        # 保存目标函数值
        f_values_iter.append(solver.Objective().Value())

        # 检查收敛性
        diff = np.linalg.norm(H_opt_df.values - H_prev.values)
        print(f"Iteration {iteration + 1}: Change in H = {diff}")
        diff_iter.append(diff)

        H_prev = H_opt_df.copy()
        distance_k = calculate_total_distance(H_opt_df, D_sub)
        distances_iter.append(distance_k)

        # 更新 POI 收入分布和社会暴露矩阵
        poi_income_dist = calculate_poi_income_distribution(H_opt_df, selected_cbgs, selected_pois, cbg_income_dist_dict, income_levels)
        for poi in selected_pois:
            visitor_dict = {cbg: flow for cbg, flow in zip(selected_cbgs, H_opt_df[poi].values) if flow > 0}
            if not visitor_dict:
                for cbg in selected_cbgs:
                    S_sub.at[cbg, poi] = 0
                continue
            Q_j = poi_income_dist[poi]
            for cbg in selected_cbgs:
                if H_opt_df.at[cbg, poi] > 0:
                    if cbg in cbg_income_dist_dict:
                        P_i = np.array([cbg_income_dist_dict[cbg][level] for level in income_levels])
                        exposure = calculate_social_exposure(P_i, Q_j)
                        S_sub.at[cbg, poi] = exposure
                else:
                    S_sub.at[cbg, poi] = 0

        iteration += 1
        print(f'迭代次数{iteration}')
        total_sum = S_sub.sum().sum()
        social_iter.append(total_sum)
        print(f"S_sub 矩阵的所有元素之和为: {total_sum}")
        

        # 补全社交暴露值
        for poi in selected_pois:
            Q_j = poi_income_dist[poi]
            for cbg in selected_cbgs:
                P_i = np.array([cbg_income_dist_dict[cbg][level] for level in income_levels])
                if cbg in cbg_income_dist_dict and S_sub.at[cbg, poi] == 0:
                    S_sub.at[cbg, poi] = calculate_social_exposure(P_i, Q_j)
    else:
        print("优化失败")
        break

end_time = time.time()
elapsed_time = end_time - start_time
print(f"程序运行时间：{elapsed_time} 秒")


A_sub = flow_matrix.loc[selected_cbgs, selected_pois]

# 计算信息熵
def calculate_poi_entropy(flow_matrix, selected_cbgs, selected_pois, cbg_income_dist_dict, income_levels):
    # 计算所有 POI 的收入分布
    poi_income_dist = calculate_poi_income_distribution(flow_matrix, selected_cbgs, selected_pois, cbg_income_dist_dict, income_levels)
    entropy_values = {}
    for poi in selected_pois:
        Q_j = poi_income_dist[poi]  # 获取该POI的收入分布
        # 如果总流量为0（Q_j 全为0），熵设为0；否则计算熵
        entropy_values[poi] = entropy(Q_j, base=2) if np.any(Q_j > 0) else 0.0
    return entropy_values


# 输出结果
if status == pywraplp.Solver.OPTIMAL:
    # 计算信息熵
    entropy_before = calculate_poi_entropy(A_sub, selected_cbgs, selected_pois, cbg_income_dist_dict, income_levels)
    entropy_after = calculate_poi_entropy(H_opt_df, selected_cbgs, selected_pois, cbg_income_dist_dict, income_levels)
    
    print("\nPOI 信息熵（流量改变前后）：")
    entropy_df = pd.DataFrame({
        'Entropy_Before': pd.Series(entropy_before),
        'Entropy_After': pd.Series(entropy_after)
    })
    print(entropy_df)
    entropy_before_total = sum(entropy_before.values())
    entropy_after_total = sum(entropy_after.values())
    entropy_summary = pd.DataFrame({'Total': [entropy_before_total, entropy_after_total]}, 
                                   index=['Before', 'After'])
    print("\n总信息熵对比：")
    print(entropy_summary)
    
    # 计算出行总距离
    distance_before = calculate_total_distance(A_sub, D_sub)
    distance_after = calculate_total_distance(H_opt_df, D_sub)
    distance_summary = pd.DataFrame({
        'Total_Distance': [distance_before, distance_after]
    }, index=['Before', 'After'])
    print("\n出行总距离对比（公里）：")
    print(distance_summary)
else:
    print("优化失败")

print("\n总社交暴露对比：")
print(f"Before {social_iter[0]}")
print(f"After {social_iter[-1]}")



# 设置 Seaborn 风格为 white，去掉灰色背景
sns.set_style("white")
#plt.style.use('ggplot') 
# 设置字体和大小
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Times New Roman'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10

# 创建子图
fig, axs = plt.subplots(2, 2, figsize=(14, 10),dpi =300)

# 使用 Seaborn 的颜色调色板
colors = sns.color_palette("husl", 4)

# 1. 绘制 f_values_iter
axs[0, 0].plot(f_values_iter, color=colors[0], linewidth=2, marker='^', markersize=4)
axs[0, 0].set_title('Function Values', pad=10)
axs[0, 0].set_xlabel('Iteration')
axs[0, 0].set_ylabel('F-value')
axs[0, 0].grid(True, linestyle='--', alpha=0.7)
axs[0, 0].legend(loc='best')

# 2. 绘制 diff_iter
axs[0, 1].plot(diff_iter, color=colors[1], linewidth=2, marker='o', markersize=4)
axs[0, 1].set_title('Difference between the traffic flow $\|H^{k+1}-H^{k}\|_2$', pad=10)
axs[0, 1].set_xlabel('Iteration')
axs[0, 1].set_ylabel('Difference')
axs[0, 1].grid(True, linestyle='--', alpha=0.7)
axs[0, 1].legend(loc='best')

# 3. 绘制 distances_iter
axs[1, 0].plot(distances_iter, color=colors[2], linewidth=2, marker='d', markersize=4)
axs[1, 0].set_title('Distance', pad=10)
axs[1, 0].set_xlabel('Iteration')
axs[1, 0].set_ylabel('Distance')
axs[1, 0].grid(True, linestyle='--', alpha=0.7)
axs[1, 0].legend(loc='best')

# 4. 绘制 social_iter
axs[1, 1].plot(social_iter, color=colors[3], linewidth=2, marker='s', markersize=4)
axs[1, 1].set_title('Social Exposure', pad=10)
axs[1, 1].set_xlabel('Iteration')
axs[1, 1].set_ylabel('Sum of Social Exposure Matrix')
axs[1, 1].grid(True, linestyle='--', alpha=0.7)
axs[1, 1].legend(loc='best')

# 调整间距
plt.tight_layout()

# 保存图片
# plt.savefig('iteration_plots.png', dpi=300, bbox_inches='tight')

# 显示图形
plt.show()

# 定义 POI 对应的 NAICS 代码映射
poi_naics_map = {
    'Other Individual and Family Services': '624190',
    'Promoters of Performing Arts, Sports, and Similar Events with Facilities': '711310',
    'Museums': '712110',
    'Fitness and Recreational Sports Centers': '713940',
    'Drinking Places (Alcoholic Beverages)': '722410',
    'Religious Organizations': '813110',
    'Religious_Organizations_christian': '813110'
}

# 获取当前类别对应的 NAICS 代码
naics_code = poi_naics_map.get(category, "unknown")


# 构造保存结果的 DataFrame
results_df = pd.DataFrame({
    'f_values_iter': pd.Series(f_values_iter),   # 101维
    'distances_iter': pd.Series(distances_iter), # 101维
    'social_iter': pd.Series(social_iter),       # 101维
    'diff_iter': pd.Series(diff_iter)            # 100维
})

# 保存结果文件
results_path = f"{cat_dir}/results_regu_{city}_{naics_code}.csv"
results_df.to_csv(results_path, index=False)

print(f"\n结果已保存至: {results_path}")

# ====== 将最终 H_opt_df 保存为文件（含正则） ======
import os
import pickle
outfile_regu = os.path.join(cat_dir, f'H_opt_df_regu_{city}_{naics_code}.pkl')
with open(outfile_regu, 'wb') as _f:
    pickle.dump(H_opt_df, _f)
print("Saved final H_opt_df (with regularization) to:", outfile_regu)
