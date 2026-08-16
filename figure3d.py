# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 14:58:19 2026

@author: JZS
"""

import pandas as pd
import numpy as np
from ortools.linear_solver import pywraplp
from scipy.stats import entropy
import time
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from networkx.algorithms import community
import pickle
import pickle


ks = np.arange(0.01, 1.01, 0.01)
markers = ['o', 's', '^']

# ---------- 每 5 点一个标记 ----------
def sparse_marker(arr, step=5):
    """返回与 arr 同长数组，仅每 step 位保留原值，其余置 nan"""
    out = np.full_like(arr, np.nan, dtype=float)
    out[::step] = arr[::step]
    return out

# 读取第二个文件
with open('k_matrices_boston_family_budget_net_metrics.pkl', 'rb') as f:
    metrics_datas = pickle.load(f)

df_metrics_datas = pd.read_csv('metrics_datas.csv')
disparity_cbgs_finals = df_metrics_datas['disparity_cbgs_finals'].values
disparity_pois_finals = df_metrics_datas['disparity_pois_finals'].values
entropy_cbgs_finals = df_metrics_datas['entropy_cbgs_finals'].values
entropy_pois_finals = df_metrics_datas['entropy_pois_finals'].values
degree_finals = df_metrics_datas['degree_finals'].values
clustering_finals = df_metrics_datas['clustering_finals'].values
betweenness_finals = df_metrics_datas['betweenness_finals'].values
modularity_finals = df_metrics_datas['modularity_finals'].values

metrics_datas = [
    disparity_cbgs_finals, disparity_pois_finals,
    entropy_cbgs_finals, entropy_pois_finals,
    degree_finals, clustering_finals,
    betweenness_finals, modularity_finals
]

# 读取 metrics_initials 并恢复原始变量名（标量）
df_metrics_initials = pd.read_csv('metrics_initials.csv')
initial_disparity_cbgs = df_metrics_initials['initial_disparity_cbgs'].iloc[0]
initial_disparity_pois = df_metrics_initials['initial_disparity_pois'].iloc[0]
initial_entropy_cbgs = df_metrics_initials['initial_entropy_cbgs'].iloc[0]
initial_entropy_pois = df_metrics_initials['initial_entropy_pois'].iloc[0]
initial_degree = df_metrics_initials['initial_degree'].iloc[0]
initial_clustering = df_metrics_initials['initial_clustering'].iloc[0]
initial_betweenness = df_metrics_initials['initial_betweenness'].iloc[0]
initial_modularity = df_metrics_initials['initial_modularity'].iloc[0]

metrics_initials = [
    initial_disparity_cbgs, initial_disparity_pois,
    initial_entropy_cbgs, initial_entropy_pois,
    initial_degree, initial_clustering,
    initial_betweenness, initial_modularity
]

colors = ['#3498db','#e74c3c']   
colors = ["#4C78A8", "#C76B6B"]   # blue / purple / red

# ---------- 绘制网络指标图 ----------
fig, axs = plt.subplots(2, 2, figsize=(10, 10), dpi=300)
axs = axs.flatten()

metrics_titles = [
    'Average Degree', 'Clustering Coefficient',
    'Betweenness Centrality', 'Modularity'
]
metrics_datas = [
    degree_finals, clustering_finals,
    betweenness_finals, modularity_finals
]
metrics_initials = [
    initial_degree, initial_clustering,
    initial_betweenness, initial_modularity
]

for i, (ax, data, title, initial_val) in enumerate(zip(axs, metrics_datas, metrics_titles, metrics_initials)):
    # 绘制最终值
    ax.plot(ks, data, color=colors[0], linewidth=2, label='Final')
    ax.plot(ks, sparse_marker(data), color=colors[0],
            marker=markers[0], markersize=6, markeredgewidth=1,
            markerfacecolor='w', linestyle='None', clip_on=False)
    
    # 绘制初始值（常量线）
    ax.axhline(y=initial_val, color=colors[1], linestyle='--', linewidth=2, label='Initial')
    
    ax.set_xlabel('$k$')
    #ax.set_ylabel(title)
    ax.set_title(title)
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.legend(loc='center right')
    ax.set_xlim(ks.min(), ks.max())

plt.tight_layout()
plt.savefig('figure3d.pdf',
            format='pdf',
            dpi=300,
            bbox_inches='tight',
            transparent=False,
            backend='pdf')
plt.show()


# ---------- 绘制网络指标图 + 输出 from-to 数值 ----------
fig, axs = plt.subplots(2, 2, figsize=(10, 10), dpi=300)
axs = axs.flatten()

metrics_titles = [
    'Average Degree', 
    'Clustering Coefficient',
    'Mean Betweenness Centrality', 
    'Modularity'
]

metrics_datas = [
    degree_finals, 
    clustering_finals,
    betweenness_finals, 
    modularity_finals
]

metrics_initials = [
    initial_degree, 
    initial_clustering,
    initial_betweenness, 
    initial_modularity
]

# 选择正文中报告的代表性预算
# 建议用 0.20，因为前文已经将 20% budget 作为重要政策节点
report_k = 0.20
report_idx = int(np.argmin(np.abs(ks - report_k)))
actual_report_k = float(ks[report_idx])

summary_rows = []

for i, (ax, data, title, initial_val) in enumerate(
    zip(axs, metrics_datas, metrics_titles, metrics_initials)
):
    final_val = float(data[report_idx])
    initial_val = float(initial_val)
    abs_change = final_val - initial_val
    pct_change = (abs_change / initial_val * 100) if initial_val != 0 else np.nan

    min_val = float(np.nanmin(data))
    max_val = float(np.nanmax(data))
    min_k = float(ks[int(np.nanargmin(data))])
    max_k = float(ks[int(np.nanargmax(data))])

    direction = "increases" if abs_change > 0 else "decreases"

    summary_rows.append({
        "metric": title,
        "initial": initial_val,
        f"value_at_k_{actual_report_k:.2f}": final_val,
        "absolute_change": abs_change,
        "percent_change": pct_change,
        "direction_at_report_k": direction,
        "min_over_budgets": min_val,
        "k_at_min": min_k,
        "max_over_budgets": max_val,
        "k_at_max": max_k
    })
    colors = ["#4C78A8", "#C76B6B"]   # blue / purple / red
    # 最终曲线
    # ax.plot(ks, data, color='#3498db', linewidth=2, label='Optimized')
    # ax.plot(
    #     ks, sparse_marker(data), color='#3498db',
    #     marker='o', markersize=6, markeredgewidth=1,
    #     markerfacecolor='w', linestyle='None', clip_on=False
    # )
    ax.plot(ks, data, color='#4C78A8', linewidth=2, label='Optimized')
    ax.plot(
        ks, sparse_marker(data), color='#4C78A8',
        marker='o', markersize=6, markeredgewidth=1,
        markerfacecolor='w', linestyle='None', clip_on=False
    )

    # 初始值
    # ax.axhline(
    #     y=initial_val, color='#C76B6B',
    #     linestyle='--', linewidth=2, label='Baseline'
    # )
    ax.axhline(
        y=initial_val, color='#C76B6B',
        linestyle='--', linewidth=2, label='Baseline'
    )

    # 报告点 k = 0.20
    ax.scatter(
        actual_report_k, final_val,
        s=70, color='#3498db',
        edgecolor='black', linewidth=0.8, zorder=5
    )

    # 在图里直接标出 initial -> optimized
    annotation = (
        f"Baseline: {initial_val:.3g}\n"
        f"$\gamma$={actual_report_k:.2f}: {final_val:.3g}\n"
        f"Δ: {abs_change:+.3g} ({pct_change:+.2f}%)"
        if not np.isnan(pct_change)
        else
        f"Baseline: {initial_val:.3g}\n"
        f"k={actual_report_k:.2f}: {final_val:.3g}\n"
        f"Δ: {abs_change:+.3g}"
    )

    ax.text(
        0.98, 0.45, annotation,
        transform=ax.transAxes,
        ha='right', va='top',
        fontsize=8,
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                  edgecolor='0.7', alpha=0.9)
    )

    ax.set_xlabel('$budget$')
    ax.set_title(title)
    ax.grid(True, linestyle='--', alpha=0.4)
    # 固定到右上角，不自动调整
    ax.legend(
    loc='center right',
    bbox_to_anchor=(0.98, 0.55),
    fontsize=8,
    frameon=True,      # 显示边框
    fancybox=True,     # 圆角边框
    shadow=False
    )
    ax.set_xlim(ks.min(), ks.max())

plt.tight_layout()
plt.savefig(
    'figure3d.pdf',
    format='pdf',
    dpi=300,
    bbox_inches='tight',
    transparent=False,
    backend='pdf'
)
plt.show()

summary_df = pd.DataFrame(summary_rows)

print("\n=== Figure 3d network metric changes ===")
print(summary_df.to_string(index=False))

