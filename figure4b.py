# -*- coding: utf-8 -*-
"""
Created on Sun Feb  1 09:51:35 2026

@author: JZS
"""
import os
import pickle
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt



# ---------------- 配置（按需修改路径） ----------------
city = 'boston'
category = 'Other Individual and Family Services'
cat_dir = f'matrices_A_D_S_Distribution/{category.replace(" ", "_")}'
outdir = 'k_flow_change_outputs'
os.makedirs(outdir, exist_ok=True)


# ---------------- 加载基础数据（与优化脚本相同） ----------------
flow_matrix = pd.read_csv(f'{cat_dir}/flow_matrix.csv', index_col=0)
social_exposure_matrix_js = pd.read_csv(f'{cat_dir}/social_exposure_matrix.csv', index_col=0)
cbg_income_dist_df = pd.read_csv(f'{cat_dir}/cbg_income_level_distribution_{city}_msa.csv', dtype={'GEOID': np.int64})
cbg_income_dist_dict = cbg_income_dist_df.set_index('GEOID').to_dict(orient='index')
R = pd.read_csv(f'{cat_dir}/pred_rownorm_int_preserve.csv', index_col=0)

# shapefile 只是为了 pad_len 保持一致
boston_msa_cbg = gpd.read_file('geo_data/tl_2021_boston_msa_bg/tl_2021_boston_msa_bg.shp')
pad_len = int(boston_msa_cbg['GEOID'].str.len().max())

income_levels = ['low_income_pct', 'lower_middle_income_pct', 'upper_middle_income_pct', 'high_income_pct']

# 复现 selected_pois & selected_cbgs（确保 baseline 与原脚本一致）
poi_total_flow = flow_matrix.sum(axis=0)
poi_num = flow_matrix.shape[1]
selected_pois = poi_total_flow.sort_values(ascending=False).head(poi_num).index.tolist()

selected_cbgs = set()
for poi in selected_pois:
    cbgs_with_flow = flow_matrix.index[flow_matrix[poi] > 0].tolist()
    selected_cbgs.update(cbgs_with_flow)
selected_cbgs = list(selected_cbgs)

A_sub_full = flow_matrix.loc[selected_cbgs, selected_pois]
# 把 baseline 的索引格式化为字符串（zfill）
baseline = A_sub_full.copy()
baseline.index = [str(x).zfill(pad_len) for x in baseline.index]

def load_H(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Cannot find file: {path}. 请确认你已在两个优化脚本中保存 H_opt_df，并把路径填对。")
    with open(path, 'rb') as f:
        H = pickle.load(f)
    # 如果保存成 DataFrame，直接返回，否则尝试转成 DataFrame
    if isinstance(H, pd.DataFrame):
        return H
    else:
        return pd.DataFrame(H, index=baseline.index, columns=baseline.columns)

# ----------- 稳健对齐 H 的函数（替换原来的 reindex 行） -----------
def align_H_to_baseline(H_df, baseline_df, pad_len):
    """
    把 H_df 对齐到 baseline_df 的行列（baseline.index 是 zero-padded 的 GEOID strings）。
    - 仅对 H_df.index 做 zfill(pad_len) 的尝试，不对列 zfill（POI id 不应该填充）。
    - 返回对齐好的 DataFrame（缺失处填 0），并打印调试信息。
    """
    H = H_df.copy()

    # 保证 index/columns 都为字符串（避免 int vs str 导致不匹配）
    H.index = H.index.map(lambda x: str(x))
    H.columns = H.columns.map(lambda x: str(x))

    H_try = H.reindex(index=baseline_df.index, columns=baseline_df.columns)
    if H_try.notna().values.any():
        # 有匹配到真实值，直接返回（缺失处会 later fillna(0)）
        H_aligned = H_try.fillna(0)
        print(f"align_H_to_baseline: direct string match found, nonzero sum = {H_aligned.values.sum():.4f}")
        return H_aligned



# 为图准备的 flow 数据（确保 index / columns 与 baseline 对齐）
R_aligned = align_H_to_baseline(R, baseline, pad_len)
baseline_df = baseline.copy()  # already zfilled

# 选择用于绘图的节点顺序（用 baseline 的总流量排序以保证一致性）
top_cbgs = baseline_df.sum(axis=1).sort_values(ascending=False).index.tolist()
top_pois = baseline_df.sum(axis=0).sort_values(ascending=False).index.tolist()
cbgs_all = [str(x) for x in top_cbgs]
pois_all = [str(x) for x in top_pois]

# 仅保留同时存在于 baseline 与 R 矩阵的节点（应该都存在，但稳健处理）
def get_present_lists(df_ref, df_other):
    idx = [str(x) for x in df_ref.index]
    cols = [str(x) for x in df_ref.columns]
    idx_present = [c for c in idx if (c in df_ref.index) and (c in df_other.index)]
    cols_present = [p for p in cols if (p in df_ref.columns) and (p in df_other.columns)]
    return idx_present, cols_present

# 因为我们对两表对齐，直接用 baseline 的索引/列即可
cbgs_present = [str(x) for x in baseline_df.index]
pois_present = [str(x) for x in baseline_df.columns]

# 位置布置（与之前的风格一致：CBG 在上，POI 在下）
n_cbgs = len(cbgs_present)
n_pois = len(pois_present)
x_cbgs = np.linspace(0.03, 0.97, n_cbgs) if n_cbgs > 0 else np.array([])
x_pois = np.linspace(0.03, 0.97, n_pois) if n_pois > 0 else np.array([])
y_cbgs = np.ones_like(x_cbgs) * 1.0
y_pois = np.zeros_like(x_pois) * 0.0

# 辅助：从 DataFrame 构建边列表 (i, j, value) —— 与你原实现保持一致（兼容单列/单行）
def build_edges_from_df(df, cbgs_present, pois_present):
    df_str = df.copy()
    df_str.index = df_str.index.map(lambda x: str(x))
    df_str.columns = df_str.columns.map(lambda x: str(x))
    edges = []
    for i, cbg in enumerate(cbgs_present):
        row = None
        try:
            row = df_str.loc[cbg, pois_present]
        except Exception:
            try:
                if len(pois_present) > 0:
                    row = df_str.loc[cbg, pois_present[0]]
            except Exception:
                row = None
        if row is None:
            continue
        row_vals = np.atleast_1d(np.array(row, dtype=float))
        for j, v in enumerate(row_vals):
            if np.isnan(v):
                continue
            if v > 0:
                edges.append((i, j, float(v)))
    return edges

# 线宽映射（直接复用）
def map_widths(edges, min_lw=0.1, max_lw=5.0):
    if len(edges) == 0:
        return np.array([])
    vals = np.array([e[2] for e in edges], dtype=float)
    vmin, vmax = vals.min(), vals.max()
    if vmax == vmin:
        return np.full_like(vals, (min_lw + max_lw) / 2.0, dtype=float)
    return min_lw + (vals - vmin) / (vmax - vmin) * (max_lw - min_lw)

# 贝塞尔画边函数（跟你原始实现保持一致）
def draw_edges_to_ax(ax, edges, widths, x_cbgs, y_cbgs, x_pois, y_pois, edge_alpha=0.7, color='#87CEFA'):
    if len(edges) == 0:
        ax.text(0.5, 0.5, 'No positive flows to plot', ha='center', va='center', fontsize=12, color=color)
        return
    vals = np.array([e[2] for e in edges], dtype=float)
    order = np.argsort(vals)  # 由细到粗绘制
    t = np.linspace(0.0, 1.0, 120)
    for idx in order:
        i, j, v = edges[idx]
        lw = float(widths[idx])
        # 若索引越界则跳过（稳健）
        if i >= len(x_cbgs) or j >= len(x_pois):
            continue
        x0, y0 = float(x_cbgs[i]), float(y_cbgs[i])
        x1, y1 = float(x_pois[j]), float(y_pois[j])
        xm = (x0 + x1) / 2.0
        dx = x1 - x0
        bend = 0.15 + 0.35 * abs(dx)
        ym = 0.35 + bend * 0.6
        P0 = np.array([x0, y0])
        P1 = np.array([xm, ym])
        P2 = np.array([x1, y1])
        curve = ((1 - t)**2)[:, None] * P0 + 2 * ((1 - t) * t)[:, None] * P1 + (t**2)[:, None] * P2
        ax.plot(curve[:, 0], curve[:, 1], linewidth=lw, color=color, alpha=edge_alpha,
                solid_capstyle='round', zorder=1)

# 替换：更短的 draw_edges_to_ax（曲线两端裁掉 t_margin）
def draw_edges_to_ax(ax, edges, widths, x_cbgs, y_cbgs, x_pois, y_pois, edge_alpha=0.7, color='#87CEFA'):
    if len(edges) == 0:
        ax.text(0.5, 0.5, 'No positive flows to plot', ha='center', va='center', fontsize=12, color=color)
        return
    vals = np.array([e[2] for e in edges], dtype=float)
    order = np.argsort(vals)  # 由细到粗绘制

    # 控制端点留白的比例（0.0-0.45，越大端点离节点越远）
    t_margin = 0.01
    t_margin = min(max(0.0, t_margin), 0.1)
    t = np.linspace(t_margin, 1.0 - t_margin, 120)

    for idx in order:
        i, j, v = edges[idx]
        lw = float(widths[idx])
        # 若索引越界则跳过（稳健）
        if i >= len(x_cbgs) or j >= len(x_pois):
            continue
        x0, y0 = float(x_cbgs[i]), float(y_cbgs[i])
        x1, y1 = float(x_pois[j]), float(y_pois[j])

        # 中间控制点（保持你原来的弯曲策略）
        xm = (x0 + x1) / 2.0
        dx = x1 - x0
        bend = 0.15 + 0.35 * abs(dx)
        ym = 0.35 + bend * 0.6

        P0 = np.array([x0, y0])
        P1 = np.array([xm, ym])
        P2 = np.array([x1, y1])

        # 计算裁切后的二次贝塞尔曲线（t 从 t_margin 到 1-t_margin）
        curve = ((1 - t)**2)[:, None] * P0 + 2 * ((1 - t) * t)[:, None] * P1 + (t**2)[:, None] * P2

        # 绘制，保证边在节点下面（nodes zorder=5），边 zorder < 5
        ax.plot(curve[:, 0], curve[:, 1], linewidth=lw, color=color, alpha=edge_alpha,
                solid_capstyle='round', zorder=2)


# 控制绘制密度
max_edges = 40000
def cap_edges(edges):
    if len(edges) == 0:
        return edges
    edges = sorted(edges, key=lambda x: x[2], reverse=True)
    return edges[:min(len(edges), max_edges)]

# 绘图：二子图并排

edges_baseline = build_edges_from_df(baseline_df, cbgs_present, pois_present)
edges_R = build_edges_from_df(R_aligned, cbgs_present, pois_present)


edges_baseline = cap_edges(edges_baseline)
edges_R = cap_edges(edges_R)
# 计算宽度
widths_baseline = map_widths(edges_baseline)
widths_R = map_widths(edges_R)



fig, axes = plt.subplots(1, 2, figsize=(10, 5), dpi=300)
ax0, ax1 = axes
COLOR = '#3498db'  # 保持你原来的浅蓝配色
# edgecolors = ["#4C78A8", "#8E5EA2", "#C76B6B"]
# Baseline
ax0.set_title('Observed Baseline Flow F (CBG → POI)', fontsize=14, pad=1)
ax0.axis('off')
draw_edges_to_ax(ax0, edges_baseline, widths_baseline, x_cbgs, y_cbgs, x_pois, y_pois, edge_alpha=0.7, color=COLOR)
ax0.scatter(x_cbgs, y_cbgs, s=10, marker='o', color='#3498db', alpha=0.95, zorder=5, edgecolors='none')
# ax0.scatter(x_pois, y_pois, s=20, marker='o', color='#e74c3f', alpha=0.95, zorder=5, edgecolors='none')
ax0.scatter(x_pois, y_pois, s=20, marker='o', color='#7c5bb8', alpha=0.95, zorder=5, edgecolors='none')



# R
ax1.set_title('Behaviorally Informed Reference Flow R (CBG → POI)', fontsize=14, pad=1)
ax1.axis('off')
draw_edges_to_ax(ax1, edges_R, widths_R, x_cbgs, y_cbgs, x_pois, y_pois, edge_alpha=0.7, color=COLOR)
ax1.scatter(x_cbgs, y_cbgs, s=10, marker='o', color='#3498db',  alpha=0.95, zorder=5, edgecolors='none')
# ax1.scatter(x_pois, y_pois, s=20, marker='o', color='#e74c3f',  alpha=0.95, zorder=5, edgecolors='none')
ax1.scatter(x_pois, y_pois, s=20, marker='o', color='#7c5bb8',  alpha=0.95, zorder=5, edgecolors='none')

plt.tight_layout()
plt.savefig('figure4b.pdf',
            format='pdf',
            dpi=300,
            bbox_inches='tight',
            transparent=False,
            backend='pdf')
plt.show()



