# -*- coding: utf-8 -*-
"""
Created on Sat Jan 31 21:20:43 2026

@author: JZS
"""

import os
import pickle
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests

# ---------------- 配置（按需修改路径） ----------------
city = 'boston'
category = 'Other Individual and Family Services'
cat_dir = f'matrices_A_D_S_Distribution/{category.replace(" ", "_")}'
outdir = 'k_flow_change_outputs'
os.makedirs(outdir, exist_ok=True)

# 这两个文件名要对应前两步脚本保存时的文件名
no_regu_file = os.path.join(cat_dir, f'H_opt_df_no_regu_{city}_624190.pkl')
regu_file    = os.path.join(cat_dir, f'H_opt_df_regu_{city}_624190.pkl')

# ---------------- 加载基础数据（与优化脚本相同） ----------------
flow_matrix = pd.read_csv(f'{cat_dir}/flow_matrix.csv', index_col=0)
social_exposure_matrix_js = pd.read_csv(f'{cat_dir}/social_exposure_matrix.csv', index_col=0)

cbg_income_dist_df = pd.read_csv(
    f'{cat_dir}/cbg_income_level_distribution_{city}_msa.csv',
    dtype={'GEOID': np.int64}
)
cbg_income_dist_dict = cbg_income_dist_df.set_index('GEOID').to_dict(orient='index')

# shapefile 只是为了 pad_len 保持一致
boston_msa_cbg = gpd.read_file('geo_data/tl_2021_boston_msa_bg/tl_2021_boston_msa_bg.shp')
pad_len = int(boston_msa_cbg['GEOID'].str.len().max())

income_levels = [
    'low_income_pct',
    'lower_middle_income_pct',
    'upper_middle_income_pct',
    'high_income_pct'
]

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


# ----------------- 辅助函数 -----------------
def load_H(path):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Cannot find file: {path}. 请确认你已在两个优化脚本中保存 H_opt_df，并把路径填对。"
        )

    with open(path, 'rb') as f:
        H = pickle.load(f)

    # 如果保存成 DataFrame，直接返回，否则尝试转成 DataFrame
    if isinstance(H, pd.DataFrame):
        return H
    else:
        return pd.DataFrame(H, index=baseline.index, columns=baseline.columns)


def compute_flow_change_and_groups(H_df, baseline_df):
    # 格式化 H 的行索引为 zfill pad_len
    H_local = H_df.copy()
    H_local.index = [str(x).zfill(pad_len) for x in H_local.index]

    # 对齐行列（以 baseline 为准）
    H_aligned = H_local.reindex(
        index=baseline_df.index,
        columns=baseline_df.columns,
        fill_value=0
    )
    A_aligned = baseline_df.reindex(
        index=baseline_df.index,
        columns=baseline_df.columns,
        fill_value=0
    )

    diff_abs = (H_aligned - A_aligned).abs()

    # 一个访问从 POI A 转移到 POI B，会产生两个绝对差值，所以乘 0.5
    flow_change = 0.5 * diff_abs.sum(axis=1)

    fc_selected = flow_change.reindex(baseline_df.index).fillna(0.0)

    changed_geo = fc_selected[fc_selected > 0].index.tolist()
    unchanged_geo = fc_selected[fc_selected == 0].index.tolist()

    return fc_selected, changed_geo, unchanged_geo


def extract_income_df_for_geoids(geoids, cbg_income_dist_dict, income_levels):
    rows = []
    ids = []

    for g in geoids:
        try:
            key = int(str(g))
        except:
            try:
                key = int(str(g).lstrip('0'))
            except:
                continue

        if key in cbg_income_dist_dict:
            row = cbg_income_dist_dict[key]
            vals = [row.get(l, np.nan) for l in income_levels]
            rows.append(vals)
            ids.append(str(g).zfill(pad_len))

    if len(rows) == 0:
        return pd.DataFrame(columns=income_levels)

    return pd.DataFrame(rows, index=ids, columns=income_levels)


# ---------------- 读取两个 H 矩阵 ----------------
H_no = load_H(no_regu_file)
H_reg = load_H(regu_file)

# 计算 flow_change 与 changed / unchanged
fc_no, changed_no, unchanged_no = compute_flow_change_and_groups(H_no, baseline)
fc_reg, changed_reg, unchanged_reg = compute_flow_change_and_groups(H_reg, baseline)

print("Counts (no regu): changed", len(changed_no), "unchanged", len(unchanged_no))
print("Counts (regu):    changed", len(changed_reg), "unchanged", len(unchanged_reg))

# 提取收入分布 DataFrames（并仅保留在 selected_cbgs 中有 income 数据的）
income_all = extract_income_df_for_geoids(
    baseline.index,
    cbg_income_dist_dict,
    income_levels
)

income_changed_no = extract_income_df_for_geoids(
    changed_no,
    cbg_income_dist_dict,
    income_levels
)

income_unchanged_no = extract_income_df_for_geoids(
    unchanged_no,
    cbg_income_dist_dict,
    income_levels
)

income_changed_reg = extract_income_df_for_geoids(
    changed_reg,
    cbg_income_dist_dict,
    income_levels
)

income_unchanged_reg = extract_income_df_for_geoids(
    unchanged_reg,
    cbg_income_dist_dict,
    income_levels
)

# ---------------- Statistical tests for Fig.4c ----------------

high_no_changed = income_changed_no['high_income_pct'].dropna()
high_no_unchanged = income_unchanged_no['high_income_pct'].dropna()

high_reg_changed = income_changed_reg['high_income_pct'].dropna()
high_reg_unchanged = income_unchanged_reg['high_income_pct'].dropna()


# Mann-Whitney U tests
u_no, p_no = mannwhitneyu(
    high_no_changed,
    high_no_unchanged,
    alternative='two-sided'
)

u_reg, p_reg = mannwhitneyu(
    high_reg_changed,
    high_reg_unchanged,
    alternative='two-sided'
)


# Holm correction
p_values = [p_no, p_reg]

_, p_corrected, _, _ = multipletests(
    p_values,
    method='holm'
)

p_no_adj = p_corrected[0]
p_reg_adj = p_corrected[1]


print("\n========== Fig4c statistical tests ==========")

print("No regularization")
print("U =", u_no)
print("adjusted p =", p_no_adj)

print("\nBehavioral anchoring")
print("U =", u_reg)
print("adjusted p =", p_reg_adj)

# ---------------- 绘图：两个子图并排（左：无正则；右：有正则） ----------------
fig, axes = plt.subplots(
    1, 2,
    figsize=(10, 5),
    dpi=300,
    constrained_layout=True
)

def blend_with_white(hex_color, alpha):
    """
    将原始颜色按 alpha 叠加到白色背景上，返回等效的不透明颜色。
    alpha 越大，颜色越接近原色；alpha 越小，颜色越浅。
    """
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)

    r_new = int(round(alpha * r + (1 - alpha) * 255))
    g_new = int(round(alpha * g + (1 - alpha) * 255))
    b_new = int(round(alpha * b + (1 - alpha) * 255))

    return f"#{r_new:02X}{g_new:02X}{b_new:02X}"

BASE_BLUE = "#3498DB"
BASE_PURPLE = "#7C5BB8"
BASE_RED = "#e74c3f"

BOX_ALPHA = 0.5   # 你主要调这里：0.25 更浅，0.45 更深，0.60 更接近原色

COLOR_UNCHANGED = blend_with_white(BASE_BLUE, BOX_ALPHA)
COLOR_CHANGED = blend_with_white(BASE_PURPLE, BOX_ALPHA)


def plot_changed_vs_unchanged(
    ax,
    inc_changed_df,
    inc_unchanged_df,
    title,
    n_changed,
    n_unchanged,
    p_value=None
):
    # 准备 boxplot 的数据：每个 income_level 两个箱（unchanged left, changed right）
    income_levels_local = income_levels
    box_data = []
    positions = []
    labels_pos = []

    offset = 0.18
    box_width = 0.30
    n_levels = len(income_levels_local)

    for i, lvl in enumerate(income_levels_local):
        arr_unch = (
            inc_unchanged_df[lvl].dropna().values
            if lvl in inc_unchanged_df.columns
            else np.array([])
        )

        arr_ch = (
            inc_changed_df[lvl].dropna().values
            if lvl in inc_changed_df.columns
            else np.array([])
        )

        if arr_unch.size == 0:
            arr_unch = np.array([np.nan])

        if arr_ch.size == 0:
            arr_ch = np.array([np.nan])

        box_data.append(arr_unch)
        positions.append(i - offset)

        box_data.append(arr_ch)
        positions.append(i + offset)

        labels_pos.append(i)

    # 画箱线
    bp = ax.boxplot(
        box_data,
        positions=positions,
        widths=box_width,
        patch_artist=True,
        showfliers=False
    )

    # 颜色
    colors = []
    # for _ in range(n_levels):
    #     colors.append('#a7c7e7')  # unchanged
    #     colors.append('#cfbaf0')  # changed
    for _ in range(n_levels):
        colors.append(COLOR_UNCHANGED)  # unchanged
        colors.append(COLOR_CHANGED)  # changed

    for patch, col in zip(bp['boxes'], colors):
        patch.set_facecolor(col)
        patch.set_edgecolor('black')
        patch.set_linewidth(0.8)

    for median in bp['medians']:
        median.set_color('black')
        median.set_linewidth(1.2)

    # 均值小方块
    means = [
        np.nanmean(arr) if np.isfinite(np.nanmean(arr)) else np.nan
        for arr in box_data
    ]

    for pos, m in zip(positions, means):
        if np.isfinite(m):
            ax.scatter(
                pos,
                m,
                marker='s',
                s=60,
                facecolors='white',
                edgecolors='black',
                zorder=10
            )

    ax.set_xticks(labels_pos)
    ax.set_xticklabels(['low', 'lower_mid', 'upper_mid', 'high'])
    ax.set_ylabel('Income share')
    ax.set_title(title)

    # 样本数放在 panel 内部左上角，不挤占 x 轴
    ax.text(
        0.04, 0.85,
        f"changed = {n_changed}\nunchanged = {n_unchanged}",
        transform=ax.transAxes,
        ha='left',
        va='top',
        fontsize=7.5
    )
    # significance annotation for high-income share
    if p_value is not None:

        if p_value < 0.001:
            p_text = r"$p<0.001$"
        else:
            p_text = rf"$p={p_value:.3f}$"
            # high-income is the fourth category
        x1 = 3 - offset
        x2 = 3 + offset
                
        y_max = max(
            inc_changed_df['high_income_pct'].max(),
            inc_unchanged_df['high_income_pct'].max()
            )
                
        y = y_max + 0.05
                
        ax.plot(
            [x1, x1, x2, x2],
            [y-0.01, y, y, y-0.01],
            lw=1,
            color='black'
            )
                
        ax.text(
            (x1+x2)/2,
            y+0.01,
            p_text,
            ha='center',
            va='bottom',
            fontsize=8
            )
                


# 左：无正则
plot_changed_vs_unchanged(
    axes[0],
    income_changed_no,
    income_unchanged_no,
    'No regularization',
    len(changed_no),
    len(unchanged_no),
    p_no_adj
)

# 右：有正则
plot_changed_vs_unchanged(
    axes[1],
    income_changed_reg,
    income_unchanged_reg,
    'With behavioural anchoring',
    len(changed_reg),
    len(unchanged_reg),
    p_reg_adj
)
# 总图标题
fig.suptitle(
    'Income composition of adjusted and unchanged origins',
    fontsize=14,
    y=1.04
)

# legend
p1 = mpatches.Patch(
    facecolor=COLOR_UNCHANGED,
    edgecolor='black',
    label='unchanged'
)

p2 = mpatches.Patch(
    facecolor=COLOR_CHANGED,
    edgecolor='black',
    label='changed'
)

# 为避免和左上角样本数重叠，把图例放到右上角
axes[0].legend(handles=[p1, p2], loc='upper left')
axes[1].legend(handles=[p1, p2], loc='upper left')

plt.savefig(
    'figure4c.pdf',
    format='pdf',
    dpi=300,
    bbox_inches='tight',
    transparent=False,
    backend='pdf'
)

plt.show()