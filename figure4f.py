# -*- coding: utf-8 -*-
"""
Created on Sat Jun 27 15:27:26 2026

@author: JZS
"""
import os
import pickle
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# ---------------- 配置（按需修改路径/变量） ----------------
city = 'boston'
category = 'Other Individual and Family Services'
cat_dir = f'matrices_A_D_S_Distribution/{category.replace(" ", "_")}'

# 必要的数据文件（请确认这些文件存在）
flow_matrix = pd.read_csv(f'{cat_dir}/flow_matrix.csv', index_col=0)
no_regu_file = os.path.join(cat_dir, f'H_opt_df_no_regu_{city}_624190.pkl')
regu_file    = os.path.join(cat_dir, f'H_opt_df_regu_{city}_624190.pkl')
R_pre = pd.read_csv(f'{cat_dir}/pred_rownorm_int_preserve.csv', index_col=0)

cbg_income_dist_df = pd.read_csv(f'{cat_dir}/cbg_income_level_distribution_{city}_msa.csv', dtype={'GEOID': np.int64})
cbg_income_dist_dict = cbg_income_dist_df.set_index('GEOID').to_dict(orient='index')
income_levels = ['low_income_pct', 'lower_middle_income_pct', 'upper_middle_income_pct', 'high_income_pct']

cbsa = gpd.read_file('geo_data/tl_2021_us_cbsa/tl_2021_us_cbsa.shp').to_crs('EPSG:4326')
bos_msa = cbsa[cbsa['GEOID'] == '14460'].copy()  
boston_msa_cbg = gpd.read_file('geo_data/tl_2021_boston_msa_bg/tl_2021_boston_msa_bg.shp')
pad_len = int(boston_msa_cbg['GEOID'].str.len().max())



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
baseline = A_sub_full.copy()
baseline.index = [str(x).zfill(pad_len) for x in baseline.index]
baseline.columns = [str(x) for x in baseline.columns]  # POI ids as strings


# ---------------- 辅助函数 ----------------
def load_H(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Cannot find file: {path}. 请确认 H 的文件路径。")
    with open(path, 'rb') as f:
        H = pickle.load(f)
    if isinstance(H, pd.DataFrame):
        H = H.copy()
        H.index = [str(x).zfill(pad_len) for x in H.index]
        H.columns = [str(x) for x in H.columns]
        return H
    else:
        return pd.DataFrame(H, index=baseline.index, columns=baseline.columns)

def align_H_to_baseline(H_df, baseline_df):
    """
    将任意 H_df 对齐到 baseline（rows: zero-padded GEOID strings; cols: POI strings）。
    - 尝试直接 string match；若发现全部为 NaN，则尝试把 H.index 执行 zfill，再试一次。
    - 返回 fillna(0) 的对齐矩阵。
    """
    H = H_df.copy()
    # make strings
    H.index = H.index.map(lambda x: str(x))
    H.columns = H.columns.map(lambda x: str(x))
    # direct reindex
    H_try = H.reindex(index=baseline_df.index, columns=baseline_df.columns)
    if H_try.notna().values.any():
        return H_try.fillna(0.0)
    # try padded indices (for indices that are numeric as strings)
    H_index_padded = H.copy()
    try:
        H_index_padded.index = H_index_padded.index.map(lambda x: str(x).zfill(pad_len))
    except Exception:
        pass
    H_try2 = H_index_padded.reindex(index=baseline_df.index, columns=baseline_df.columns)
    return H_try2.fillna(0.0)
# ---------------- 加载并对齐三个矩阵 ----------------
H_no_pre = load_H(no_regu_file)
H_reg_pre = load_H(regu_file)
R_pre = R_pre.copy()
R_pre.index = R_pre.index.map(lambda x: str(x))
R_pre.columns = R_pre.columns.map(lambda x: str(x))

H_no = align_H_to_baseline(H_no_pre, baseline)
H_reg = align_H_to_baseline(H_reg_pre, baseline)
R = align_H_to_baseline(R_pre, baseline)

# ---------------- 计算 CBG 的 P matrix（n_cbgs x K） ----------------
# 从 cbg_income_dist_dict 构造 DataFrame，并对 baseline.index 做 reindex
def build_income_matrix(baseline_index, cbg_income_dist_dict, income_levels):
    rows = []
    idxs = []
    for g in baseline_index:
        # try integer key variants
        try:
            key = int(g)
        except:
            try:
                key = int(g.lstrip('0'))
            except:
                key = None
        if key is not None and key in cbg_income_dist_dict:
            row = cbg_income_dist_dict[key]
            vals = [row.get(l, np.nan) for l in income_levels]
            rows.append(vals)
            idxs.append(g)
        else:
            # if missing, fill zeros
            rows.append([0.0] * len(income_levels))
            idxs.append(g)
    P_df = pd.DataFrame(rows, index=idxs, columns=income_levels)
    # convert percents to fractions if values look like percents (>1)
    # safe handling: if values sum > 1.5 then assume percentages
    row_sums = P_df.sum(axis=1)
    if (row_sums > 1.5).any():
        P_df = P_df / 100.0
    # ensure rows sum to 1 (if not, normalize where row sum>0)
    rs = P_df.sum(axis=1)
    nonzero = rs > 0
    P_df.loc[nonzero, :] = P_df.loc[nonzero, :].div(rs[nonzero], axis=0)
    return P_df

P_df = build_income_matrix(baseline.index, cbg_income_dist_dict, income_levels)
P_mat = P_df.values  # n x K

#===========================================================================================
#----------------------辅助函数--------------------------------------------------
def shannon_entropy(p):
    p = np.array(p, dtype=float)
    if p.sum() == 0:
        return 0.0
    p = p / p.sum()
    p = p[p > 0]
    if p.size == 0:
        return 0.0
    return -(p * np.log2(p)).sum()

def gini_from_values(x):
    x = np.array(x, dtype=float).flatten()
    if x.size == 0:
        return np.nan
    if np.all(x == 0):
        return 0.0
    x = x[x >= 0]
    x_sorted = np.sort(x)
    n = x_sorted.size
    index = np.arange(1, n+1)
    return (2.0 * np.sum(index * x_sorted) / (n * np.sum(x_sorted))) - (n + 1) / n

def compute_high_income_visit_dist(H_df, income_share_series):
    H_local = H_df.reindex(index=income_share_series.index, columns=H_df.columns, fill_value=0)
    hi_visits = (H_local.multiply(income_share_series, axis=0)).sum(axis=0)
    total = hi_visits.sum()
    if total == 0:
        return hi_visits * 0.0
    return hi_visits / total

# -------------- high-income visit dist / entropy / gini (for baseline/no/reg) --------------
high_dist_A = compute_high_income_visit_dist(baseline, P_df['high_income_pct'])
high_dist_no = compute_high_income_visit_dist(H_no, P_df['high_income_pct'])
high_dist_reg = compute_high_income_visit_dist(H_reg, P_df['high_income_pct'])

entropy_A_high = shannon_entropy(high_dist_A.values)
entropy_no_high = shannon_entropy(high_dist_no.values)
entropy_reg_high = shannon_entropy(high_dist_reg.values)

gini_A_high = gini_from_values(high_dist_A.values)
gini_no_high = gini_from_values(high_dist_no.values)
gini_reg_high = gini_from_values(high_dist_reg.values)


upmi_dist_A = compute_high_income_visit_dist(baseline, P_df['upper_middle_income_pct'])
upmi_dist_no = compute_high_income_visit_dist(H_no, P_df['upper_middle_income_pct'])
upmi_dist_reg = compute_high_income_visit_dist(H_reg, P_df['upper_middle_income_pct'])

entropy_A_upmi = shannon_entropy(upmi_dist_A.values)
entropy_no_upmi = shannon_entropy(upmi_dist_no.values)
entropy_reg_upmi = shannon_entropy(upmi_dist_reg.values)

gini_A_upmi = gini_from_values(upmi_dist_A.values)
gini_no_upmi = gini_from_values(upmi_dist_no.values)
gini_reg_upmi = gini_from_values(upmi_dist_reg.values)


lomi_dist_A = compute_high_income_visit_dist(baseline, P_df['lower_middle_income_pct'])
lomi_dist_no = compute_high_income_visit_dist(H_no, P_df['lower_middle_income_pct'])
lomi_dist_reg = compute_high_income_visit_dist(H_reg, P_df['lower_middle_income_pct'])

entropy_A_lomi = shannon_entropy(lomi_dist_A.values)
entropy_no_lomi = shannon_entropy(lomi_dist_no.values)
entropy_reg_lomi = shannon_entropy(lomi_dist_reg.values)

gini_A_lomi = gini_from_values(lomi_dist_A.values)
gini_no_lomi = gini_from_values(lomi_dist_no.values)
gini_reg_lomi = gini_from_values(lomi_dist_reg.values)


low_dist_A = compute_high_income_visit_dist(baseline, P_df['low_income_pct'])
low_dist_no = compute_high_income_visit_dist(H_no, P_df['low_income_pct'])
low_dist_reg = compute_high_income_visit_dist(H_reg, P_df['low_income_pct'])

entropy_A_low = shannon_entropy(low_dist_A.values)
entropy_no_low = shannon_entropy(low_dist_no.values)
entropy_reg_low = shannon_entropy(low_dist_reg.values)

gini_A_low = gini_from_values(low_dist_A.values)
gini_no_low = gini_from_values(low_dist_no.values)
gini_reg_low = gini_from_values(low_dist_reg.values)

# 颜色
colors = ['#3498db', '#e74c3f', '#7c5bb8']  # 蓝 / 紫 / 红
labels = [
    'Observed baseline',
    'Behaviorally anchored',
    'Unregularized'
]

# 创建独立图
fig, ax = plt.subplots(figsize=(5, 5), dpi=300)

def plot_lorenz(ax, p, label, color):
    p_vals = np.array(p.fillna(0.0))
    if p_vals.sum() == 0:
        p_sorted = np.zeros_like(p_vals)
    else:
        p_sorted = np.sort(p_vals)[::-1]
    cum = np.cumsum(p_sorted)
    cum = cum / (cum[-1] if cum[-1] > 0 else 1)
    ax.plot(np.linspace(0, 1, len(p_sorted)), cum,
            label=label, linewidth=2, color=color)

# 画三条曲线
plot_lorenz(ax, high_dist_A.fillna(0), labels[0], colors[0])
plot_lorenz(ax, high_dist_reg.fillna(0), labels[1], colors[1])
plot_lorenz(ax, high_dist_no.fillna(0), labels[2], colors[2])

# 45° 参考线
ax.plot([0, 1], [0, 1], color='grey', linestyle='--', linewidth=1, zorder=0)

# 标注 Entropy & Gini
txt = (
    f"Entropy: {entropy_A_high:.3f} / {entropy_reg_high:.3f} / {entropy_no_high:.3f}\n"
    f"Gini: {gini_A_high:.3f} / {gini_reg_high:.3f} / {gini_no_high:.3f}\n"
    f"Baseline / anchored / unregularized"
)
# 右对齐：x=1 表示最右侧，ha='right' 控制文字右边缘对齐
ax.text(0.98, 0.02, txt, transform=ax.transAxes, fontsize=9,
        ha='right',  # 关键参数
        bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

# 轴标签与标题
ax.set_xlabel('Cumulative fraction of POIs')
ax.set_ylabel('Cumulative fraction of high-income visits')
ax.set_title('Lorenz-like curves for high-income visit distribution')
ax.legend(
    frameon=False,
    loc='upper left'
)
ax.grid(alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig('figure4f.pdf',
            format='pdf',
            dpi=300,
            bbox_inches='tight',
            transparent=False,
            backend='pdf')
plt.show()



#=========================================================================================================




from typing import Tuple, List, Optional



def plot_lorenz_curves_for_income(
    baseline_dist: pd.Series,
    reg_dist: pd.Series,
    no_reg_dist: pd.Series,
    entropy_values: Tuple[float, float, float],
    gini_values: Tuple[float, float, float],
    income_category_name: str,
    colors: Optional[List[str]] = None,
    labels: Optional[List[str]] = None
) -> Tuple[plt.Figure, plt.Axes]:

    # 默认颜色与标签
    if colors is None:
        colors = ['#3498db', '#e74c3f', '#7c5bb8']  # 蓝 / 红 / 紫
    if labels is None:
        labels = ['baseline', 'regu', 'no_regu']
    
    # 创建画布
    fig, ax = plt.subplots(figsize=(5, 5), dpi=300)
    
    # 辅助函数：绘制单条 Lorenz 曲线
    def plot_lorenz(ax, p, label, color):
        p_vals = np.array(p.fillna(0.0))
        if p_vals.sum() == 0:
            p_sorted = np.zeros_like(p_vals)
        else:
            p_sorted = np.sort(p_vals)[::-1]  # 降序排列
        cum = np.cumsum(p_sorted)
        cum = cum / (cum[-1] if cum[-1] > 0 else 1)
        ax.plot(np.linspace(0, 1, len(p_sorted)), cum,
                label=label, linewidth=2, color=color)
    
    # 绘制三条曲线
    plot_lorenz(ax, baseline_dist.fillna(0), labels[0], colors[0])
    plot_lorenz(ax, reg_dist.fillna(0), labels[1], colors[1])
    plot_lorenz(ax, no_reg_dist.fillna(0), labels[2], colors[2])
    
    # 45° 参考线
    ax.plot([0, 1], [0, 1], color='grey', linestyle='--', linewidth=1, zorder=0)
    
    # 标注 Entropy & Gini
    txt = (f"Entropy (base/reg/no): {entropy_values[0]:.3f}/{entropy_values[1]:.3f}/{entropy_values[2]:.3f}\n"
           f"Gini (base/reg/no): {gini_values[0]:.3f}/{gini_values[1]:.3f}/{gini_values[2]:.3f}")
    ax.text(0.98, 0.02, txt, transform=ax.transAxes, fontsize=9,
            ha='right', va='bottom',
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
    
    # 轴标签与标题
    ax.set_xlabel('Cumulative fraction of POIs')
    ax.set_ylabel(f'Cumulative fraction of {income_category_name} visits')
    ax.set_title(f'Lorenz-like curves for {income_category_name} visit distribution')
    
    # 图例与网格
    ax.legend(frameon=False)
    ax.grid(alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.show()
    
    return ax
    


plot_lorenz_curves_for_income(
    baseline_dist=high_dist_A,
    reg_dist=high_dist_reg,
    no_reg_dist=high_dist_no,
    entropy_values=(entropy_A_high, entropy_reg_high, entropy_no_high),
    gini_values=(gini_A_high, gini_reg_high, gini_no_high),
    income_category_name='high-income'
)
plot_lorenz_curves_for_income(
    baseline_dist=upmi_dist_A,
    reg_dist=upmi_dist_reg,
    no_reg_dist=upmi_dist_no,
    entropy_values=(entropy_A_upmi, entropy_reg_upmi, entropy_no_upmi),
    gini_values=(gini_A_upmi, gini_reg_upmi, gini_no_upmi),
    income_category_name='upper-middle-income'
)
plot_lorenz_curves_for_income(
    baseline_dist=lomi_dist_A,
    reg_dist=lomi_dist_reg,
    no_reg_dist=lomi_dist_no,
    entropy_values=(entropy_A_lomi, entropy_reg_lomi, entropy_no_lomi),
    gini_values=(gini_A_lomi, gini_reg_lomi, gini_no_lomi),
    income_category_name='lower-middle-income'
)
plot_lorenz_curves_for_income(
    baseline_dist=low_dist_A,
    reg_dist=low_dist_reg,
    no_reg_dist=low_dist_no,
    entropy_values=(entropy_A_low, entropy_reg_low, entropy_no_low),
    gini_values=(gini_A_low, gini_reg_low, gini_no_low),
    income_category_name='low-income'
)


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def plot_lorenz_on_axes(ax, baseline_dist, reg_dist, no_reg_dist,
                        entropy_values, gini_values, income_category_name,
                        colors=None, labels=None, show_legend=False):
    """
    Plot Lorenz-like curves on a given axes object for a single income category.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to plot on
    baseline_dist, reg_dist, no_reg_dist : pandas.Series
        Visit distributions for three scenarios
    entropy_values : tuple of 3 floats
        Entropy values for (baseline, regulation, no-regulation)
    gini_values : tuple of 3 floats
        Gini values for (baseline, regulation, no-regulation)
    income_category_name : str
        Name of income category for labels
    colors : list of 3 colors, optional
        Colors for the three scenarios
    labels : list of 3 str, optional
        Labels for the three scenarios
    show_legend : bool
        Whether to show legend on this axes
    """
    if colors is None:
        colors = ['#3498db', '#e74c3f', '#7c5bb8']
    if labels is None:
        labels = ['baseline', 'regu', 'no_regu']
    
    def plot_single_lorenz(p, label, color):
        p_vals = np.array(p.fillna(0.0))
        if p_vals.sum() == 0:
            p_sorted = np.zeros_like(p_vals)
        else:
            p_sorted = np.sort(p_vals)[::-1]  # 降序排列
        cum = np.cumsum(p_sorted)
        cum = cum / (cum[-1] if cum[-1] > 0 else 1)
        ax.plot(np.linspace(0, 1, len(p_sorted)), cum,
                label=label, linewidth=2, color=color)
    
    # 绘制三条洛伦兹曲线
    for dist, label, color in zip([baseline_dist, reg_dist, no_reg_dist], labels, colors):
        plot_single_lorenz(dist, label, color)
    
    # 45°参考线
    ax.plot([0, 1], [0, 1], color='grey', linestyle='--', linewidth=1, zorder=0)
    
    # 标注Entropy和Gini系数
    txt = (f"Entropy (base/reg/no): {entropy_values[0]:.3f}/{entropy_values[1]:.3f}/{entropy_values[2]:.3f}\n"
           f"Gini (base/reg/no): {gini_values[0]:.3f}/{gini_values[1]:.3f}/{gini_values[2]:.3f}")
    ax.text(0.98, 0.02, txt, transform=ax.transAxes, fontsize=8,
            ha='right', va='bottom',
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
    
    # 设置标签
    ax.set_xlabel('Cumulative fraction of POIs', fontsize=10)
    ax.set_ylabel(f'Cumulative fraction of {income_category_name} visits', fontsize=10)
    ax.set_title(f'{income_category_name}', fontsize=11, fontweight='bold')
    
    # 图例与网格
    if show_legend:
        ax.legend(frameon=False, loc='upper left')
    ax.grid(alpha=0.3, linestyle='--')
    
    return ax




def plot_all_lorenz_grid_2x2():
    """
    创建2x2组图，展示四个收入类别的洛伦兹曲线。
    假设所有数据变量已在全局环境中定义。
    """
    # 配置四个收入类别的数据
    income_configs = [
        {
            'name': 'high-income',
            'baseline': high_dist_A,
            'reg': high_dist_reg,
            'no_reg': high_dist_no,
            'entropy': (entropy_A_high, entropy_reg_high, entropy_no_high),
            'gini': (gini_A_high, gini_reg_high, gini_no_high)
        },
        {
            'name': 'upper-middle-income',
            'baseline': upmi_dist_A,
            'reg': upmi_dist_reg,
            'no_reg': upmi_dist_no,
            'entropy': (entropy_A_upmi, entropy_reg_upmi, entropy_no_upmi),
            'gini': (gini_A_upmi, gini_reg_upmi, gini_no_upmi)
        },
        {
            'name': 'lower-middle-income',
            'baseline': lomi_dist_A,
            'reg': lomi_dist_reg,
            'no_reg': lomi_dist_no,
            'entropy': (entropy_A_lomi, entropy_reg_lomi, entropy_no_lomi),
            'gini': (gini_A_lomi, gini_reg_lomi, gini_no_lomi)
        },
        {
            'name': 'low-income',
            'baseline': low_dist_A,
            'reg': low_dist_reg,
            'no_reg': low_dist_no,
            'entropy': (entropy_A_low, entropy_reg_low, entropy_no_low),
            'gini': (gini_A_low, gini_reg_low, gini_no_low)
        }
    ]
    
    # 创建2x2网格
    fig, axes = plt.subplots(2, 2, figsize=(10, 10), dpi=300)
    axes = axes.flatten()
    
    colors = ['#3498db', '#e74c3f', '#7c5bb8']
    labels = ['baseline', 'regu', 'no_regu']
    
    # 绘制每个子图
    for idx, config in enumerate(income_configs):
        # 只在第一个子图显示图例
        show_legend = (idx == 0)
        
        plot_lorenz_on_axes(
            ax=axes[idx],
            baseline_dist=config['baseline'],
            reg_dist=config['reg'],
            no_reg_dist=config['no_reg'],
            entropy_values=config['entropy'],
            gini_values=config['gini'],
            income_category_name=config['name'],
            colors=colors,
            labels=labels,
            show_legend=show_legend
        )
    
    # 添加全局图例
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=colors[i], label=labels[i]) for i in range(3)]
    fig.legend(handles=legend_elements, loc='upper center', 
               bbox_to_anchor=(0.5, 0.98), ncol=3, frameon=False)
    
    # 全局标题
    fig.suptitle('Lorenz-like curves for income group visit distributions', 
                 fontsize=14, y=0.995)
    plt.savefig('SI_figure4f.pdf',
                format='pdf',
                dpi=300,
                bbox_inches='tight',
                transparent=False,
                backend='pdf')
    
    plt.tight_layout()
    plt.show()


# 调用函数生成组图
plot_all_lorenz_grid_2x2()



#===========================================================================================
# Fig.4f: Reverse-ranked cumulative concentration of modeled adjustment burden
#
# Required objects already defined in the main script:
#     baseline : observed CBG--POI flow matrix F
#     H_no     : unregularized counterfactual flow matrix
#     H_reg    : L1-anchored counterfactual flow matrix
#     P_df     : CBG income-share dataframe containing 'high_income_pct'
#===========================================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#===========================================================================================
# Configuration
#===========================================================================================

SAVE_FIGURE = False
OUTPUT_FILE = "Fig4f_adjustment_burden_concentration.pdf"

# Blue-purple color scheme
unreg_color = "#3498db"
anchored_color = "#7c5bb8"

ROW_SUM_TOL = 1e-6
BURDEN_TOL = 1e-9


#===========================================================================================
# Helper functions
#===========================================================================================

def prepare_flow_matrix(matrix_df, reference_df, matrix_name):
    """
    Align a flow matrix to the baseline row and column structure
    and convert all entries to numeric values.
    """
    matrix = matrix_df.reindex(
        index=reference_df.index,
        columns=reference_df.columns
    ).fillna(0.0)

    matrix = matrix.apply(
        pd.to_numeric,
        errors="coerce"
    ).fillna(0.0)

    min_value = matrix.to_numpy(dtype=float).min()

    if min_value < -ROW_SUM_TOL:
        raise ValueError(
            f"{matrix_name} contains negative flow values. "
            f"Minimum value = {min_value:.6f}"
        )

    return matrix


def check_origin_flow_conservation(
    H_df,
    F_df,
    matrix_name,
    tol=1e-6
):
    """
    Verify origin-level flow conservation:

        sum_j H_ij = sum_j F_ij.

    This condition is required for

        0.5 * sum_j |H_ij - F_ij|

    to equal the number of reassigned visits from origin i.
    """
    row_sum_H = H_df.sum(axis=1)
    row_sum_F = F_df.sum(axis=1)

    row_error = (row_sum_H - row_sum_F).abs()
    max_error = row_error.max()

    if max_error > tol:
        problem_origin = row_error.idxmax()

        raise ValueError(
            f"{matrix_name} does not preserve origin-level visit totals.\n"
            f"Maximum row-sum error = {max_error:.6f}\n"
            f"Origin with maximum error = {problem_origin}\n"
            f"Baseline total = {row_sum_F.loc[problem_origin]:.6f}\n"
            f"Counterfactual total = {row_sum_H.loc[problem_origin]:.6f}"
        )

    return float(max_error)


def origin_adjustment_burden(H_df, F_df):
    """
    Modeled origin-level adjustment burden:

        A_i(H,F) = 0.5 * sum_j |H_ij - F_ij|.

    Under origin-level flow conservation, this quantity equals
    the number of visits reassigned from origin i.
    """
    H = H_df.reindex(
        index=F_df.index,
        columns=F_df.columns
    ).fillna(0.0)

    F = F_df.reindex(
        index=F_df.index,
        columns=F_df.columns
    ).fillna(0.0)

    H = H.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    F = F.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    burden = 0.5 * (H - F).abs().sum(axis=1)
    burden.name = "adjustment_burden"

    return burden


def gini_from_values(values):
    """
    Standard Gini coefficient for non-negative values.

    All origins are included, including origins with zero burden.
    """
    x = np.asarray(values, dtype=float).flatten()
    x = x[np.isfinite(x)]

    if x.size == 0:
        return np.nan

    if np.any(x < 0):
        raise ValueError("Gini input contains negative values.")

    if np.allclose(x, 0.0):
        return 0.0

    x_sorted = np.sort(x)
    n = x_sorted.size
    ranks = np.arange(1, n + 1, dtype=float)

    gini = (
        2.0 * np.sum(ranks * x_sorted)
        / (n * np.sum(x_sorted))
        - (n + 1.0) / n
    )

    return float(gini)


def reverse_ranked_concentration_curve(values):
    """
    Construct a reverse-ranked cumulative concentration curve.

    Origins are ranked from highest to lowest modeled adjustment burden.

    x:
        cumulative share of all origins.

    y:
        cumulative share of total reassigned visits.

    Because origins are ranked in descending order, a curve farther
    above the equal-burden line indicates greater concentration.
    """
    values_series = pd.Series(values).copy()

    values_series = pd.to_numeric(
        values_series,
        errors="coerce"
    ).fillna(0.0)

    values_series = values_series.clip(lower=0.0)
    values_sorted = values_series.sort_values(ascending=False)

    n_origins = len(values_sorted)
    total_burden = values_sorted.sum()

    if n_origins == 0:
        return (
            np.array([0.0]),
            np.array([0.0]),
            values_sorted
        )

    x = np.arange(0, n_origins + 1, dtype=float) / n_origins

    if total_burden <= 0:
        y = np.zeros(n_origins + 1, dtype=float)
    else:
        cumulative_burden = np.cumsum(
            values_sorted.to_numpy(dtype=float)
        )

        y = np.r_[
            0.0,
            cumulative_burden / total_burden
        ]

    return x, y, values_sorted


def top_burden_share(values, top_pct):
    """
    Percentage of total adjustment burden carried by the highest-burden
    top_pct percent of all origins.
    """
    values_series = pd.Series(values).copy()

    values_series = pd.to_numeric(
        values_series,
        errors="coerce"
    ).fillna(0.0)

    values_series = values_series.clip(lower=0.0)

    total_burden = values_series.sum()
    n_origins = len(values_series)

    if total_burden <= 0 or n_origins == 0:
        return np.nan

    n_top = int(np.ceil(n_origins * top_pct / 100.0))
    n_top = max(1, min(n_top, n_origins))

    top_total = (
        values_series
        .sort_values(ascending=False)
        .iloc[:n_top]
        .sum()
    )

    return float(top_total / total_burden * 100.0)


def changed_origin_share(values, tol=1e-9):
    """
    Percentage of all origins with positive modeled adjustment burden.
    """
    values_series = pd.Series(values).copy()

    values_series = pd.to_numeric(
        values_series,
        errors="coerce"
    ).fillna(0.0)

    if len(values_series) == 0:
        return np.nan

    return float((values_series > tol).mean() * 100.0)


def top_mean_high_income(
    burden_values,
    high_income_series,
    top_pct=10
):
    """
    Mean high-income share among the highest-burden origins.

    The income values may be stored either on a 0--1 scale
    or a 0--100 percentage scale.
    """
    burden = pd.Series(burden_values).copy()

    burden = pd.to_numeric(
        burden,
        errors="coerce"
    ).fillna(0.0)

    burden = burden.clip(lower=0.0)

    high_income = high_income_series.reindex(burden.index)

    high_income = pd.to_numeric(
        high_income,
        errors="coerce"
    )

    n_origins = len(burden)
    n_top = int(np.ceil(n_origins * top_pct / 100.0))
    n_top = max(1, min(n_top, n_origins))

    top_indices = (
        burden
        .sort_values(ascending=False)
        .iloc[:n_top]
        .index
    )

    selected_income = high_income.loc[top_indices].dropna()

    if selected_income.empty:
        return np.nan

    mean_value = selected_income.mean()

    # Convert shares on the 0--1 scale into percentages.
    if high_income.dropna().max() <= 1.5:
        mean_value *= 100.0

    return float(mean_value)


def discrete_top_fraction(n_origins, top_pct):
    """
    Obtain the actual horizontal-axis location associated with the
    discrete number of origins selected for a top-percentage statistic.
    """
    if n_origins <= 0:
        raise ValueError("The number of origins must be positive.")

    n_top = int(np.ceil(n_origins * top_pct / 100.0))
    n_top = max(1, min(n_top, n_origins))

    return n_top, n_top / n_origins


#===========================================================================================
# Prepare and validate matrices
#===========================================================================================

baseline_num = baseline.apply(
    pd.to_numeric,
    errors="coerce"
).fillna(0.0)

if baseline_num.empty:
    raise ValueError("The baseline flow matrix is empty.")

if (
    baseline_num.to_numpy(dtype=float) < -ROW_SUM_TOL
).any():
    raise ValueError(
        "The baseline flow matrix contains negative values."
    )

H_no_num = prepare_flow_matrix(
    H_no,
    baseline_num,
    matrix_name="Unregularized matrix H_no"
)

H_reg_num = prepare_flow_matrix(
    H_reg,
    baseline_num,
    matrix_name="L1-anchored matrix H_reg"
)

max_row_error_no = check_origin_flow_conservation(
    H_no_num,
    baseline_num,
    matrix_name="Unregularized matrix H_no",
    tol=ROW_SUM_TOL
)

max_row_error_reg = check_origin_flow_conservation(
    H_reg_num,
    baseline_num,
    matrix_name="L1-anchored matrix H_reg",
    tol=ROW_SUM_TOL
)


#===========================================================================================
# Calculate modeled origin-level adjustment burden
#===========================================================================================

burden_no = origin_adjustment_burden(
    H_no_num,
    baseline_num
)

burden_reg = origin_adjustment_burden(
    H_reg_num,
    baseline_num
)

total_flow = baseline_num.to_numpy(dtype=float).sum()

if total_flow <= 0:
    raise ValueError("Total baseline flow must be positive.")

total_burden_no = burden_no.sum()
total_burden_reg = burden_reg.sum()

# Share of baseline visits that must be reassigned
departure_no_pct = (
    total_burden_no / total_flow * 100.0
)

departure_reg_pct = (
    total_burden_reg / total_flow * 100.0
)

burden_reduction_pct = (
    (total_burden_no - total_burden_reg)
    / total_burden_no
    * 100.0
    if total_burden_no > 0
    else np.nan
)


#===========================================================================================
# Calculate concentration statistics
#===========================================================================================

top10_no = top_burden_share(
    burden_no,
    top_pct=10
)

top20_no = top_burden_share(
    burden_no,
    top_pct=20
)

top10_reg = top_burden_share(
    burden_reg,
    top_pct=10
)

top20_reg = top_burden_share(
    burden_reg,
    top_pct=20
)

gini_no = gini_from_values(
    burden_no.to_numpy(dtype=float)
)

gini_reg = gini_from_values(
    burden_reg.to_numpy(dtype=float)
)

changed_no = changed_origin_share(
    burden_no,
    tol=BURDEN_TOL
)

changed_reg = changed_origin_share(
    burden_reg,
    tol=BURDEN_TOL
)

if "high_income_pct" not in P_df.columns:
    raise KeyError(
        "P_df must contain the column 'high_income_pct'."
    )

top10_hi_no = top_mean_high_income(
    burden_no,
    P_df["high_income_pct"],
    top_pct=10
)

top10_hi_reg = top_mean_high_income(
    burden_reg,
    P_df["high_income_pct"],
    top_pct=10
)


#===========================================================================================
# Print diagnostics
#===========================================================================================

print(
    "\n========== Fig.4f modeled adjustment-burden concentration =========="
)

print(f"Number of origins: {len(baseline_num):d}")
print(f"Number of POIs: {len(baseline_num.columns):d}")
print(f"Total baseline flow: {total_flow:.6f}")

print("\n--- Origin-level flow conservation checks ---")

print(
    f"Maximum row-sum error, unregularized: "
    f"{max_row_error_no:.8f}"
)

print(
    f"Maximum row-sum error, L1-anchored:   "
    f"{max_row_error_reg:.8f}"
)

print("\n--- Aggregate modeled adjustment burden ---")

print(
    f"Unregularized adjustment burden: "
    f"{total_burden_no:.6f} reassigned visits "
    f"({departure_no_pct:.6f}% of baseline visits)"
)

print(
    f"L1-anchored adjustment burden: "
    f"{total_burden_reg:.6f} reassigned visits "
    f"({departure_reg_pct:.6f}% of baseline visits)"
)

print(
    f"Relative reduction in total modeled burden: "
    f"{burden_reduction_pct:.2f}%"
)

print(
    "\n--- Concentration across all origins, "
    "including zero-burden origins ---"
)

print(
    f"Unregularized: "
    f"top 10% carry {top10_no:.2f}%; "
    f"top 20% carry {top20_no:.2f}%; "
    f"Gini = {gini_no:.3f}; "
    f"changed origins = {changed_no:.2f}%"
)

print(
    f"L1-anchored:  "
    f"top 10% carry {top10_reg:.2f}%; "
    f"top 20% carry {top20_reg:.2f}%; "
    f"Gini = {gini_reg:.3f}; "
    f"changed origins = {changed_reg:.2f}%"
)

print("\n--- Income structure of highest-burden origins ---")

print(
    f"Mean high-income share among top 10% burden origins, "
    f"unregularized: {top10_hi_no:.2f}%"
)

print(
    f"Mean high-income share among top 10% burden origins, "
    f"L1-anchored:  {top10_hi_reg:.2f}%"
)


#===========================================================================================
# Construct reverse-ranked cumulative concentration curves
#===========================================================================================

x_no, y_no, burden_no_sorted = (
    reverse_ranked_concentration_curve(burden_no)
)

x_reg, y_reg, burden_reg_sorted = (
    reverse_ranked_concentration_curve(burden_reg)
)

n_origins = len(burden_no)

n20, x20 = discrete_top_fraction(
    n_origins,
    top_pct=20
)

# Curve arrays begin at (0,0), so y[n20] is the cumulative
# burden share carried by the highest-burden n20 origins.
y20_no = y_no[n20]
y20_reg = y_reg[n20]


#===========================================================================================
# Plot
#===========================================================================================

fig, ax = plt.subplots(
    figsize=(5.2, 5.0),
    dpi=300
)

# Unregularized curve
ax.plot(
    x_no,
    y_no,
    color=unreg_color,
    linewidth=2.5,
    solid_capstyle="round",
    label=(
        "Unregularized "
        f"({departure_no_pct:.1f}% of baseline visits)"
    ),
    zorder=3
)

# L1-anchored curve
ax.plot(
    x_reg,
    y_reg,
    color=anchored_color,
    linewidth=2.5,
    linestyle="--",
    dash_capstyle="butt",
    label=(
        fr"$L_1$-anchored "
        f"({departure_reg_pct:.1f}% of baseline visits)"
    ),
    zorder=3
)

# Equal-burden reference line
ax.plot(
    [0.0, 1.0],
    [0.0, 1.0],
    color="0.55",
    linestyle=":",
    linewidth=1.25,
    label="Equal burden distribution",
    zorder=1
)

# Exact discrete position corresponding to the top 20% of origins
ax.axvline(
    x=x20,
    color="0.80",
    linestyle="--",
    linewidth=0.85,
    zorder=0
)

# Mark the cumulative burden shares at the top-20% threshold
ax.scatter(
    [x20],
    [y20_no],
    s=26,
    color=unreg_color,
    edgecolors="white",
    linewidths=0.55,
    zorder=4
)

ax.scatter(
    [x20],
    [y20_reg],
    s=26,
    color=anchored_color,
    edgecolors="white",
    linewidths=0.55,
    zorder=4
)


# Statistical annotation:
# extensive margin + upper-tail concentration + overall inequality
annotation_text = (
    f"Adjusted origins\n"
    f"Unreg.: {changed_no:.1f}%   "
    f"Anchored: {changed_reg:.1f}%\n"
    f"Top 20% burden share\n"
    f"Unreg.: {top20_no:.1f}%   "
    f"Anchored: {top20_reg:.1f}%"
)

ax.text(
    0.98,
    0.04,
    annotation_text,
    transform=ax.transAxes,
    ha="right",
    va="bottom",
    fontsize=8.15,
    linespacing=1.18,
    bbox=dict(
        facecolor="white",
        alpha=0.90,
        edgecolor="none",
        pad=3.8
    ),
    zorder=5
)

# Axis limits
ax.set_xlim(0.0, 1.0)
ax.set_ylim(0.0, 1.02)

# Axis labels
ax.set_xlabel(
    "Cumulative share of CBGs\n"
    "ranked from highest to lowest burden",
    fontsize=10
)

ax.set_ylabel(
    "Cumulative share of total reassigned visits",
    fontsize=10
)

# No panel title: explanation should be placed in the full Fig.4 caption.

# Tick formatting
ax.tick_params(
    axis="both",
    which="major",
    labelsize=9
)

# Explicit tick locations
ax.set_xticks(
    np.linspace(0.0, 1.0, 6)
)

ax.set_yticks(
    np.linspace(0.0, 1.0, 6)
)

# Legend
ax.legend(
    frameon=False,
    loc="upper left",
    fontsize=8.25,
    handlelength=2.7,
    handletextpad=0.7,
    borderaxespad=0.65,
    labelspacing=0.55
)

# Light grid
ax.grid(
    alpha=0.20,
    linestyle="--",
    linewidth=0.6
)

# Light but visible borders
for spine in ax.spines.values():
    spine.set_linewidth(0.8)
    spine.set_color("0.70")

plt.tight_layout()


#===========================================================================================
# Save and show
#===========================================================================================

if SAVE_FIGURE:
    fig.savefig(
        OUTPUT_FILE,
        dpi=600,
        bbox_inches="tight"
    )

    print(f"\nFigure saved to: {OUTPUT_FILE}")

plt.show()