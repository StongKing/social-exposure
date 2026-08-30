# -*- coding: utf-8 -*-
"""
Created on Sat Jan 31 21:54:54 2026

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

# ---------------- 重新计算 Social Exposure 矩阵 S 为每个 H（baseline, H_no, H_reg） ----------------
# 公式向量化说明：
# For H (n x m), compute V = H.sum(axis=0)  (length m)
# Q_mat (m x K) = (H.T @ P_mat) / V[:,None]   (handle V==0 -> Q zeros)
# S = 1 - P_mat @ Q_mat.T   (n x m), then for columns where V==0 set S[:,j]=0
def compute_S_from_H(H_df, P_mat):
    H = H_df.reindex(index=P_df.index, columns=baseline.columns).fillna(0.0)
    H_vals = H.values.astype(float)  # n x m
    V = H_vals.sum(axis=0)  # m
    # compute numerator: (H.T @ P_mat) -> m x K
    if H_vals.size == 0:
        m = 0
        return pd.DataFrame([], index=H.index, columns=H.columns)
    Q_num = H_vals.T @ P_mat  # m x K
    # handle V
    V_safe = V.copy()
    V_safe[V_safe == 0] = 1.0  # to avoid divide by zero
    Q_mat = Q_num / V_safe[:, None]  # m x K
    # For POIs with V==0, set Q row to zeros
    zero_cols = (V == 0)
    if zero_cols.any():
        Q_mat[zero_cols, :] = 0.0
    # compute S = 1 - P @ Q.T
    S_vals = 1.0 - (P_mat @ Q_mat.T)  # n x m
    # for zero columns set S to 0 (or choose other sentinel) — we choose 0 to indicate no exposure
    if zero_cols.any():
        S_vals[:, zero_cols] = 0.0
    S_df = pd.DataFrame(S_vals, index=H.index, columns=H.columns)
    return S_df

# compute S for the three matrices
S_baseline = compute_S_from_H(baseline, P_mat)
S_no = compute_S_from_H(H_no, P_mat)
S_reg = compute_S_from_H(H_reg, P_mat)
# (optionally) S_R if you want exposure for R: not required by you, so omitted.

# ---------------- 根据各自 S 计算 aggregate exposures 和 per-CBG exposure ----------------
def aggregate_exposure(H_df, S_df):
    H_al = H_df.reindex(index=S_df.index, columns=S_df.columns).fillna(0.0)
    S_al = S_df.reindex(index=H_al.index, columns=H_al.columns).fillna(0.0)
    return (H_al * S_al).sum().sum()

def per_cbg_exposure(H_df, S_df):
    H_al = H_df.reindex(index=S_df.index, columns=S_df.columns).fillna(0.0)
    S_al = S_df.reindex(index=H_al.index, columns=H_al.columns).fillna(0.0)
    numer = (H_al * S_al).sum(axis=1)
    denom = H_al.sum(axis=1).replace(0, np.nan)
    return (numer / denom).fillna(0.0)

# 创建二元掩码（非零元素变为1）
baseline_bin = (baseline != 0).astype(int)
H_no_bin = (H_no != 0).astype(int)
H_reg_bin = (H_reg != 0).astype(int)

# Hadamard点乘
result_baseline = S_baseline * baseline_bin
result_no = S_no * H_no_bin
result_reg = S_reg * H_reg_bin

print("result_baseline",result_baseline.sum().sum())
print("result_no",result_no.sum().sum())
print("result_reg",result_reg.sum().sum())

agg_exp_baseline = aggregate_exposure(baseline, result_baseline)
agg_exp_no = aggregate_exposure(H_no, result_no)
agg_exp_reg = aggregate_exposure(H_reg, result_reg)

exp_per_cbg_baseline = per_cbg_exposure(baseline, S_baseline)
exp_per_cbg_no = per_cbg_exposure(H_no, S_no)
exp_per_cbg_reg = per_cbg_exposure(H_reg, S_reg)

# ---------------- 重新构建 per_cbg_df（含新的 exposure） ----------------
dev_no = (H_no - R).abs().sum(axis=1)
dev_reg = (H_reg - R).abs().sum(axis=1)

fc_no = (H_no - baseline).abs().sum(axis=1)
fc_reg = (H_reg - baseline).abs().sum(axis=1)

per_cbg_df = pd.DataFrame({
    'exp_A': exp_per_cbg_baseline,
    'exp_no': exp_per_cbg_no,
    'exp_reg': exp_per_cbg_reg,
    'dev_no': dev_no,
    'dev_reg': dev_reg,
    'fc_no': fc_no,
    'fc_reg': fc_reg,
    'high_income_pct': P_df['high_income_pct']
}).reindex(baseline.index).fillna(0.0)
per_cbg_df['delta_exp_reg_vs_no'] = per_cbg_df['exp_reg'] - per_cbg_df['exp_no']



# 1. 准备数据 (保持不变)
gdf = boston_msa_cbg.copy()
gdf_all = boston_msa_cbg.copy()
gdf['GEOID_z'] = gdf['GEOID'].astype(str).str.zfill(pad_len)
gdf = gdf.set_index('GEOID_z').reindex(baseline.index)
gdf['fc_reg'] = per_cbg_df['fc_reg']
gdf['high_income_pct'] = per_cbg_df['high_income_pct']
gdf = gpd.GeoDataFrame(gdf.reset_index(), geometry='geometry')


# 2. 自定义浅色蓝紫配色
cmap_bp = LinearSegmentedColormap.from_list('light_blue_purple', 
                                            ['#3498db', '#7c5bb8'])  # 更浅的蓝色和紫色
# 3. 绘图
fig, ax = plt.subplots(figsize=(5, 5), dpi=300)

# 绘制所有CBG多边形背景 (浅灰色主题)
gdf_all.plot(ax=ax, color='whitesmoke', edgecolor='#cccccc', 
             linewidth=0.8, alpha=1.0, zorder=1)

# 绘制MSA边界 (黑色轮廓)
bos_msa.boundary.plot(ax=ax, color='black', linewidth=1.0, zorder=3)

# 筛选有变化的CBG并绘制散点
gdf_nonzero = gdf[gdf['fc_reg'] > 0].copy()
if not gdf_nonzero.empty:
    # 改进size计算：平方根变换使大小差异更显著，范围20-280
    max_fc = gdf_nonzero['fc_reg'].max()
    sizes = (gdf_nonzero['fc_reg'] / max_fc) * 300-20
    
    # 绘制散点 (zorder=5确保在最上层)
    sc = ax.scatter(gdf_nonzero.geometry.centroid.x, gdf_nonzero.geometry.centroid.y,
                    s=sizes, c=gdf_nonzero['high_income_pct'], cmap=cmap_bp, 
                    alpha=0.85, edgecolors='k', linewidths=0.3, zorder=5)
    
    # 颜色条 (简洁样式)
    cbar = plt.colorbar(sc, ax=ax, fraction=0.05, pad=0.02, shrink=0.8, aspect=25)

# 设置视图范围和比例 (参考第一段代码风格)
minx, miny, maxx, maxy = gdf_all.total_bounds
dx = maxx - minx
dy = maxy - miny
pad_x = dx * 0.03
pad_y = dy * 0.03
ax.set_xlim(minx - pad_x, maxx + pad_x)
ax.set_ylim(miny - pad_y, maxy + pad_y)
ax.set_aspect('equal')

# 移除坐标轴
ax.axis('off')
# 标题
ax.set_title('Boston: high-income share (color) & absolute flow-change (size)', 
             fontsize=10, pad=6)
plt.tight_layout()
plt.savefig('figure4e.pdf',
            format='pdf',
            dpi=300,
            bbox_inches='tight',
            transparent=False,
            backend='pdf')
plt.show()



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
labels = ['baseline', 'regu', 'no_regu']

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
txt = (f"Entropy (base/reg/no): {entropy_A_high:.3f}/{entropy_reg_high:.3f}/{entropy_no_high:.3f}\n"
       f"Gini (base/reg/no): {gini_A_high:.3f}/{gini_reg_high:.3f}/{gini_no_high:.3f}")
# 右对齐：x=1 表示最右侧，ha='right' 控制文字右边缘对齐
ax.text(0.98, 0.02, txt, transform=ax.transAxes, fontsize=9,
        ha='right',  # 关键参数
        bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

# 轴标签与标题
ax.set_xlabel('Cumulative fraction of POIs')
ax.set_ylabel('Cumulative fraction of high-income visits')
ax.set_title('Lorenz-like curves for high-income visit distribution')
ax.legend(frameon=False)
ax.grid(alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig('figure4f.pdf',
            format='pdf',
            dpi=300,
            bbox_inches='tight',
            transparent=False,
            backend='pdf')
plt.show()