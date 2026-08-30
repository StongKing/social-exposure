# -*- coding: utf-8 -*-
"""
Created on Fri Oct 24 11:18:53 2025

@author: 13670
"""

# -*- coding: utf-8 -*-
"""
Matrix-style split-triangle plot for multiple POIs × cities.
Created on ...
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from matplotlib import cm
from matplotlib.colors import TwoSlopeNorm
import matplotlib.ticker as mtick
import textwrap

# ------------------ 配置 ------------------
city_configs = {
    "newyork":   {"core_shp": "tl_2021_place_new_core_bg/tl_2021_new_core_bg.shp"},
    "losangeles":{"core_shp": "tl_2021_place_los_core_bg/tl_2021_los_core_bg.shp"},
    "chicago":   {"core_shp": "tl_2021_place_chi_core_bg/tl_2021_chi_core_bg.shp"},
    "houston":   {"core_shp": "tl_2021_place_hou_core_bg/tl_2021_hou_core_bg.shp"},
    "atlanta":   {"core_shp": "tl_2021_place_atl_core_bg/tl_2021_atl_core_bg.shp"},
    "seattle":   {"core_shp": "tl_2021_place_sea_core_bg/tl_2021_sea_core_bg.shp"},
    "boston":    {"core_shp": "tl_2021_place_bos_core_bg/tl_2021_bos_core_bg.shp"},
    "fresno":    {"core_shp": "tl_2021_place_fre_core_bg/tl_2021_fre_core_bg.shp"},
    "baltimore": {"core_shp": "tl_2021_place_bal_core_bg/tl_2021_bal_core_bg.shp"},
    "tulsa":     {"core_shp": "tl_2021_place_tul_core_bg/tl_2021_tul_core_bg.shp"},
    "tyler":     {"core_shp": "tl_2021_place_tyl_core_bg/tl_2021_tyl_core_bg.shp"},
    "champaign": {"core_shp": "tl_2021_place_cha_core_bg/tl_2021_cha_core_bg.shp"},
    "billings":  {"core_shp": "tl_2021_place_bil_core_bg/tl_2021_bil_core_bg.shp"},
    "sebring":   {"core_shp": "tl_2021_place_seb_core_bg/tl_2021_seb_core_bg.shp"},
    "cheyenne":  {"core_shp": "tl_2021_place_che_core_bg/tl_2021_che_core_bg.shp"},
}
pretty_names = {k: k.title() if k not in ("newyork","losangeles") else ("New York" if k=="newyork" else "Los Angeles") 
                for k in city_configs.keys()}

# POI 映射（5 类）
naics_map = {
    '624190':  'Other_Individual_and_Family_Services',
    '711310':  'Promoters_of_Performing_Arts,_Sports,_and_Similar_Events_with_Facilities',
    '813110':  'Religious_Organizations_catholic',
    '712110':  'Museums',
    '713940':  'Fitness_and_Recreational_Sports_Centers',
    '722410':  'Drinking_Places_(Alcoholic_Beverages)'
}
# poi_codes = list(naics_map.keys())
# poi_labels = [naics_map[c] for c in poi_codes]
# # 将下划线替换为空格
# poi_labels = [label.replace('_', ' ') for label in poi_labels]

poi_codes = list(naics_map.keys())
poi_labels = [naics_map[c] for c in poi_codes]

# 将下划线替换为空格
poi_labels = [label.replace('_', ' ') for label in poi_labels]

# 修改最终绘图显示名称
poi_labels = [
    "Religious Organizations (Catholic)"
    if label == "Religious Organizations catholic"
    else label
    for label in poi_labels
]


# 文件路径模式（与您原来一致）
base_pattern = "matrices_A_D_S_Distribution_{city}_core/{poi_folder}/results_{city}_{poi_naics}.csv"

# 待匹配的列名候选
col_keys = {
    'f_values': ['f_values_iter', 'f_values', 'f_value', 'f_valuesiter'],
    'distances': ['distances_iter', 'distances', 'distance', 'distancesiter'],
    'social': ['social_iter', 'social', 'socialiter']
}

cities = list(city_configs.keys())

# ------------------ 辅助函数 ------------------
def find_col(df_cols, candidates):
    cols = [c.strip().lower() for c in df_cols]
    for cand in candidates:
        cand_low = cand.strip().lower()
        for i, c in enumerate(cols):
            if cand_low == c or cand_low in c or c in cand_low:
                return list(df_cols)[i]
    return None

# ------------------ 读取数据到矩阵 ------------------
n_pois = len(poi_codes)
n_cities = len(cities)

social_matrix = np.full((n_pois, n_cities), np.nan, dtype=float)
dist_matrix   = np.full((n_pois, n_cities), np.nan, dtype=float)
missing = []

for pi, poi_code in enumerate(poi_codes):
    poi_folder = naics_map[poi_code]
    for ci, city in enumerate(cities):
        fp = base_pattern.format(city=city, poi_folder=poi_folder, poi_naics=poi_code)
        if not os.path.exists(fp):
            missing.append((city, poi_code, fp))
            continue
        try:
            df = pd.read_csv(fp)
        except Exception as e:
            missing.append((city, poi_code, fp))
            continue
        df.columns = df.columns.map(lambda x: str(x).strip())
        # social
        col_s = find_col(df.columns, col_keys['social'])
        if col_s is not None:
            vals = df[col_s].dropna().values
            if vals.size>0:
                s0, s1 = float(vals[0]), float(vals[-1])
                if s0 != 0:
                    social_matrix[pi,ci] = (s1 - s0) / s0
        # distances
        col_d = find_col(df.columns, col_keys['distances'])
        if col_d is not None:
            vals = df[col_d].dropna().values
            if vals.size>0:
                s0, s1 = float(vals[0]), float(vals[-1])
                if s0 != 0:
                    dist_matrix[pi,ci] = (s1 - s0) / s0

if missing:
    print(f"Warning: {len(missing)} files missing or unreadable (city, poi_code, path).")

# # ------------------ color normalization ------------------
# cmap_social = cm.get_cmap('Purples')   # 社交暴露（增大用紫色系）
# cmap_dist   = cm.get_cmap('Greens_r') # 距离变化（这里用绿系，您可以改为其他）


# # 获取原始颜色映射
# cmap_social = cm.get_cmap('Purples')  # 社交暴露（增大用紫色系）
# #cmap_dist = cm.get_cmap('Blues_r')   # 距离变化（这里用绿系，您可以改为其他）

# # 获取颜色映射的完整范围
# social_full_range = cmap_social(np.linspace(0, 1, cmap_social.N))
# dist_full_range = cmap_dist(np.linspace(0, 1, cmap_dist.N))

# # 获取颜色映射的前一半
# social_half_range = cmap_social(np.linspace(0, 0.5, cmap_social.N // 2))
# dist_half_range = cmap_dist(np.linspace(0.5, 1, cmap_dist.N // 2))

# # 创建新的颜色映射
# cmap_social_half = mcolors.LinearSegmentedColormap.from_list('social_half', social_half_range)
# cmap_dist_half = mcolors.LinearSegmentedColormap.from_list('dist_half', dist_half_range)

# cmap_social = cmap_social_half   # 社交暴露（增大用紫色系）
# cmap_dist   = cmap_dist_half # 距离变化（这里用绿系，您可以改为其他）
# #

# # 自动计算颜色范围
# vmin_social, vmax_social = np.nanmin(social_matrix), np.nanmax(social_matrix)
# vmin_dist, vmax_dist = np.nanmin(dist_matrix), np.nanmax(dist_matrix)

# # 创建归一化对象
# norm_social = mcolors.Normalize(vmin=vmin_social, vmax=vmax_social)
# norm_dist = mcolors.Normalize(vmin=vmin_dist, vmax=vmax_dist)

# ------------------ Fig.2b-consistent light-half color normalization ------------------
# Use light 0%--50% visual range only.
# This keeps Fig.2e consistent with Fig.2b and avoids overly dark colorbar tails.

color_static = "#4C78A8"   # Fig.2b blue
color_dynamic = "#8E5EA2"  # Fig.2b purple
neutral_light = "#F6F8FC"


def make_light_half_cmap(name, end_color, n=256):
    """
    Build a light-half sequential colormap.

    It mimics using only the first 50% of a sequential palette:
    light neutral -> pale color -> Fig.2b reference color.

    No extra dark tail is added.
    """
    return mcolors.LinearSegmentedColormap.from_list(
        name,
        [
            neutral_light,
            mcolors.to_rgba(end_color, alpha=0.18),
            mcolors.to_rgba(end_color, alpha=0.36),
            mcolors.to_rgba(end_color, alpha=0.58),
            end_color
        ],
        N=n
    )


# Social exposure: larger increase = stronger purple, but only up to Fig.2b purple.
cmap_social = make_light_half_cmap(
    name="fig2b_light_purple_half",
    end_color=color_dynamic
)

# Distance: larger distance reduction = stronger blue, but only up to Fig.2b blue.
# Because distance changes are usually negative, we normalize the magnitude of reduction.
cmap_dist = make_light_half_cmap(
    name="fig2b_light_blue_half",
    end_color=color_static
)


# ------------------ data transformation for color mapping ------------------
# Social exposure uses the original relative change.
social_plot_matrix = social_matrix.copy()

# Distance is converted to reduction magnitude:
#   distance change = -20%  -> plotted value = +20%
#   distance change =  +5%  -> plotted value = 0
# This makes darker blue mean larger distance saving.
dist_reduction_matrix = np.where(
    np.isfinite(dist_matrix),
    np.maximum(-dist_matrix, 0),
    np.nan
)


def safe_norm_from_data(values, start_at_zero=True):
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]

    if vals.size == 0:
        return mcolors.Normalize(vmin=0.0, vmax=1.0)

    vmax = float(np.nanmax(vals))

    if start_at_zero:
        vmin = 0.0
    else:
        vmin = float(np.nanmin(vals))

    if np.isclose(vmin, vmax):
        vmax = vmin + 1e-6

    return mcolors.Normalize(vmin=vmin, vmax=vmax)


# Colorbar only covers the useful data range.
# No unused diverging side is included.
norm_social = safe_norm_from_data(social_plot_matrix, start_at_zero=True)
norm_dist = safe_norm_from_data(dist_reduction_matrix, start_at_zero=True)


# ------------------ 绘图 ------------------
fig_w = 17
fig_h = 7 + n_pois * 0.19
fig = plt.figure(figsize=(fig_w, fig_h), dpi=300)
ax = fig.add_subplot(111)

ax.set_xlim(-0.5, n_cities - 0.5)
ax.set_ylim(-0.5, n_pois - 0.5)
ax.set_xticks(np.arange(n_cities))


# 使用 textwrap.wrap() 限制每行字符数（例如，每行最多 20 个字符）
poi_labels_wrapped = ['\n'.join(textwrap.wrap(label, width=20)) for label in poi_labels]

# 绘图时，使用换行后的标签
ax.set_yticks(np.arange(n_pois))
ax.set_yticklabels(poi_labels_wrapped[::-1], fontsize=12)  # reverse so第0行在顶部
ax.invert_yaxis()

# cell size
cell_size = 0.8
half = cell_size / 2.0

# 画每个小方块并按对角线分两三角
for i in range(n_pois):
    for j in range(n_cities):
        y = i
        x = j
        bl = (x - half, y + half)
        tl = (x - half, y - half)
        tr = (x + half, y - half)
        br = (x + half, y + half)
        soc_val = social_matrix[n_pois - 1 - i, j]  # 因为绘制时y反转，索引需翻转
        dis_val = dist_matrix[n_pois - 1 - i, j]

        # social triangle
        if not np.isnan(soc_val):
            color = cmap_social(norm_social(soc_val))
            tri_soc = plt.Polygon([tl, tr, br], closed=True, facecolor=color, edgecolor='k', linewidth=0.3)
            ax.add_patch(tri_soc)
            centroid_soc = np.mean(np.array([tl, tr, br]), axis=0)
            #ax.text(centroid_soc[0], centroid_soc[1], f"{soc_val * 100:+.0f}%", ha='center', va='center', fontsize=6, color='k')

        # dist triangle
        if not np.isnan(dis_val):
            dis_reduction_val = max(-dis_val, 0)
            color = cmap_dist(norm_dist(dis_reduction_val))
            tri_dis = plt.Polygon([tl, bl, br], closed=True, facecolor=color, edgecolor='k', linewidth=0.3)
            ax.add_patch(tri_dis)
            centroid_dis = np.mean(np.array([tl, bl, br]), axis=0)
            #ax.text(centroid_dis[0], centroid_dis[1], f"{dis_val * 100:+.0f}%", ha='center', va='center', fontsize=6, color='k')

        # 没数据时画小灰点在方形中心
        if np.isnan(soc_val) and np.isnan(dis_val):
            ax.plot(x, y, marker='o', markersize=4, color='lightgray', markeredgecolor='k', zorder=6)

# 外观调整
fig.suptitle(
    'POI × City — Social Exposure (upper triangle) & Distance (lower triangle) relative change',
    fontsize=14,
    y=1
)
ax.set_aspect('equal')
ax.set_xticks(np.arange(n_cities))
ax.set_xticklabels([pretty_names.get(c, c) for c in cities], rotation=0, ha='center',fontsize=12)

# 创建颜色条
sm_s = cm.ScalarMappable(norm=norm_social, cmap=cmap_social)
sm_s.set_array([])

sm_d = cm.ScalarMappable(norm=norm_dist, cmap=cmap_dist)
sm_d.set_array([])

cax1 = fig.add_axes([0.12, 0.92, 0.38, 0.02])  # [left, bottom, width, height]
cb1 = fig.colorbar(sm_s, cax=cax1, orientation='horizontal')
cb1.set_label('Social Exposure relative change (%)', fontsize=12, labelpad=3)
cb1.ax.xaxis.set_label_position('top')
cb1.ax.xaxis.set_ticks_position('bottom')
cb1.ax.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
cb1.ax.tick_params(axis='x', labelsize=12) 

cax2 = fig.add_axes([0.55, 0.92, 0.33, 0.02])
cb2 = fig.colorbar(sm_d, cax=cax2, orientation='horizontal')
cb2.set_label('Distance relative change (%)', fontsize=12)
cb2.ax.xaxis.set_label_position('top')
cb2.ax.xaxis.set_ticks_position('bottom')
cb2.ax.xaxis.set_major_formatter(
    mtick.FuncFormatter(lambda x, pos: "0%" if abs(x) < 1e-12 else f"{-100 * x:.0f}%")
)
# 调整刻度数字大小
cb2.ax.tick_params(axis='x', labelsize=12)  

plt.subplots_adjust(top=0.9, left=0.08, right=0.98, bottom=0.12)
plt.savefig('figure2e.pdf',
            format='pdf',        # 显式指定
            dpi=300,             # 矢量格式里 dpi 仅影响嵌入的预览位图
            bbox_inches='tight', # 去掉白边
            transparent=True)    # 可选：透明背景
plt.show()