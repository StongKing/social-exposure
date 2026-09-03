# -*- coding: utf-8 -*-
"""
Figure 4e

Boston:
    high-income share (color)
    absolute flow-change (size)

Required public inputs
----------------------
    figure4e_cbg_data.csv

    geo_data/
        tl_2021_us_cbsa/
            tl_2021_us_cbsa.shp

        tl_2021_boston_msa_bg/
            tl_2021_boston_msa_bg.shp

Output
------
    figure4e.pdf

@author: JZS
"""

import os

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

from matplotlib.colors import LinearSegmentedColormap


# ============================================================
# Configuration
# ============================================================

data_path = (
    'figure4e_cbg_data.csv'
)


cbsa_path = (
    'geo_data/'
    'tl_2021_us_cbsa/'
    'tl_2021_us_cbsa.shp'
)


cbg_path = (
    'geo_data/'
    'tl_2021_boston_msa_bg/'
    'tl_2021_boston_msa_bg.shp'
)


# ============================================================
# Check files
# ============================================================

for path in [

    data_path,

    cbsa_path,

    cbg_path

]:

    if not os.path.exists(path):

        raise FileNotFoundError(
            f'Cannot find required file:\n{path}'
        )


# ============================================================
# Load public Fig.4e data
# ============================================================

per_cbg_df = pd.read_csv(

    data_path,

    dtype={
        'GEOID':
            str
    }
)


required_columns = [

    'GEOID',

    'fc_reg',

    'high_income_pct'
]


missing_columns = [

    col

    for col
    in required_columns

    if col
    not in per_cbg_df.columns
]


if len(
    missing_columns
) > 0:

    raise ValueError(
        'Missing required columns in '
        'figure4e_cbg_data.csv:\n'
        f'{missing_columns}'
    )


# ============================================================
# Load geographic data
# ============================================================

cbsa = (
    gpd.read_file(
        cbsa_path
    )
    .to_crs(
        'EPSG:4326'
    )
)


bos_msa = (
    cbsa[
        cbsa[
            'GEOID'
        ] == '14460'
    ]
    .copy()
)


boston_msa_cbg = gpd.read_file(
    cbg_path
)


boston_msa_cbg[
    'GEOID'
] = (
    boston_msa_cbg[
        'GEOID'
    ]
    .astype(str)
)


pad_len = int(
    boston_msa_cbg[
        'GEOID'
    ]
    .str.len()
    .max()
)


# ============================================================
# Standardize GEOID
# ============================================================

per_cbg_df[
    'GEOID'
] = (
    per_cbg_df[
        'GEOID'
    ]
    .astype(str)
    .str.zfill(
        pad_len
    )
)


per_cbg_df = (
    per_cbg_df
    .set_index(
        'GEOID'
    )
)


# ============================================================
# Prepare data
#
# Equivalent to the original:
#
# gdf = boston_msa_cbg.copy()
# gdf_all = boston_msa_cbg.copy()
# gdf['GEOID_z'] = ...
# gdf = gdf.set_index(...).reindex(baseline.index)
# gdf['fc_reg'] = ...
# gdf['high_income_pct'] = ...
# ============================================================

gdf = (
    boston_msa_cbg
    .copy()
)


gdf_all = (
    boston_msa_cbg
    .copy()
)


gdf[
    'GEOID_z'
] = (
    gdf[
        'GEOID'
    ]
    .astype(str)
    .str.zfill(
        pad_len
    )
)


gdf = (
    gdf
    .set_index(
        'GEOID_z'
    )
    .reindex(
        per_cbg_df.index
    )
)


gdf[
    'fc_reg'
] = (
    per_cbg_df[
        'fc_reg'
    ]
)


gdf[
    'high_income_pct'
] = (
    per_cbg_df[
        'high_income_pct'
    ]
)

gdf.index.name = 'GEOID_z'
gdf = gpd.GeoDataFrame(
    
    gdf.reset_index(),

    geometry='geometry'
)


# ============================================================
# Original Fig.4e plotting code
# ============================================================

# 2. 自定义浅色蓝紫配色
cmap_bp = LinearSegmentedColormap.from_list(
    'light_blue_purple',
    ['#3498db', '#7c5bb8']
)


# 3. 绘图
fig, ax = plt.subplots(
    figsize=(5, 5),
    dpi=300
)


# 绘制所有CBG多边形背景 (浅灰色主题)
gdf_all.plot(
    ax=ax,
    color='whitesmoke',
    edgecolor='#cccccc',
    linewidth=0.8,
    alpha=1.0,
    zorder=1
)


# 绘制MSA边界 (黑色轮廓)
bos_msa.boundary.plot(
    ax=ax,
    color='black',
    linewidth=1.0,
    zorder=3
)


# 筛选有变化的CBG并绘制散点
gdf_nonzero = (
    gdf[
        gdf[
            'fc_reg'
        ] > 0
    ]
    .copy()
)


if not gdf_nonzero.empty:

    # 改进size计算：平方根变换使大小差异更显著，范围20-280
    max_fc = (
        gdf_nonzero[
            'fc_reg'
        ]
        .max()
    )


    sizes = (
        (
            gdf_nonzero[
                'fc_reg'
            ]
            /
            max_fc
        )
        *
        300
        -
        20
    )


    # 绘制散点 (zorder=5确保在最上层)
    sc = ax.scatter(

        gdf_nonzero
        .geometry
        .centroid
        .x,

        gdf_nonzero
        .geometry
        .centroid
        .y,

        s=sizes,

        c=gdf_nonzero[
            'high_income_pct'
        ],

        cmap=cmap_bp,

        alpha=0.85,

        edgecolors='k',

        linewidths=0.3,

        zorder=5
    )


    # 颜色条 (简洁样式)
    cbar = plt.colorbar(

        sc,

        ax=ax,

        fraction=0.05,

        pad=0.02,

        shrink=0.8,

        aspect=25
    )


# 设置视图范围和比例 (参考第一段代码风格)
minx, miny, maxx, maxy = (
    gdf_all.total_bounds
)


dx = (
    maxx
    -
    minx
)


dy = (
    maxy
    -
    miny
)


pad_x = (
    dx
    *
    0.03
)


pad_y = (
    dy
    *
    0.03
)


ax.set_xlim(
    minx - pad_x,
    maxx + pad_x
)


ax.set_ylim(
    miny - pad_y,
    maxy + pad_y
)


ax.set_aspect(
    'equal'
)


# 移除坐标轴
ax.axis(
    'off'
)


# 标题
ax.set_title(
    'Boston: high-income share (color) & absolute flow-change (size)',
    fontsize=10,
    pad=6
)


plt.tight_layout()


plt.savefig(
    'figure4e.pdf',
    format='pdf',
    dpi=300,
    bbox_inches='tight',
    transparent=False,
    backend='pdf'
)


plt.show()