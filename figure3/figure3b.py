import os
import pickle
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec

# --- load data (assumes you already have these files in working dir) ---
file_path = 'k_matrices_boston_family_budget.pkl'
with open(file_path, 'rb') as f:
    k_matrices = pickle.load(f)

city = 'boston'
category = 'Other Individual and Family Services'
cat_dir = f'matrices_A_D_S_Distribution/{category.replace(" ", "_")}'

flow_matrix = pd.read_csv(f'{cat_dir}/flow_matrix.csv', index_col=0)
social_exposure_matrix_js = pd.read_csv(f'{cat_dir}/social_exposure_matrix.csv', index_col=0)

boston_msa_cbg = gpd.read_file('geo_data/tl_2021_boston_msa_bg/tl_2021_boston_msa_bg.shp')
boston_msa_cbg['GEOID'] = boston_msa_cbg['GEOID'].astype(str)
pad_len = int(boston_msa_cbg['GEOID'].str.len().max())

# select POIs/CBGs as in your script
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

# ---- parameters ----
ks_to_plot = [0.01, 0.05, 0.20]   # middle and right
outdir = 'k_flow_change_outputs'
os.makedirs(outdir, exist_ok=True)
dpi = 300
figsize = (10, 10)
cmap_name = 'Blues'

# ------------------------------------------------------------
# Zoom circle position and size
# ------------------------------------------------------------

USE_MANUAL_ZOOM_CENTER = True

# 如果 boston_msa_cbg 是经纬度坐标，这里填 lon/lat；
# Boston downtown 附近：
ZOOM_CENTER_X = -71.0000
ZOOM_CENTER_Y = 42.6521

# 如果你的 shp 仍是经纬度坐标，这里的 r 是“度”，不是 km。
# 0.08–0.15 通常可作为 Boston 局部放大半径试验。
ZOOM_RADIUS = 0.15

# 如果不手动设置，则继续使用 MSA 中心和 zoom_frac
zoom_frac = 0.30  # fraction of MSA bounding box for zoom window (0.3 = 30%)


pad_len = pad_len if 'pad_len' in globals() else None

# helper: find nearest key in k_matrices if exact not present
def find_best_key(k_val, k_matrices):
    best, bestd = None, 1e9
    for kk in k_matrices.keys():
        try:
            d = abs(float(kk) - float(k_val))
            if d < bestd:
                bestd, best = d, kk
        except:
            pass
    return best

# compute per-CBG flow_change for the ks_to_plot
flow_change = {}
for k in ks_to_plot:
    mat = k_matrices.get(k)
    if mat is None:
        bk = find_best_key(k, k_matrices)
        mat = k_matrices.get(bk) if bk is not None else None
    if mat is None:
        raise RuntimeError(f"未找到 k={k} 的矩阵")
    H = mat.copy()
    try:
        if pad_len is not None:
            H.index = [str(x).zfill(pad_len) for x in H.index]
    except:
        H.index = H.index.astype(str)
    H_aligned = H.reindex(index=baseline.index, columns=baseline.columns, fill_value=0.0)
    A_aligned = baseline.reindex(index=baseline.index, columns=baseline.columns, fill_value=0.0)
    fc = 0.5 * (H_aligned - A_aligned).abs().sum(axis=1)
    flow_change[k] = fc

# prepare geo and add flow_change columns
geo = boston_msa_cbg.copy()
for k, series in flow_change.items():
    geo[f'fc_k_{k:.2f}'] = series.reindex(geo['GEOID']).fillna(0.0).values

# also add a zero baseline column to show as left map (all zeros)
geo['zero'] = 0.0

# unified vmin/vmax for the two zoom maps
all_vals = np.concatenate([geo[f'fc_k_{k:.2f}'].values for k in ks_to_plot])
vmax = np.nanmax(all_vals)
if not np.isfinite(vmax) or vmax <= 0:
    vmax = 1.0

# colormap (trim darker part for readability)
cmap = plt.get_cmap(cmap_name)
new_cmap = mcolors.LinearSegmentedColormap.from_list('custom', cmap(np.linspace(0.20, 1, 256)))

# determine center and zoom extents
minx, miny, maxx, maxy = boston_msa_cbg.total_bounds

if USE_MANUAL_ZOOM_CENTER:
    cx = ZOOM_CENTER_X
    cy = ZOOM_CENTER_Y
    r = ZOOM_RADIUS
else:
    cx = (minx + maxx) / 2.0
    cy = (miny + maxy) / 2.0
    width = (maxx - minx) * zoom_frac
    height = (maxy - miny) * zoom_frac
    r = max(width, height) / 2.0

# zoom extents：圆外接正方形范围
x0, x1 = cx - r, cx + r
y0, y1 = cy - r, cy + r

# ---------- plotting ----------
# 用 gridspec_kw 控制每列宽度比例，并减少子图间水平间距
widths = [1,1]   
wspace = 0.2              # 子图间水平空白（越小紧凑）


fig = plt.figure(figsize=(10, 10), dpi=dpi)
# 2x2 网格，但调整行列比例让左上角更大
gs = gridspec.GridSpec(2, 2, 
                       width_ratios=[1, 1],   # 第一列更宽
                       height_ratios=[1, 1],  # 第一行更高
                       wspace=0,              # 水平间距
                       hspace=0)              # 垂直间距


edgecolors = ["#4C78A8", "#8E5EA2", "#C76B6B"]

# 创建子图
# 左上角大图
ax0 = fig.add_subplot(gs[0, 0])

# 右上角小图（嵌套1x1网格）
inner1 = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=gs[0, 1])
ax1 = fig.add_subplot(inner1[0, 0])

# 左下角小图（嵌套1x1网格）
inner2 = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=gs[1, 0])
ax2 = fig.add_subplot(inner2[0, 0])

# 右下角小图（嵌套1x1网格）
inner3 = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=gs[1, 1])
ax3 = fig.add_subplot(inner3[0, 0])

# 核心：手动缩小后三个子图（缩放因子设为0.8）
def shrink_axes(axes_list, factor):
    """将 axes 列表中的子图在父单元格内居中缩小"""
    for ax in axes_list:
        pos = ax.get_position()  # 获取当前位置 (左, 底, 宽, 高)
        new_width = pos.width * factor
        new_height = pos.height * factor
        new_left = pos.x0 + (pos.width - new_width) / 2
        new_bottom = pos.y0 + (pos.height - new_height) / 2
        ax.set_position([new_left, new_bottom, new_width, new_height])

shrink_axes([ax1, ax2, ax3], factor=0.75)

# # LEFT: reference map with zero values and boundary, and draw circle to show zoom area
# #ax0 = axes[0, 0]
# # plot zero column (gives uniform color)
# geo.plot(column='zero', ax=ax0, cmap=new_cmap, vmin=0, vmax=vmax, linewidth=0.2, edgecolor='0.6')
# # draw thin CBG boundaries on top for clarity
# geo.boundary.plot(ax=ax0, linewidth=0.25, edgecolor='0.6')
# # draw MSA outer boundary thicker
# msa_boundary = boston_msa_cbg.dissolve()
# msa_boundary.boundary.plot(ax=ax0, edgecolor='black', linewidth=0.7)
# # draw circle showing zoom area (data coords)
# circle0 = mpatches.Circle((cx, cy), r, edgecolor='black', facecolor='none', linewidth=1.2, linestyle='--', zorder=4)
# ax0.add_patch(circle0)
# ax0.set_title('Boston MSA reference Zoom area outlined', pad=8)
# ax0.set_xlim(minx, maxx)
# ax0.set_ylim(miny, maxy)
# ax0.set_aspect('equal', adjustable='box')
# ax0.axis('off')

# ------------------------------------------------------------
# FIRST PANEL: full Boston MSA under k = 0.01
# ------------------------------------------------------------
k0 = 0.01
colname0 = f'fc_k_{k0:.2f}'

# plot actual reassigned visits per CBG
geo.plot(
    column=colname0,
    ax=ax0,
    cmap=new_cmap,
    vmin=0,
    vmax=vmax,
    linewidth=0.1,
    edgecolor='0.6',
    missing_kwds={'color': 'lightgrey'}
)

# draw thin CBG boundaries
geo.boundary.plot(
    ax=ax0,
    linewidth=0.1,
    edgecolor='0.6'
)

# draw MSA outer boundary
msa_boundary = boston_msa_cbg.dissolve()
msa_boundary.boundary.plot(
    ax=ax0,
    edgecolor='black',
    linewidth=0.7
)

# indicate the zoom area
circle0 = mpatches.Circle(
    (cx, cy),
    r,
    edgecolor=edgecolors[0],   # 与 k=0.01 局部图保持一致
    facecolor='none',
    linewidth=1.5,
    linestyle='--',
    zorder=4
)
ax0.add_patch(circle0)

ax0.set_title('Boston MSA reallocation budget = 1%\n(zoom area outlined)', pad=8)

ax0.set_xlim(minx, maxx)
ax0.set_ylim(miny, maxy)
ax0.set_aspect('equal', adjustable='box')
ax0.axis('off')

# edgecolors = ['#3498db','#7c5bb8','#e74c3c']

edgecolors = ["#4C78A8", "#8E5EA2", "#C76B6B"]   # blue / purple / red
plot_axes = [ax1, ax2, ax3]

# MIDDLE and RIGHT: zoomed maps for each k clipped to a circular patch centered at (cx,cy)
for i, k in enumerate(ks_to_plot):
    ax = plot_axes[i]
    colname = f'fc_k_{k:.2f}'

    # plot choropleth (full), capture returned collections
    # We plot without axes turning off yet (we will clip)
    g = geo.plot(column=colname, ax=ax, cmap=new_cmap, vmin=0, vmax=vmax,
                 linewidth=0.2, edgecolor='0.6', missing_kwds={'color': 'lightgrey'})

    # draw the MSA boundary on this axis (for context)
    msa_boundary.boundary.plot(ax=ax, edgecolor='black', linewidth=0.4)

    # create circle patch in data coordinates
    circ = mpatches.Circle((cx, cy), r, transform=ax.transData)

    # Attempt to clip the last drawn PolyCollection to the circle.
    # geopandas adds collections to ax.collections; the latest Polygon collection is usually at the end.
    try:
        # take all patch-like collections and set their clip path
        for coll in ax.collections:
            try:
                coll.set_clip_path(circ)
            except Exception:
                pass
    except Exception:
        # fallback: do nothing (map still shown without circular mask)
        pass

    # draw circular outline on top to show crop boundary (in axes coordinates, but using data coords)
    circ_outline = mpatches.Circle((cx, cy), r, edgecolor=edgecolors[i], facecolor='none', linewidth=2, linestyle='--', zorder=5, transform=ax.transData)
    ax.add_patch(circ_outline)

    # set zoom extents to the square containing the circle so circle is centered in subplot
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect('equal', adjustable='box')
    ax.set_title(f'Reallocation budget = {k:.0%}', pad=8)
    ax.axis('off')

# Add a single colorbar on the right
sm = plt.cm.ScalarMappable(cmap=new_cmap, norm=plt.Normalize(vmin=0, vmax=vmax))
sm._A = []
# place colorbar to the right of figure
cax = fig.add_axes([0.9, 0.2, 0.02, 0.6])
cb = plt.colorbar(sm, cax=cax)
cb.set_label(r'Reassigned visits per CBG, $\frac{1}{2}\sum_j |H_{ij}(b)-F_{ij}|$')
plt.savefig('figure3b.pdf',
            format='pdf',
            dpi=300,
            bbox_inches='tight',
            transparent=False,
            backend='pdf')
#fig.suptitle('Center zoom comparison  k=0.01 vs k=0.20', fontsize=14)
plt.show()
