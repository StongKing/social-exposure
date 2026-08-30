# -*- coding: utf-8 -*-
"""
Created on Wed Jan 28 21:35:04 2026

@author: JZS
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Show 2D fill_between non-overlap regions but put them on two separate planes:
- P (blue) on plane y=y1, highlight where p1>p2 by filling between p1 and p2 on that plane.
- Q (green) on plane y=y2, highlight where p2>p1 by filling between p2 and p1 on that plane.
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# ---------------- parameters ----------------
mu1, sigma1 = 3.0, 1.6
mu2, sigma2 = 3.5, 1.5
n_points = 2000

vline1 = None  # if None -> mu1
vline2 = None  # if None -> mu2

y1 = 1.0
y2 = 3.0

out_png = "Social Exposure Diagram 1.png"
# --------------------------------------------

def normal_pdf(x, mu, sigma):
    return (1.0 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

# x grid & densities
x_min = min(mu1 - 4*sigma1, mu2 - 4*sigma2)
x_max = max(mu1 + 4*sigma1, mu2 + 4*sigma2)
x = np.linspace(x_min, x_max, n_points)

p1 = normal_pdf(x, mu1, sigma1)
p2 = normal_pdf(x, mu2, sigma2)
p1 /= np.trapz(p1, x)
p2 /= np.trapz(p2, x)

# defaults for vlines
vline1 = 1.5
vline2 = 2.0

# find contiguous True segments of a boolean mask
def find_segments(mask):
    idx = np.where(mask)[0]
    if idx.size == 0:
        return []
    splits = np.where(np.diff(idx) != 1)[0]
    segs = []
    start = int(idx[0])
    for s in splits:
        end = int(idx[s])
        segs.append((start, end))
        start = int(idx[s+1])
    segs.append((start, int(idx[-1])))
    return segs

# masks: where p1>p2 (blue region), where p2>p1 (green region)
mask1 = p1 > p2
mask2 = p2 > p1

segs1 = find_segments(mask1)
segs2 = find_segments(mask2)

# build polygons for mask1 on plane y1: top = p1, bottom = p2 (so it matches fill_between(p1,p2))
polys_plane_y1 = []
for (s, e) in segs1:
    xs = x[s:e+1]
    top = p1[s:e+1]
    bot = p2[s:e+1]
    verts = [(float(xs[i]), float(y1), float(top[i])) for i in range(len(xs))]
    # go back along bottom boundary
    for i in range(len(xs)-1, -1, -1):
        verts.append((float(xs[i]), float(y1), float(bot[i])))
    polys_plane_y1.append(verts)

# build polygons for mask2 on plane y2: top = p2, bottom = p1
polys_plane_y2 = []
for (s, e) in segs2:
    xs = x[s:e+1]
    top = p2[s:e+1]
    bot = p1[s:e+1]
    verts = [(float(xs[i]), float(y2), float(top[i])) for i in range(len(xs))]
    for i in range(len(xs)-1, -1, -1):
        verts.append((float(xs[i]), float(y2), float(bot[i])))
    polys_plane_y2.append(verts)

# ---------------- plotting ----------------
from mpl_toolkits.mplot3d import Axes3D  # noqa

fig = plt.figure(figsize=(10,7), dpi=200)
ax = fig.add_subplot(111, projection='3d')

# # plot main curves on their planes
# ax.plot(x, np.full_like(x, y1), p1, color='#1f77b4', lw=2.2, label=f'P (y={y1})')
# ax.plot(x, np.full_like(x, y2), p2, color='#2ca02c', lw=2.2, label=f'Q (y={y2})')


# # add filled regions as polygons on respective planes
# if polys_plane_y1:
#     poly1 = Poly3DCollection(polys_plane_y1, facecolors='#87CEEB', edgecolors='none', alpha=0.85)
#     ax.add_collection3d(poly1)
# if polys_plane_y2:
#     poly2 = Poly3DCollection(polys_plane_y2, facecolors='#90EE90', edgecolors='none', alpha=0.85)
#     ax.add_collection3d(poly2)

# plot main curves on their planes
ax.plot(x, np.full_like(x, y1), p1, color='#4C78A8', lw=2.2, label=f'P (y={y1})')
ax.plot(x, np.full_like(x, y2), p2, color='#8E5EA2', lw=2.2, label=f'Q (y={y2})')


if polys_plane_y1:
    poly1 = Poly3DCollection(polys_plane_y1, facecolors='#9BB7D4', edgecolors='none', alpha=0.85)
    ax.add_collection3d(poly1)
if polys_plane_y2:
    poly2 = Poly3DCollection(polys_plane_y2, facecolors='#C4A5CF', edgecolors='none', alpha=0.85)
    ax.add_collection3d(poly2)



# labels & ticks
ax.set_xlabel('x', labelpad=8)
ax.set_ylabel('y (depth)', labelpad=8)
ax.set_zlabel('density (z)', labelpad=8)

ax.set_xticks(np.linspace(round(x_min,1), round(x_max,1), 6))
max_z = max(p1.max(), p2.max())
ax.set_zticks(np.linspace(0, max_z, 6))
ax.set_yticks([y1, y2])

# box aspect (attempt cube-like)
x_span = x_max - x_min
try:
    ax.set_box_aspect((x_span, x_span, x_span))
except Exception:
    pass

# camera - you can change elev/azim
ax.view_init(elev=18, azim=-70)

#ax.set_title('Separate-plane non-overlap: blue on y=1 (p1>p2), green on y=3 (p2>p1)', pad=14)
# ---------- 2. 三无：无标记、无坐标轴、无网格 ----------
ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
ax.axis('off')          # 一键隐藏坐标轴+刻度+标签
ax.grid(False)          # 关闭网格线
ax.set_xlim(-4, 13)                 # 视觉边界 1→3
ax.set_ylim(1, 3)                 # 视觉边界 1→3
ax.set_zlim(0, 0.5)                 # 视觉边界 1→3
plt.tight_layout(pad=0)
# plt.savefig(out_png, dpi=300, bbox_inches='tight')
# 替换掉你原来的 plt.show() 部分
out_pdf = "Social Exposure Diagram 1.pdf"
plt.savefig(out_pdf,
            format='pdf',
            dpi=300,
            bbox_inches='tight',
            pad_inches=0,
            transparent=True)
plt.show()   # 如不需要预览可注释
print(f"Saved to: {out_png}")




import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# ---------------- parameters ----------------
mu1, sigma1 = 3.0, 1.6
mu2, sigma2 = 6.0, 2.0
n_points = 2000

vline1 = None  # if None -> mu1
vline2 = None  # if None -> mu2

y1 = 1.0
y2 = 2

out_png = "Social Exposure Diagram 2.png"
# --------------------------------------------

def normal_pdf(x, mu, sigma):
    return (1.0 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

# x grid & densities
x_min = min(mu1 - 4*sigma1, mu2 - 4*sigma2)
x_max = max(mu1 + 4*sigma1, mu2 + 4*sigma2)
x = np.linspace(x_min, x_max, n_points)

p1 = normal_pdf(x, mu1, sigma1)
p2 = normal_pdf(x, mu2, sigma2)
p1 /= np.trapz(p1, x)
p2 /= np.trapz(p2, x)

# defaults for vlines
if vline1 is None:
    vline1 = mu1
if vline2 is None:
    vline2 = mu2

# find contiguous True segments of a boolean mask
def find_segments(mask):
    idx = np.where(mask)[0]
    if idx.size == 0:
        return []
    splits = np.where(np.diff(idx) != 1)[0]
    segs = []
    start = int(idx[0])
    for s in splits:
        end = int(idx[s])
        segs.append((start, end))
        start = int(idx[s+1])
    segs.append((start, int(idx[-1])))
    return segs

# masks: where p1>p2 (blue region), where p2>p1 (green region)
mask1 = p1 > p2
mask2 = p2 > p1

segs1 = find_segments(mask1)
segs2 = find_segments(mask2)

# build polygons for mask1 on plane y1: top = p1, bottom = p2 (so it matches fill_between(p1,p2))
polys_plane_y1 = []
for (s, e) in segs1:
    xs = x[s:e+1]
    top = p1[s:e+1]
    bot = p2[s:e+1]
    verts = [(float(xs[i]), float(y1), float(top[i])) for i in range(len(xs))]
    # go back along bottom boundary
    for i in range(len(xs)-1, -1, -1):
        verts.append((float(xs[i]), float(y1), float(bot[i])))
    polys_plane_y1.append(verts)

# build polygons for mask2 on plane y2: top = p2, bottom = p1
polys_plane_y2 = []
for (s, e) in segs2:
    xs = x[s:e+1]
    top = p2[s:e+1]
    bot = p1[s:e+1]
    verts = [(float(xs[i]), float(y2), float(top[i])) for i in range(len(xs))]
    for i in range(len(xs)-1, -1, -1):
        verts.append((float(xs[i]), float(y2), float(bot[i])))
    polys_plane_y2.append(verts)

# ---------------- plotting ----------------
from mpl_toolkits.mplot3d import Axes3D  # noqa

fig = plt.figure(figsize=(10,7), dpi=200)
ax = fig.add_subplot(111, projection='3d')

# # plot main curves on their planes
# ax.plot(x, np.full_like(x, y1), p1, color='#1f77b4', lw=2.2, label=f'P (y={y1})')
# ax.plot(x, np.full_like(x, y2), p2, color='#2ca02c', lw=2.2, label=f'Q (y={y2})')


# # add filled regions as polygons on respective planes
# if polys_plane_y1:
#     poly1 = Poly3DCollection(polys_plane_y1, facecolors='#87CEEB', edgecolors='none', alpha=0.85)
#     ax.add_collection3d(poly1)
# if polys_plane_y2:
#     poly2 = Poly3DCollection(polys_plane_y2, facecolors='#90EE90', edgecolors='none', alpha=0.85)
#     ax.add_collection3d(poly2)


# plot main curves on their planes
ax.plot(x, np.full_like(x, y1), p1, color='#4C78A8', lw=2.2, label=f'P (y={y1})')
ax.plot(x, np.full_like(x, y2), p2, color='#8E5EA2', lw=2.2, label=f'Q (y={y2})')


if polys_plane_y1:
    poly1 = Poly3DCollection(polys_plane_y1, facecolors='#9BB7D4', edgecolors='none', alpha=0.85)
    ax.add_collection3d(poly1)
if polys_plane_y2:
    poly2 = Poly3DCollection(polys_plane_y2, facecolors='#C4A5CF', edgecolors='none', alpha=0.85)
    ax.add_collection3d(poly2)


# labels & ticks
ax.set_xlabel('x', labelpad=8)
ax.set_ylabel('y (depth)', labelpad=8)
ax.set_zlabel('density (z)', labelpad=8)

ax.set_xticks(np.linspace(round(x_min,1), round(x_max,1), 6))
max_z = max(p1.max(), p2.max())
ax.set_zticks(np.linspace(0, max_z, 6))
ax.set_yticks([y1, y2])

# box aspect (attempt cube-like)
x_span = x_max - x_min
try:
    ax.set_box_aspect((x_span, x_span, x_span))
except Exception:
    pass

# camera - you can change elev/azim
ax.view_init(elev=18, azim=-70)
ax.set_xlim(-4, 13)                 # 视觉边界 1→3
ax.set_ylim(1, 3)                 # 视觉边界 1→3
ax.set_zlim(0, 0.5)                 # 视觉边界 1→3

#ax.set_title('Separate-plane non-overlap: blue on y=1 (p1>p2), green on y=3 (p2>p1)', pad=14)
# ---------- 2. 三无：无标记、无坐标轴、无网格 ----------
ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
ax.axis('off')          # 一键隐藏坐标轴+刻度+标签
ax.grid(False)          # 关闭网格线

plt.tight_layout()
out_pdf = "Social Exposure Diagram 2.pdf"
plt.savefig(out_pdf,
            format='pdf',
            dpi=300,
            bbox_inches='tight',
            pad_inches=0,
            transparent=True)
plt.show()   # 如不需要预览可注释

print(f"Saved to: {out_png}")