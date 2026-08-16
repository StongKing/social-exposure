# -*- coding: utf-8 -*-
"""
Figure 1b schematic: CBG -> POI network-flow reallocation.

Updated version:
    1. Network uses 4 CBGs / 5 POIs.
    2. CBG order is from low income to high income.
    3. Higher income uses darker color.
    4. Before:
       - low-income CBG visits 3 POIs
       - lower-middle-income CBG visits 2 POIs
       - upper-middle-income CBG visits 2 POIs
       - high-income CBG visits 1 POI
       - based on the current flow ratio, POI4 and POI5 are each visited by one CBG.
    5. After:
       - low-income CBG visits 4 POIs
       - lower-middle-income CBG visits 3 POIs
       - upper-middle-income CBG visits 4 POIs
       - high-income CBG visits 4 POIs
    6. Each CBG keeps total outgoing flow = 100 before and after.
    7. POI pie charts are computed automatically from incoming simulated flows.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

from matplotlib.patches import Polygon, Wedge, Circle, PathPatch
from matplotlib.path import Path


# =============================================================================
# 1. Global style
# =============================================================================

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False

SHOW_NODE_LABELS = False
SAVE_FIGURES = False   # 改成 True 则保存 png/pdf


# Income colors: low -> high, higher income = darker color
CBG_COLORS = {
    "low": "#e4f3ff",
    "lower_middle": "#bfe4f7",
    "upper_middle": "#8ecae6",
    "high": "#4f9fce",
}

CBG_COLORS = {
    "low": "#B3D9F5",
    "lower_middle": "#ABD0EA",
    "upper_middle": "#A3C4DF",
    "high": "#9BB7D4",
}

POI_PIE_COLORS = [
    CBG_COLORS["low"],
    CBG_COLORS["lower_middle"],
    CBG_COLORS["upper_middle"],
    CBG_COLORS["high"],
]

EDGE_COLORS = CBG_COLORS.copy()

INCOME_ORDER = ["low", "lower_middle", "upper_middle", "high"]


CBG_BORDER_COLOR = "#000000"
POI_BORDER_COLOR = "#90EE90"

POI_BORDER_COLOR = "#8E5EA2"

POI_EMPTY_FACE_COLOR = "#ffffff"


# =============================================================================
# 2. Geometry helpers
# =============================================================================

def map_xy(local_xy, panel):
    """
    Map local panel coordinates [0,1] x [0,1] to absolute figure coordinates.
    """
    x0, y0, w, h = panel
    return np.array([x0 + local_xy[0] * w, y0 + local_xy[1] * h], dtype=float)


def normalize(v):
    n = np.linalg.norm(v)
    if n < 1e-12:
        return np.array([0.0, 0.0])
    return v / n


def perpendicular(v):
    return np.array([-v[1], v[0]], dtype=float)


def irregular_polygon(center, radius=0.16, n=14, seed=0, anisotropy=None):
    """
    More irregular CBG-like polygon.
    """
    rng = np.random.default_rng(seed)

    if anisotropy is None:
        anisotropy = (
            rng.uniform(0.82, 1.22),
            rng.uniform(0.82, 1.22),
        )

    base_angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    angle_jitter = rng.uniform(-0.5, 0.5, size=n)
    angles = np.sort(base_angles + angle_jitter + 0.41 * seed)

    radial_jitter = rng.uniform(0.38, 1.6, size=n)
    radial_jitter = (
        0.52 * radial_jitter
        + 0.24 * np.roll(radial_jitter, 1)
        + 0.24 * np.roll(radial_jitter, -1)
    )

    x0 = radius * anisotropy[0] * radial_jitter * np.cos(angles)
    y0 = radius * anisotropy[1] * radial_jitter * np.sin(angles)

    shear = rng.uniform(-0.2, 0.2)
    x = center[0] + x0 + shear * y0
    y = center[1] + y0

    return np.c_[x, y]


def bezier_path(p0, p1, p2, p3):
    """
    Cubic Bezier path.
    """
    verts = [tuple(p0), tuple(p1), tuple(p2), tuple(p3)]
    codes = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4]
    return Path(verts, codes)


# =============================================================================
# 3. Draw node functions
# =============================================================================

def draw_cbg(ax, panel, name, local_xy, income_group, seed):
    """
    Draw one CBG polygon.
    """
    xy = map_xy(local_xy, panel)
    poly = irregular_polygon(xy, radius=0.30, seed=seed)

    patch = Polygon(
        poly,
        closed=True,
        facecolor=CBG_COLORS[income_group],
        edgecolor=CBG_BORDER_COLOR,
        linewidth=1.10,
        zorder=8,
    )

    patch.set_path_effects(
        [pe.SimplePatchShadow(offset=(2, -2), alpha=0.18), pe.Normal()]
    )

    ax.add_patch(patch)

    if SHOW_NODE_LABELS:
        ax.text(
            xy[0],
            xy[1] - 0.43,
            name,
            ha="center",
            va="top",
            fontsize=15,
            color="#111111",
            zorder=11,
        )


def draw_poi_pie(ax, panel, name, local_xy, shares, radius=0.27):
    """
    Draw POI pie chart.

    If shares is None, the POI has no incoming flow in that panel.
    """
    xy = map_xy(local_xy, panel)

    if shares is None:
        ax.add_patch(
            Circle(
                xy,
                radius,
                facecolor=POI_EMPTY_FACE_COLOR,
                edgecolor=POI_BORDER_COLOR,
                linewidth=1.80,
                alpha=1.0,
                zorder=10,
            )
        )
    else:
        start = 90.0

        for share, color in zip(shares, POI_PIE_COLORS):
            if share <= 0:
                continue

            ax.add_patch(
                Wedge(
                    xy,
                    radius,
                    start,
                    start + share * 360.0,
                    facecolor=color,
                    edgecolor=POI_BORDER_COLOR,
                    linewidth=0.45,
                    alpha=1.0,
                    zorder=10,
                )
            )

            start += share * 360.0

        ax.add_patch(
            Circle(
                xy,
                radius,
                facecolor="none",
                edgecolor=POI_BORDER_COLOR,
                linewidth=2,
                zorder=11,
            )
        )

    if SHOW_NODE_LABELS:
        ax.text(
            xy[0],
            xy[1] + radius + 0.11,
            name,
            ha="center",
            va="bottom",
            fontsize=15,
            color="#111111",
            zorder=12,
        )


# =============================================================================
# 4. Flow drawing
# =============================================================================

def draw_flow(
    ax,
    panel,
    cbg_xy,
    poi_xy,
    value,
    income_group,
    rad,
    max_value,
    source_radius=0.31,
    target_radius=0.28,
):
    """
    Draw flow as a cubic Bezier curve.

    This version:
    - does not bend around one common center
    - uses moderate curvature
    - keeps relatively thick flow lines
    - allows crossing / separation through rad
    """
    src = map_xy(cbg_xy, panel)
    dst = map_xy(poi_xy, panel)

    u = normalize(dst - src)

    p0 = src + u * (source_radius + 0.04)
    p3 = dst - u * (target_radius + 0.04)

    chord = p3 - p0
    t = normalize(perpendicular(chord))

    p1 = p0 + 0.32 * chord
    p2 = p0 + 0.68 * chord

    offset = 0.40 * rad
    p1 = p1 + t * offset
    p2 = p2 + t * offset

    path = bezier_path(p0, p1, p2, p3)

    q = np.sqrt(value / max_value)
    lw = 2.80 + 8.80 * q
    alpha = 0.28 + 0.52 * q
    color = EDGE_COLORS[income_group]

    # Soft halo
    ax.add_patch(
        PathPatch(
            path,
            facecolor="none",
            edgecolor=color,
            linewidth=lw + 2.30,
            alpha=0.075,
            zorder=2,
            capstyle="round",
            joinstyle="round",
        )
    )

    # Main curve
    ax.add_patch(
        PathPatch(
            path,
            facecolor="none",
            edgecolor=color,
            linewidth=lw,
            alpha=alpha,
            zorder=3,
            capstyle="round",
            joinstyle="round",
        )
    )


# =============================================================================
# 5. Flow -> POI pie composition
# =============================================================================

def compute_poi_shares(flows, cbgs, poi_names):
    """
    Compute POI pie shares automatically from incoming simulated flows.

    If a POI has no incoming flow, return None for that POI.
    """
    counts = {poi: {k: 0.0 for k in INCOME_ORDER} for poi in poi_names}

    for cbg, poi, value, _ in flows:
        group = cbgs[cbg]["income"]
        counts[poi][group] += value

    shares = {}

    for poi in poi_names:
        total = sum(counts[poi][k] for k in INCOME_ORDER)

        if total <= 0:
            shares[poi] = None
        else:
            shares[poi] = [counts[poi][k] / total for k in INCOME_ORDER]

    return shares


def compute_cbg_outflow_totals(flows):
    """
    Compute total outgoing flow for each CBG.
    """
    totals = {}

    for cbg, _, value, _ in flows:
        totals[cbg] = totals.get(cbg, 0.0) + value

    return totals


def compute_cbg_degree(flows):
    """
    Count how many POIs each CBG visits.
    """
    visited = {}

    for cbg, poi, value, _ in flows:
        if value <= 0:
            continue

        if cbg not in visited:
            visited[cbg] = set()

        visited[cbg].add(poi)

    return {cbg: len(pois) for cbg, pois in visited.items()}


def compute_poi_degree(flows, poi_names):
    """
    Count how many CBGs visit each POI.
    Includes POIs with zero incoming flow.
    """
    visited = {poi: set() for poi in poi_names}

    for cbg, poi, value, _ in flows:
        if value <= 0:
            continue
        visited[poi].add(cbg)

    return {poi: len(cbgs_) for poi, cbgs_ in visited.items()}


# =============================================================================
# 6. Simulated network data
# =============================================================================
# 4 CBGs + 5 POIs
# CBG order: low -> lower-middle -> upper-middle -> high
# Each CBG has total outgoing flow = 100 before and after.

cbgs = {
    "CBG1": {"xy": (0.13, 0.81), "income": "low",          "seed": 101},
    "CBG2": {"xy": (0.13, 0.61), "income": "lower_middle", "seed": 102},
    "CBG3": {"xy": (0.13, 0.43), "income": "upper_middle", "seed": 103},
    "CBG4": {"xy": (0.13, 0.24), "income": "high",         "seed": 104},
}

poi_positions = {
    "POI1": (0.87, 0.90),
    "POI2": (0.87, 0.70),
    "POI3": (0.87, 0.50),
    "POI4": (0.87, 0.30),
    "POI5": (0.87, 0.10),
}


# -----------------------------------------------------------------------------
# Before, according to your current ratio:
# - CBG1 low income visits 3 POIs
# - CBG2 lower-middle income visits 2 POIs
# - CBG3 upper-middle income visits 2 POIs
# - CBG4 high income visits 1 POI
# - Each CBG outflow = 100
#
# POI degree pattern:
#   POI1: 2 CBGs
#   POI2: 2 CBGs
#   POI3: 2 CBGs
#   POI4: 1 CBG
#   POI5: 1 CBG
# -----------------------------------------------------------------------------
flows_before = [
    # Low income: 3 POIs
    ("CBG1", "POI1", 25, -0.36),
    ("CBG1", "POI2", 30, -0.14),
    ("CBG1", "POI4", 45,  0.34),

    # Lower-middle income: 2 POIs
    ("CBG2", "POI1", 55, -0.08),
    ("CBG2", "POI2", 45,  0.12),

    # Upper-middle income: 2 POIs
    ("CBG3", "POI3", 60,  0.32),
    ("CBG3", "POI4", 40,  0.32),

    # High income: 1 POI
    ("CBG4", "POI3", 40, 0.16),
    ("CBG4", "POI5", 60, 0.86),
]


# -----------------------------------------------------------------------------
# After, according to your current ratio:
# - CBG1 low income visits 4 POIs
# - CBG2 lower-middle income visits 3 POIs
# - CBG3 upper-middle income visits 4 POIs
# - CBG4 high income visits 4 POIs
# - Each CBG outflow = 100
# -----------------------------------------------------------------------------
flows_after = [
    # Low income: 4 POIs
    ("CBG1", "POI1", 22, -0.34),
    ("CBG1", "POI2", 24, -0.16),
    ("CBG1", "POI4", 24,  0.12),
    ("CBG1", "POI5", 30,  0.34),

    # Lower-middle income: 3 POIs
    ("CBG2", "POI1", 30, -0.12),
    ("CBG2", "POI2", 35,  0.06),
    ("CBG2", "POI3", 35,  0.24),

    # Upper-middle income: 4 POIs
    ("CBG3", "POI1", 20, -0.04),
    ("CBG3", "POI2", 24,  0.14),
    ("CBG3", "POI3", 26,  0.32),
    ("CBG3", "POI4", 30,  0.50),

    # High income: 4 POIs
    ("CBG4", "POI2", 18,  0.12),
    ("CBG4", "POI3", 28,  0.30),
    ("CBG4", "POI4", 26,  0.48),
    ("CBG4", "POI5", 28,  0.66),
]


# =============================================================================
# 7. Consistency checks
# =============================================================================

assert compute_cbg_outflow_totals(flows_before) == {
    "CBG1": 100.0,
    "CBG2": 100.0,
    "CBG3": 100.0,
    "CBG4": 100.0,
}

assert compute_cbg_outflow_totals(flows_after) == {
    "CBG1": 100.0,
    "CBG2": 100.0,
    "CBG3": 100.0,
    "CBG4": 100.0,
}

assert compute_cbg_degree(flows_before) == {
    "CBG1": 3,
    "CBG2": 2,
    "CBG3": 2,
    "CBG4": 2,
}

assert compute_cbg_degree(flows_after) == {
    "CBG1": 4,
    "CBG2": 3,
    "CBG3": 4,
    "CBG4": 4,
}

assert sum(v for _, _, v, _ in flows_before) == 400
assert sum(v for _, _, v, _ in flows_after) == 400

poi_names = list(poi_positions.keys())

poi_degree_before = compute_poi_degree(flows_before, poi_names)
assert poi_degree_before == {
    "POI1": 2,
    "POI2": 2,
    "POI3": 2,
    "POI4": 2,
    "POI5": 1,
}

poi_degree_after = compute_poi_degree(flows_after, poi_names)
assert poi_degree_after == {
    "POI1": 3,
    "POI2": 4,
    "POI3": 3,
    "POI4": 3,
    "POI5": 2,
}

poi_shares_before = compute_poi_shares(flows_before, cbgs, poi_names)
poi_shares_after = compute_poi_shares(flows_after, cbgs, poi_names)

def draw_column_headers(ax, panel):
    """
    Add column headers above the CBG and POI columns.
    """
    cbg_header_xy = map_xy((0.13, 0.95), panel)
    poi_header_xy = map_xy((0.87, 0.95), panel)

    ax.text(
        cbg_header_xy[0],
        cbg_header_xy[1],
        "CBGs",
        ha="center",
        va="bottom",
        fontsize=18,
        color="#111111",
        zorder=30,
    )

    ax.text(
        poi_header_xy[0],
        poi_header_xy[1],
        "POIs",
        ha="center",
        va="bottom",
        fontsize=18,
        color="#111111",
        zorder=30,
    )

# =============================================================================
# 8. Canvas / layout
# =============================================================================

fig = plt.figure(figsize=(17, 6.8), dpi=300)
ax = fig.add_axes([0, 0, 1, 1])

ax.set_xlim(0, 16.5)
ax.set_ylim(0.65, 6.95)
ax.set_aspect("equal", adjustable="box")
ax.axis("off")

left_panel = (0.05, 0.30, 7.0, 7.00)
right_panel = (9.20, 0.30, 7.0, 7.00)

max_flow_value = max(
    max(v for _, _, v, _ in flows_before),
    max(v for _, _, v, _ in flows_after),
)


# =============================================================================
# 9. Draw flows first
# =============================================================================

for cbg, poi, value, rad in sorted(flows_before, key=lambda x: x[2]):
    draw_flow(
        ax,
        left_panel,
        cbgs[cbg]["xy"],
        poi_positions[poi],
        value,
        cbgs[cbg]["income"],
        rad,
        max_flow_value,
    )

for cbg, poi, value, rad in sorted(flows_after, key=lambda x: x[2]):
    draw_flow(
        ax,
        right_panel,
        cbgs[cbg]["xy"],
        poi_positions[poi],
        value,
        cbgs[cbg]["income"],
        rad,
        max_flow_value,
    )


# =============================================================================
# 10. Draw nodes
# =============================================================================

for name, meta in cbgs.items():
    draw_cbg(ax, left_panel, name, meta["xy"], meta["income"], meta["seed"])
    draw_cbg(ax, right_panel, name, meta["xy"], meta["income"], meta["seed"])

for poi in poi_names:
    draw_poi_pie(
        ax,
        left_panel,
        poi,
        poi_positions[poi],
        poi_shares_before[poi],
        radius=0.27,
    )

    draw_poi_pie(
        ax,
        right_panel,
        poi,
        poi_positions[poi],
        poi_shares_after[poi],
        radius=0.27,
    )

draw_column_headers(ax, left_panel)
draw_column_headers(ax, right_panel)
# =============================================================================
# 11. Central vertical legend
# =============================================================================

legend_x_icon = 7.50
legend_x_text = 8.00
legend_r = 0.270

legend_y = [5.95, 4.95, 3.95, 2.95, 1.85]

# Order from low income to high income
legend_items = [
    ("Low income", "low", cbgs["CBG1"]["seed"]),
    ("Lower-middle\nincome", "lower_middle", cbgs["CBG2"]["seed"]),
    ("Upper-middle\nincome", "upper_middle", cbgs["CBG3"]["seed"]),
    ("High income", "high", cbgs["CBG4"]["seed"]),
]

for i, (label, group, seed) in enumerate(legend_items):
    y = legend_y[i]

    poly = irregular_polygon((legend_x_icon, y), radius=legend_r, seed=seed)

    patch = Polygon(
        poly,
        closed=True,
        facecolor=CBG_COLORS[group],
        edgecolor=CBG_BORDER_COLOR,
        linewidth=1.10,
        zorder=20,
    )

    patch.set_path_effects([
        pe.SimplePatchShadow(offset=(1.0, -1.0), alpha=0.12),
        pe.Normal(),
    ])

    ax.add_patch(patch)

    ax.text(
        legend_x_text,
        y,
        label,
        ha="left",
        va="center",
        fontsize=15,
        color="#111111",
        zorder=21,
        linespacing=0.98,
    )


# POI pie legend
pie_y = legend_y[4]
example_shares = [0.25, 0.25, 0.25, 0.25]
start = 90.0

for share, color in zip(example_shares, POI_PIE_COLORS):
    ax.add_patch(
        Wedge(
            (legend_x_icon, pie_y),
            legend_r,
            start,
            start + share * 360.0,
            facecolor=color,
            edgecolor=POI_BORDER_COLOR,
            linewidth=0.45,
            alpha=1.0,
            zorder=20,
        )
    )

    start += share * 360.0

ax.add_patch(
    Circle(
        (legend_x_icon, pie_y),
        legend_r,
        facecolor="none",
        edgecolor=POI_BORDER_COLOR,
        linewidth=2,
        zorder=21,
    )
)

ax.text(
    legend_x_text,
    pie_y + 0.05,
    "POI visitor\ncomposition",
    ha="left",
    va="center",
    fontsize=15,
    color="#111111",
    zorder=21,
    linespacing=0.98,
)


# =============================================================================
# 12. Save / show
# =============================================================================

plt.savefig('figure1b.pdf',
            format='pdf',
            dpi=300,
            bbox_inches='tight',
            transparent=False,
            backend='pdf')

plt.show()