# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as mpatches


# ============================================================
# Load preprocessed projected networks
# ============================================================

nodes_df = pd.read_csv(
    'figure3c_nodes.csv'
)


initial_edges_df = pd.read_csv(
    'figure3c_initial_edges.csv'
)


optimized_edges_df = pd.read_csv(
    'figure3c_optimized_edges.csv'
)


summary_df = pd.read_csv(
    'figure3c_summary.csv'
)


# ============================================================
# Restore node types
#
# Original Fig.3c reads GEOID from the income file as np.int64.
# ============================================================

nodes_df['GEOID'] = (
    nodes_df['GEOID']
    .astype(np.int64)
)


if len(initial_edges_df) > 0:

    initial_edges_df['source'] = (
        initial_edges_df['source']
        .astype(np.int64)
    )

    initial_edges_df['target'] = (
        initial_edges_df['target']
        .astype(np.int64)
    )


if len(optimized_edges_df) > 0:

    optimized_edges_df['source'] = (
        optimized_edges_df['source']
        .astype(np.int64)
    )

    optimized_edges_df['target'] = (
        optimized_edges_df['target']
        .astype(np.int64)
    )


# ============================================================
# Reconstruct projected networks
#
# Node order and edge insertion order are preserved.
# ============================================================

G_initial = nx.Graph()


for node in (
    nodes_df
    .sort_values('order')['GEOID']
    .tolist()
):

    G_initial.add_node(
        node
    )


for _, row in initial_edges_df.iterrows():

    G_initial.add_edge(
        int(row['source']),
        int(row['target']),
        weight=row['weight']
    )


G_optimized = nx.Graph()


for node in (
    nodes_df
    .sort_values('order')['GEOID']
    .tolist()
):

    G_optimized.add_node(
        node
    )


for _, row in optimized_edges_df.iterrows():

    G_optimized.add_edge(
        int(row['source']),
        int(row['target']),
        weight=row['weight']
    )


# ============================================================
# Original plotting definitions
# ============================================================

# 离散化的四种颜色（与您定义的 cmap 对应）

discrete_colors = [
    '#2166ac',
    '#67a9cf',
    '#fddbc7',
    '#b2182b'
]


def income_to_color(score):
    """
    将 0-1 的 income score 映射到 4 个离散颜色索引：
    [0,0.25) -> idx 0,
    [0.25,0.5) -> idx 1,
    [0.5,0.75) -> idx 2,
    [0.75,1] -> idx 3
    """

    try:
        s = float(score)

    except:
        s = 0.0

    # 确保 s 在 [0,1]，避免异常值

    s = max(
        0.0,
        min(
            1.0,
            s
        )
    )

    idx = min(
        int(s * 4),
        3
    )

    return discrete_colors[idx]


def plot_optimization_networks(
    initial_G,
    optimized_G,
    cbg_income_dist_dict,
    income_levels,
    figsize=(8, 14),
    node_size=200,
    font_size=8
):
    """
    专门为优化结果绘制网络结构对比图

    Parameters:
    -----------
    initial_G : Graph
        原始流量矩阵对应的 CBG-CBG 投影网络

    optimized_G : Graph
        优化后流量矩阵对应的 CBG-CBG 投影网络

    cbg_income_dist_dict : dict
        CBG收入分布字典

    income_levels : list
        收入水平列表
    """

    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=figsize
    )


    # 创建收入水平颜色映射

    cmap = LinearSegmentedColormap.from_list(
        'income_cmap',
        [
            '#2166ac',
            '#67a9cf',
            '#fddbc7',
            '#b2182b'
        ]
    )


    # 计算每个CBG的综合收入水平（使用高收入比例作为代表）

    cbg_income_scores = {}


    for cbg in initial_G.nodes():

        if cbg in cbg_income_dist_dict:

            # 使用高收入比例作为收入水平的代表

            cbg_income_scores[cbg] = (
                cbg_income_dist_dict[
                    cbg
                ].get(
                    'high_income_pct',
                    0
                )
            )

        else:

            cbg_income_scores[cbg] = 0


    # 创建初始网络并计算固定布局

    fixed_pos = (
        create_income_stratified_layout(
            initial_G,
            cbg_income_scores
        )
    )


    # 图1: 优化前网络

    plot_single_optimization_network(
        initial_G,
        cbg_income_scores,
        ax1,
        cmap,
        node_size,
        font_size,
        'Before Optimization',
        fixed_pos
    )


    # 图2: 优化后网络

    plot_single_optimization_network(
        optimized_G,
        cbg_income_scores,
        ax2,
        cmap,
        node_size,
        font_size,
        'After Optimization (100% Budget)',
        fixed_pos
    )


    # 添加图例

    add_optimization_legend(
        fig,
        cmap
    )


    plt.tight_layout()

    return fig


def plot_single_optimization_network(
    G,
    cbg_income_scores,
    ax,
    cmap,
    node_size,
    font_size,
    title,
    fixed_pos
):
    """绘制单个优化网络图"""


    if len(G.nodes()) == 0:

        ax.text(
            0.5,
            0.5,
            'No Network Connections',
            ha='center',
            va='center',
            transform=ax.transAxes,
            fontsize=12
        )

        ax.set_title(
            title,
            fontsize=14
        )

        ax.axis(
            'off'
        )

        return


    # 使用固定布局

    pos = fixed_pos


    # 绘制边（先画收入差小的，后画收入差大的，使大差异边覆盖在上面）

    edges = list(
        G.edges(
            data=True
        )
    )


    if edges:

        # 计算边权重并归一化宽度

        edge_weights = [
            data.get(
                'weight',
                1
            )
            for _, _, data in edges
        ]


        max_weight = (
            max(edge_weights)
            if edge_weights
            else 1
        )


        edge_widths = [
            w * 2 / max_weight
            for w in edge_weights
        ]


        # 收集每条边的额外信息：
        # income_diff, edge tuple, color, width

        edge_infos = []


        for (
            u,
            v,
            data
        ), width in zip(
            edges,
            edge_widths
        ):

            income_u = (
                cbg_income_scores.get(
                    u,
                    0.5
                )
            )

            income_v = (
                cbg_income_scores.get(
                    v,
                    0.5
                )
            )

            income_diff = abs(
                income_u
                - income_v
            )


            # 根据收入差决定颜色（>0.3 为大差异）

            if income_diff > 0.3:

                color = '#e74c3c'

            else:

                color = '#3498db'


            edge_infos.append(
                (
                    income_diff,
                    (u, v),
                    color,
                    width
                )
            )


        # 按 income_diff 升序排序（小差异先画）

        edge_infos.sort(
            key=lambda x: x[0]
        )


        # 准备绘制参数（保持绘制顺序）

        edgelist = [
            info[1]
            for info in edge_infos
        ]

        edge_colors = [
            info[2]
            for info in edge_infos
        ]

        edge_widths_sorted = [
            info[3]
            for info in edge_infos
        ]


        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=edgelist,
            width=edge_widths_sorted,
            alpha=0.7,
            edge_color=edge_colors,
            ax=ax
        )


    # 绘制节点 - 根据收入水平着色

    node_colors = [
        cmap(
            cbg_income_scores[node]
        )
        for node in G.nodes()
    ]


    nodes = nx.draw_networkx_nodes(
        G,
        pos,
        node_size=node_size,
        node_color=node_colors,
        edgecolors='black',
        linewidths=0.8,
        ax=ax
    )


    # 设置图形属性

    ax.set_title(
        title,
        fontsize=14,
        pad=20
    )

    ax.axis(
        'off'
    )


    # 添加网络指标文本框

    metrics = (
        calculate_optimization_metrics(
            G,
            None,
            cbg_income_scores
        )
    )


    textstr = '\n'.join(
        [
            f'{k}: {v:.2f}'
            for k, v in metrics.items()
        ]
    )


    props = dict(
        boxstyle='round',
        facecolor='lightblue',
        alpha=0.8
    )


    ax.text(
        0.02,
        0.98,
        textstr,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment='top',
        bbox=props
    )


    ax.set_aspect(
        'equal',
        adjustable='box'
    )


def create_income_stratified_layout(
    G,
    cbg_income_scores
):
    """
    创建收入分层布局 - 低收入节点在中心，高收入节点在外围
    """

    if len(G.nodes()) == 0:

        return {}


    pos = {}


    # 将节点按收入水平分组

    low_income_nodes = []

    medium_low_nodes = []

    medium_high_nodes = []

    high_income_nodes = []


    for node in G.nodes():

        income = (
            cbg_income_scores.get(
                node,
                0
            )
        )


        if income < 0.25:

            low_income_nodes.append(
                node
            )

        elif income < 0.5:

            medium_low_nodes.append(
                node
            )

        elif income < 0.75:

            medium_high_nodes.append(
                node
            )

        else:

            high_income_nodes.append(
                node
            )


    # 定义各收入层的半径

    radii = {
        'low': 0.1,
        'medium_low': 0.3,
        'medium_high': 0.6,
        'high': 0.9
    }


    # 放置低收入节点在最内圈

    angle_step = (
        2 * np.pi
        / max(
            1,
            len(low_income_nodes)
        )
    )


    for i, node in enumerate(
        low_income_nodes
    ):

        angle = (
            i * angle_step
        )

        pos[node] = (
            radii['low']
            * np.cos(angle),

            radii['low']
            * np.sin(angle)
        )


    # 放置中低收入节点在内圈

    angle_step = (
        2 * np.pi
        / max(
            1,
            len(medium_low_nodes)
        )
    )


    for i, node in enumerate(
        medium_low_nodes
    ):

        angle = (
            i * angle_step
        )

        pos[node] = (
            radii['medium_low']
            * np.cos(angle),

            radii['medium_low']
            * np.sin(angle)
        )


    # 放置中高收入节点在中圈

    angle_step = (
        2 * np.pi
        / max(
            1,
            len(medium_high_nodes)
        )
    )


    for i, node in enumerate(
        medium_high_nodes
    ):

        angle = (
            i * angle_step
        )

        pos[node] = (
            radii['medium_high']
            * np.cos(angle),

            radii['medium_high']
            * np.sin(angle)
        )


    # 放置高收入节点在外圈

    angle_step = (
        2 * np.pi
        / max(
            1,
            len(high_income_nodes)
        )
    )


    for i, node in enumerate(
        high_income_nodes
    ):

        angle = (
            i * angle_step
        )

        pos[node] = (
            radii['high']
            * np.cos(angle),

            radii['high']
            * np.sin(angle)
        )


    # 添加轻微随机扰动，避免节点重叠

    for node in pos:

        pos[node] = (
            pos[node][0]
            + np.random.normal(
                0,
                0.02
            ),

            pos[node][1]
            + np.random.normal(
                0,
                0.02
            )
        )


    return pos


def calculate_optimization_metrics(
    G,
    flow_df,
    cbg_income_scores
):
    """计算优化网络的指标"""

    if len(G.nodes()) == 0:

        return {
            'Nodes': 0,
            'Edges': 0,
            'Avg Degree': 0,
            'Assortativity': 0
        }


    try:

        metrics = {
            'Edges':
                G.number_of_edges(),
        }


        # 计算同配性（收入水平）

        try:

            # 为网络节点添加收入属性

            for node in G.nodes():

                G.nodes[node]['income'] = (
                    cbg_income_scores.get(
                        node,
                        0.5
                    )
                )


            assortativity = (
                nx.numeric_assortativity_coefficient(
                    G,
                    'income'
                )
            )

        except:

            metrics[
                'Income Assort.'
            ] = 0


        # 计算跨收入连接比例

        try:

            cross_income_edges = 0

            total_edges = (
                G.number_of_edges()
            )


            for u, v in G.edges():

                income_u = (
                    cbg_income_scores.get(
                        u,
                        0.5
                    )
                )

                income_v = (
                    cbg_income_scores.get(
                        v,
                        0.5
                    )
                )


                # 如果收入差异大于0.3，认为是跨收入连接

                if abs(
                    income_u
                    - income_v
                ) > 0.3:

                    cross_income_edges += 1


            metrics['Cross Income '] = (
                cross_income_edges
                / total_edges
                if total_edges > 0
                else 0
            )

        except:

            metrics[
                'Cross Income %'
            ] = 0


        # 计算模块化

        try:

            from networkx.algorithms import community

            communities = (
                community
                .greedy_modularity_communities(
                    G
                )
            )

            modularity = (
                community.modularity(
                    G,
                    communities
                )
            )

        except:

            metrics[
                'Modularity'
            ] = 0


    except Exception as e:

        metrics = {
            'Edges': 0,
            'Cross Income %': 0
        }


    return metrics


def add_optimization_legend(
    fig,
    cmap
):
    """添加优化网络图例（节点用点表示，边用线段表示）"""

    from matplotlib.lines import Line2D


    # 收入水平图例（用原点表示）

    income_labels = [
        'Low Income',
        'Middle-Low',
        'Middle-High',
        'High Income'
    ]


    income_handles = []


    for i, label in enumerate(
        income_labels
    ):

        color = cmap(
            i / 3.0
        )


        # 使用 Line2D 画点作为图例项（不画线）

        marker_handle = Line2D(
            [0],
            [0],
            marker='o',
            color='w',
            markerfacecolor=color,
            markeredgecolor='black',
            markersize=10,
            linestyle='None'
        )


        income_handles.append(
            marker_handle
        )


    # 边颜色图例（用线段表示）

    edge_labels = [
        'Low Income Diff',
        'High Income Diff'
    ]


    edge_colors = [
        '#3498db',
        '#e74c3c'
    ]


    edge_handles = []


    for color, label in zip(
        edge_colors,
        edge_labels
    ):

        line_handle = Line2D(
            [0],
            [0],
            color=color,
            lw=3,
            solid_capstyle='round'
        )


        edge_handles.append(
            line_handle
        )


    # 组合图例（保持原来的布局）

    all_handles = (
        income_handles
        + edge_handles
    )


    all_labels = (
        income_labels
        + edge_labels
    )


    fig.legend(
        handles=all_handles,
        labels=all_labels,
        loc='lower center',
        ncol=3,
        bbox_to_anchor=(
            0.5,
            -0.01
        ),
        fontsize=10,
        frameon=True,
        fancybox=True,
        shadow=True
    )


# ============================================================
# Load public CBG income distribution
# ============================================================

city = 'boston'

category = (
    'Other Individual and Family Services'
)


cbg_income_dist_df = pd.read_csv(
    f'matrices_A_D_S_Distribution/'
    f'cbg_income_level_distribution_{city}_msa.csv',
    dtype={
        'GEOID':
            np.int64
    }
)


cbg_income_dist_dict = (
    cbg_income_dist_df
    .set_index('GEOID')
    .to_dict(
        orient='index'
    )
)


income_levels = [
    'low_income_pct',
    'lower_middle_income_pct',
    'upper_middle_income_pct',
    'high_income_pct'
]


# ============================================================
# Draw network comparison
# ============================================================

fig = plot_optimization_networks(
    G_initial,
    G_optimized,
    cbg_income_dist_dict,
    income_levels,
    figsize=(5, 10),
    node_size=120,
    font_size=9
)


plt.savefig(
    'figure3c.pdf',
    format='pdf',
    dpi=300,
    bbox_inches='tight',
    transparent=False,
    backend='pdf'
)


plt.show()


# ============================================================
# Print original statistics
# ============================================================

print(
    "\n网络统计信息:"
)


print(
    f"初始网络 - CBG数量: "
    f"{int(summary_df.loc[0, 'initial_cbg_count'])}, "
    f"POI数量: "
    f"{int(summary_df.loc[0, 'initial_poi_count'])}"
)


print(
    f"优化网络 - CBG数量: "
    f"{int(summary_df.loc[0, 'optimized_cbg_count'])}, "
    f"POI数量: "
    f"{int(summary_df.loc[0, 'optimized_poi_count'])}"
)


print(
    f"总流量变化: "
    f"{summary_df.loc[0, 'total_flow_change']:.2f}"
)