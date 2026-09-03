# -*- coding: utf-8 -*-

"""
Preprocessing code for Fig.3c.

Private inputs:
    matrices_A_D_S_Distribution/
        Other_Individual_and_Family_Services/
            flow_matrix.csv

    k_matrices_boston_family_budget.pkl

Public outputs:
    figure3c_nodes.csv
    figure3c_initial_edges.csv
    figure3c_optimized_edges.csv
    figure3c_summary.csv

The preprocessing preserves the original Fig.3c operations:

1. Select the top 44 POIs according to baseline total flow.
2. Calculate each CBG's total flow on those POIs.
3. Select the top 100 CBGs.
4. Extract the initial and 100%-budget optimized matrices.
5. Project each CBG×POI matrix into a CBG–CBG network:
       two CBGs are connected if they visit the same POI;
       edge weight = number of shared visited POIs.
"""

import pandas as pd
import numpy as np
import networkx as nx
import pickle


# ============================================================
# Network projection
# ============================================================

def create_optimization_projection(flow_df, threshold=0.02):
    """
    为优化结果创建CBG到CBG的投影网络
    基于共同访问POI
    """
    G = nx.Graph()

    cbgs = flow_df.index.tolist()
    pois = flow_df.columns.tolist()

    # 添加节点
    for cbg in cbgs:
        G.add_node(cbg)

    for poi in pois:

        cbgs_with_flow = (
            flow_df.index[
                flow_df[poi] > 0
            ]
            .tolist()
        )

        for i in range(len(cbgs_with_flow)):

            for j in range(
                i + 1,
                len(cbgs_with_flow)
            ):

                cbg1 = cbgs_with_flow[i]
                cbg2 = cbgs_with_flow[j]

                if G.has_edge(cbg1, cbg2):

                    G[cbg1][cbg2]['weight'] += 1

                else:

                    G.add_edge(
                        cbg1,
                        cbg2,
                        weight=1
                    )

    return G


# ============================================================
# User parameters
# ============================================================

city = 'boston'

category = 'Other Individual and Family Services'

cat_dir = (
    f'matrices_A_D_S_Distribution/'
    f'{category.replace(" ", "_")}'
)


FLOW_PATH = (
    f'{cat_dir}/flow_matrix.csv'
)

OPT_PATH = (
    'k_matrices_boston_family_budget.pkl'
)


OUT_NODES = (
    'figure3c_nodes.csv'
)

OUT_INITIAL_EDGES = (
    'figure3c_initial_edges.csv'
)

OUT_OPTIMIZED_EDGES = (
    'figure3c_optimized_edges.csv'
)

OUT_SUMMARY = (
    'figure3c_summary.csv'
)


# ============================================================
# Load baseline flow matrix
# ============================================================

flow_matrix = pd.read_csv(
    FLOW_PATH,
    index_col=0
)


# ============================================================
# Select POIs and CBGs
# Same logic as the original code
# ============================================================

poi_total_flow = (
    flow_matrix.sum(axis=0)
)

poi_num = (
    flow_matrix.shape[1]
)

selected_pois = (
    poi_total_flow
    .sort_values(ascending=False)
    .head(poi_num)
    .index
    .tolist()
)


selected_cbgs = set()

for poi in selected_pois:

    cbgs_with_flow = (
        flow_matrix.index[
            flow_matrix[poi] > 0
        ]
        .tolist()
    )

    selected_cbgs.update(
        cbgs_with_flow
    )


selected_cbgs = list(
    selected_cbgs
)


# ============================================================
# Load optimization results
# ============================================================

with open(
    OPT_PATH,
    'rb'
) as f:

    k_matrices = pickle.load(f)


# 获取100% budget优化结果

optimized_flow_df = (
    k_matrices.get(1.0)
)


if optimized_flow_df is None:

    raise RuntimeError(
        "未找到 100% budget (1.0) 的优化矩阵"
    )


# ============================================================
# Original Fig.3c selection
# ============================================================

# 1. 选前 10 大流量的 POI
# Original code actually uses head(44)

top10_pois = (
    poi_total_flow
    .sort_values(ascending=False)
    .head(44)
    .index
)


# 2. 在这些 POI 上，每个 CBG 的总流出量

cbg_flow_on_top10 = (
    flow_matrix
    .loc[:, top10_pois]
    .sum(axis=1)
)


# 3. 选前 10 大流量的 CBG
# Original code actually uses head(100)

top10_cbgs = (
    cbg_flow_on_top10
    .sort_values(ascending=False)
    .head(100)
    .index
)


# 4. 初始流量矩阵

initial_flow_df = (
    flow_matrix.loc[
        top10_cbgs,
        top10_pois
    ]
)


# 获取优化后的流量矩阵

optimized_flow_df = (
    optimized_flow_df.loc[
        top10_cbgs,
        top10_pois
    ]
)


# ============================================================
# Create projected networks
# Same operation as original figure3c.py
# ============================================================

G_initial = (
    create_optimization_projection(
        initial_flow_df
    )
)


G_optimized = (
    create_optimization_projection(
        optimized_flow_df
    )
)


# ============================================================
# Save node order
#
# This is important because create_income_stratified_layout()
# places nodes according to G.nodes() order.
# ============================================================

nodes_df = pd.DataFrame({
    'order': np.arange(
        len(G_initial.nodes())
    ),
    'GEOID': list(
        G_initial.nodes()
    )
})


nodes_df.to_csv(
    OUT_NODES,
    index=False
)


# ============================================================
# Save initial edges
#
# Preserve original G.edges(data=True) order.
# ============================================================

initial_edges = []


for u, v, data in G_initial.edges(
    data=True
):

    initial_edges.append({
        'source': u,
        'target': v,
        'weight': data.get(
            'weight',
            1
        )
    })


initial_edges_df = pd.DataFrame(
    initial_edges,
    columns=[
        'source',
        'target',
        'weight'
    ]
)


initial_edges_df.to_csv(
    OUT_INITIAL_EDGES,
    index=False
)


# ============================================================
# Save optimized edges
# ============================================================

optimized_edges = []


for u, v, data in G_optimized.edges(
    data=True
):

    optimized_edges.append({
        'source': u,
        'target': v,
        'weight': data.get(
            'weight',
            1
        )
    })


optimized_edges_df = pd.DataFrame(
    optimized_edges,
    columns=[
        'source',
        'target',
        'weight'
    ]
)


optimized_edges_df.to_csv(
    OUT_OPTIMIZED_EDGES,
    index=False
)


# ============================================================
# Save original final statistics
# ============================================================

summary_df = pd.DataFrame([{

    'initial_cbg_count':
        len(initial_flow_df.index),

    'initial_poi_count':
        len(initial_flow_df.columns),

    'optimized_cbg_count':
        len(optimized_flow_df.index),

    'optimized_poi_count':
        len(optimized_flow_df.columns),

    'total_flow_change':
        (
            optimized_flow_df
            .sum()
            .sum()
            -
            initial_flow_df
            .sum()
            .sum()
        )

}])


summary_df.to_csv(
    OUT_SUMMARY,
    index=False
)


# ============================================================
# Print information
# ============================================================

print(
    "\n========== FIG.3C PREPROCESSING =========="
)

print(
    f"Initial projected network: "
    f"{G_initial.number_of_nodes()} nodes, "
    f"{G_initial.number_of_edges()} edges"
)

print(
    f"Optimized projected network: "
    f"{G_optimized.number_of_nodes()} nodes, "
    f"{G_optimized.number_of_edges()} edges"
)

print(
    f"\n[SAVE] {OUT_NODES}"
)

print(
    f"[SAVE] {OUT_INITIAL_EDGES}"
)

print(
    f"[SAVE] {OUT_OPTIMIZED_EDGES}"
)

print(
    f"[SAVE] {OUT_SUMMARY}"
)


print(
    "\n网络统计信息:"
)

print(
    f"初始网络 - CBG数量: "
    f"{len(initial_flow_df.index)}, "
    f"POI数量: "
    f"{len(initial_flow_df.columns)}"
)

print(
    f"优化网络 - CBG数量: "
    f"{len(optimized_flow_df.index)}, "
    f"POI数量: "
    f"{len(optimized_flow_df.columns)}"
)

print(
    f"总流量变化: "
    f"{optimized_flow_df.sum().sum() - initial_flow_df.sum().sum():.2f}"
)

print(
    "=========================================="
)