import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as mpatches
import pickle



# 离散化的四种颜色（与您定义的 cmap 对应）
discrete_colors = ['#2166ac', '#67a9cf', '#fddbc7', '#b2182b']

def income_to_color(score):
    """
    将 0-1 的 income score 映射到 4 个离散颜色索引：
    [0,0.25) -> idx 0, [0.25,0.5) -> idx 1, [0.5,0.75) -> idx 2, [0.75,1] -> idx 3
    """
    try:
        s = float(score)
    except:
        s = 0.0
    # 确保 s 在 [0,1]，避免异常值
    s = max(0.0, min(1.0, s))
    idx = min(int(s * 4), 3)  # int(0.999*4)=3
    return discrete_colors[idx]

def plot_optimization_networks(initial_flow_df, optimized_flow_df, cbg_income_dist_dict, income_levels, 
                              figsize=(8, 14), node_size=200, font_size=8):
    """
    专门为优化结果绘制网络结构对比图
    
    Parameters:
    -----------
    initial_flow_df : DataFrame
        原始流量矩阵 (CBGs × POIs)
    optimized_flow_df : DataFrame
        优化后流量矩阵 (CBGs × POIs)
    cbg_income_dist_dict : dict
        CBG收入分布字典
    income_levels : list
        收入水平列表
    """
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
    
    # 创建收入水平颜色映射
    cmap = LinearSegmentedColormap.from_list('income_cmap', ['#2166ac', '#67a9cf', '#fddbc7', '#b2182b'])
    
    # 计算每个CBG的综合收入水平（使用高收入比例作为代表）
    cbg_income_scores = {}
    for cbg in initial_flow_df.index:
        if cbg in cbg_income_dist_dict:
            # 使用高收入比例作为收入水平的代表
            cbg_income_scores[cbg] = cbg_income_dist_dict[cbg].get('high_income_pct', 0)
        else:
            cbg_income_scores[cbg] = 0
    
    # 创建初始网络并计算固定布局
    G_initial = create_optimization_projection(initial_flow_df)
    fixed_pos = create_income_stratified_layout(G_initial, cbg_income_scores)
    
    # 图1: 优化前网络
    plot_single_optimization_network(initial_flow_df, cbg_income_scores, ax1, cmap, 
                                   node_size, font_size, 'Before Optimization', fixed_pos)
    
    # 图2: 优化后网络  
    plot_single_optimization_network(optimized_flow_df, cbg_income_scores, ax2, cmap,
                                   node_size, font_size, 'After Optimization (100% Budget)', fixed_pos)
    
    # 添加图例
    add_optimization_legend(fig, cmap)
    
    plt.tight_layout()
    return fig

def plot_single_optimization_network(flow_df, cbg_income_scores, ax, cmap, 
                                   node_size, font_size, title, fixed_pos):
    """绘制单个优化网络图"""
    
    # 创建CBG到CBG的投影网络（基于共同访问POI）
    G = create_optimization_projection(flow_df)
    
    if len(G.nodes()) == 0:
        ax.text(0.5, 0.5, 'No Network Connections', ha='center', va='center', 
                transform=ax.transAxes, fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.axis('off')
        return
    
    # 使用固定布局
    pos = fixed_pos
    # 绘制边（先画收入差小的，后画收入差大的，使大差异边覆盖在上面）
    edges = list(G.edges(data=True))
    if edges:
        # 计算边权重并归一化宽度
        edge_weights = [data.get('weight', 1) for _, _, data in edges]
        max_weight = max(edge_weights) if edge_weights else 1
        edge_widths = [w * 2 / max_weight for w in edge_weights]  # 归一化边宽度

        # 收集每条边的额外信息：income_diff, edge tuple, color, width
        edge_infos = []
        for (u, v, data), width in zip(edges, edge_widths):
            income_u = cbg_income_scores.get(u, 0.5)
            income_v = cbg_income_scores.get(v, 0.5)
            income_diff = abs(income_u - income_v)
            # # 根据收入差决定颜色（>0.3 为大差异）
            if income_diff > 0.3:
                color = '#e74c3c'  # 红色 - 高收入差异
            else:
                color = '#3498db'  # 蓝色 - 低收入差异
            # 根据收入差决定颜色（>0.3 为大差异）
            # if income_diff > 0.3:
            #     color = '#C76B6B'  # 红色 - 高收入差异
            # else:
            #     color = '#4C78A8'  # 蓝色 - 低收入差异
            edge_infos.append((income_diff, (u, v), color, width))

        # 按 income_diff 升序排序（小差异先画）
        edge_infos.sort(key=lambda x: x[0])

        # 准备绘制参数（保持绘制顺序）
        edgelist = [info[1] for info in edge_infos]
        edge_colors = [info[2] for info in edge_infos]
        edge_widths_sorted = [info[3] for info in edge_infos]

        nx.draw_networkx_edges(G, pos, edgelist=edgelist,
                          width=edge_widths_sorted,
                          alpha=0.7,
                          edge_color=edge_colors,
                          ax=ax)

    
    # # 绘制边
    # edges = list(G.edges(data=True))
    # if edges:
    #     edge_weights = [data['weight'] for _, _, data in edges]
    #     max_weight = max(edge_weights) if edge_weights else 1
    #     edge_widths = [w *2/ max_weight for w in edge_weights]  # 归一化边宽度
        
    #     # 根据连接节点的收入差异给边上色
    #     edge_colors = []
    #     for u, v, data in edges:
    #         income_u = cbg_income_scores.get(u, 0.5)
    #         income_v = cbg_income_scores.get(v, 0.5)
    #         income_diff = abs(income_u - income_v)
    #         #edge_colors.append('gray')
            
    #         # 收入差异大的边用红色，小的用蓝色
    #         if income_diff > 0.3:
    #             edge_colors.append('#e74c3c')  # 红色 - 高收入差异
    #         # elif income_diff > 0.15:
    #         #     edge_colors.append('#f39c12')  # 橙色 - 中等收入差异
    #         else:
    #             edge_colors.append('#3498db')  # 蓝色 - 低收入差异
        
    #     nx.draw_networkx_edges(G, pos, edgelist=edges,
    #                           width=edge_widths,
    #                           alpha=0.7,
    #                           edge_color=edge_colors,
    #                           ax=ax)
    
    # 绘制节点 - 根据收入水平着色
    node_colors = [cmap(cbg_income_scores[node]) for node in G.nodes()]
    # 使用离散颜色表为节点着色（替代原来的 cmap 渐变映射）
    #node_colors = [income_to_color(cbg_income_scores.get(node, 0.0)) for node in G.nodes()]
    nodes = nx.draw_networkx_nodes(G, pos, 
                                  node_size=node_size,
                                  node_color=node_colors,
                                  edgecolors='black',
                                  linewidths=0.8,
                                  ax=ax)
    
    # 设置图形属性
    ax.set_title(title, fontsize=14,pad=20) 
    ax.axis('off')
    
    # 添加网络指标文本框
    metrics = calculate_optimization_metrics(G, flow_df, cbg_income_scores)
    textstr = '\n'.join([f'{k}: {v:.2f}' for k, v in metrics.items()])
    props = dict(boxstyle='round', facecolor='lightblue', alpha=0.8)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=props)
    
    # 添加收入分层指示
    #ax.text(0.02, 0.02, "Inner: Low Income\nOuter: High Income", transform=ax.transAxes, fontsize=8, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))
    ax.set_aspect('equal', adjustable='box')

def create_income_stratified_layout(G, cbg_income_scores):
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
        income = cbg_income_scores.get(node, 0)
        if income < 0.25:
            low_income_nodes.append(node)
        elif income < 0.5:
            medium_low_nodes.append(node)
        elif income < 0.75:
            medium_high_nodes.append(node)
        else:
            high_income_nodes.append(node)
    
    # 定义各收入层的半径
    radii = {
        'low': 0.1,        # 最内圈
        'medium_low': 0.3, # 内圈
        'medium_high': 0.6, # 中圈
        'high': 0.9        # 外圈
    }
    
    # 放置低收入节点在最内圈
    angle_step = 2 * np.pi / max(1, len(low_income_nodes))
    for i, node in enumerate(low_income_nodes):
        angle = i * angle_step
        pos[node] = (radii['low'] * np.cos(angle), radii['low'] * np.sin(angle))
    
    # 放置中低收入节点在内圈
    angle_step = 2 * np.pi / max(1, len(medium_low_nodes))
    for i, node in enumerate(medium_low_nodes):
        angle = i * angle_step
        pos[node] = (radii['medium_low'] * np.cos(angle), radii['medium_low'] * np.sin(angle))
    
    # 放置中高收入节点在中圈
    angle_step = 2 * np.pi / max(1, len(medium_high_nodes))
    for i, node in enumerate(medium_high_nodes):
        angle = i * angle_step
        pos[node] = (radii['medium_high'] * np.cos(angle), radii['medium_high'] * np.sin(angle))
    
    # 放置高收入节点在外圈
    angle_step = 2 * np.pi / max(1, len(high_income_nodes))
    for i, node in enumerate(high_income_nodes):
        angle = i * angle_step
        pos[node] = (radii['high'] * np.cos(angle), radii['high'] * np.sin(angle))
    
    # 添加轻微随机扰动，避免节点重叠
    for node in pos:
        pos[node] = (pos[node][0] + np.random.normal(0, 0.02), 
                     pos[node][1] + np.random.normal(0, 0.02))
    
    return pos

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
        cbgs_with_flow = flow_df.index[flow_df[poi] > 0].tolist()
        for i in range(len(cbgs_with_flow)):
            for j in range(i+1, len(cbgs_with_flow)):
                cbg1 = cbgs_with_flow[i]
                cbg2 = cbgs_with_flow[j]
                if G.has_edge(cbg1, cbg2):
                    G[cbg1][cbg2]['weight'] += 1
                else:
                    G.add_edge(cbg1, cbg2, weight=1)
    
    return G

def calculate_optimization_metrics(G, flow_df, cbg_income_scores):
    """计算优化网络的指标"""
    if len(G.nodes()) == 0:
        return {'Nodes': 0, 'Edges': 0, 'Avg Degree': 0, 'Assortativity': 0}
    
    try:
        metrics = {
            #'Nodes': G.number_of_nodes(),
            'Edges': G.number_of_edges(),
            #'Avg Degree': sum(dict(G.degree()).values()) / G.number_of_nodes(),
        }
        
        # 计算同配性（收入水平）
        try:
            # 为网络节点添加收入属性
            for node in G.nodes():
                G.nodes[node]['income'] = cbg_income_scores.get(node, 0.5)
            assortativity = nx.numeric_assortativity_coefficient(G, 'income')
            #metrics['Income Assort.'] = assortativity
        except:
            metrics['Income Assort.'] = 0
            
        # 计算跨收入连接比例
        try:
            cross_income_edges = 0
            total_edges = G.number_of_edges()
            for u, v in G.edges():
                income_u = cbg_income_scores.get(u, 0.5)
                income_v = cbg_income_scores.get(v, 0.5)
                # 如果收入差异大于0.3，认为是跨收入连接
                if abs(income_u - income_v) > 0.3:
                    cross_income_edges += 1
            metrics['Cross Income '] = cross_income_edges / total_edges if total_edges > 0 else 0
        except:
            metrics['Cross Income %'] = 0
            
        # 计算模块化
        try:
            from networkx.algorithms import community
            communities = community.greedy_modularity_communities(G)
            modularity = community.modularity(G, communities)
            #metrics['Modularity'] = modularity
        except:
            metrics['Modularity'] = 0
            
    except Exception as e:
        #metrics = {'Nodes': 0, 'Edges': 0, 'Avg Degree': 0, 'Income Assort.': 0, 'Cross Income %': 0, 'Modularity': 0}
        metrics = {'Edges': 0, 'Cross Income %': 0}
    
    return metrics

# def add_optimization_legend(fig, cmap):
#     """添加优化网络图例"""
#     # 收入水平图例
#     income_labels = ['Low Income', 'Middle-Low', 'Middle-High', 'High Income']
#     income_handles = []
    
#     for i, label in enumerate(income_labels):
#         color = cmap(i / 3.0)
#         patch = mpatches.Patch(color=color, label=label)
#         income_handles.append(patch)
    
#     # 边颜色图例
#     # edge_labels = ['Low Income Diff', 'Medium Income Diff', 'High Income Diff']
#     # edge_colors = ['#3498db', '#f39c12', '#e74c3c']
#     edge_labels = ['Low Income Diff', 'High Income Diff']
#     edge_colors = ['#3498db', '#e74c3c']
#     edge_handles = []
    
#     for color, label in zip(edge_colors, edge_labels):
#         patch = mpatches.Patch(color=color, label=label)
#         edge_handles.append(patch)
    
#     # 组合图例
#     all_handles = income_handles + edge_handles
    
#     fig.legend(handles=all_handles, loc='lower center', ncol=3, 
#               bbox_to_anchor=(0.5, -0.01), fontsize=10, frameon=True,
#               fancybox=True, shadow=True)

def add_optimization_legend(fig, cmap):
    """添加优化网络图例（节点用点表示，边用线段表示）"""
    from matplotlib.lines import Line2D

    # 收入水平图例（用原点表示）
    income_labels = ['Low Income', 'Middle-Low', 'Middle-High', 'High Income']
    income_handles = []
    
    for i, label in enumerate(income_labels):
        color = cmap(i / 3.0)
        # 使用 Line2D 画点作为图例项（不画线）
        marker_handle = Line2D([0], [0],
                               marker='o',
                               color='w',  # 线颜色设为白色以隐藏线段
                               markerfacecolor=color,
                               markeredgecolor='black',
                               markersize=10,
                               linestyle='None')
        income_handles.append(marker_handle)
    
    # 边颜色图例（用线段表示）
    edge_labels = ['Low Income Diff', 'High Income Diff']
    edge_colors = ['#3498db', '#e74c3c']
    # edge_colors = ["#4C78A8", "#C76B6B"]   # blue / red
    edge_handles = []
    
    for color, label in zip(edge_colors, edge_labels):
        # 使用 Line2D 画线段作为图例项
        line_handle = Line2D([0], [0],
                             color=color,
                             lw=3,
                             solid_capstyle='round')
        edge_handles.append(line_handle)
    
    # 组合图例（保持原来的布局）
    all_handles = income_handles + edge_handles
    all_labels = income_labels + edge_labels
    
    fig.legend(handles=all_handles, labels=all_labels,
               loc='lower center', ncol=3,
               bbox_to_anchor=(0.5, -0.01), fontsize=10,
               frameon=True, fancybox=True, shadow=True)


# 使用示例 - 加载您的优化结果
city = 'boston'
category = 'Other Individual and Family Services'

# 加载CBG收入分布
cbg_income_dist_df = pd.read_csv(f'matrices_A_D_S_Distribution/cbg_income_level_distribution_{city}_msa.csv', dtype={'GEOID': np.int64})
cbg_income_dist_dict = cbg_income_dist_df.set_index('GEOID').to_dict(orient='index')
income_levels = ['low_income_pct', 'lower_middle_income_pct', 'upper_middle_income_pct', 'high_income_pct']

# 加载流量矩阵
cat_dir = f'matrices_A_D_S_Distribution/{category.replace(" ", "_")}'
flow_matrix = pd.read_csv(f'{cat_dir}/flow_matrix.csv', index_col=0)

# 选取POI和CBG（与您优化代码中相同的逻辑）
poi_total_flow = flow_matrix.sum(axis=0)
poi_num = flow_matrix.shape[1]
selected_pois = poi_total_flow.sort_values(ascending=False).head(poi_num).index.tolist()

selected_cbgs = set()
for poi in selected_pois:
    cbgs_with_flow = flow_matrix.index[flow_matrix[poi] > 0].tolist()
    selected_cbgs.update(cbgs_with_flow)
selected_cbgs = list(selected_cbgs)


# 加载优化结果
try:
    with open('k_matrices_boston_family_budget.pkl', 'rb') as f:
        k_matrices = pickle.load(f)
    
    # 获取100% budget优化结果
    optimized_flow_df = k_matrices.get(1.0)

except FileNotFoundError:
    print("优化结果文件未找到，使用随机数据演示")
    # 创建演示数据
    np.random.seed(42)
    optimized_flow_df = initial_flow_df.copy()
    # 模拟优化效果：增加跨收入群体的连接
    for i in range(min(20, len(selected_cbgs))):
        for j in range(min(15, len(selected_pois))):
            if np.random.random() < 0.3:
                optimized_flow_df.iloc[i, j] += np.random.exponential(0.5)


# 1. 选前 10 大流量的 POI
top10_pois = poi_total_flow.sort_values(ascending=False).head(44).index

# 2. 在这些 POI 上，每个 CBG 的总流出量
cbg_flow_on_top10 = flow_matrix.loc[:, top10_pois].sum(axis=1)

# 3. 选前 10 大流量的 CBG
top10_cbgs = cbg_flow_on_top10.sort_values(ascending=False).head(100).index

# 4. 10×10 初始流量矩阵
initial_flow_df = flow_matrix.loc[top10_cbgs, top10_pois]

# 获取初始流量矩阵
#initial_flow_df = flow_matrix.loc[selected_cbgs, selected_pois]


optimized_flow_df = optimized_flow_df.loc[top10_cbgs, top10_pois]

# 绘制网络对比图
fig = plot_optimization_networks(initial_flow_df, optimized_flow_df, 
                               cbg_income_dist_dict, income_levels,
                               figsize=(5, 10), node_size=120, font_size=9)
plt.savefig('figure3c.pdf',
            format='pdf',
            dpi=300,
            bbox_inches='tight',
            transparent=False,
            backend='pdf')
plt.show()

# 1. 选前 10 大流量的 POI
top10_pois = poi_total_flow.sort_values(ascending=False).head(44).index

# 2. 在这些 POI 上，每个 CBG 的总流出量
cbg_flow_on_top10 = flow_matrix.loc[:, top10_pois].sum(axis=1)

# 3. 选前 10 大流量的 CBG
top10_cbgs = cbg_flow_on_top10.sort_values(ascending=False).head(100).index

# 4. 10×10 初始流量矩阵
initial_flow_df = flow_matrix.loc[top10_cbgs, top10_pois]

# 获取初始流量矩阵
#initial_flow_df = flow_matrix.loc[selected_cbgs, selected_pois]


# 打印一些统计信息
print("\n网络统计信息:")
print(f"初始网络 - CBG数量: {len(initial_flow_df.index)}, POI数量: {len(initial_flow_df.columns)}")
print(f"优化网络 - CBG数量: {len(optimized_flow_df.index)}, POI数量: {len(optimized_flow_df.columns)}")
print(f"总流量变化: {optimized_flow_df.sum().sum() - initial_flow_df.sum().sum():.2f}")