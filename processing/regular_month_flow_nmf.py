# -*- coding: utf-8 -*-
"""
Created on Sat Nov  8 16:03:45 2025

@author: JZS
"""

# -*- coding: utf-8 -*-
"""
生成基于观测流量的月度参考矩阵 R
思路：
1) 直接对 observed flow matrix 做 NMF，提取 CBG--POI 潜在亲和力；
2) 将 NMF 亲和力、距离接近度和收入相似度作为目的地选择特征；
3) 在给定每个 CBG 总访问量的条件下，用条件多项式似然估计特征系数；
4) 用 softmax 得到目的地选择概率，并生成浮点期望流量矩阵；
5) 使用 largest-remainder 方法确定性整数化，严格保持每个 CBG 行和；
6) 保存浮点矩阵、整数矩阵、估计系数和拟合诊断。
"""
import os
import json
from math import radians, sin, cos, atan2, sqrt
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from sklearn.decomposition import NMF
from scipy.optimize import minimize
from scipy.special import logsumexp
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------- 可调参数 -----------------
DATA_DIR = '.'  # 数据所在目录（shapefile 和 csv）
OUTPUT_DIR = 'matrices_A_D_S_Distribution'
NAICS_MAP = {'624190':  'Other Individual and Family Services',}
NMF_COMPONENTS = 10          
NMF_MAX_ITER = 2000
NMF_TOL = 1e-5
RIDGE_LAMBDA = 1e-6
USE_INCOME_SIMILARITY = True
INCOME_COEF_NONNEGATIVE = True
RNG_SEED = 42               # 仅用于 NMF 初始化
PLOT_TOP = 100              # 可视化时选取 top N CBG/POI
# ------------------------------------------------


# ----------------- 公共函数 -----------------
def haversine(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    return 6371.0 * (2 * atan2(sqrt(a), sqrt(1-a)))

def parse_visitor_home_cbgs(s):
    if pd.isna(s):
        return {}
    try:
        return json.loads(s)
    except Exception:
        return {}

def get_income_distribution(visitor_dict, income_dist_dict, income_levels):
    visitor_income_dist = np.zeros(len(income_levels), dtype=float)
    total_visits = 0.0
    for cbg, visits in (visitor_dict or {}).items():
        if cbg in income_dist_dict:
            cbg_dist = np.array([income_dist_dict[cbg][lvl] for lvl in income_levels], dtype=float)
            visitor_income_dist += cbg_dist * float(visits)
            total_visits += float(visits)
    if total_visits > 0:
        visitor_income_dist /= total_visits
    return visitor_income_dist


def standardize_feature(x, eps=1e-12):
    """全矩阵标准化，改善条件多项式估计的数值稳定性。"""
    x = np.asarray(x, dtype=float)
    finite = np.isfinite(x)
    if not finite.any():
        raise ValueError("Feature contains no finite values.")
    mean = float(x[finite].mean())
    std = float(x[finite].std())
    if std < eps:
        return np.zeros_like(x), mean, std
    z = (x - mean) / std
    z[~finite] = 0.0
    return z, mean, std


def fit_conditional_multinomial(flow, feature_arrays, feature_names,
                                bounds, ridge_lambda=1e-6):
    """
    F_i· | O_i ~ Multinomial(O_i, p_i·)
    p_ij = softmax(sum_r theta_r x_ijr)
    """
    F = np.asarray(flow, dtype=float)
    X = np.stack(feature_arrays, axis=2).astype(float)
    row_totals = F.sum(axis=1)
    valid_rows = row_totals > 0

    Fv = F[valid_rows]
    Xv = X[valid_rows]
    Ov = row_totals[valid_rows]

    def objective(theta):
        utility = np.einsum("ijk,k->ij", Xv, theta)
        log_denom = logsumexp(utility, axis=1)
        log_likelihood = np.sum(Fv * utility) - np.dot(Ov, log_denom)

        probabilities = np.exp(utility - log_denom[:, None])
        residual = Fv - Ov[:, None] * probabilities
        gradient_ll = np.einsum("ij,ijk->k", residual, Xv)

        penalty = 0.5 * ridge_lambda * float(np.dot(theta, theta))
        negative_ll = -log_likelihood + penalty
        gradient = -gradient_ll + ridge_lambda * theta
        return negative_ll, gradient

    x0 = np.zeros(len(feature_names), dtype=float)
    if len(x0) > 0:
        x0[0] = 1.0
    if len(x0) > 1:
        x0[1] = 1.0

    result = minimize(
        fun=lambda theta: objective(theta)[0],
        x0=x0,
        jac=lambda theta: objective(theta)[1],
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 2000, "ftol": 1e-12, "gtol": 1e-8},
    )
    if not result.success:
        raise RuntimeError(
            "Conditional multinomial estimation failed: "
            f"{result.message}"
        )

    theta = result.x
    utility_all = np.einsum("ijk,k->ij", X, theta)
    log_probability_all = utility_all - logsumexp(
        utility_all, axis=1, keepdims=True
    )
    probability_all = np.exp(log_probability_all)
    probability_all[~valid_rows, :] = 0.0

    fitted_ll = -objective(theta)[0] + 0.5 * ridge_lambda * float(np.dot(theta, theta))
    n_choices = F.shape[1]
    null_ll = float(
        np.sum(row_totals[valid_rows]) * (-np.log(max(n_choices, 1)))
    )
    k = len(theta)

    diagnostics = {
        "converged": bool(result.success),
        "message": str(result.message),
        "conditional_log_likelihood": float(fitted_ll),
        "uniform_null_log_likelihood": null_ll,
        "mcfadden_pseudo_r2": float(1.0 - fitted_ll / null_ll) if null_ll != 0 else np.nan,
        "AIC": float(2 * k - 2 * fitted_ll),
        "number_of_parameters": int(k),
        "number_of_origins": int(valid_rows.sum()),
        "number_of_destinations": int(F.shape[1]),
        "total_visits": float(row_totals.sum()),
    }

    coef_df = pd.DataFrame({
        "feature": feature_names,
        "coefficient": theta,
    })
    return probability_all, coef_df, diagnostics


def largest_remainder_integerize(expected_flow, row_totals):
    """确定性整数化并严格保持每个 origin 的整数行和。"""
    expected = np.asarray(expected_flow, dtype=float)
    target_totals = np.rint(np.asarray(row_totals, dtype=float)).astype(int)

    result = np.floor(np.maximum(expected, 0.0) + 1e-12).astype(int)

    for i in range(expected.shape[0]):
        need = int(target_totals[i] - result[i].sum())
        if need > 0:
            fractional = expected[i] - np.floor(expected[i])
            order = np.argsort(-fractional, kind="mergesort")
            result[i, order[:need]] += 1
        elif need < 0:
            removable = np.where(result[i] > 0)[0]
            fractional = expected[i, removable] - np.floor(expected[i, removable])
            order = removable[np.argsort(fractional, kind="mergesort")]
            result[i, order[:(-need)]] -= 1

    if not np.array_equal(result.sum(axis=1), target_totals):
        raise RuntimeError("Integerization failed to preserve row totals.")
    return result

# ----------------- 读取数据（与你的原始流程保持一致） -----------------
# CBG shapefile -> 用于 centroid 经纬度
boston_msa_cbg = gpd.read_file(os.path.join(DATA_DIR, 'geo_data/tl_2021_boston_msa_bg/tl_2021_boston_msa_bg.shp')).to_crs('EPSG:4326')
boston_msa_cbg['centroid'] = boston_msa_cbg.geometry.centroid
boston_msa_cbg['lon'] = boston_msa_cbg['centroid'].x
boston_msa_cbg['lat'] = boston_msa_cbg['centroid'].y

# POI / visits / income
poi_attr = pd.read_csv(os.path.join(DATA_DIR, 'poi_boston_msa_all.csv'), dtype={'GEOID': str}, low_memory=False)
poi_visits = pd.read_csv(os.path.join(DATA_DIR, 'filtered_boston_msa_all.csv'), low_memory=False)
income_dist_df = pd.read_csv(os.path.join(DATA_DIR, 'cbg_income_level_distribution_boston_msa.csv'), dtype={'GEOID': str})
income_dist_dict = income_dist_df.set_index('GEOID').to_dict(orient='index')

# NAICS filter
sorted_prefixes = sorted(NAICS_MAP.keys(), key=lambda x: -len(x))
def match_naics(code):
    code_str = str(code)
    for prefix in sorted_prefixes:
        if code_str.startswith(prefix):
            return prefix
    return None

poi_attr['naics_prefix'] = poi_attr['naics_code'].apply(match_naics)
valid_geoids = set(income_dist_df['GEOID'])

poi_filtered = poi_attr[poi_attr['naics_prefix'].notna()].copy()
poi_filtered = poi_filtered[poi_filtered['GEOID'].isin(valid_geoids)]

poi = pd.merge(poi_filtered, poi_visits, on='placekey', how='left')
poi['visitor_home_cbgs_dict'] = poi['visitor_home_cbgs'].apply(parse_visitor_home_cbgs)
geometry = [Point(xy) for xy in zip(poi.longitude, poi.latitude)]
poi_gdf = gpd.GeoDataFrame(poi, crs='EPSG:4326', geometry=geometry)

cbg_coords = boston_msa_cbg.set_index('GEOID')[['lon', 'lat']].to_dict(orient='index')
income_levels = ['low_income_pct', 'lower_middle_income_pct', 'upper_middle_income_pct', 'high_income_pct']

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ----------------- 主流程：对每个 NAICS 类别处理 -----------------
for prefix, cname in NAICS_MAP.items():
    print(f"Processing category {cname} ({prefix}) ...")
    poi_cat = poi_gdf[poi_gdf['naics_prefix'] == prefix].copy()
    poi_cat = poi_cat[poi_cat['visits_by_day'].notna()].copy()
    if poi_cat.empty:
        print("No POIs for this category, skip.")
        continue

    # 列表与矩阵初始化（使用字符串形式的索引）
    all_pois = poi_cat['placekey'].astype(str).tolist()
    all_cbgs = sorted(list(valid_geoids))  # 字符串 GEOID
    # flow_matrix: 月度观测（CBG x POI）
    flow_matrix = pd.DataFrame(0.0, index=all_cbgs, columns=all_pois)
    distance_matrix = pd.DataFrame(1e6, index=all_cbgs, columns=all_pois)  # 默认大距离
    income_matrix = pd.DataFrame(0.0, index=all_cbgs, columns=all_pois)

    # 填充 flow_matrix 和距离 / 收入相似度
    for poi_key in all_pois:
        rows = poi_cat[poi_cat['placekey'].astype(str) == str(poi_key)]
        if rows.empty:
            continue
        row = rows.iloc[0]
        visitor_dict = row['visitor_home_cbgs_dict'] or {}
        poi_lon = float(row['longitude'])
        poi_lat = float(row['latitude'])
        Q_j = get_income_distribution(visitor_dict, income_dist_dict, income_levels)

        # 填距离 & income similarity（向量化会更快，但这里清晰可靠）
        for cbg in all_cbgs:
            if cbg in cbg_coords:
                cbg_lon = cbg_coords[cbg]['lon']; cbg_lat = cbg_coords[cbg]['lat']
                distance_matrix.at[cbg, poi_key] = haversine(cbg_lon, cbg_lat, poi_lon, poi_lat)
            else:
                distance_matrix.at[cbg, poi_key] = 1e6
            if cbg in income_dist_dict:
                P_i = np.array([income_dist_dict[cbg][lvl] for lvl in income_levels], dtype=float)
                denom = (np.linalg.norm(P_i) * np.linalg.norm(Q_j)) + 1e-12
                inc_sim = float(np.dot(P_i, Q_j) / denom) if denom > 0 else 0.0
                income_matrix.at[cbg, poi_key] = inc_sim
            else:
                income_matrix.at[cbg, poi_key] = 0.0

        # 填充观测月度流量（visitor_home_cbgs）
        for cbg, v in (visitor_dict or {}).items():
            cbg_s = str(cbg)
            if cbg_s in flow_matrix.index:
                try:
                    flow_matrix.at[cbg_s, poi_key] = float(v)
                except Exception:
                    flow_matrix.at[cbg_s, poi_key] = 0.0

    # 1) 根据 POI 总流量排序
    poi_total_flow = flow_matrix.sum(axis=0)
    poi_num = poi_total_flow.size
    selected_pois = poi_total_flow.sort_values(ascending=False).head(poi_num).index.tolist()

    
    # 2) 收集访问这些 POI 的所有 CBG（将 index 中满足 flow>0 的项转 int，再汇总）
    selected_cbgs = set()
    for poi in selected_pois:
        # note: flow_matrix.index[flow_matrix[poi] > 0] 返回 index labels (strings)
        cbgs_with_flow = [int(cbg) for cbg in flow_matrix.index[flow_matrix[poi] > 0]]
        selected_cbgs.update(cbgs_with_flow)
    selected_cbgs = list(selected_cbgs)

    # 3) 将索引回退为 string 并尝试对齐 by string
    flow_matrix.index = flow_matrix.index.astype(str)
    distance_matrix.index = distance_matrix.index.astype(str)
    income_matrix.index = income_matrix.index.astype(str)
    selected_cbgs = [str(x) for x in selected_cbgs]

    # 生成子矩阵（selected_cbgs × selected_pois）
    flow_matrix = flow_matrix.loc[selected_cbgs, selected_pois].astype(float)
    distance_matrix = distance_matrix.loc[selected_cbgs, selected_pois].astype(float)
    income_matrix = income_matrix.loc[selected_cbgs, selected_pois].astype(float)

    # ----------------- NMF + 条件多项式目的地选择 -----------------
    m, n = flow_matrix.shape
    print(f"子矩阵维度: {m} CBG x {n} POI。")

    base = flow_matrix.values.astype(float)
    distance_values = distance_matrix.values.astype(float)
    income_values = income_matrix.values.astype(float)
    eps = 1e-10

    # 1) 直接对观测计数矩阵做 KL-NMF。
    n_components = min(NMF_COMPONENTS, m, n)
    print(f"拟合 KL-NMF：rank={n_components} ...")
    nmf_model = NMF(
        n_components=n_components,
        init="nndsvda",
        solver="mu",
        beta_loss="kullback-leibler",
        random_state=RNG_SEED,
        max_iter=NMF_MAX_ITER,
        tol=NMF_TOL,
    )
    W = nmf_model.fit_transform(base)
    H_nmf = nmf_model.components_
    recon = np.maximum(W @ H_nmf, 0.0)

    # 2) 构造目的地选择特征。
    nmf_affinity_raw = np.log(recon + eps)
    distance_closeness_raw = -np.log1p(distance_values)

    nmf_affinity, nmf_mean, nmf_std = standardize_feature(
        nmf_affinity_raw
    )
    distance_closeness, dist_mean, dist_std = standardize_feature(
        distance_closeness_raw
    )

    feature_arrays = [nmf_affinity, distance_closeness]
    feature_names = ["log_nmf_affinity", "distance_closeness"]
    bounds = [(0.0, None), (0.0, None)]

    scaling_records = [
        {"feature": "log_nmf_affinity", "mean": nmf_mean, "std": nmf_std},
        {"feature": "distance_closeness", "mean": dist_mean, "std": dist_std},
    ]

    if USE_INCOME_SIMILARITY:
        income_similarity, inc_mean, inc_std = standardize_feature(
            income_values
        )
        feature_arrays.append(income_similarity)
        feature_names.append("income_similarity")
        bounds.append(
            (0.0, None) if INCOME_COEF_NONNEGATIVE else (None, None)
        )
        scaling_records.append({
            "feature": "income_similarity",
            "mean": inc_mean,
            "std": inc_std,
        })

    # 3) 条件于每个 CBG 总访问量，用聚合访问计数直接估计系数。
    probabilities, coefficient_df, fit_diagnostics = fit_conditional_multinomial(
        flow=base,
        feature_arrays=feature_arrays,
        feature_names=feature_names,
        bounds=bounds,
        ridge_lambda=RIDGE_LAMBDA,
    )

    print("条件多项式模型估计系数：")
    print(coefficient_df.to_string(index=False))
    print("拟合诊断：")
    for key, value in fit_diagnostics.items():
        print(f"  {key}: {value}")

    # 4) 生成浮点期望流量 R_float_ij = O_i * p_ij。
    orig_row_sum_1d = base.sum(axis=1)
    pred_rownorm = orig_row_sum_1d[:, None] * probabilities
    pred_rownorm_df = pd.DataFrame(
        pred_rownorm,
        index=flow_matrix.index,
        columns=flow_matrix.columns,
    )

    # 5) largest-remainder 确定性整数化。
    print("使用 largest-remainder 方法进行确定性整数化 ...")
    pred_rownorm_int_array = largest_remainder_integerize(
        pred_rownorm,
        orig_row_sum_1d,
    )
    pred_rownorm_int = pd.DataFrame(
        pred_rownorm_int_array,
        index=flow_matrix.index,
        columns=flow_matrix.columns,
        dtype=int,
    )

    float_row_error = np.max(
        np.abs(pred_rownorm_df.sum(axis=1).values - orig_row_sum_1d)
    )
    int_row_error = np.max(
        np.abs(
            pred_rownorm_int.sum(axis=1).values
            - np.rint(orig_row_sum_1d).astype(int)
        )
    )
    print(f"浮点矩阵最大行和误差: {float_row_error:.6e}")
    print(f"整数矩阵最大行和误差: {int_row_error:d}")


    # ----------------- 保存结果 -----------------
    out_cat_dir = os.path.join(OUTPUT_DIR, cname.replace(' ', '_'))
    os.makedirs(out_cat_dir, exist_ok=True)

    pred_rownorm_df.to_csv(
        os.path.join(out_cat_dir, 'pred_rownorm_float_conditional_mnl.csv')
    )
    # 保留原文件名，避免下游优化脚本改路径。
    pred_rownorm_int.to_csv(
        os.path.join(out_cat_dir, 'pred_rownorm_int_preserve.csv')
    )
    coefficient_df.to_csv(
        os.path.join(out_cat_dir, 'conditional_mnl_coefficients.csv'),
        index=False,
    )
    pd.DataFrame(scaling_records).to_csv(
        os.path.join(out_cat_dir, 'conditional_mnl_feature_scaling.csv'),
        index=False,
    )
    pd.DataFrame([fit_diagnostics]).to_csv(
        os.path.join(out_cat_dir, 'conditional_mnl_fit_diagnostics.csv'),
        index=False,
    )

    print("已保存以下文件到目录:", out_cat_dir)
    print("  - pred_rownorm_float_conditional_mnl.csv")
    print("  - pred_rownorm_int_preserve.csv")
    print("  - conditional_mnl_coefficients.csv")
    print("  - conditional_mnl_feature_scaling.csv")
    print("  - conditional_mnl_fit_diagnostics.csv")


    # ----------------- 可视化比较：原始 vs 两种预测 -----------------
    print("生成可视化图表 ...")
    # 选 top 若干 POI / CBG 以便画图
    top_pois = flow_matrix.sum(axis=0).sort_values(ascending=False).head(poi_num).index.tolist()
    top_cbgs = flow_matrix.sum(axis=1).sort_values(ascending=False).head(PLOT_TOP).index.tolist()

    # 1) 热力图：原始 - 预测（保行和版）
    diff_preserve = flow_matrix.loc[top_cbgs, top_pois] - pred_rownorm_int.loc[top_cbgs, top_pois]
    plt.figure(figsize=(20,15))
    sns.heatmap(diff_preserve.fillna(0), cmap='vlag', center=0)
    plt.title('原始 - 预测（保行和整数版）(top CBG x top POI)')
    plt.tight_layout()
    #plt.savefig(os.path.join(out_cat_dir, 'heatmap_orig_minus_pred_preserve_top.png'), dpi=300)
    plt.show()

    
    # 热力图：原始 
    diff_preserve = flow_matrix.loc[top_cbgs, top_pois]
    plt.figure(figsize=(20,15),dpi=300)
    sns.heatmap(diff_preserve.fillna(0), cmap='vlag', center=0)
    plt.title('Original (top CBG x top POI)')
    plt.tight_layout()
    plt.show()

    # 热力图 预测（保行和）
    diff_round = pred_rownorm_int.loc[top_cbgs, top_pois]
    plt.figure(figsize=(20,15),dpi=300)
    sns.heatmap(diff_round.fillna(0), cmap='vlag', center=0)
    plt.title('Predict (top CBG x top POI)')
    plt.tight_layout()
    plt.show()
    

    # 3) POI 总量比较条形图（原始 vs 保行和版 vs 四舍五入版）
    orig_poi_totals = flow_matrix.sum(axis=0)
    pres_poi_totals = pred_rownorm_int.sum(axis=0)
    poi_comp = pd.DataFrame({'orig': orig_poi_totals, 'preserve_int': pres_poi_totals})
    poi_comp = poi_comp.sort_values('orig', ascending=False).head(PLOT_TOP)  # 画前100个 POI
    plt.figure(figsize=(14,6))
    poi_comp.plot(kind='bar')
    plt.title('POI ：Original vs Predict')
    plt.tight_layout()
    #plt.savefig(os.path.join(out_cat_dir, 'poi_totals_orig_vs_pred.png'), dpi=300)
    plt.show()
    
    flow_o = flow_matrix.copy()
    flow_o_str = flow_o.copy()
    flow_o_str.index = flow_o_str.index.map(lambda x: str(x))
    flow_o_str.columns = flow_o_str.columns.map(lambda x: str(x))

    flow_p = pred_rownorm_int.copy()
    flow_p_str = flow_p.copy()
    flow_p_str.index = flow_p_str.index.map(lambda x: str(x))
    flow_p_str.columns = flow_p_str.columns.map(lambda x: str(x))

    # 把传入的 top 列表都转换为 str 用于匹配（但保留原 top 供参考）
    top_cbgs = flow_matrix.sum(axis=1).sort_values(ascending=False).head(flow_matrix.shape[0]).index.tolist()
    top_pois = flow_matrix.sum(axis=0).sort_values(ascending=False).head(flow_matrix.shape[1]).index.tolist()
    cbgs_all = [str(x) for x in top_cbgs]
    pois_all = [str(x) for x in top_pois]

    # 优先选择同时存在于两个矩阵的节点（保证左右两个子图节点一致）
    cbgs_present = [c for c in cbgs_all if (c in flow_o_str.index) and (c in flow_p_str.index)]
    pois_present = [p for p in pois_all if (p in flow_o_str.columns) and (p in flow_p_str.columns)]

    # 位置布置（按 cbgs_present / pois_present 顺序）
    n_cbgs = len(cbgs_present)
    n_pois = len(pois_present)
    x_cbgs = np.linspace(0.03, 0.97, n_cbgs)
    x_pois = np.linspace(0.03, 0.97, n_pois)
    y_cbgs = np.ones_like(x_cbgs) * 1.0
    y_pois = np.zeros_like(x_pois) * 0.0

    # 收集边（兼容单列/单行返回为标量的情况）
    edges_o = []  # 原始矩阵边 (i, j, value)
    for i, cbg in enumerate(cbgs_present):
        row = None
        try:
            row = flow_o_str.loc[cbg, pois_present]
        except Exception:
            # 兼容：如果 pois_present 只有一个元素或定位返回标量
            try:
                row = flow_o_str.loc[cbg, pois_present[0]]
            except Exception:
                row = None
        if row is None:
            continue
        row_vals = np.atleast_1d(np.array(row, dtype=float))
        for j, v in enumerate(row_vals):
            if np.isnan(v):
                continue
            if v > 0:
                edges_o.append((i, j, float(v)))

    edges_p = []  # 预测（保行和整数）矩阵的边
    for i, cbg in enumerate(cbgs_present):
        row = None
        try:
            row = flow_p_str.loc[cbg, pois_present]
        except Exception:
            try:
                row = flow_p_str.loc[cbg, pois_present[0]]
            except Exception:
                row = None
        if row is None:
            continue
        row_vals = np.atleast_1d(np.array(row, dtype=float))
        for j, v in enumerate(row_vals):
            if np.isnan(v):
                continue
            if v > 0:
                edges_p.append((i, j, float(v)))

    print(f"DEBUG: 原始矩阵正流量边数 = {len(edges_o)}")
    print(f"DEBUG: 预测矩阵正流量边数 = {len(edges_p)}")


    # 控制绘制密度：最多绘制多少条边（可调整）
    max_edges = 40000
    if len(edges_o) > 0:
        edges_o.sort(key=lambda x: x[2], reverse=True)
        edges_o = edges_o[:min(len(edges_o), max_edges)]
    if len(edges_p) > 0:
        edges_p.sort(key=lambda x: x[2], reverse=True)
        edges_p = edges_p[:min(len(edges_p), max_edges)]

    # 将边按值映射到线宽（各自单独映射）
    def map_widths(edges, min_lw=0.2, max_lw=10.0):
        if len(edges) == 0:
            return np.array([])
        vals = np.array([e[2] for e in edges], dtype=float)
        vmin, vmax = vals.min(), vals.max()
        if vmax == vmin:
            return np.full_like(vals, (min_lw + max_lw) / 2.0, dtype=float)
        return min_lw + (vals - vmin) / (vmax - vmin) * (max_lw - min_lw)

    widths_o = map_widths(edges_o)
    widths_p = map_widths(edges_p)

    # 绘图：两个子图并排
    fig, axes = plt.subplots(1, 2, figsize=(24, 10), dpi=300)
    ax0, ax1 = axes
    # --- 统一浅蓝色配色 ---
    COLOR = '#87CEFA'  # 浅蓝色（可换成其它 hex，如 '#6fb3ff'）

    # 绘制边到 axis（修正后的贝塞尔表达式 + 固定颜色）
    def draw_edges_to_ax(ax, edges, widths, x_cbgs, y_cbgs, x_pois, y_pois, edge_alpha=0.7, color=COLOR):
        if len(edges) == 0:
            ax.text(0.5, 0.5, 'No positive flows to plot', ha='center', va='center', fontsize=12, color=color)
            return
        # 为了保证细线先画、粗线在上，按值升序画
        vals = np.array([e[2] for e in edges], dtype=float)
        order = np.argsort(vals)
        t = np.linspace(0.0, 1.0, 120)
        for idx in order:
            i, j, v = edges[idx]
            lw = float(widths[idx])
            x0, y0 = float(x_cbgs[i]), float(y_cbgs[i])
            x1, y1 = float(x_pois[j]), float(y_pois[j])
            xm = (x0 + x1) / 2.0
            dx = x1 - x0
            bend = 0.15 + 0.35 * abs(dx)
            ym = 0.35 + bend * 0.6
            P0 = np.array([x0, y0])
            P1 = np.array([xm, ym])
            P2 = np.array([x1, y1])
            curve = ((1 - t)**2)[:, None] * P0 + 2 * ((1 - t) * t)[:, None] * P1 + (t**2)[:, None] * P2
            ax.plot(curve[:, 0], curve[:, 1], linewidth=lw, color=color, alpha=edge_alpha,
                    solid_capstyle='round', zorder=1)

    # 左图：原始
    ax0.set_title('Original Flow (CBG → POI)', fontsize=20, pad=2)

    ax0.axis('off')
    draw_edges_to_ax(ax0, edges_o, widths_o, x_cbgs, y_cbgs, x_pois, y_pois, edge_alpha=0.7, color=COLOR)
    ax0.scatter(x_cbgs, y_cbgs, s=40, marker='o', alpha=0.95, zorder=5, edgecolors='none')
    ax0.scatter(x_pois, y_pois, s=60, marker='o', alpha=0.95, zorder=5, edgecolors='none')

    # 右图：预测（保行和整数版）
    ax1.set_title('Recommended Flow (CBG → POI)', fontsize=20, pad=2)

    ax1.axis('off')
    draw_edges_to_ax(ax1, edges_p, widths_p, x_cbgs, y_cbgs, x_pois, y_pois, edge_alpha=0.7, color=COLOR)
    ax1.scatter(x_cbgs, y_cbgs, s=40, marker='o', alpha=0.95, zorder=5, edgecolors='none')
    ax1.scatter(x_pois, y_pois, s=60, marker='o', alpha=0.95, zorder=5, edgecolors='none')

    plt.tight_layout()
    plt.show()

print("全部完成。")

# ----------------- 修正后的直接绘图 -----------------

# ---- debug 信息 ----
# print("DEBUG: flow_matrix.index type / sample:", type(flow_matrix.index[0]), list(flow_matrix.index[:5]))
# print("DEBUG: flow_matrix.columns type / sample:", type(flow_matrix.columns[0]), list(flow_matrix.columns[:5]))
# print("DEBUG: pred_rownorm_int.index type / sample:", type(pred_rownorm_int.index[0]), list(pred_rownorm_int.index[:5]))
# print("DEBUG: pred_rownorm_int.columns type / sample:", type(pred_rownorm_int.columns[0]), list(pred_rownorm_int.columns[:5]))
# print("DEBUG: top_cbgs (first 10):", top_cbgs[:10])
# print("DEBUG: top_pois (first 10):", top_pois[:10])

city = 'boston'
category = 'Other Individual and Family Services'
cat_dir = f'matrices_A_D_S_Distribution/{category.replace(" ", "_")}'

R_pre = pd.read_csv(f'{cat_dir}/pred_rownorm_int_preserve.csv', index_col=0)


pred_rownorm_int = R_pre







# 将两个 DataFrame 做字符串视图（不修改原 DF）
flow_o = flow_matrix.copy()
flow_o_str = flow_o.copy()
flow_o_str.index = flow_o_str.index.map(lambda x: str(x))
flow_o_str.columns = flow_o_str.columns.map(lambda x: str(x))

flow_p = pred_rownorm_int.copy()
flow_p_str = flow_p.copy()
flow_p_str.index = flow_p_str.index.map(lambda x: str(x))
flow_p_str.columns = flow_p_str.columns.map(lambda x: str(x))

# 把传入的 top 列表都转换为 str 用于匹配（但保留原 top 供参考）
top_cbgs = flow_matrix.sum(axis=1).sort_values(ascending=False).head(flow_matrix.shape[0]).index.tolist()
top_pois = flow_matrix.sum(axis=0).sort_values(ascending=False).head(flow_matrix.shape[1]).index.tolist()
cbgs_all = [str(x) for x in top_cbgs]
pois_all = [str(x) for x in top_pois]

# 优先选择同时存在于两个矩阵的节点（保证左右两个子图节点一致）
cbgs_present = [c for c in cbgs_all if (c in flow_o_str.index) and (c in flow_p_str.index)]
pois_present = [p for p in pois_all if (p in flow_o_str.columns) and (p in flow_p_str.columns)]

# 位置布置（按 cbgs_present / pois_present 顺序）
n_cbgs = len(cbgs_present)
n_pois = len(pois_present)
x_cbgs = np.linspace(0.03, 0.97, n_cbgs)
x_pois = np.linspace(0.03, 0.97, n_pois)
y_cbgs = np.ones_like(x_cbgs) * 1.0
y_pois = np.zeros_like(x_pois) * 0.0

# 收集边（兼容单列/单行返回为标量的情况）
edges_o = []  # 原始矩阵边 (i, j, value)
for i, cbg in enumerate(cbgs_present):
    row = None
    try:
        row = flow_o_str.loc[cbg, pois_present]
    except Exception:
        # 兼容：如果 pois_present 只有一个元素或定位返回标量
        try:
            row = flow_o_str.loc[cbg, pois_present[0]]
        except Exception:
            row = None
    if row is None:
        continue
    row_vals = np.atleast_1d(np.array(row, dtype=float))
    for j, v in enumerate(row_vals):
        if np.isnan(v):
            continue
        if v > 0:
            edges_o.append((i, j, float(v)))

edges_p = []  # 预测（保行和整数）矩阵的边
for i, cbg in enumerate(cbgs_present):
    row = None
    try:
        row = flow_p_str.loc[cbg, pois_present]
    except Exception:
        try:
            row = flow_p_str.loc[cbg, pois_present[0]]
        except Exception:
            row = None
    if row is None:
        continue
    row_vals = np.atleast_1d(np.array(row, dtype=float))
    for j, v in enumerate(row_vals):
        if np.isnan(v):
            continue
        if v > 0:
            edges_p.append((i, j, float(v)))

print(f"DEBUG: 原始矩阵正流量边数 = {len(edges_o)}")
print(f"DEBUG: 预测矩阵正流量边数 = {len(edges_p)}")


# 控制绘制密度：最多绘制多少条边（可调整）
max_edges = 40000
if len(edges_o) > 0:
    edges_o.sort(key=lambda x: x[2], reverse=True)
    edges_o = edges_o[:min(len(edges_o), max_edges)]
if len(edges_p) > 0:
    edges_p.sort(key=lambda x: x[2], reverse=True)
    edges_p = edges_p[:min(len(edges_p), max_edges)]

# 将边按值映射到线宽（各自单独映射）
def map_widths(edges, min_lw=0.2, max_lw=10.0):
    if len(edges) == 0:
        return np.array([])
    vals = np.array([e[2] for e in edges], dtype=float)
    vmin, vmax = vals.min(), vals.max()
    if vmax == vmin:
        return np.full_like(vals, (min_lw + max_lw) / 2.0, dtype=float)
    return min_lw + (vals - vmin) / (vmax - vmin) * (max_lw - min_lw)

widths_o = map_widths(edges_o)
widths_p = map_widths(edges_p)

# 绘图：两个子图并排
fig, axes = plt.subplots(1, 2, figsize=(24, 10), dpi=300)
ax0, ax1 = axes
# --- 统一浅蓝色配色 ---
COLOR = '#87CEFA'  # 浅蓝色（可换成其它 hex，如 '#6fb3ff'）

colors = ['#3498db', '#e74c3f', '#7c5bb8']

# 绘制边到 axis（修正后的贝塞尔表达式 + 固定颜色）
def draw_edges_to_ax(ax, edges, widths, x_cbgs, y_cbgs, x_pois, y_pois, edge_alpha=0.7, color=COLOR):
    if len(edges) == 0:
        ax.text(0.5, 0.5, 'No positive flows to plot', ha='center', va='center', fontsize=12, color=color)
        return
    # 为了保证细线先画、粗线在上，按值升序画
    vals = np.array([e[2] for e in edges], dtype=float)
    order = np.argsort(vals)
    t = np.linspace(0.0, 1.0, 120)
    for idx in order:
        i, j, v = edges[idx]
        lw = float(widths[idx])
        x0, y0 = float(x_cbgs[i]), float(y_cbgs[i])
        x1, y1 = float(x_pois[j]), float(y_pois[j])
        xm = (x0 + x1) / 2.0
        dx = x1 - x0
        bend = 0.15 + 0.35 * abs(dx)
        ym = 0.35 + bend * 0.6
        P0 = np.array([x0, y0])
        P1 = np.array([xm, ym])
        P2 = np.array([x1, y1])
        curve = ((1 - t)**2)[:, None] * P0 + 2 * ((1 - t) * t)[:, None] * P1 + (t**2)[:, None] * P2
        ax.plot(curve[:, 0], curve[:, 1], linewidth=lw, color=color, alpha=edge_alpha,
                solid_capstyle='round', zorder=1)

# 左图：原始
ax0.set_title('Original Flow (CBG → POI)', fontsize=20, pad=2)

ax0.axis('off')
draw_edges_to_ax(ax0, edges_o, widths_o, x_cbgs, y_cbgs, x_pois, y_pois, edge_alpha=0.7, color=COLOR)
ax0.scatter(x_cbgs, y_cbgs, s=40, marker='o', alpha=0.95, zorder=5, edgecolors='none', color='#3498db')
ax0.scatter(x_pois, y_pois, s=60, marker='o', alpha=0.95, zorder=5, edgecolors='none', color='#e74c3f')

# 右图：预测（保行和整数版）
ax1.set_title('Recommended Flow (CBG → POI)', fontsize=20, pad=2)

ax1.axis('off')
draw_edges_to_ax(ax1, edges_p, widths_p, x_cbgs, y_cbgs, x_pois, y_pois, edge_alpha=0.7, color=COLOR)
ax1.scatter(x_cbgs, y_cbgs, s=40, marker='o', alpha=0.95, zorder=5, edgecolors='none', color='#3498db')
ax1.scatter(x_pois, y_pois, s=60, marker='o', alpha=0.95, zorder=5, edgecolors='none', color='#e74c3f')

plt.tight_layout()
plt.show()
print("DEBUG: plotted bipartite compare with single light-blue color:", COLOR)

