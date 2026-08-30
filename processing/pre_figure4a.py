# -*- coding: utf-8 -*-
"""
Figure 4a 数据预处理程序

功能：
1. 读取各类 POI 的原始流量矩阵和社会暴露矩阵；
2. 将 POI 维度聚合到 CBG 层面；
3. 计算每个 CBG 的：
   - 总出发量 total_outflow
   - 基于流量加权的平均社会暴露 avg_exposure_cbg
   - 高收入人口占比 high_income_pct
4. 每类 POI 单独保存一个聚合 CSV；
5. 不输出原始 GEOID，仅输出匿名 cbg_record_id。

生成的聚合 CSV 可与绘图程序一起上传，
绘图程序不再需要访问任何原始流量矩阵或社会暴露矩阵。
"""

from pathlib import Path
import numpy as np
import pandas as pd


# ============================================================
# 用户可修改参数
# ============================================================

CITY = "boston"

# 原始数据根目录
BASE_DIR = Path("matrices_A_D_S_Distribution")

# 聚合结果保存目录
OUTPUT_DIR = Path("figure4a_aggregated_data")

# 收入分布文件
INCOME_CSV = BASE_DIR / f"cbg_income_level_distribution_{CITY}_msa.csv"


# ============================================================
# NAICS 与类别目录名称
# ============================================================

NAICS_MAP = {
    "624190": "Other Individual and Family Services",
    "711310": "Promoters of Performing Arts, Sports, and Similar Events with Facilities",
    "712110": "Museums",
    "713940": "Fitness and Recreational Sports Centers",
    "722410": "Drinking Places (Alcoholic Beverages)",
    "813110": "Religious_Organizations_catholic",
}


# ============================================================
# GEOID 匹配函数
# ============================================================

def geoid_candidates(value):
    """
    为一个 GEOID 生成若干可能的字符串形式。

    用于处理以下常见差异：
    1. 字符串和整数格式不同；
    2. CSV 读取后出现 '.0'；
    3. 前导零被删除。
    """
    if pd.isna(value):
        return []

    raw = str(value).strip()

    if not raw:
        return []

    candidates = [raw]

    # 处理类似 250250001001.0 的形式
    if raw.endswith(".0"):
        candidates.append(raw[:-2])

    # 去掉前导零后的形式
    no_leading_zero = raw.lstrip("0")
    if no_leading_zero:
        candidates.append(no_leading_zero)

    # 尝试转为整数后再转回字符串
    try:
        integer_form = str(int(float(raw)))
        candidates.append(integer_form)
    except (ValueError, TypeError, OverflowError):
        pass

    # 保持顺序并去重
    return list(dict.fromkeys(candidates))


def build_income_lookup(income_df):
    """
    建立 GEOID -> high_income_pct 的容错查询字典。
    """
    lookup = {}

    for _, row in income_df.iterrows():
        geoid = row["GEOID"]
        high_income_pct = pd.to_numeric(
            row["high_income_pct"],
            errors="coerce"
        )

        if pd.isna(high_income_pct):
            continue

        for key in geoid_candidates(geoid):
            lookup[key] = float(high_income_pct)

    return lookup


def get_high_income_pct(geoid, income_lookup):
    """
    根据 GEOID 查询 high_income_pct。
    若无法匹配，则返回 NaN。
    """
    for key in geoid_candidates(geoid):
        if key in income_lookup:
            return income_lookup[key]

    return np.nan


# ============================================================
# 矩阵标签标准化
# ============================================================

def standardize_matrix_labels(df):
    """
    将矩阵行名和列名统一转换为去除首尾空格的字符串。
    """
    df = df.copy()
    df.index = df.index.astype(str).str.strip()
    df.columns = df.columns.astype(str).str.strip()
    return df


def convert_matrix_to_numeric(df):
    """
    将矩阵值转换为数值，无法转换的值设为 0。
    """
    return df.apply(pd.to_numeric, errors="coerce").fillna(0.0)


# ============================================================
# 主程序
# ============================================================

def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # 读取收入分布
    # --------------------------------------------------------

    if not INCOME_CSV.exists():
        raise FileNotFoundError(
            f"未找到收入分布文件：{INCOME_CSV}"
        )

    income_df = pd.read_csv(
        INCOME_CSV,
        dtype={"GEOID": str}
    )

    required_income_columns = {"GEOID", "high_income_pct"}

    missing_income_columns = required_income_columns.difference(
        income_df.columns
    )

    if missing_income_columns:
        raise ValueError(
            "收入分布文件缺少必要字段："
            f"{sorted(missing_income_columns)}"
        )

    income_lookup = build_income_lookup(income_df)

    print(
        f"已读取收入分布文件：{INCOME_CSV}\n"
        f"原始记录数：{len(income_df):,}\n"
        f"可匹配收入记录键数量：{len(income_lookup):,}"
    )

    # 用于记录所有类别的处理结果
    summary_records = []

    # --------------------------------------------------------
    # 逐类别处理
    # --------------------------------------------------------

    for naics, category in NAICS_MAP.items():

        category_dir = BASE_DIR / category.replace(" ", "_")

        flow_path = category_dir / "flow_matrix.csv"
        social_path = category_dir / "social_exposure_matrix.csv"

        output_path = (
            OUTPUT_DIR /
            f"{naics}_aggregated_cbg_metrics.csv"
        )

        summary_record = {
            "naics": naics,
            "category": category,
            "flow_path": str(flow_path),
            "social_path": str(social_path),
            "output_file": str(output_path),
            "n_original_cbgs": np.nan,
            "n_selected_cbgs": np.nan,
            "n_pois": np.nan,
            "missing_income_count": np.nan,
            "total_flow": np.nan,
            "status": "failed",
            "note": "",
        }

        print("\n" + "=" * 80)
        print(f"正在处理：{naics} - {category}")
        print("=" * 80)

        try:
            # ------------------------------------------------
            # 检查输入文件
            # ------------------------------------------------

            if not flow_path.exists():
                raise FileNotFoundError(
                    f"未找到流量矩阵：{flow_path}"
                )

            if not social_path.exists():
                raise FileNotFoundError(
                    f"未找到社会暴露矩阵：{social_path}"
                )

            # ------------------------------------------------
            # 读取矩阵
            # ------------------------------------------------

            flow_matrix = pd.read_csv(
                flow_path,
                index_col=0
            )

            social_exposure_matrix = pd.read_csv(
                social_path,
                index_col=0
            )

            flow_matrix = standardize_matrix_labels(flow_matrix)
            social_exposure_matrix = standardize_matrix_labels(
                social_exposure_matrix
            )

            flow_matrix = convert_matrix_to_numeric(flow_matrix)
            social_exposure_matrix = convert_matrix_to_numeric(
                social_exposure_matrix
            )

            if flow_matrix.empty:
                raise ValueError("流量矩阵为空。")

            summary_record["n_original_cbgs"] = flow_matrix.shape[0]
            summary_record["n_pois"] = flow_matrix.shape[1]

            # ------------------------------------------------
            # 保留全部 POI，并按照总流量降序排列
            # ------------------------------------------------

            poi_total_flow = flow_matrix.sum(axis=0)

            selected_pois = (
                poi_total_flow
                .sort_values(ascending=False)
                .index
                .tolist()
            )

            # ------------------------------------------------
            # 仅保留至少存在一次正流量的 CBG
            # ------------------------------------------------

            positive_flow_mask = (
                flow_matrix[selected_pois] > 0
            ).any(axis=1)

            selected_cbgs = flow_matrix.index[
                positive_flow_mask
            ].tolist()

            if not selected_cbgs:
                raise ValueError(
                    "该类别没有任何总出发量大于 0 的 CBG。"
                )

            # ------------------------------------------------
            # 构建并对齐流量矩阵 A 和暴露矩阵 S
            # ------------------------------------------------

            A = flow_matrix.loc[
                selected_cbgs,
                selected_pois
            ].copy()

            S = social_exposure_matrix.reindex(
                index=selected_cbgs,
                columns=selected_pois
            ).fillna(0.0)

            # ------------------------------------------------
            # 计算 CBG 层面的聚合指标
            # ------------------------------------------------

            total_outflow = A.sum(axis=1)

            weighted_exposure_sum = (A * S).sum(axis=1)

            avg_exposure_cbg = (
                weighted_exposure_sum
                / total_outflow.replace(0, np.nan)
            )

            high_income_pct = [
                get_high_income_pct(
                    geoid=geoid,
                    income_lookup=income_lookup
                )
                for geoid in selected_cbgs
            ]

            # ------------------------------------------------
            # 构建可上传的聚合数据
            #
            # 不保存原始 GEOID，避免暴露原始空间标识。
            # ------------------------------------------------

            aggregated_df = pd.DataFrame({
                "cbg_record_id": np.arange(
                    1,
                    len(selected_cbgs) + 1
                ),
                "naics": naics,
                "category": category,
                "total_outflow": total_outflow.to_numpy(
                    dtype=float
                ),
                "avg_exposure_cbg": avg_exposure_cbg.to_numpy(
                    dtype=float
                ),
                "high_income_pct": np.asarray(
                    high_income_pct,
                    dtype=float
                ),
            })

            # 无穷值统一转为缺失值
            aggregated_df = aggregated_df.replace(
                [np.inf, -np.inf],
                np.nan
            )

            # ------------------------------------------------
            # 保存当前类别
            # ------------------------------------------------

            aggregated_df.to_csv(
                output_path,
                index=False,
                encoding="utf-8-sig",
                float_format="%.10f"
            )

            missing_income_count = int(
                aggregated_df["high_income_pct"]
                .isna()
                .sum()
            )

            summary_record.update({
                "n_selected_cbgs": len(aggregated_df),
                "missing_income_count": missing_income_count,
                "total_flow": float(
                    aggregated_df["total_outflow"].sum()
                ),
                "status": "success",
                "note": "",
            })

            print(f"聚合结果已保存：{output_path}")
            print(f"CBG 数量：{len(aggregated_df):,}")
            print(
                "缺失 high_income_pct 的 CBG 数量："
                f"{missing_income_count:,}"
            )
            print(
                "总流量："
                f"{aggregated_df['total_outflow'].sum():,.2f}"
            )

        except Exception as error:
            summary_record["note"] = str(error)

            print(f"处理失败：{error}")

        summary_records.append(summary_record)

    # --------------------------------------------------------
    # 保存处理汇总
    # --------------------------------------------------------

    summary_df = pd.DataFrame(summary_records)

    summary_path = OUTPUT_DIR / "figure4a_preprocessing_summary.csv"

    summary_df.to_csv(
        summary_path,
        index=False,
        encoding="utf-8-sig"
    )

    print("\n" + "=" * 80)
    print("全部类别处理结束")
    print("=" * 80)
    print(f"聚合数据目录：{OUTPUT_DIR.resolve()}")
    print(f"处理汇总文件：{summary_path.resolve()}")

    success_count = int(
        (summary_df["status"] == "success").sum()
    )

    print(
        f"成功处理 {success_count}/{len(NAICS_MAP)} 个类别。"
    )


if __name__ == "__main__":
    main()