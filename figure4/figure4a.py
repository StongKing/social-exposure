# -*- coding: utf-8 -*-
"""
Figure 4a 独立绘图程序

只读取预处理程序生成的 CBG 层面聚合 CSV，

"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr


# ============================================================
# 用户可修改参数
# ============================================================

aggregated_dir = Path("figure4a_aggregated_data")

dpi = 300
figsize_per_panel = (5, 5)

output_pdf = "figure4a.pdf"
output_statistics_csv = "figure4a_statistics.csv"


# ============================================================
# 类别名称：严格保持原始程序
# ============================================================

naics_map = {
    "624190": "Other Individual and Family Services",
    "711310": "Promoters of Performing Arts, Sports, and Similar Events with Facilities",
    "712110": "Museums",
    "713940": "Fitness and Recreational Sports Centers",
    "722410": "Drinking Places (Alcoholic Beverages)",
    "813110": "Religious Organizations (Catholic)",
}

def significance_stars(p):
    if np.isnan(p):
        return ""
    elif p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    else:
        return ""

# ============================================================
# 创建图形：严格保持原始布局
# ============================================================

categories = list(naics_map.items())

ncols = 3
nrows = 2

fig, axes = plt.subplots(
    nrows=nrows,
    ncols=ncols,
    figsize=(
        figsize_per_panel[0] * ncols,
        figsize_per_panel[1] * nrows
    ),
    dpi=dpi
)

axes = axes.flatten()

summary = []


# ============================================================
# 逐类别读取聚合 CSV 并绘图
# ============================================================

for idx, (naics, category) in enumerate(categories):

    ax = axes[idx]

    csv_path = (
        aggregated_dir /
        f"{naics}_aggregated_cbg_metrics.csv"
    )

    record = {
        "naics": naics,
        "category": category,
        "n_cbgs": np.nan,
        "missing_income_count": np.nan,
        "spearman_rho": np.nan,
        "spearman_p": np.nan,
        "note": "",
    }

    try:
        if not csv_path.exists():
            raise FileNotFoundError(
                f"Missing aggregated CSV: {csv_path}"
            )

        # ----------------------------------------------------
        # 读取聚合结果
        # ----------------------------------------------------

        df = pd.read_csv(csv_path)

        required_columns = [
            "total_outflow",
            "avg_exposure_cbg",
            "high_income_pct",
        ]

        missing_columns = [
            column
            for column in required_columns
            if column not in df.columns
        ]

        if missing_columns:
            raise ValueError(
                f"Missing required columns: {missing_columns}"
            )

        # 转换为数值
        for column in required_columns:
            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            )

        df = df.replace(
            [np.inf, -np.inf],
            np.nan
        )

        missing_income_mask = df["high_income_pct"].isna()

        record["n_cbgs"] = len(df)
        record["missing_income_count"] = int(
            missing_income_mask.sum()
        )

        # 与原始程序一致：
        # 仅排除 high_income_pct 缺失的记录
        df_nonmissing = df[
            ~missing_income_mask
        ].copy()

        # 同时排除暴露值无效的极少数情况
        df_nonmissing = df_nonmissing.dropna(
            subset=["avg_exposure_cbg"]
        )

        # ----------------------------------------------------
        # Spearman 相关系数
        # ----------------------------------------------------

        if len(df_nonmissing) >= 3:

            x = df_nonmissing[
                "high_income_pct"
            ].astype(float)

            y = df_nonmissing[
                "avg_exposure_cbg"
            ].astype(float)

            spearman_res = spearmanr(
                x,
                y,
                nan_policy="omit"
            )

            record["spearman_rho"] = float(
                spearman_res.correlation
            )

            record["spearman_p"] = float(
                spearman_res.pvalue
            )

        else:
            record["note"] = (
                "Too few non-missing obs for Spearman"
            )

        # ----------------------------------------------------
        # 绘图：完全复制原始 regplot 参数
        # ----------------------------------------------------

        if len(df_nonmissing) >= 1:

            sns.regplot(
                x="high_income_pct",
                y="avg_exposure_cbg",
                data=df_nonmissing,
                lowess=False,
                scatter=True,
                scatter_kws={
                    "s": 8,
                    "alpha": 0.6,
                    "color": "#3498db",
                },
                line_kws={
                    "color": "#e74c3f",
                    "linewidth": 3,
                },
                ax=ax
            )

        else:
            ax.text(
                0.5,
                0.5,
                "No valid non-missing data",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=10
            )

        # ----------------------------------------------------
        # 坐标轴和标题：完全保持原始形式
        # ----------------------------------------------------

        ax.set_xlabel("High income share")

        ax.set_ylabel(
            "Baseline flow-weighted social exposure"
        )

        title = category

        if not np.isnan(record["spearman_rho"]):
            stars = significance_stars(record["spearman_p"])

            title += (
                f'\nSpearman ρ='
                f'{record["spearman_rho"]:.3f}'
                f'{stars}'
                )

        if record["note"]:
            title += f'\n{record["note"]}'

        ax.set_title(
            title,
            fontsize=9
        )

        ax.grid(
            True,
            linewidth=0.25,
            alpha=0.5
        )

    except Exception as error:

        err_txt = (
            "Error loading / processing\n"
            f"{str(error)}"
        )

        ax.text(
            0.5,
            0.5,
            err_txt,
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=9,
            wrap=True
        )

        ax.set_title(
            f"{category}\n(errored)",
            fontsize=9
        )

        ax.set_xticks([])
        ax.set_yticks([])

        record["note"] = f"Error: {str(error)}"

    summary.append(record)


# ============================================================
# 关闭多余面板
# ============================================================

for j in range(
    len(categories),
    nrows * ncols
):
    axes[j].axis("off")


# ============================================================
# 保存统计结果
# ============================================================

summary_df = pd.DataFrame(summary)

summary_df.to_csv(
    output_statistics_csv,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 保存图片：严格保持原始设置
# ============================================================

plt.tight_layout()

plt.savefig(
    output_pdf,
    format="pdf",
    dpi=300,
    bbox_inches="tight",
    transparent=False,
    backend="pdf"
)

plt.show()

print(f"Figure saved: {output_pdf}")
print(f"Statistics saved: {output_statistics_csv}")