# -*- coding: utf-8 -*-
"""
Fig. 1c data aggregation script

Purpose
-------
1. Read the original city-level flow, distance, and income-distribution data.
2. Compute city–POI-category metrics.
3. Compute the city-level aggregate used in the figure.
4. Save only the aggregated results to the current working directory.

This script does not produce any figure.

Output
------
fig1c_aggregated_data.csv

The output contains city–POI-category summary statistics only. It does not
contain original CBG-level matrices, GEOIDs, POI-level OD records, or other
record-level source data.
"""

import glob
import os
import warnings

import numpy as np
import pandas as pd


# ============================================================
# 0. Adjustable parameters
# ============================================================

PROJECT_ROOT = r"d:\mobility_social_exposure"

DMAX_KM = 50
DISTANCE_SCALE = 1.0

PRINT_PROGRESS = True
USE_ACTIVE_WITHIN_DMAX_FOR_BASELINE = True

# Save directly to the directory from which this script is run.
OUTPUT_CSV = os.path.join(os.getcwd(), "fig1c_aggregated_data.csv")


# ============================================================
# 1. City, POI, and income settings
# ============================================================

CITY_LIST = [
    "newyork",
    "losangeles",
    "chicago",
    "houston",
    "atlanta",
    "seattle",
    "boston",
    "fresno",
    "baltimore",
    "tulsa",
    "tyler",
    "champaign",
    "billings",
    "sebring",
    "cheyenne",
]

PRETTY_NAMES = {
    "newyork": "New York",
    "losangeles": "Los Angeles",
    "chicago": "Chicago",
    "houston": "Houston",
    "atlanta": "Atlanta",
    "seattle": "Seattle",
    "boston": "Boston",
    "fresno": "Fresno",
    "baltimore": "Baltimore",
    "tulsa": "Tulsa",
    "tyler": "Tyler",
    "champaign": "Champaign",
    "billings": "Billings",
    "sebring": "Sebring",
    "cheyenne": "Cheyenne",
}

INCOME_LEVELS = [
    "low_income_pct",
    "lower_middle_income_pct",
    "upper_middle_income_pct",
    "high_income_pct",
]

POI_NAMES = [
    "Other_Individual_and_Family_Services",
    "Promoters_of_Performing_Arts,_Sports,_and_Similar_Events_with_Facilities",
    "Museums",
    "Fitness_and_Recreational_Sports_Centers",
    "Drinking_Places_(Alcoholic_Beverages)",
    "Religious_Organizations",
]

POI_PRETTY = {
    "Other_Individual_and_Family_Services": "Individual &\nFamily Services",
    "Promoters_of_Performing_Arts,_Sports,_and_Similar_Events_with_Facilities":
        "Performing Arts\nFacilities",
    "Museums": "Museums",
    "Fitness_and_Recreational_Sports_Centers": "Fitness\nCenters",
    "Drinking_Places_(Alcoholic_Beverages)": "Drinking\nPlaces",
    "Religious_Organizations": "Religious\nOrganizations",
}

POI_PRETTY_ONE_LINE = {
    "Other_Individual_and_Family_Services": "Individual & Family Services",
    "Promoters_of_Performing_Arts,_Sports,_and_Similar_Events_with_Facilities":
        "Performing Arts Facilities",
    "Museums": "Museums",
    "Fitness_and_Recreational_Sports_Centers": "Fitness Centers",
    "Drinking_Places_(Alcoholic_Beverages)": "Drinking Places",
    "Religious_Organizations": "Religious Organizations",
}

POI_TO_CODE = {
    "Other_Individual_and_Family_Services": "624190",
    "Promoters_of_Performing_Arts,_Sports,_and_Similar_Events_with_Facilities":
        "711310",
    "Museums": "712110",
    "Fitness_and_Recreational_Sports_Centers": "713940",
    "Drinking_Places_(Alcoholic_Beverages)": "722410",
    "Religious_Organizations": "813110",
}


# ============================================================
# 2. Path and reading functions
# ============================================================

def get_base_dir(city_name):
    return os.path.join(
        PROJECT_ROOT,
        f"matrices_A_D_S_Distribution_{city_name}_core",
    )


def get_poi_dir(city_name, poi_name):
    return os.path.join(get_base_dir(city_name), poi_name)


def normalize_geoid(x):
    """
    Normalize CBG GEOID:
        250250001011.0 -> '250250001011'
    """
    if pd.isna(x):
        return None

    try:
        return str(int(float(x)))
    except (TypeError, ValueError, OverflowError):
        return str(x)


def read_matrix_csv(path):
    df = pd.read_csv(path, header=0, index_col=0)
    df.index = df.index.astype(str).map(normalize_geoid)
    df.columns = df.columns.astype(str)
    df = df.apply(pd.to_numeric, errors="coerce")

    # Prevent duplicate rows after GEOID normalization.
    if df.index.duplicated().any():
        df = df.groupby(level=0).sum()

    return df


def load_city_poi_case(city_name, poi_name):
    poi_dir = get_poi_dir(city_name, poi_name)

    if not os.path.isdir(poi_dir):
        raise FileNotFoundError(f"POI directory not found: {poi_dir}")

    flow_path = os.path.join(poi_dir, "flow_matrix.csv")
    distance_path = os.path.join(poi_dir, "distance_matrix.csv")

    if not os.path.isfile(flow_path):
        raise FileNotFoundError(f"flow_matrix.csv not found: {flow_path}")

    if not os.path.isfile(distance_path):
        raise FileNotFoundError(
            f"distance_matrix.csv not found: {distance_path}"
        )

    flow = read_matrix_csv(flow_path)
    distance = read_matrix_csv(distance_path) * DISTANCE_SCALE

    if PRINT_PROGRESS:
        print(f"\n[LOAD] {city_name} | {poi_name}")
        print(f"  flow    : {flow_path} | shape={flow.shape}")
        print(f"  distance: {distance_path} | shape={distance.shape}")

    return flow, distance


def find_income_file(city_name):
    """
    Find the CBG income-distribution file for one city.
    """
    base_dir = get_base_dir(city_name)

    candidates = [
        os.path.join(
            base_dir,
            f"cbg_income_level_distribution_{city_name}_msa_core.csv",
        )
    ]

    for path in candidates:
        if os.path.isfile(path):
            return path

    patterns = [
        os.path.join(
            PROJECT_ROOT,
            "**",
            f"cbg_income_level_distribution_{city_name}_msa.csv",
        ),
        os.path.join(
            PROJECT_ROOT,
            "**",
            f"cbg_income_level_distribution_{city_name}_core.csv",
        ),
    ]

    for pattern in patterns:
        matches = glob.glob(pattern, recursive=True)
        if matches:
            return matches[0]

    raise FileNotFoundError(
        f"Cannot find income distribution file for {city_name}. "
        "Checked common paths and recursive patterns."
    )


def load_income_distribution(city_name):
    income_path = find_income_file(city_name)

    if PRINT_PROGRESS:
        print(f"[LOAD income] {city_name}: {income_path}")

    df = pd.read_csv(income_path)

    if "GEOID" not in df.columns:
        raise ValueError(
            f"Income file must contain GEOID column: {income_path}"
        )

    missing_cols = [
        column for column in INCOME_LEVELS if column not in df.columns
    ]
    if missing_cols:
        raise ValueError(
            f"Income file missing columns {missing_cols}: {income_path}"
        )

    df["GEOID_str"] = df["GEOID"].apply(normalize_geoid)

    income = df.set_index("GEOID_str")[INCOME_LEVELS].copy()
    income = income.apply(pd.to_numeric, errors="coerce").fillna(0)

    if income.index.duplicated().any():
        income = income.groupby(level=0).mean()

    row_sum = income.sum(axis=1).replace(0, np.nan)
    income = income.div(row_sum, axis=0).fillna(0)

    return income


# ============================================================
# 3. Exposure and alignment functions
# ============================================================

def compute_all_pair_unmasked_exposure(flow, income):
    """
    Estimate each POI's visitor income composition:

        Q_j = sum_i F_ij P_i / sum_i F_ij

    Then compute all-pair potential cross-income exposure:

        S_ij = 1 - dot(P_i, Q_j)
    """
    flow = flow.copy()
    flow.index = flow.index.map(normalize_geoid)

    common_cbgs = sorted(set(flow.index) & set(income.index))
    if not common_cbgs:
        raise ValueError(
            "No common CBGs between flow matrix and income distribution."
        )

    flow = flow.loc[common_cbgs].copy()
    income_aligned = income.loc[common_cbgs, INCOME_LEVELS].copy()

    poi_total_flow = flow.sum(axis=0)
    valid_pois = poi_total_flow[poi_total_flow > 0].index.tolist()
    if not valid_pois:
        raise ValueError("No POI has positive total flow.")

    flow = flow[valid_pois].copy()

    flow_values = flow.values.astype(float)
    income_values = income_aligned.values.astype(float)
    poi_total_flow_values = flow_values.sum(axis=0)

    poi_income_composition = (
        flow_values.T @ income_values
    ) / poi_total_flow_values[:, None]

    composition_sum = poi_income_composition.sum(axis=1, keepdims=True)
    poi_income_composition = np.divide(
        poi_income_composition,
        composition_sum,
        out=np.zeros_like(poi_income_composition),
        where=composition_sum > 0,
    )

    exposure_values = 1.0 - (
        income_values @ poi_income_composition.T
    )

    exposure = pd.DataFrame(
        exposure_values,
        index=flow.index,
        columns=flow.columns,
    )

    return exposure, flow


def align_flow_distance_exposure(flow, distance, exposure):
    flow = flow.copy()
    distance = distance.copy()
    exposure = exposure.copy()

    flow.index = flow.index.map(normalize_geoid)
    distance.index = distance.index.map(normalize_geoid)
    exposure.index = exposure.index.map(normalize_geoid)

    common_rows = sorted(
        set(flow.index) & set(distance.index) & set(exposure.index)
    )
    common_columns = sorted(
        set(flow.columns) & set(distance.columns) & set(exposure.columns)
    )

    if not common_rows:
        raise ValueError("No common CBG rows among F, D, and S.")

    if not common_columns:
        raise ValueError("No common POI columns among F, D, and S.")

    return (
        flow.loc[common_rows, common_columns].copy(),
        distance.loc[common_rows, common_columns].copy(),
        exposure.loc[common_rows, common_columns].copy(),
    )


def weighted_mean(values, weights):
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)

    mask = (
        np.isfinite(values)
        & np.isfinite(weights)
        & (weights > 0)
    )

    if mask.sum() == 0:
        return np.nan

    return np.sum(values[mask] * weights[mask]) / np.sum(weights[mask])


# ============================================================
# 4. City–POI-category statistic
# ============================================================

def summarize_relative_opportunity_city_poi(
    city_name,
    poi_name,
    income,
):
    """
    Compute the share of unused feasible links satisfying:

        delta exposure > 0
        delta distance <= 0

    Active reference baselines are flow-weighted means.
    """
    flow_raw, distance_raw = load_city_poi_case(city_name, poi_name)

    exposure_all, flow_income = compute_all_pair_unmasked_exposure(
        flow_raw,
        income,
    )

    flow, distance, exposure = align_flow_distance_exposure(
        flow_income,
        distance_raw,
        exposure_all,
    )

    flow_values = flow.values.astype(float)
    distance_values = distance.values.astype(float)
    exposure_values = exposure.values.astype(float)

    valid = (
        np.isfinite(flow_values)
        & np.isfinite(distance_values)
        & np.isfinite(exposure_values)
    )
    distance_feasible = (
        valid
        & (distance_values >= 0)
        & (distance_values <= DMAX_KM)
    )

    active_all = valid & (flow_values > 0)

    if USE_ACTIVE_WITHIN_DMAX_FOR_BASELINE:
        active_reference = active_all & distance_feasible
    else:
        active_reference = active_all

    unused_feasible = distance_feasible & (flow_values <= 0)

    n_all_valid_pairs = int(valid.sum())
    n_feasible_pairs = int(distance_feasible.sum())
    n_active_reference = int(active_reference.sum())
    n_unused_feasible = int(unused_feasible.sum())

    total_active_reference_flow = float(
        np.nansum(flow_values[active_reference])
    )
    total_all_active_flow = float(
        np.nansum(flow_values[active_all])
    )

    if n_active_reference == 0 or total_active_reference_flow <= 0:
        raise ValueError(
            "No active reference links or no positive active reference flow."
        )

    if n_unused_feasible == 0:
        raise ValueError("No unused feasible links.")

    active_weighted_exposure = weighted_mean(
        exposure_values[active_reference],
        flow_values[active_reference],
    )
    active_weighted_distance = weighted_mean(
        distance_values[active_reference],
        flow_values[active_reference],
    )

    if (
        not np.isfinite(active_weighted_exposure)
        or not np.isfinite(active_weighted_distance)
    ):
        raise ValueError(
            "Invalid active weighted exposure or distance baseline."
        )

    delta_exposure_unused = (
        exposure_values[unused_feasible] - active_weighted_exposure
    )
    delta_distance_unused = (
        distance_values[unused_feasible] - active_weighted_distance
    )

    higher_exposure = delta_exposure_unused > 0
    not_farther = delta_distance_unused <= 0
    second_quadrant = higher_exposure & not_farther

    return {
        "city": city_name,
        "city_label": PRETTY_NAMES.get(city_name, city_name.title()),
        "poi_category": poi_name,
        "poi_label": POI_PRETTY.get(poi_name, poi_name),
        "poi_label_one_line": POI_PRETTY_ONE_LINE.get(
            poi_name,
            poi_name,
        ),
        "poi_code": POI_TO_CODE.get(poi_name, poi_name),
        "n_all_valid_pairs": n_all_valid_pairs,
        "n_feasible_pairs": n_feasible_pairs,
        "n_active_ref": n_active_reference,
        "n_unused_feasible": n_unused_feasible,
        "total_active_ref_flow": total_active_reference_flow,
        "total_all_active_flow": total_all_active_flow,
        "active_weighted_exposure_ref": active_weighted_exposure,
        "active_weighted_distance_ref": active_weighted_distance,
        "share_higher_exposure": float(np.mean(higher_exposure)),
        "share_not_farther": float(np.mean(not_farther)),
        "share_second_quadrant": float(np.mean(second_quadrant)),
        "mean_delta_exposure": float(
            np.nanmean(delta_exposure_unused)
        ),
        "median_delta_exposure": float(
            np.nanmedian(delta_exposure_unused)
        ),
        "mean_delta_distance": float(
            np.nanmean(delta_distance_unused)
        ),
        "median_delta_distance": float(
            np.nanmedian(delta_distance_unused)
        ),
    }


# ============================================================
# 5. City-level aggregate
# ============================================================

def unused_link_weighted_city_aggregate(group):
    """
    Reproduce the effective aggregation used by the original script.

    The city aggregate is weighted by the number of unused feasible links
    in each POI category:

        sum_c share_c * n_unused_c / sum_c n_unused_c
    """
    shares = group["share_second_quadrant"].to_numpy(dtype=float)
    weights = group["n_unused_feasible"].to_numpy(dtype=float)

    mask = (
        np.isfinite(shares)
        & np.isfinite(weights)
        & (weights > 0)
    )

    if mask.sum() == 0:
        city_share = np.nan
    else:
        city_share = (
            np.sum(shares[mask] * weights[mask])
            / np.sum(weights[mask])
        )

    return pd.Series({
        "city_aggregate_share": city_share,
        "city_n_categories": int(mask.sum()),
        "city_n_all_valid_pairs": int(
            group["n_all_valid_pairs"].sum()
        ),
        "city_n_feasible_pairs": int(
            group["n_feasible_pairs"].sum()
        ),
        "city_n_active_ref": int(group["n_active_ref"].sum()),
        "city_n_unused_feasible": int(
            group["n_unused_feasible"].sum()
        ),
        "city_total_active_ref_flow": float(
            group["total_active_ref_flow"].sum()
        ),
        "city_total_all_active_flow": float(
            group["total_all_active_flow"].sum()
        ),
    })


# ============================================================
# 6. Main workflow
# ============================================================

def main():
    summary_rows = []
    income_cache = {}

    for city_name in CITY_LIST:
        base_dir = get_base_dir(city_name)

        if not os.path.isdir(base_dir):
            warnings.warn(
                f"City directory not found, skipped: {base_dir}"
            )
            continue

        try:
            if city_name not in income_cache:
                income_cache[city_name] = load_income_distribution(
                    city_name
                )
            income = income_cache[city_name]
        except Exception as exc:
            warnings.warn(
                f"Skipped city {city_name}: "
                f"cannot load income file: {exc}"
            )
            continue

        for poi_name in POI_NAMES:
            try:
                row = summarize_relative_opportunity_city_poi(
                    city_name=city_name,
                    poi_name=poi_name,
                    income=income,
                )
                summary_rows.append(row)

                if PRINT_PROGRESS:
                    print(
                        "  second-quadrant share="
                        f"{row['share_second_quadrant'] * 100:.3f}% | "
                        "delta S > 0="
                        f"{row['share_higher_exposure'] * 100:.3f}% | "
                        "delta D <= 0="
                        f"{row['share_not_farther'] * 100:.3f}% | "
                        "active flow="
                        f"{row['total_active_ref_flow']:.1f}"
                    )

            except Exception as exc:
                warnings.warn(
                    f"Skipped {city_name} | {poi_name}: {exc}"
                )

    if not summary_rows:
        raise RuntimeError(
            "No city-category case was successfully loaded. "
            "Check PROJECT_ROOT, income files, and matrix folders."
        )

    summary = pd.DataFrame(summary_rows)

    summary["city"] = pd.Categorical(
        summary["city"],
        categories=CITY_LIST,
        ordered=True,
    )
    summary["poi_category"] = pd.Categorical(
        summary["poi_category"],
        categories=POI_NAMES,
        ordered=True,
    )

    summary = (
        summary
        .sort_values(["city", "poi_category"])
        .reset_index(drop=True)
    )

    city_summary = (
        summary
        .groupby(["city", "city_label"], observed=True)
        .apply(unused_link_weighted_city_aggregate)
        .reset_index()
    )

    city_summary["city"] = pd.Categorical(
        city_summary["city"],
        categories=CITY_LIST,
        ordered=True,
    )
    city_summary = (
        city_summary
        .sort_values("city")
        .reset_index(drop=True)
    )

    # Convert categorical merge keys back to strings for a clean CSV.
    summary["city"] = summary["city"].astype(str)
    summary["poi_category"] = summary["poi_category"].astype(str)
    city_summary["city"] = city_summary["city"].astype(str)

    aggregated = summary.merge(
        city_summary,
        on=["city", "city_label"],
        how="left",
        validate="many_to_one",
    )

    # Record the aggregation rule explicitly in the released table.
    aggregated["city_aggregate_weight"] = "n_unused_feasible"

    aggregated.to_csv(
        OUTPUT_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    print("\n========== Aggregation completed ==========")
    print(f"Saved file: {os.path.abspath(OUTPUT_CSV)}")
    print(f"Rows: {len(aggregated)}")
    print(
        f"Cities: {aggregated['city'].nunique()} | "
        f"city-category cases: {len(aggregated)}"
    )

    display_columns = [
        "city_label",
        "city_n_categories",
        "city_n_unused_feasible",
        "city_total_active_ref_flow",
        "city_aggregate_share",
    ]
    print(
        city_summary[display_columns]
        .assign(
            city_aggregate_share_pct=lambda frame:
                frame["city_aggregate_share"] * 100
        )
        .drop(columns=["city_aggregate_share"])
        .round(3)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
