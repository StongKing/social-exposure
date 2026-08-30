# -*- coding: utf-8 -*-
"""
Prepare the released data package used by Fig. 1d.

This script is the only one that reads the confidential original flow
magnitudes and the original poi_boston_msa_all.csv attribute file. It exports:
1. binary flow-presence packages and derived baselines;
2. a minimal POI plotting table containing only ID, coordinates, and category.

@author: JZS
"""

import glob
import os

import numpy as np
import pandas as pd


# ============================================================
# 0. Adjustable parameters
# ============================================================

PROJECT_ROOT = r"d:\mobility_social_exposure"

DMAX_KM = 50
DISTANCE_SCALE = 1.0
USE_ACTIVE_WITHIN_DMAX_FOR_BASELINE = True

MSA_MATRIX_DIR = os.path.join(
    PROJECT_ROOT,
    "matrices_A_D_S_Distribution",
)

# The released binary-flow package is saved directly under the directory
# from which this script is run.
OUTPUT_DIR = os.path.join(
    os.getcwd(),
    "fig1d_binary_flow_package",
)
METADATA_PATH = os.path.join(
    OUTPUT_DIR,
    "category_flow_metadata.csv",
)

# Original POI attribute file. It is read only by this preparation script.
POI_SOURCE_PATH = os.path.join(
    PROJECT_ROOT,
    "poi_boston_msa_all.csv",
)
POI_LOCATION_PATH = os.path.join(
    OUTPUT_DIR,
    "poi_boston_msa_plot_locations.csv",
)
POI_ID_COL = "placekey"

PRINT_PROGRESS = True


# ============================================================
# 1. POI and income settings
# ============================================================

INCOME_LEVELS = [
    "low_income_pct",
    "lower_middle_income_pct",
    "upper_middle_income_pct",
    "high_income_pct",
]

POI_NAMES = [
    "Fitness_and_Recreational_Sports_Centers",
    "Religious_Organizations",
    "Drinking_Places_(Alcoholic_Beverages)",
    "Museums",
    "Promoters_of_Performing_Arts,_Sports,_and_Similar_Events_with_Facilities",
    "Other_Individual_and_Family_Services",
]

POI_TO_CODE = {
    "Fitness_and_Recreational_Sports_Centers": "713940",
    "Religious_Organizations": "813110",
    "Drinking_Places_(Alcoholic_Beverages)": "722410",
    "Museums": "712110",
    "Promoters_of_Performing_Arts,_Sports,_and_Similar_Events_with_Facilities": "711310",
    "Other_Individual_and_Family_Services": "624190",
}

POI_CODES = tuple(POI_TO_CODE.values())


# ============================================================
# 2. Reading and alignment helpers
# ============================================================

def normalize_geoid(value):
    """Normalize values such as 250250001011.0 to '250250001011'."""
    if pd.isna(value):
        return None

    try:
        return str(int(float(value)))
    except (TypeError, ValueError, OverflowError):
        return str(value)


def read_matrix_csv(path):
    matrix = pd.read_csv(path, header=0, index_col=0)
    matrix.index = matrix.index.astype(str).map(normalize_geoid)
    matrix.columns = matrix.columns.astype(str)
    matrix = matrix.apply(pd.to_numeric, errors="coerce")

    if matrix.index.duplicated().any():
        matrix = matrix.groupby(level=0).sum()

    return matrix


def find_income_file_boston_msa():
    candidates = [
        os.path.join(
            MSA_MATRIX_DIR,
            "cbg_income_level_distribution_boston_msa.csv",
        ),
        os.path.join(
            MSA_MATRIX_DIR,
            "cbg_income_level_distribution_boston_core.csv",
        ),
        os.path.join(
            PROJECT_ROOT,
            "cbg_income_level_distribution_boston_msa.csv",
        ),
        os.path.join(
            PROJECT_ROOT,
            "cbg_income_level_distribution_boston_core.csv",
        ),
    ]

    for path in candidates:
        if os.path.isfile(path):
            return path

    patterns = [
        os.path.join(
            PROJECT_ROOT,
            "**",
            "cbg_income_level_distribution_boston_msa.csv",
        ),
        os.path.join(
            PROJECT_ROOT,
            "**",
            "cbg_income_level_distribution_boston_core.csv",
        ),
    ]

    for pattern in patterns:
        matches = glob.glob(pattern, recursive=True)
        if matches:
            return matches[0]

    raise FileNotFoundError(
        "Cannot find the Boston MSA income-distribution file."
    )


def load_income_distribution_boston_msa():
    path = find_income_file_boston_msa()
    income = pd.read_csv(path)

    if "GEOID" not in income.columns:
        raise ValueError(f"Income file must contain GEOID: {path}")

    missing = [
        column for column in INCOME_LEVELS
        if column not in income.columns
    ]
    if missing:
        raise ValueError(
            f"Income file is missing columns {missing}: {path}"
        )

    income["GEOID_str"] = income["GEOID"].apply(normalize_geoid)

    distribution = (
        income
        .set_index("GEOID_str")[INCOME_LEVELS]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0)
    )

    if distribution.index.duplicated().any():
        distribution = distribution.groupby(level=0).mean()

    row_sum = distribution.sum(axis=1).replace(0, np.nan)
    distribution = distribution.div(row_sum, axis=0).fillna(0)

    if PRINT_PROGRESS:
        print(f"[LOAD income] {path}")

    return distribution


def get_poi_dir(poi_name):
    return os.path.join(MSA_MATRIX_DIR, poi_name)


def load_original_case(poi_name):
    poi_dir = get_poi_dir(poi_name)
    flow_path = os.path.join(poi_dir, "flow_matrix.csv")
    distance_path = os.path.join(poi_dir, "distance_matrix.csv")

    if not os.path.isfile(flow_path):
        raise FileNotFoundError(
            f"Original flow matrix not found: {flow_path}"
        )

    if not os.path.isfile(distance_path):
        raise FileNotFoundError(
            f"Distance matrix not found: {distance_path}"
        )

    flow = read_matrix_csv(flow_path)
    distance = read_matrix_csv(distance_path) * DISTANCE_SCALE

    if PRINT_PROGRESS:
        print(f"\n[LOAD] {poi_name}")
        print(f"  original flow: {flow_path} | shape={flow.shape}")
        print(f"  distance     : {distance_path} | shape={distance.shape}")

    return flow, distance


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
# 2A. Prepare a minimal POI location file for plotting
# ============================================================

def match_poi_code(naics_value):
    """Return one of the six retained NAICS codes, or None."""
    if pd.isna(naics_value):
        return None

    code_text = str(naics_value).strip()
    if code_text.endswith(".0"):
        code_text = code_text[:-2]

    for code in sorted(POI_CODES, key=len, reverse=True):
        if code_text.startswith(code):
            return code

    return None


def prepare_poi_plot_locations():
    """
    Extract only the fields required by the plotting program:

        poi_id, longitude, latitude, poi_code

    No POI name, address, brand, visit record, or other attribute is exported.
    The plotting program will clip these points to the public Boston MSA
    boundary before drawing.
    """
    if not os.path.isfile(POI_SOURCE_PATH):
        raise FileNotFoundError(
            f"Original POI attribute file not found: {POI_SOURCE_PATH}"
        )

    header = pd.read_csv(POI_SOURCE_PATH, nrows=0)
    required_columns = {
        POI_ID_COL,
        "longitude",
        "latitude",
        "naics_code",
    }
    missing = sorted(required_columns - set(header.columns))
    if missing:
        raise ValueError(
            "POI source file is missing columns: " + ", ".join(missing)
        )

    poi = pd.read_csv(
        POI_SOURCE_PATH,
        usecols=[POI_ID_COL, "longitude", "latitude", "naics_code"],
        dtype={POI_ID_COL: str, "naics_code": str},
        low_memory=False,
    )

    poi["poi_code"] = poi["naics_code"].apply(match_poi_code)
    poi = poi[poi["poi_code"].notna()].copy()

    poi["longitude"] = pd.to_numeric(poi["longitude"], errors="coerce")
    poi["latitude"] = pd.to_numeric(poi["latitude"], errors="coerce")
    poi["poi_id"] = poi[POI_ID_COL].astype(str).str.strip()

    poi = poi.dropna(subset=["longitude", "latitude"]).copy()
    poi = poi[poi["poi_id"].ne("") & poi["poi_id"].ne("nan")].copy()
    poi = poi.drop_duplicates(subset=["poi_id"], keep="first")

    # Only the four plotting fields are released.
    output = poi[["poi_id", "longitude", "latitude", "poi_code"]].copy()
    output = output.sort_values(
        ["poi_code", "poi_id"],
        kind="mergesort",
    ).reset_index(drop=True)

    output.to_csv(
        POI_LOCATION_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    if PRINT_PROGRESS:
        print("\n[POI location extraction]")
        print(f"  source: {POI_SOURCE_PATH}")
        print(f"  saved : {POI_LOCATION_PATH}")
        print(f"  retained POIs: {len(output):,}")
        print(
            output.groupby("poi_code")
            .size()
            .rename("n_pois")
            .to_string()
        )

    return output


# ============================================================
# 3. Original-flow calculations retained as derived quantities
# ============================================================

def compute_exposure_and_q_from_original_flow(flow, income):
    """
    Calculate the same POI visitor composition and exposure matrix as the
    original Fig. 1d script.
    """
    flow = flow.copy()
    flow.index = flow.index.map(normalize_geoid)

    common_cbgs = sorted(set(flow.index) & set(income.index))
    if not common_cbgs:
        raise ValueError(
            "No common CBGs between flow and income data."
        )

    flow = flow.loc[common_cbgs].copy()
    income_aligned = income.loc[common_cbgs, INCOME_LEVELS].copy()

    poi_total_flow = flow.sum(axis=0)
    valid_pois = poi_total_flow[poi_total_flow > 0].index.tolist()
    if not valid_pois:
        raise ValueError("No POI has positive total flow.")

    flow = flow[valid_pois].copy()

    flow_values = flow.to_numpy(dtype=float)
    income_values = income_aligned.to_numpy(dtype=float)

    # This matches the original code. The source matrices are expected not
    # to contain NaN values in retained cells.
    poi_total_flow_values = flow_values.sum(axis=0)
    q_values = (
        flow_values.T @ income_values
    ) / poi_total_flow_values[:, None]

    q_sum = q_values.sum(axis=1, keepdims=True)
    q_values = np.divide(
        q_values,
        q_sum,
        out=np.zeros_like(q_values),
        where=q_sum > 0,
    )

    exposure_values = 1.0 - (income_values @ q_values.T)

    exposure = pd.DataFrame(
        exposure_values,
        index=flow.index,
        columns=flow.columns,
    )
    q = pd.DataFrame(
        q_values,
        index=flow.columns,
        columns=INCOME_LEVELS,
    )

    return exposure, flow, q


def align_flow_distance_exposure(flow, distance, exposure):
    flow = flow.copy()
    distance = distance.copy()
    exposure = exposure.copy()

    flow.index = flow.index.map(normalize_geoid)
    distance.index = distance.index.map(normalize_geoid)
    exposure.index = exposure.index.map(normalize_geoid)

    common_rows = sorted(
        set(flow.index)
        & set(distance.index)
        & set(exposure.index)
    )
    common_columns = sorted(
        set(flow.columns)
        & set(distance.columns)
        & set(exposure.columns)
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


def prepare_one_category(poi_name, income):
    original_flow, distance_raw = load_original_case(poi_name)

    exposure_all, flow_income, q = (
        compute_exposure_and_q_from_original_flow(
            original_flow,
            income,
        )
    )

    flow, distance, exposure = align_flow_distance_exposure(
        flow_income,
        distance_raw,
        exposure_all,
    )
    q = q.loc[flow.columns, INCOME_LEVELS].copy()

    flow_values = flow.to_numpy(dtype=float)
    distance_values = distance.to_numpy(dtype=float)
    exposure_values = exposure.to_numpy(dtype=float)

    # Keep a separate validity mask so an invalid/missing original cell is
    # not accidentally converted into an ordinary zero-flow cell.
    flow_valid = np.isfinite(flow_values)
    flow_binary = (flow_valid & (flow_values > 0)).astype(np.uint8)

    valid = (
        flow_valid
        & np.isfinite(distance_values)
        & np.isfinite(exposure_values)
    )
    distance_feasible = (
        valid
        & (distance_values >= 0)
        & (distance_values <= DMAX_KM)
    )
    active_all = valid & (flow_binary > 0)

    if USE_ACTIVE_WITHIN_DMAX_FOR_BASELINE:
        active_reference = active_all & distance_feasible
    else:
        active_reference = active_all

    unused_feasible = distance_feasible & (flow_binary == 0)

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
            f"No valid active reference links for {poi_name}."
        )

    if n_unused_feasible == 0:
        raise ValueError(
            f"No unused feasible links for {poi_name}."
        )

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
            f"Invalid original-flow baseline for {poi_name}."
        )

    poi_code = POI_TO_CODE[poi_name]
    package_path = os.path.join(
        OUTPUT_DIR,
        f"{poi_code}_binary_flow_package.npz",
    )

    # Unicode arrays allow allow_pickle=False in the plotting script.
    cbg_ids = np.asarray(flow.index.astype(str), dtype=str)
    poi_ids = np.asarray(flow.columns.astype(str), dtype=str)

    np.savez_compressed(
        package_path,
        flow_binary=flow_binary,
        flow_valid=flow_valid.astype(np.uint8),
        cbg_ids=cbg_ids,
        poi_ids=poi_ids,
        q_values=q.to_numpy(dtype=np.float64),
    )

    if PRINT_PROGRESS:
        original_positive = flow_values[flow_values > 0]
        print(f"  saved: {package_path}")
        print(
            f"  binary active links={int(flow_binary.sum()):,} | "
            f"matrix shape={flow_binary.shape}"
        )
        print(
            "  original positive magnitudes were read only in memory and "
            f"were not exported (count={len(original_positive):,})."
        )

    return {
        "poi_category": poi_name,
        "poi_code": poi_code,
        "package_file": os.path.basename(package_path),
        "n_cbgs": int(flow.shape[0]),
        "n_pois": int(flow.shape[1]),
        "n_valid_pairs": int(valid.sum()),
        "n_active_all": int(active_all.sum()),
        "n_active_ref": n_active_reference,
        "n_unused_feasible": n_unused_feasible,
        "total_active_ref_flow": total_active_reference_flow,
        "total_all_active_flow": total_all_active_flow,
        "active_weighted_exposure_ref": active_weighted_exposure,
        "active_weighted_distance_ref": active_weighted_distance,
        "dmax_km": float(DMAX_KM),
        "distance_scale": float(DISTANCE_SCALE),
        "use_active_within_dmax_for_baseline": bool(
            USE_ACTIVE_WITHIN_DMAX_FOR_BASELINE
        ),
        "flow_encoding": "0=no observed flow; 1=positive observed flow",
    }


# ============================================================
# 4. Main workflow
# ============================================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Export the minimal, non-sensitive POI coordinate table first.
    prepare_poi_plot_locations()

    income = load_income_distribution_boston_msa()
    metadata_rows = []

    for poi_name in POI_NAMES:
        metadata_rows.append(
            prepare_one_category(
                poi_name=poi_name,
                income=income,
            )
        )

    metadata = pd.DataFrame(metadata_rows)
    metadata.to_csv(
        METADATA_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    readme_path = os.path.join(OUTPUT_DIR, "README.txt")
    with open(readme_path, "w", encoding="utf-8") as file:
        file.write(
            "Fig. 1d binary-flow package\n"
            "============================\n\n"
            "Each NPZ file contains:\n"
            "  flow_binary : uint8 matrix; 0=no flow, 1=positive flow\n"
            "  flow_valid  : uint8 matrix; 1=valid original cell\n"
            "  cbg_ids     : matrix row identifiers\n"
            "  poi_ids     : matrix column identifiers\n"
            "  q_values    : derived POI visitor-income composition\n\n"
            "The package does not contain original positive flow magnitudes.\n"
            "category_flow_metadata.csv stores only category-level derived "
            "baselines and totals required to reproduce the original figure.\n\n"
            "poi_boston_msa_plot_locations.csv contains only:\n"
            "  poi_id, longitude, latitude, poi_code\n"
            "It does not contain POI names, addresses, brands, visit records, "
            "or other original POI attributes.\n"
        )

    print("\n========== Binary-flow preparation completed ==========")
    print(f"Output directory: {os.path.abspath(OUTPUT_DIR)}")
    print(f"Metadata: {os.path.abspath(METADATA_PATH)}")
    print(f"POI locations: {os.path.abspath(POI_LOCATION_PATH)}")
    print(metadata[[
        "poi_code",
        "n_cbgs",
        "n_pois",
        "n_active_ref",
        "n_unused_feasible",
        "active_weighted_exposure_ref",
        "active_weighted_distance_ref",
    ]].round(6).to_string(index=False))


if __name__ == "__main__":
    main()