# -*- coding: utf-8 -*-
"""
Prepare aggregated source data for Fig. 2e.

This script reads the original city × POI optimization-result files and
extracts only the aggregated relative changes required to reproduce Fig. 2e.

Input:
    matrices_A_D_S_Distribution_{city}_core/
        {poi_folder}/
            results_{city}_{poi_naics}.csv

Output:
    figure2e_source_data.csv

The output file contains only city × POI-level aggregated results and does
not contain CBG × POI mobility matrices or optimization allocation matrices.
"""

import os
import numpy as np
import pandas as pd


# ============================================================
# 1. CONFIGURATION
# ============================================================

# City order is kept exactly the same as in the original Fig. 2e code.
cities = [
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


# POI mapping is kept exactly the same as in the original Fig. 2e code.
naics_map = {
    "624190": "Other_Individual_and_Family_Services",
    "711310": "Promoters_of_Performing_Arts,_Sports,_and_Similar_Events_with_Facilities",
    "813110": "Religious_Organizations_catholic",
    "712110": "Museums",
    "713940": "Fitness_and_Recreational_Sports_Centers",
    "722410": "Drinking_Places_(Alcoholic_Beverages)",
}


# Original result-file path pattern.
base_pattern = (
    "matrices_A_D_S_Distribution_{city}_core/"
    "{poi_folder}/"
    "results_{city}_{poi_naics}.csv"
)


# Possible column names used in the original files.
col_keys = {
    "distances": [
        "distances_iter",
        "distances",
        "distance",
        "distancesiter",
    ],
    "social": [
        "social_iter",
        "social",
        "socialiter",
    ],
}


# Output file.
OUTPUT_FILE = "figure2e_source_data.csv"


# ============================================================
# 2. HELPER FUNCTIONS
# ============================================================

def find_col(df_cols, candidates):
    """
    Find a matching column name.

    This reproduces the matching logic used in the original Fig. 2e code.
    """
    cols = [str(c).strip().lower() for c in df_cols]

    for cand in candidates:
        cand_low = cand.strip().lower()

        for i, c in enumerate(cols):
            if cand_low == c or cand_low in c or c in cand_low:
                return list(df_cols)[i]

    return None


def calculate_relative_change(df, candidates):
    """
    Extract the first and last non-missing values from the requested column
    and calculate:

        (final - initial) / initial

    This is exactly the quantity used by the original Fig. 2e code.

    Returns
    -------
    relative_change : float
        Relative change, or NaN if unavailable.
    initial_value : float
        Initial value, or NaN.
    final_value : float
        Final value, or NaN.
    matched_column : str or None
        Name of the matched column.
    """

    col = find_col(df.columns, candidates)

    if col is None:
        return np.nan, np.nan, np.nan, None

    values = pd.to_numeric(
        df[col],
        errors="coerce"
    ).dropna().values

    if values.size == 0:
        return np.nan, np.nan, np.nan, col

    initial_value = float(values[0])
    final_value = float(values[-1])

    if initial_value == 0:
        return np.nan, initial_value, final_value, col

    relative_change = (
        final_value - initial_value
    ) / initial_value

    return (
        relative_change,
        initial_value,
        final_value,
        col,
    )


def make_display_name(poi_folder):
    """
    Convert the original folder label to the display label used in Fig. 2e.
    """
    label = poi_folder.replace("_", " ")

    if label == "Religious Organizations catholic":
        label = "Religious Organizations (Catholic)"

    return label


# ============================================================
# 3. EXTRACT FIG. 2e SOURCE DATA
# ============================================================

records = []
missing_files = []
read_errors = []
column_warnings = []


for poi_code, poi_folder in naics_map.items():

    poi_category = make_display_name(poi_folder)

    for city in cities:

        fp = base_pattern.format(
            city=city,
            poi_folder=poi_folder,
            poi_naics=poi_code,
        )

        print("=" * 80)
        print(f"[CITY] {city}")
        print(f"[POI ] {poi_code} | {poi_category}")
        print(f"[FILE] {fp}")

        # ----------------------------------------------------
        # File does not exist
        # ----------------------------------------------------
        if not os.path.exists(fp):

            print("[MISSING] File not found.")

            missing_files.append(
                (city, poi_code, fp)
            )

            # Keep the city × POI combination in the source-data
            # table, but mark the result as unavailable.
            records.append(
                {
                    "city": city,
                    "poi_code": poi_code,
                    "poi_category": poi_category,
                    "social_relative_change": np.nan,
                    "distance_relative_change": np.nan,
                }
            )

            continue

        # ----------------------------------------------------
        # Read file
        # ----------------------------------------------------
        try:
            df = pd.read_csv(fp)

        except Exception as e:

            print(f"[ERROR] Cannot read file: {e}")

            read_errors.append(
                (city, poi_code, fp, str(e))
            )

            records.append(
                {
                    "city": city,
                    "poi_code": poi_code,
                    "poi_category": poi_category,
                    "social_relative_change": np.nan,
                    "distance_relative_change": np.nan,
                }
            )

            continue

        # Strip possible whitespace from column names.
        df.columns = df.columns.map(
            lambda x: str(x).strip()
        )

        # ----------------------------------------------------
        # Social exposure relative change
        # ----------------------------------------------------
        (
            social_change,
            social_initial,
            social_final,
            social_col,
        ) = calculate_relative_change(
            df,
            col_keys["social"],
        )

        # ----------------------------------------------------
        # Distance relative change
        # ----------------------------------------------------
        (
            distance_change,
            distance_initial,
            distance_final,
            distance_col,
        ) = calculate_relative_change(
            df,
            col_keys["distances"],
        )

        # ----------------------------------------------------
        # Diagnostics
        # ----------------------------------------------------
        if social_col is None:

            column_warnings.append(
                (
                    city,
                    poi_code,
                    "social",
                    fp,
                )
            )

            print(
                "[WARNING] Social-exposure column "
                "was not found."
            )

        else:

            print(
                f"[SOCIAL] column={social_col}"
            )
            print(
                f"         initial={social_initial}"
            )
            print(
                f"         final={social_final}"
            )
            print(
                f"         relative change={social_change}"
            )

        if distance_col is None:

            column_warnings.append(
                (
                    city,
                    poi_code,
                    "distance",
                    fp,
                )
            )

            print(
                "[WARNING] Distance column "
                "was not found."
            )

        else:

            print(
                f"[DISTANCE] column={distance_col}"
            )
            print(
                f"           initial={distance_initial}"
            )
            print(
                f"           final={distance_final}"
            )
            print(
                f"           relative change={distance_change}"
            )

        # ----------------------------------------------------
        # Store only the data actually required by Fig. 2e
        # ----------------------------------------------------
        records.append(
            {
                "city": city,
                "poi_code": poi_code,
                "poi_category": poi_category,
                "social_relative_change": social_change,
                "distance_relative_change": distance_change,
            }
        )


# ============================================================
# 4. SAVE PUBLIC SOURCE DATA
# ============================================================

source_df = pd.DataFrame(records)


# Keep deterministic city and POI order.
city_order = {
    city: i
    for i, city in enumerate(cities)
}

poi_order = {
    poi_code: i
    for i, poi_code in enumerate(naics_map.keys())
}


source_df["_city_order"] = (
    source_df["city"]
    .map(city_order)
)

source_df["_poi_order"] = (
    source_df["poi_code"]
    .map(poi_order)
)


source_df = source_df.sort_values(
    ["_poi_order", "_city_order"]
).drop(
    columns=[
        "_city_order",
        "_poi_order",
    ]
).reset_index(
    drop=True
)


source_df.to_csv(
    OUTPUT_FILE,
    index=False,
)


# ============================================================
# 5. FINAL SUMMARY
# ============================================================

print("\n")
print("=" * 80)
print("FIG. 2e SOURCE-DATA PREPARATION COMPLETE")
print("=" * 80)

print(
    f"Expected city × POI combinations: "
    f"{len(cities)} × {len(naics_map)} "
    f"= {len(cities) * len(naics_map)}"
)

print(
    f"Rows written: {len(source_df)}"
)

print(
    f"Missing files: {len(missing_files)}"
)

print(
    f"Unreadable files: {len(read_errors)}"
)

print(
    f"Missing required columns: "
    f"{len(column_warnings)}"
)

print(
    f"\nOutput file:\n{os.path.abspath(OUTPUT_FILE)}"
)


# ------------------------------------------------------------
# Missing-file details
# ------------------------------------------------------------
if missing_files:

    print("\n")
    print("=" * 80)
    print("MISSING FILES")
    print("=" * 80)

    for city, poi_code, fp in missing_files:

        print(
            f"{city:12s} | "
            f"{poi_code} | "
            f"{fp}"
        )


# ------------------------------------------------------------
# Read-error details
# ------------------------------------------------------------
if read_errors:

    print("\n")
    print("=" * 80)
    print("UNREADABLE FILES")
    print("=" * 80)

    for city, poi_code, fp, error in read_errors:

        print(
            f"{city:12s} | "
            f"{poi_code} | "
            f"{fp}"
        )

        print(
            f"    ERROR: {error}"
        )


# ------------------------------------------------------------
# Column-warning details
# ------------------------------------------------------------
if column_warnings:

    print("\n")
    print("=" * 80)
    print("MISSING REQUIRED COLUMNS")
    print("=" * 80)

    for (
        city,
        poi_code,
        variable,
        fp,
    ) in column_warnings:

        print(
            f"{city:12s} | "
            f"{poi_code} | "
            f"{variable} | "
            f"{fp}"
        )


# ------------------------------------------------------------
# Preview
# ------------------------------------------------------------
print("\n")
print("=" * 80)
print("SOURCE-DATA PREVIEW")
print("=" * 80)

print(
    source_df.to_string(
        index=False
    )
)