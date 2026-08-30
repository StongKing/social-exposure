# -*- coding: utf-8 -*-
"""
Figure1a_pre
"""
import os
import numpy as np
import pandas as pd
import geopandas as gpd
PRINT_PROGRESS = True
CBSA_SHP_PATH = 'geo_data/tl_2021_us_cbsa/tl_2021_us_cbsa.shp'
OUTPUT_CSV = 'us_15_cities_aggregated.csv'
CITY_LIST = [('newyork', '35620'), ('losangeles', '31080'), ('chicago', '16980'), ('houston', '26420'), ('atlanta', '12060'), ('seattle', '42660'), ('boston', '14460'), ('fresno', '23420'), ('baltimore', '12580'), ('tulsa', '46140'), ('tyler', '46340'), ('champaign', '16580'), ('billings', '13740'), ('sebring', '42700'), ('cheyenne', '16940')]
PRETTY_NAMES = {'newyork': 'New York', 'losangeles': 'Los Angeles', 'chicago': 'Chicago', 'houston': 'Houston', 'atlanta': 'Atlanta', 'seattle': 'Seattle', 'boston': 'Boston', 'fresno': 'Fresno', 'baltimore': 'Baltimore', 'tulsa': 'Tulsa', 'tyler': 'Tyler', 'champaign': 'Champaign', 'billings': 'Billings', 'sebring': 'Sebring', 'cheyenne': 'Cheyenne'}
POI_NAMES = ['Other_Individual_and_Family_Services', 'Promoters_of_Performing_Arts,_Sports,_and_Similar_Events_with_Facilities', 'Museums', 'Fitness_and_Recreational_Sports_Centers', 'Drinking_Places_(Alcoholic_Beverages)', 'Religious_Organizations']

def read_matrix_sum(file_path):
    df = pd.read_csv(file_path, header=0, index_col=0)
    numeric = df.apply(pd.to_numeric, errors='coerce')
    return float(np.nansum(numeric.to_numpy(dtype=float)))

def read_flow_summary(file_path):
    df = pd.read_csv(file_path, header=0, index_col=0)
    numeric = df.apply(pd.to_numeric, errors='coerce')
    total_flow = float(np.nansum(numeric.to_numpy(dtype=float)))
    total_pois = int(df.shape[1])
    return (total_flow, total_pois)

def get_cbsa_centroid_lonlat(cbsa_gdf, geoid):
    matches = cbsa_gdf[cbsa_gdf['GEOID'].astype(str) == str(geoid)].copy()
    if matches.empty:
        return (np.nan, np.nan)
    matches_proj = matches.to_crs('EPSG:5070')
    if hasattr(matches_proj.geometry, 'union_all'):
        merged_geometry = matches_proj.geometry.union_all()
    else:
        merged_geometry = matches_proj.geometry.unary_union
    centroid_proj = merged_geometry.centroid
    centroid_wgs84 = gpd.GeoSeries([centroid_proj], crs='EPSG:5070').to_crs('EPSG:4326').iloc[0]
    return (float(centroid_wgs84.x), float(centroid_wgs84.y))
if not os.path.isfile(CBSA_SHP_PATH):
    raise FileNotFoundError(f'未找到 CBSA shapefile：{CBSA_SHP_PATH}')
cbsa = gpd.read_file(CBSA_SHP_PATH).to_crs('EPSG:4326')
if 'GEOID' not in cbsa.columns:
    raise KeyError('CBSA shapefile 中不存在 GEOID 字段。')
records = []
for city_name, geoid in CITY_LIST:
    base_dir = f'matrices_A_D_S_Distribution_{city_name}_core'
    total_distance = 0.0
    total_se = 0.0
    total_se_js = 0.0
    total_flow = 0.0
    total_pois = 0
    loaded_poi_categories = 0
    missing_poi_categories = []
    failed_files = []
    lon, lat = get_cbsa_centroid_lonlat(cbsa, geoid)
    if PRINT_PROGRESS:
        print('=' * 72)
        print(f'Processing city: {city_name}')
        print(f'Source data directory: {base_dir}')
    if not os.path.isdir(base_dir):
        if PRINT_PROGRESS:
            print(f'[WARN] City directory not found: {base_dir}')
        records.append({'city': city_name, 'city_name': PRETTY_NAMES.get(city_name, city_name.title()), 'cbsa_geoid': geoid, 'lon': lon, 'lat': lat, 'total_distance': 0.0, 'total_se': 0.0, 'total_se_js': 0.0, 'total_flow': 0.0, 'total_pois': 0, 'loaded_poi_categories': 0, 'missing_poi_categories': len(POI_NAMES), 'failed_file_count': 0, 'data_available': False})
        continue
    for poi_name in POI_NAMES:
        poi_dir = os.path.join(base_dir, poi_name)
        if not os.path.isdir(poi_dir):
            missing_poi_categories.append(poi_name)
            continue
        loaded_poi_categories += 1
        file_map = {'distance': os.path.join(poi_dir, 'distance_matrix.csv'), 'se': os.path.join(poi_dir, 'social_exposure_matrix.csv'), 'se_js': os.path.join(poi_dir, 'social_exposure_matrix_js.csv'), 'flow': os.path.join(poi_dir, 'flow_matrix.csv')}
        if os.path.isfile(file_map['distance']):
            try:
                total_distance += read_matrix_sum(file_map['distance'])
            except Exception as exc:
                failed_files.append(file_map['distance'])
                if PRINT_PROGRESS:
                    print(f"[WARN] Failed to read distance file: {file_map['distance']}")
                    print(f'       Reason: {exc}')
        else:
            failed_files.append(file_map['distance'])
            if PRINT_PROGRESS:
                print(f"[WARN] File not found: {file_map['distance']}")
        if os.path.isfile(file_map['se']):
            try:
                total_se += read_matrix_sum(file_map['se'])
            except Exception as exc:
                failed_files.append(file_map['se'])
                if PRINT_PROGRESS:
                    print(f"[WARN] Failed to read social exposure file: {file_map['se']}")
                    print(f'       Reason: {exc}')
        else:
            failed_files.append(file_map['se'])
            if PRINT_PROGRESS:
                print(f"[WARN] File not found: {file_map['se']}")
        if os.path.isfile(file_map['se_js']):
            try:
                total_se_js += read_matrix_sum(file_map['se_js'])
            except Exception as exc:
                failed_files.append(file_map['se_js'])
                if PRINT_PROGRESS:
                    print(f"[WARN] Failed to read JS exposure file: {file_map['se_js']}")
                    print(f'       Reason: {exc}')
        else:
            failed_files.append(file_map['se_js'])
            if PRINT_PROGRESS:
                print(f"[WARN] File not found: {file_map['se_js']}")
        if os.path.isfile(file_map['flow']):
            try:
                category_flow, category_pois = read_flow_summary(file_map['flow'])
                total_flow += category_flow
                total_pois += category_pois
            except Exception as exc:
                failed_files.append(file_map['flow'])
                if PRINT_PROGRESS:
                    print(f"[WARN] Failed to read flow file: {file_map['flow']}")
                    print(f'       Reason: {exc}')
        else:
            failed_files.append(file_map['flow'])
            if PRINT_PROGRESS:
                print(f"[WARN] File not found: {file_map['flow']}")
    if PRINT_PROGRESS and missing_poi_categories:
        print(f'[WARN] Missing {len(missing_poi_categories)} POI category directories: {missing_poi_categories}')
    records.append({'city': city_name, 'city_name': PRETTY_NAMES.get(city_name, city_name.title()), 'cbsa_geoid': geoid, 'lon': lon, 'lat': lat, 'total_distance': float(total_distance), 'total_se': float(total_se), 'total_se_js': float(total_se_js), 'total_flow': float(total_flow), 'total_pois': int(total_pois), 'loaded_poi_categories': int(loaded_poi_categories), 'missing_poi_categories': int(len(missing_poi_categories)), 'failed_file_count': int(len(failed_files)), 'data_available': bool(loaded_poi_categories > 0)})
    if PRINT_PROGRESS:
        print(f'[OK] {city_name}: distance={total_distance:.6f}, SE={total_se:.6f}, SE_JS={total_se_js:.6f}, flow={total_flow:.0f}, POIs={total_pois}')
df_agg = pd.DataFrame.from_records(records)
numeric_columns = ['lon', 'lat', 'total_distance', 'total_se', 'total_se_js', 'total_flow', 'total_pois', 'loaded_poi_categories', 'missing_poi_categories', 'failed_file_count']
for column in numeric_columns:
    df_agg[column] = pd.to_numeric(df_agg[column], errors='coerce')
df_agg.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig', float_format='%.10f')
print('=' * 72)
print(f'City-level aggregated results saved to: {os.path.abspath(OUTPUT_CSV)}')
print(f'Saved {len(df_agg)} cities.')


