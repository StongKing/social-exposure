Fig. 1d binary-flow package
============================

Each NPZ file contains:
  flow_binary : uint8 matrix; 0=no flow, 1=positive flow
  flow_valid  : uint8 matrix; 1=valid original cell
  cbg_ids     : matrix row identifiers
  poi_ids     : matrix column identifiers
  q_values    : derived POI visitor-income composition

The package does not contain original positive flow magnitudes.
category_flow_metadata.csv stores only category-level derived baselines and totals required to reproduce the original figure.

poi_boston_msa_plot_locations.csv contains only:
  poi_id, longitude, latitude, poi_code
It does not contain POI names, addresses, brands, visit records, or other original POI attributes.
