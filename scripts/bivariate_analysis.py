# baltimoreparcel/scripts/bivariate_analysis.py
"""
Use the base_lag_panel to conduct bivariate analysis on change in log_real_value and other variables across space and time.
"""
import sys
import arcpy
import geopandas as gpd
import pandas as pd
from multiprocessing import Pool, cpu_count
from pathlib import Path
import time

from baltimoreparcel.utils import Logger, info, warn, error, success, process_step
from datetime import datetime
from baltimoreparcel.directories import LOGS_DIR, GBD_DIR, DATA_DIR
from baltimoreparcel.gis_utils import arcstr
from baltimoreparcel.config import ALL_YEARS
from baltimoreparcel.scripts.merge_time_layers import merge_layers

arcpy.env.overwriteOutput = True
arcpy.env.workspace = arcstr(GBD_DIR)
print(arcstr(GBD_DIR))
# sys.exit()

lag_panel_name = "base_lag_panel"
temp_path = Path(__file__).resolve().parents[2] / "temp"
temp_path.mkdir(exist_ok=True)
temp_gpkg = temp_path / "lag_panel.gpkg"
lag_panel_gpkg_path = arcstr(temp_gpkg / "lag_panel")

NEIGHBORHOOD_SETTINGS = [
    # {"type": "K_NEAREST_NEIGHBORS", "param": 4},
    # {"type": "K_NEAREST_NEIGHBORS", "param": 8},
    # {"type": "K_NEAREST_NEIGHBORS", "param": 20},
    # {"type": "K_NEAREST_NEIGHBORS", "param": 40},
    {"type": "K_NEAREST_NEIGHBORS", "param": 50},
    # {"type": "DISTANCE_BAND", "param": 50},   # Units = meters by default
    # {"type": "DISTANCE_BAND", "param": 100},
    # {"type": "DISTANCE_BAND", "param": 150},
    # {"type": "CONTIGUITY_EDGES_ONLY", "param": None},  # Rook
    # {"type": "CONTIGUITY_EDGES_CORNERS", "param": None},  # Queen
]
BASE_VAR = "LOG_REAL_NFMTTLVL_CHNG"
DEPENDENT_VARS_BASE = ["LOG_REAL_NFMIMPVL_CHNG", "ZONING_CHNG", "LOG_REAL_NFMLNDVL_CHNG"]

def short_name(var):
    mapping = {
        "LOG_REAL_NFMTTLVL_CHNG": "TTLCHG",
        "LOG_REAL_NFMIMPVL_CHNG": "IMPCHG",
        "LOG_REAL_NFMLNDVL_CHNG": "LNDCHG",
        "ZONING_CHNG": "ZCHG"
    }
    return mapping.get(var, var[:6])

def sanitize_output_name(field1, field2, nb_setting, year=None, lag_year=None):
    f1 = short_name(field1)
    f2 = short_name(field2)
    if year is None and lag_year is None:
        return f"biv_{f1}_{f2}_{nb_setting}"
    else:
        return f"biv_{f1}_{f2}_{nb_setting}_{year}_{lag_year}"

def safe_delete(out_path, max_attempts=3):
    if arcpy.Exists(out_path):
        for attempt in range(1, max_attempts + 1):
            try:
                arcpy.management.Delete(out_path)
                return True
            except Exception as e:
                warn(f"Attempt {attempt}: Failed to delete {out_path}: {e}")
                time.sleep(2)
        error(f"Could not delete locked feature class: {out_path}")
        return False
    return True

def get_base_lag_year(year: int, base: str = BASE_VAR, df: gpd.GeoDataFrame = None) -> int | None:
    """
    Return the most recent lag year < `year` where `base_lag_column_{lag}` exists in the dataframe.
    """
    potential_lags = sorted([int(col.split('_')[-1]) for col in df.columns if col.startswith(base)], reverse=True)
    # print(f"Potential lags for {base}: {potential_lags}")
    for lag in potential_lags:
        if lag < year:
            return lag
    return None

def process_year_setting(args):
    lag_panel_cols, response_var, explan_var, year, lag_year, nb_settings, suffix, output_name_base = args
    run_bivariate_analysis_for_setting(
        lag_panel_cols, response_var, explan_var, year, lag_year, nb_settings, suffix, output_name_base
    )

def run_bivariate_analysis_for_setting(lag_panel_cols, response_var, explan_var, year, lag_year, nb_settings, suffix, output_name_base):
    output_name_base2 = f"{output_name_base}"#_{suffix}"

    analysis_field1 = f"{response_var}_{year}"
    analysis_field2 = f"{explan_var}_{lag_year}"
    out_path = arcstr(GBD_DIR / f"{output_name_base2}_{year}_{lag_year}")

    # Check that both columns exist
    if analysis_field1 not in lag_panel_cols or analysis_field2 not in lag_panel_cols:
        missing = []
        if analysis_field1 not in lag_panel_cols:
            missing.append(analysis_field1)
        if analysis_field2 not in lag_panel_cols:
            missing.append(analysis_field2)
        warn(f"Skipping {year} due to missing fields: {', '.join(missing)}")
        return None
    
    if not safe_delete(out_path):
        return None
    
    # Get neighborhood settings
    nb_type = nb_settings["type"]
    nb_param = nb_settings["param"]
    
    info(f"Running BivariateSpatialAssociation for {year} vs {lag_year} ({suffix})...")
    
    try:
        arcpy.stats.BivariateSpatialAssociation(
            in_features=lag_panel_gpkg_path,
            analysis_field1=analysis_field1,
            analysis_field2=analysis_field2,
            out_features=out_path,
            neighborhood_type=nb_type,
            num_neighbors=nb_param if nb_type == "K_NEAREST_NEIGHBORS" else None,
            local_weighting_scheme="UNWEIGHTED",
            num_permutations=199
        )
    except Exception as e:
        error(f"Failed spatial association for {year} vs {lag_year} ({suffix}): {e}")
        print(lag_panel_gpkg_path)
        print(analysis_field1)
        print(analysis_field2)
        print(out_path)
        print(nb_type, nb_param)

        
        # sys.exit()
        return None
    # try:
    #     # Now join fields from lag_panel_name into the output
    #     join_fields = ["ACCTID", "NEIGHBORHOOD", "GEOGCODE"]
    #     # print(f"Joining fields {join_fields} from {lag_panel_name} to {out_path}")
    #     arcpy.management.JoinField(
    #         in_data=out_path,
    #         in_field="OBJECTID",
    #         join_table=lag_panel_gpkg_path,
    #         join_field="OBJECTID",
    #         fields=join_fields
    #     )
    # except Exception as e:
    #     warn(f"Analysis completed but failed to join fields to {out_path}: {e}")
    #     return None
    
    success(f"{year} vs {lag_year} ({suffix}) --> Output: {out_path}")
    return output_name_base2
        

def bivariate_neighborhood_pipeline(lag_panel_gdf=None, neighborhood_definitions=NEIGHBORHOOD_SETTINGS, x_var_to_test=DEPENDENT_VARS_BASE, n_proc=1):
    if lag_panel_gdf is None:
        lag_panel_path = arcstr(GBD_DIR / lag_panel_name)
        if not arcpy.Exists(lag_panel_path):
            error(f"Lag panel feature class '{lag_panel_path}' does not exist.")
            exit(1)

        info(f"Reading lag panel from {lag_panel_path}...")
        lag_panel_gdf = gpd.read_file(str(GBD_DIR), layer=lag_panel_name)
        info(f"lag_panel shape: {lag_panel_gdf.shape=}")
    
    lag_panel_gdf.to_file(temp_gpkg, layer="lag_panel", driver="GPKG")


    lag_panel_cols = lag_panel_gdf.columns.tolist()

    response_var = BASE_VAR
    merged_outputs = []
    merge_plan = [] # Store output names for merging later
    tasks = []

    for explan_var in x_var_to_test:
        # expl_var = var.split('_')[2]
        info(f"Running bivariate analysis for {response_var} vs {explan_var}...")
        for setting in neighborhood_definitions:
            nb_setting = f"{setting['type'].lower()}{setting['param']}" if setting['param'] else setting['type'].lower()
            nb_setting = nb_setting.replace("_k_nearest_neighbors", "_knn")

            output_name_base = sanitize_output_name(response_var, explan_var, nb_setting)
            output_name_base = output_name_base.replace("_k_nearest_neighbors", "_knn")
            merge_plan.append(output_name_base)

            for year in ALL_YEARS:
                lag_year = get_base_lag_year(year, base=response_var, df=lag_panel_gdf)
                if lag_year is None:
                    warn(f"No lag year found for {year} with base {response_var}")
                    continue
                tasks.append((
                    lag_panel_cols, response_var, explan_var, year, lag_year, setting, nb_setting, output_name_base
                ))

    # Run in parallel
    info(f"Dispatching {len(tasks)} bivariate analysis tasks to {n_proc} processes...")
    with Pool(processes=n_proc) as pool:
        pool.map(process_year_setting, tasks)
    
    success(f"Completed {len(tasks)} bivariate analysis tasks.")
    info(f"Merging outputs for {len(merge_plan)} settings...")
    # Merge results per output_name_base
    for output_name_base in merge_plan:
        try:
            merged_fc_name = f"{output_name_base}_merged"
            merge_layers(
                prefix=output_name_base,
                output_fc=merged_fc_name,
                convert_time=True,
            )
            merged_outputs.append(merged_fc_name)
        except Exception as e:
            error(f"Failed to merge {output_name_base}: {e}")

    success("Bivariate analysis completed for all specified variables, years, and neighborhood definitions.")
    return merged_outputs

def evaluate_neighborhood_significant_count(merged_feature_paths: list[str]) -> pd.DataFrame:
    """
    Evaluate a list of merged BivariateSpatialAssociation feature classes based on significance count and bin breakdown.
        - Layer name
        - Total parcels
        - Number and % of significant parcels
        - Breakdown of Gi_Bin values (High-High, Low-Low, etc.)
        - Mean GiZScore (if available)
        - Optionally: Grouped stats by GEOGCODE or NEIGHBORHOOD
    """
    summaries = []

    for layer_name in merged_feature_paths:
        layer_path = arcstr(GBD_DIR / layer_name)
        try:
            gdf = gpd.read_file(str(layer_path))
        except Exception as e:
            print(f"Failed to load {layer_name}: {e}")
            continue

        total = len(gdf)
        significant = gdf[gdf['Gi_Bin'] != 0].copy()  # ArcGIS assigns 0 to not significant
        sig_count = len(significant)
        sig_pct = round(sig_count / total * 100, 2) if total else 0.0

        # Count of each Gi_Bin category
        bin_counts = gdf['Gi_Bin'].value_counts(dropna=False).to_dict()
        bin_counts_str = {f"bin_{int(k)}": v for k, v in bin_counts.items() if pd.notna(k)}

        # Mean GiZScore (if it exists)
        mean_z = gdf['GiZScore'].mean() if 'GiZScore' in gdf.columns else None

        summaries.append({
            "layer": layer_name,
            "total_parcels": total,
            "sig_parcels": sig_count,
            "sig_pct": sig_pct,
            "mean_z": mean_z,
            **bin_counts_str
        })

    return pd.DataFrame(summaries)

if __name__ == "__main__":
    arcpy.env.overwriteOutput = True

    # Initialize logger
    now = datetime.now()
    timestamp = now.strftime("%Y%m%d %H%M")
    log_file = LOGS_DIR / f"bivariate_analysis_{timestamp}.log"
    logger = Logger(log_file)

    # Step 1: Read lag panel
    print("Reading lag panel...")
    lag_panel_path = arcstr(GBD_DIR / lag_panel_name)
    lag_panel_gdf = gpd.read_file(str(GBD_DIR), layer=lag_panel_name)
    info(f"lag_panel shape: {lag_panel_gdf.shape=}")
    print(f"Columns in lag panel: {lag_panel_gdf.columns.tolist()}")

    print("Starting bivariate neighborhood analysis...")
    outputs = bivariate_neighborhood_pipeline(lag_panel_gdf=lag_panel_gdf, n_proc=1)

    # Step 2: Evaluate significant counts
    if outputs:
        eval_results = evaluate_neighborhood_significant_count(outputs)
        print("Evaluation Results:")
        print(eval_results)
        eval_results.to_csv(DATA_DIR / "bivariate_evaluation_results.csv", index=False)

    success("Bivariate analysis completed for all specified variables and years.")
