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
lag_panel_gpkg_path = arcstr(GBD_DIR / lag_panel_name)
print(f"lag_panel_gpkg_path: {lag_panel_gpkg_path}")

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

        info(f"Cleaning output for {year} vs {lag_year} ({suffix})...")
        clean_bivariate_output(
            out_path=out_path,
            analysis_field1=analysis_field1,
            analysis_field2=analysis_field2,
            year=year,
            lag_year=lag_year,
            nb_setting=suffix,
            lag_panel_path=lag_panel_gpkg_path
        )

    except Exception as e:
        error(f"Failed spatial association for {year} vs {lag_year} ({suffix}): {e}")

        return None
        
    success(f"{year} vs {lag_year} ({suffix}) --> Output: {out_path}")
    return output_name_base2
        
def clean_bivariate_output(out_path, analysis_field1, analysis_field2, year, lag_year, nb_setting, lag_panel_path):
    """
    Clean up the BivariateSpatialAssociation output feature class by:
    1. Spatially joining key identifiers from the original lag panel
    2. Adding metadata fields
    3. Removing unnecessary fields
    4. Renaming fields for clarity
    
    Parameters:
    -----------
    out_path : str
        Path to the BivariateSpatialAssociation output feature class
    analysis_field1 : str
        Name of the response variable field (e.g., "LOG_REAL_NFMTTLVL_CHNG_2020")
    analysis_field2 : str
        Name of the explanatory variable field (e.g., "LOG_REAL_NFMIMPVL_CHNG_2019")
    year : int
        Analysis year
    lag_year : int
        Lag year for explanatory variable
    nb_setting : str
        Neighborhood setting identifier (e.g., "k_nearest_neighbors50")
    lag_panel_path : str
        Path to the original lag panel feature class for spatial join
    """

    info(f"Cleaning bivariate output: {out_path}")

    # Step 1: Create a temporary feature class for spatial join
    try:
        temp_join_fc = out_path + "_temp_join"
        if arcpy.Exists(temp_join_fc):
            arcpy.management.Delete(temp_join_fc)
        
        # Spatial join to get ACCTID, GEOGCODE, NEIGHBORHOOD back
        info("Performing spatial join to recover parcel identifiers...")
        arcpy.analysis.SpatialJoin(
            target_features=out_path,
            join_features=lag_panel_path,
            out_feature_class=temp_join_fc,
            join_operation="JOIN_ONE_TO_ONE",
            join_type="KEEP_ALL",
            match_option="INTERSECT"
        )
    except Exception as e:
        error(f"Spatial join failed for {out_path}: {e}")
        return None
    info(f"Spatial join completed: {temp_join_fc}")
    try:
        # Step 2: Add metadata fields to temp feature class
        info("Adding metadata fields...")
        metadata_fields = [
            ("ANALYSIS_YEAR", "SHORT"),
            ("LAG_YEAR", "SHORT"),
            ("NB_SETTING", "TEXT", 50),
            ("RESPONSE_VAR", "TEXT", 50),
            ("EXPLAN_VAR", "TEXT", 50)
        ]

        for field_info in metadata_fields:
            field_name = field_info[0]
            field_type = field_info[1]
            field_length = field_info[2] if len(field_info) > 2 else None
            
            if field_length:
                arcpy.management.AddField(temp_join_fc, field_name, field_type, field_length=field_length)
            else:
                arcpy.management.AddField(temp_join_fc, field_name, field_type)

        # Step 3: Populate metadata fields
        info("Populating metadata...")
        response_var_short = short_name(analysis_field1.rsplit('_', 1)[0])  # Remove year suffix
        explan_var_short = short_name(analysis_field2.rsplit('_', 1)[0])   # Remove year suffix
        
        # Step 3: Populate metadata fields
        info("Populating metadata...")
        response_var_full = analysis_field1.rsplit('_', 1)[0]  # Remove year suffix - keep full name
        explan_var_full = analysis_field2.rsplit('_', 1)[0]    # Remove year suffix - keep full name
        
        with arcpy.da.UpdateCursor(temp_join_fc, 
                                    ["ANALYSIS_YEAR", "LAG_YEAR", "NB_SETTING", "RESPONSE_VAR", "EXPLAN_VAR"]) as cursor:
            for row in cursor:
                row[0] = year
                row[1] = lag_year
                row[2] = nb_setting
                row[3] = response_var_full  # Store full variable name
                row[4] = explan_var_full    # Store full variable name
                cursor.updateRow(row)
        
        # Step 4: Define fields to keep
        essential_fields = [
            'OBJECTID', 'Shape',        # Required ArcGIS fields
            'ACCTID',                   # Parcel identifier  
            'GEOGCODE',                 # Geographic code
            'NEIGHBORHOOD',             # Neighborhood identifier
            'SOURCE_ID',                # From bivariate analysis
            'LOCAL_L',                  # Local bivariate statistic
            'P_VALUE',                  # P-value from bivariate analysis
            'SIG_LEVEL',                # Significance level
            'ASSOC_CAT',                # Association category (HH, HL, LH, LL)
            'NUM_NBRS',                 # Number of neighbors used
            'ANALYSIS_YEAR',            # Metadata fields
            'LAG_YEAR',
            'NB_SETTING',
            'RESPONSE_VAR',
            'EXPLAN_VAR'
        ]
        print(f"Fields in temp file: {arcpy.ListFields(temp_join_fc)}")
        # Step 5: Get list of fields to delete
        existing_fields = [f.name for f in arcpy.ListFields(temp_join_fc)]
        fields_to_delete = [f for f in existing_fields if f not in essential_fields]
        
        # Remove fields that are required by ArcGIS or don't exist
        protected_fields = ['OBJECTID', 'Shape', 'OBJECTID_1', 'Shape_Length', 'Shape_Area']
        fields_to_delete = [f for f in fields_to_delete if f not in protected_fields]
        
        if fields_to_delete:
            info(f"Removing {len(fields_to_delete)} unnecessary fields...")
            try:
                arcpy.management.DeleteField(temp_join_fc, fields_to_delete)
            except Exception as e:
                warn(f"Could not delete some fields: {e}")
        
        # Step 6: Replace original with cleaned version
        info("Replacing original with cleaned version...")
        if arcpy.Exists(out_path):
            arcpy.management.Delete(out_path)
        arcpy.management.Rename(temp_join_fc, out_path)
        
        # Step 7: Optional field renaming for clarity
        field_mappings = {
            'LOCAL_L': 'BIVAR_STAT',
            'P_VALUE': 'BIVAR_PVALUE', 
            'SIG_LEVEL': 'SIGNIF_LEVEL',
            'ASSOC_CAT': 'CLUSTER_TYPE',
            'NUM_NBRS': 'N_NEIGHBORS'
        }
        
        for old_name, new_name in field_mappings.items():
            if len([f for f in arcpy.ListFields(out_path) if f.name == old_name]) > 0:
                try:
                    arcpy.management.AlterField(out_path, old_name, new_name, new_name)
                except Exception as e:
                    warn(f"Could not rename {old_name} to {new_name}: {e}")
        
        success(f"Successfully cleaned bivariate output: {out_path}")
        
        # Step 8: Print summary of cleaned feature class
        result_count = int(arcpy.management.GetCount(out_path)[0])
        final_fields = [f.name for f in arcpy.ListFields(out_path)]
        info(f"Cleaned feature class has {result_count} records and {len(final_fields)} fields")
        info(f"Final fields: {', '.join(final_fields)}")
    except Exception as e:
        error(f"Failed to clean bivariate output {out_path}: {e}")
        # Clean up temp files if they exist
        if arcpy.Exists(temp_join_fc):
            try:
                arcpy.management.Delete(temp_join_fc)
            except:
                pass
        raise


def bivariate_neighborhood_pipeline(lag_panel_gdf: gpd.GeoDataFrame = None, 
                                    panel_name: str = lag_panel_name, 
                                    neighborhood_definitions=NEIGHBORHOOD_SETTINGS, 
                                    explan_vars=DEPENDENT_VARS_BASE, n_proc=1):
    """
    Main pipeline for running bivariate neighborhood analysis.
    - lag_panel_gdf: Optional GeoDataFrame of the lag panel. If None, will read the feature class 'panel_name'.
    """
    if lag_panel_gdf is None:
        #default to reading from the lag panel feature class

        lag_panel_path = arcstr(GBD_DIR / panel_name)
        if not arcpy.Exists(lag_panel_path):
            error(f"Lag panel feature class '{lag_panel_path}' does not exist.")
            exit(1)

        info(f"Reading lag panel from {lag_panel_path}...")
        lag_panel_gdf = gpd.read_file(str(GBD_DIR), layer=lag_panel_name)
        info(f"lag_panel shape: {lag_panel_gdf.shape=}")
    
    lag_panel_cols = lag_panel_gdf.columns.tolist()

    response_var = BASE_VAR
    merged_outputs = []
    merge_plan = [] # Store output names for merging later
    tasks = []

    for explan_var in explan_vars:
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
    outputs = bivariate_neighborhood_pipeline(lag_panel_gdf=lag_panel_gdf, n_proc=1,
                                              neighborhood_definitions=NEIGHBORHOOD_SETTINGS, 
                                              explan_vars=DEPENDENT_VARS_BASE)

    # Step 2: Evaluate significant counts
    if outputs:
        eval_results = evaluate_neighborhood_significant_count(outputs)
        print("Evaluation Results:")
        print(eval_results)
        eval_results.to_csv(DATA_DIR / "bivariate_evaluation_results.csv", index=False)

    success("Bivariate analysis completed for all specified variables and years.")
