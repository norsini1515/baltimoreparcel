#baltimoreparcel/scripts/asssemble_lag_panel.py
"""
Assemble Lag Panel Script for Baltimore Parcel Project
This script assembles a lag panel for property value changes in Baltimore.
"""
from datetime import datetime
import sys
import arcpy
from pathlib import Path
import geopandas as gpd
from baltimoreparcel.directories import LOGS_DIR, GBD_DIR, get_year_gpkg_dir, ensure_dir
from baltimoreparcel.utils import Logger, info, warn, error, success, process_step
from baltimoreparcel.gis_utils import arcstr
from baltimoreparcel.gis_utils import write_gpkg_layer, pivot_panel, export_to_geodb
from baltimoreparcel.config import ALL_YEARS


# Define years for processing
YEARS = ALL_YEARS  # or subset like range(2010, 2020)
############################################
FULL_PANEL_GEOPKG = "Baci_full_panel.gpkg"
FULL_PANEL_DIR = get_year_gpkg_dir("full_panel")
# Define layer names for full and change panels
# These should match the names used in the GPKG or GDB
full_panel_name = "full_panel"  # Name for full panel layer in GPKG
change_panel_name = "full_change_panel"  # Name for change panel layer in GPKG
VARIABLE = "LOG_REAL_NFMTTLVL_CHNG"  # Variable to analyze for hotspots

lag_panel_name = "base_lag_panel"
############################################
def read_panels(which: str = "both"):
    """
    Read full and/or change panel GeoDataFrames from the geodatabase.

    Parameters:
    -----------
    which : str
        One of "full", "change", or "both" (default: "both")

    Returns:
    --------
    tuple
        Depending on `which`, returns:
        - ("full") → full_panel_gdf
        - ("change") → change_panel_gdf
        - ("both") → (full_panel_gdf, change_panel_gdf)
    """
    which = which.lower()
    if which not in {"full", "change", "both"}:
        raise ValueError("Argument `which` must be one of: 'full', 'change', 'both'")

    result = ()

    if which in {"full", "both"}:
        print("Reading full panel...")
        full_panel_gdf = gpd.read_file(str(GBD_DIR), layer=full_panel_name)
        info(f"Full panel shape: {full_panel_gdf.shape=}")
        result += (full_panel_gdf,)

    if which in {"change", "both"}:
        print("Reading change panel...")
        change_panel_gdf = gpd.read_file(str(GBD_DIR), layer=change_panel_name)
        info(f"Change panel shape: {change_panel_gdf.shape=}")
        result += (change_panel_gdf,)

    return result[0] if which in {"full", "change"} else result

if __name__ == "__main__":
    arcpy.env.overwriteOutput = True

    # Initialize logger
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y%m%d %H%M")
    log_file = LOGS_DIR / f"assemble_lags_{timestamp}.log"
    logger = Logger(log_file)

    # Read full and change panels
    change_panel_gdf = read_panels("change")
    year_pairs = change_panel_gdf[["START_YR", "END_YR"]].drop_duplicates().values.tolist()

    data_columns = ['LOG_REAL_NFMTTLVL_CHNG', 'LOG_REAL_NFMIMPVL_CHNG', 'LOG_REAL_NFMLNDVL_CHNG', 'ZONING_CHNG', 'OWNNAME1_CHNG']

    for i, variable in enumerate(data_columns):
        if variable not in change_panel_gdf.columns:
            error(f"Variable '{variable}' not found in change panel. Skipping lag assembly for this variable.")
            continue

        print(f"Processing lag panel for variable: {variable}")
        variable_lag_df = pivot_panel(panel_gdf=change_panel_gdf, value_field=variable,
                             id_field="ACCTID", col_field="END_YR",
                             aggfunc="first")

        #merge change data
        if i == 0:
            lag_df = variable_lag_df.copy()
        else:
            lag_df = lag_df.merge(variable_lag_df.drop(columns=['geometry']), on=['ACCTID'], how="outer")
        print(f"Merged lag {i} DataFrame shape: {lag_df.shape=}")
    
    print(f"Final lag DataFrame shape: {lag_df.shape=}")
    
    print('merging neighborhood data...')
    neighborhood_df = change_panel_gdf[['GEOGCODE', 'ACCTID', 'NEIGHBORHOOD']].drop_duplicates()
    print(f"Neighborhood DataFrame shape: {neighborhood_df.shape=}")
    lag_df = lag_df.merge(neighborhood_df, on='ACCTID', how='left')

    print(f"After merging neighborhood data: {lag_df.shape=}")
    print(f"Columns in final lag DataFrame: {lag_df.columns.tolist()}")
    
    # Write lag panel to GPKG
    write_gpkg_layer(
        gdf=lag_df,
        year='base_lag_panel',
        name=FULL_PANEL_GEOPKG,
        directory=FULL_PANEL_DIR,
        layer=lag_panel_name
    )
    print('Exporting to project geodatabase...')
    # Step 2: Export to FileGDB using ArcPy
    temp_fc_path = export_to_geodb(
        input_gpkg_path=FULL_PANEL_DIR / FULL_PANEL_GEOPKG,
        layer_name=lag_panel_name,
        gdb_path=GBD_DIR,
        out_feature_name=lag_panel_name
    )
    print(f"Exported {lag_panel_name} to FileGDB at {temp_fc_path}")