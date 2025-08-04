#baltimoreparcel/scripts/hotspot_analysis.py
"""
# -*- coding: utf-8 -*-
# Hotspot Analysis Script for Baltimore Parcel Project
# This script performs hotspot analysis on property value changes in Baltimore.

"""
import sys
import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path
from datetime import datetime
from pyogrio.errors import DataLayerError
import arcpy

from baltimoreparcel.directories import LOGS_DIR, GBD_DIR, get_year_gpkg_dir, ensure_dir
from baltimoreparcel.gis_utils import read_vector_layer, write_gpkg_layer, pivot_panel, export_to_geodb, convert_time_fields
from baltimoreparcel.utils import Logger, info, warn, error, success, process_step
from baltimoreparcel.config import ALL_YEARS
############################################
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
############################################
# Functions
def arcstr(path):
    return str(path).replace("\\", "/")

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

def gpkg_layer_exists(gpkg_path, layer_name):
    try:
        gpd.read_file(gpkg_path, layer=layer_name, rows=1)
        return True
    except (ValueError, OSError, DataLayerError):
        return False
    
if __name__ == "__main__":
    arcpy.env.overwriteOutput = True

    # Initialize logger
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y%m%d %H%M")
    log_file = LOGS_DIR / f"hotspot_analysis_{timestamp}.log"
    logger = Logger(log_file)

    # Read full and change panels
    change_panel_gdf = read_panels("change")
    year_pairs = change_panel_gdf[["START_YR", "END_YR"]].drop_duplicates().values.tolist()
    # Process each year pair
    for year_pair in year_pairs:
        start, end = year_pair
        print(f"Processing year pair: {start} to {end}")

        year_str = f"{start}_{end}"
        name = f"change_panel_{year_str}"

        year_changes_gdf = change_panel_gdf[
            (change_panel_gdf["START_YR"] == start) &
            (change_panel_gdf["END_YR"] == end)
        ]

        info(f"Changes found for {year_str}: {len(year_changes_gdf)} records")

        # Check if layer already exists in GPKG
        layer_exists = False
        gpkg_path = FULL_PANEL_DIR / FULL_PANEL_GEOPKG
        if gpkg_layer_exists(FULL_PANEL_DIR / FULL_PANEL_GEOPKG, name):
            print(f"[SKIP {start} {end}] Layer {name} already exists in GPKG. Skipping write.")
            temp_fc_path = GBD_DIR / name
        else:
            write_gpkg_layer(
                gdf=year_changes_gdf,
                year=name,
                name=FULL_PANEL_GEOPKG,
                directory=FULL_PANEL_DIR,
                layer=name
            )
            # Step 2: Export to FileGDB using ArcPy
            temp_fc_path = export_to_geodb(
                input_gpkg_path=FULL_PANEL_DIR / FULL_PANEL_GEOPKG,
                layer_name=name,
                gdb_path=GBD_DIR,
                out_feature_name=name
            )
            print(f"Exported {name} to GPKG at {FULL_PANEL_DIR / FULL_PANEL_GEOPKG}")
        
        print(f"Feature class path: {temp_fc_path}")
        assert VARIABLE in year_changes_gdf.columns
        print(year_changes_gdf[VARIABLE].dtype)
        print(year_changes_gdf[VARIABLE].isna().sum())
        

        # Step 3: Run HotSpot Analysis
        print(f"Conducting hotspot analysis on {temp_fc_path}")
        print(f"Desired output: {GBD_DIR / f'hotspot_{year_str}'}")
        print('-'* 80)
        try:
            arcpy.stats.HotSpots(
                Input_Field=VARIABLE,
                Input_Feature_Class=arcstr(temp_fc_path),
                Output_Feature_Class=arcstr(GBD_DIR / f"hotspot_{year_str}"),
                Conceptualization_of_Spatial_Relationships="INVERSE_DISTANCE",
                Distance_Method="EUCLIDEAN_DISTANCE",
                Standardization="ROW",
                Distance_Band_or_Threshold_Distance=150,
                Self_Potential_Field=None,
                Weights_Matrix_File=None,
                Apply_False_Discovery_Rate__FDR__Correction="APPLY_FDR",
                number_of_neighbors=None
            )
        except Exception as e:
            error(f"HotSpots failed for {year_str}: {e}")
            print(arcpy.GetMessages())
            raise

        process_step(f"Hotspot analysis completed for {year_str}")