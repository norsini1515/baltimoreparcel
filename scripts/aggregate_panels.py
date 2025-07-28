# baltimoreparcel/scripts/aggregate_panels.py
"""
A script to aggregate and process full and change panels for Baltimore Parcel data.
This script reads the full panel and change panel data, aggregates them by specified rules,
and saves the results back to the geodatabase as new layers.
"""
import sys
import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path
from datetime import datetime

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
############################################
# Functions
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

def run_aggregations(panel_gdf, rules, panel_name, geom_lookup: dict[str, gpd.GeoDataFrame] = None):
    """
    Run aggregation rules on a panel GeoDataFrame.

    Parameters:
    -----------
    panel_gdf : GeoDataFrame
        The input full or change panel GeoDataFrame.
    rules : list of tuples
        Each tuple has (groupby_keys, aggregation_dict).
    panel_name : str
        Name prefix for output layer.

    Returns:
    --------
    dict of {str: GeoDataFrame}
        Dictionary mapping output layer name to aggregated GeoDataFrame.
    """
    aggregated_results = {}

    process_step(f"Running aggregations on: {panel_name}")
    print(f"Initial input shape: {panel_gdf.shape}")

    for group_keys, agg_dict in rules:
        suffix = "_".join(group_keys).lower().replace(" ", "")
        out_layer_name = f"{panel_name}_agg_{suffix}"

        info(f"Aggregating {panel_name}. Target output ==> {out_layer_name}")
        print(f"-> Grouping by: {group_keys}, aggregating with: {agg_dict}")
        
        # Determine if external geometry should be merged
        geometry_key = next((key for key in group_keys if geom_lookup and key in geom_lookup), None)


        # Ensure numeric columns are actually numeric (this avoids mean on dtype=object)
        for col in agg_dict:
            if agg_dict[col] == 'mean' and panel_gdf[col].dtype == 'object':
                warn(f"Column {col} is object type, attempting to coerce to numeric")
                panel_gdf[col] = pd.to_numeric(panel_gdf[col], errors='coerce')
        
        try:
            grouped = (
                panel_gdf
                .groupby(group_keys)
                .agg(agg_dict)
            ).reset_index()
            print(f"-> Grouped shape after aggregation: {grouped.shape}")
            print(f"-> Grouped columns: {grouped.columns.tolist()}")

            print(f"-> Grouped shape before geometry merge: {grouped.shape}")

            if geometry_key:
                print(f"-> Merging geometry for key: {geometry_key}")
                geom_df = geom_lookup[geometry_key]
                print(f"-> Geometry DataFrame shape: {geom_df.shape}, columns: {geom_df.columns.tolist()}")

                grouped = grouped.merge(geom_df[[geometry_key, "geometry"]], on=geometry_key, how="left")
                # Ensure geometry is set correctly
                grouped = gpd.GeoDataFrame(grouped, geometry="geometry", crs=panel_gdf.crs)
            else:
                # Attempt to preserve geometry from one example row in each group
                # if "GEOGCODE" in group_keys or "NEIGHBORHOOD" in group_keys:
                grouped = gpd.GeoDataFrame(grouped, geometry=None)
                grouped = grouped.merge(
                    panel_gdf[[*group_keys, "geometry"]].drop_duplicates(subset=group_keys),
                    on=group_keys,
                    how="left"
                )
                grouped = gpd.GeoDataFrame(grouped, geometry="geometry", crs=panel_gdf.crs)

            print(f"-> Grouped shape after geometry merge: {grouped.shape}")

            aggregated_results[out_layer_name] = grouped
            info(f"Finished aggregation: {out_layer_name} ({grouped.shape[0]} rows)")

        except Exception as e:
            error(f"Aggregation failed for {group_keys}: {e}")
    if len(aggregated_results) > 0:
        success(f"{len(aggregated_results)} aggregations completed for {panel_name}")
    else:
        error(f"No aggregations completed for {panel_name}. Check rules and input data.")
        raise RuntimeError("No aggregations were successfully completed.")
    return aggregated_results

############################################

full_panel_aggregation_rules = [
    #aggregation var, rules
    (['GEOGCODE', 'YEAR'], {'LANDAREA':'mean', 
                  'YEARBLT': 'mean',
                  'SQFTSTRC': 'mean',
                  'LOG_REAL_NFMLNDVL': 'mean',
                  'LOG_REAL_NFMIMPVL': 'mean',
                  'LOG_REAL_NFMTTLVL': 'mean',
                #   'NFMLNDVL': 'mean',
                #   'NFMIMPVL': 'mean',
                #   'NFMTTLVL': 'mean',
                #   'REAL_NFMLNDVL': 'mean',
                #   'REAL_NFMIMPVL': 'mean',
                #   'REAL_NFMTTLVL': 'mean',
                  }),
    (['NEIGHBORHOOD', 'YEAR'], {'LANDAREA':'mean', 
                                    'YEARBLT': 'mean',
                                    'SQFTSTRC': 'mean',
                                    'LOG_REAL_NFMLNDVL': 'mean',
                                    'LOG_REAL_NFMIMPVL': 'mean',
                                    'LOG_REAL_NFMTTLVL': 'mean',
                                    #   'NFMLNDVL': 'mean',
                                    #   'NFMIMPVL': 'mean',
                                    #   'NFMTTLVL': 'mean',
                                    #   'REAL_NFMLNDVL': 'mean',
                                    #   'REAL_NFMIMPVL': 'mean',
                                    #   'REAL_NFMTTLVL': 'mean',
                                    }),
]

change_panel_aggregation_rules = [
    #aggregation var, rules
    # (['GEOGCODE', 'START_YR', 'END_YR'], 
    #               {
    #                 'ACRES_CHNG': 'mean',
    #                 'SQFTSTRC_CHNG': 'mean',
    #                 'LOG_REAL_NFMLNDVL_CHNG': 'mean',
    #                 'LOG_REAL_NFMIMPVL_CHNG': 'mean',
    #                 'LOG_REAL_NFMTTLVL_CHNG': 'mean',
    #                 'LU_CHNG': 'count',
    #                 'ZONING_CHNG': 'count',
    #                 'OWNNAME1_CHNG': 'count',
    #               }
    # ),
    (['NEIGHBORHOOD', 'START_YR', 'END_YR'], 
                {
                    'ACRES_CHNG': 'mean',
                    'SQFTSTRC_CHNG': 'mean',
                    'LOG_REAL_NFMLNDVL_CHNG': 'mean',
                    'LOG_REAL_NFMIMPVL_CHNG': 'mean',
                    'LOG_REAL_NFMTTLVL_CHNG': 'mean',
                    'LU_CHNG': 'count',
                    'ZONING_CHNG': 'count',
                    'OWNNAME1_CHNG': 'count',
                }
    ),
]


# Main script logic
if __name__ == "__main__":
    ############################################
    # Toggle which aggregations to run
    do_full_panel = False
    do_change_panel = True
    ############################################
    process_step("Starting aggregate_panels script...")
    # Set up logging
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y%m%d %H%M")
    logger = Logger(LOGS_DIR / f"aggregate_panels_{timestamp}.log")
    ############################################
    # Ensure geometries exist.
    neigh_geom = gpd.read_file(str(GBD_DIR), layer="neighborhoods")
    neigh_geom = neigh_geom.rename(columns={"Name": "NEIGHBORHOOD"})
    neigh_geom = neigh_geom[["NEIGHBORHOOD", "geometry"]]
    
    geom_lookup = {
        "NEIGHBORHOOD": neigh_geom
    }
    for key, geom in geom_lookup.items():
        print(f"{key} geometries loaded with {len(geom)} features")
        print(f"{key} columns: {geom.columns.tolist()}")
    ############################################

    if do_full_panel:
        full_panel = read_panels("full")
        print(f"Full panel columns: {full_panel.columns.tolist()}")
        aggregated_full_panel = run_aggregations(
            full_panel, 
            full_panel_aggregation_rules, 
            panel_name="full_panel", 
            geom_lookup=geom_lookup
        )
    
    if do_change_panel:
        change_panel = read_panels("change")
        print(f"Change panel columns: {change_panel.columns.tolist()}")
        aggregated_change_panel = run_aggregations(
            change_panel, 
            change_panel_aggregation_rules, 
            panel_name="change_panel", 
            geom_lookup=geom_lookup
        )
    ############################################
    all_aggregated = {}
    if do_full_panel:
        all_aggregated.update(aggregated_full_panel)
    if do_change_panel:
        all_aggregated.update(aggregated_change_panel)
    print(f"Total aggregated layers: {len(all_aggregated)}")

    # Export to GDB
    print("Exporting aggregated layers to GDB...")
    for name, gdf in all_aggregated.items():
        print(f"Exporting {name} with shape {gdf.shape}")
        print(gdf.head())
        for col in gdf:
            if gdf[col].dtype == 'int64' and gdf[col].isna().any():
                gdf[col] = gdf[col].astype('float')
        print(f"Final dtypes before export: {gdf.dtypes=}")
        print(f"Final shape before export: {gdf.shape=}")
        print(f"Final columns before export: {gdf.columns.tolist()=}")
        write_gpkg_layer(
            gdf=gdf,
            year=name,
            name=FULL_PANEL_GEOPKG,
            directory=FULL_PANEL_DIR,
            layer=name
        )
        print(f"Exported {name} to GPKG at {FULL_PANEL_DIR / FULL_PANEL_GEOPKG}")
        layer_path = export_to_geodb(
                        input_gpkg_path=FULL_PANEL_DIR / FULL_PANEL_GEOPKG,
                        layer_name=name,
                        gdb_path=GBD_DIR,
                        out_feature_name=name,
                    )
        if not layer_path:
            error(f"Failed to export {name} to GDB. Check logs for details.")
            sys.exit(1)

        if do_full_panel:
            convert_time_fields(
                table_path=layer_path.name,  # now just the name of the FC
                field_pairs=[("YEAR", "YEAR_DATE")]
            )
        if do_change_panel:
            convert_time_fields(
                table_path=layer_path.name,
                field_pairs=[("START_YR", "START_DATE"), ("END_YR", "END_DATE")]
            )
        
        print(f"Exported {name} to GDB at {layer_path}")
    success("All aggregations and exports completed successfully.")
    logger.close()
    print("Script finished. Check logs for details.")