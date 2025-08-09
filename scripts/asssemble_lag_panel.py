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

############################################
def read_gdb_layer(layer_name: str) -> gpd.GeoDataFrame:
    """
    Read a layer from the FileGDB.
    """
    info(f"Reading layer: {layer_name}")
    try:
        return gpd.read_file(str(GBD_DIR), layer=layer_name)
    except Exception as e:
        print(f"Error reading layer {layer_name}: {e}")

def assemble_lag_panel(
    df: gpd.GeoDataFrame,
    variables: list[str],
    id_field: str = "ACCTID",
    col_field: str = "END_YR"
) -> gpd.GeoDataFrame:
    """
    Assemble a lag panel by pivoting multiple variables.

    Parameters:
    -----------
    df : GeoDataFrame
        Input change or bivariate panel.
    variables : list of str
        List of variable names to pivot into wide format.
    id_field : str
        Unique ID field (e.g., ACCTID).
    col_field : str
        Column to pivot across (e.g., END_YR).

    Returns:
    --------
    GeoDataFrame: Merged wide-format lag panel.
    """
    for i, var in enumerate(variables):
        if var not in df.columns:
            warn(f"Variable '{var}' not found in layer. Skipping.")
            continue

        info(f"Pivoting variable: {var}")
        pivoted = pivot_panel(
            panel_gdf=df,
            value_field=var,
            id_field=id_field,
            col_field=col_field,
            aggfunc="first"
        )

        if i == 0:
            lag_df = pivoted.copy()
        else:
            lag_df = lag_df.merge(pivoted.drop(columns=["geometry"]), on=id_field, how="outer")

        info(f"Merged '{var}' — shape: {lag_df.shape}")

    return lag_df

def attach_neighborhood_metadata(
    lag_df: gpd.GeoDataFrame,
    source_df: gpd.GeoDataFrame,
    id_field: str = "ACCTID"
) -> gpd.GeoDataFrame:
    """
    Attach GEOGCODE and NEIGHBORHOOD to the lag panel.

    Parameters:
    -----------
    lag_df : GeoDataFrame
        Lag panel in wide format.
    source_df : GeoDataFrame
        Source panel with neighborhood info.

    Returns:
    --------
    GeoDataFrame: Enriched lag panel with metadata.
    """
    meta_cols = ["GEOGCODE", id_field, "NEIGHBORHOOD"]
    meta_df = source_df[meta_cols].drop_duplicates()
    lag_df = lag_df.merge(meta_df, on=id_field, how="left")
    return lag_df

def write_lag_outputs(
    lag_df: gpd.GeoDataFrame,
    layer_name: str
) -> None:
    """
    Write lag panel to GPKG and FileGDB.
    """
    write_gpkg_layer(
        gdf=lag_df,
        year="custom_lag",
        name=FULL_PANEL_GEOPKG,
        directory=FULL_PANEL_DIR,
        layer=layer_name
    )

    process_step("Exporting lag panel to FileGDB...")
    temp_fc_path = export_to_geodb(
        input_gpkg_path=FULL_PANEL_DIR / FULL_PANEL_GEOPKG,
        layer_name=layer_name,
        gdb_path=GBD_DIR,
        out_feature_name=layer_name
    )

    success(f"Exported to FileGDB: {temp_fc_path}")

def run_lag_panel_pipeline(
    input_layer: str,
    variables: list[str],
    output_layer: str,
    id_field: str = "ACCTID",
    col_field: str = "END_YR"
):
    """
    Run the full lag panel workflow.

    Parameters:
    -----------
    input_layer : str
        Name of the input layer in the GDB.
    variables : list of str
        Variables to include in lag panel.
    output_layer : str
        Output layer name for lag panel.
    """
    info(f"Running lag panel pipeline for layer: {input_layer} for variables: {variables}")
    change_panel_gdf = read_gdb_layer(input_layer)

    process_step("Assembling lag panel...")
    lag_df = assemble_lag_panel(change_panel_gdf, variables, id_field, col_field)

    process_step("Attaching neighborhood metadata...")
    lag_df = attach_neighborhood_metadata(lag_df, change_panel_gdf, id_field)

    info(f"Final lag DataFrame shape: {lag_df.shape}")
    info(f"Columns: {lag_df.columns.tolist()}")

    process_step("Writing lag panel outputs...")
    write_lag_outputs(lag_df, output_layer)


if __name__ == "__main__":
    arcpy.env.overwriteOutput = True

    # Initialize logger
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y%m%d %H%M")
    log_file = LOGS_DIR / f"assemble_lags_{timestamp}.log"
    logger = Logger(log_file)
    
    # lag_panel_name = "base_lag_panel"
    # change_panel_name = "full_change_panel"
    # run_lag_panel_pipeline(
    #     input_layer=change_panel_name,
    #     variables=[
    #         "LOG_REAL_NFMTTLVL_CHNG",
    #         "LOG_REAL_NFMIMPVL_CHNG",
    #         "OWNNAME1_CHNG"
    #     ],
    #     output_layer=lag_panel_name
    # )

    # Assemble bivariate analysis panels
    # input_layer = "biv_TTLCHG_IMPCHG_knn50_merged"
    # output_layer = "biv_TTLCHG_IMPCHG_knn50_merged_panel"
    # run_lag_panel_pipeline(
    #     input_layer=input_layer,
    #     variables=[
    #         "BIVAR_PVALUE",
    #         "CLUSTER_TYPE",
    #     ],
    #     output_layer=output_layer
    # )
    input_layer = "full_panel"
    # output_layer = "LOG_REAL_NFMTTLVL_panel"
    output_layer = "LOG_REAL_NFMIMPVL_panel"
    run_lag_panel_pipeline(
        input_layer=input_layer,
        variables=[
            "LOG_REAL_NFMIMPVL",
        ],
        col_field="YEAR",
        output_layer=output_layer
    )

    success(f"Completed lag panel assembly. Output saved to: {GBD_DIR / output_layer}")