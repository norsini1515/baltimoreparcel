#scripts/eda/nfmtlvl_eda.py
'''Exploratory Data Analysis of NFMTTLVL in the panel data
===============================================================================
'''
from baltimoreparcel.gis_utils import read_vector_layer, write_gpkg_layer, pivot_panel
from baltimoreparcel.directories import get_year_gpkg_dir, GBD_DIR
from baltimoreparcel.engineer_panel import log_value, calculate_change, summarize_field, enrich_change_gdf
from baltimoreparcel.scripts.eda.plots import timeseries_histogram
from baltimoreparcel.utils import info, warn, error

import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
import pandas as pd
import numpy as np
from pathlib import Path

FULL_PANEL_GEOPKG = "Baci_full_panel.gpkg"
FULL_PANEL_DIR = get_year_gpkg_dir("full_panel")

FULL_PANEL_NAME = "full_panel"
CHANGE_PANEL_NAME = "full_change_panel"

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
        full_panel_gdf = gpd.read_file(str(GBD_DIR), layer=FULL_PANEL_NAME)
        info(f"Full panel shape: {full_panel_gdf.shape=}")
        result += (full_panel_gdf,)

    if which in {"change", "both"}:
        print("Reading change panel...")
        change_panel_gdf = gpd.read_file(str(GBD_DIR), layer=CHANGE_PANEL_NAME)
        info(f"Change panel shape: {change_panel_gdf.shape=}")
        result += (change_panel_gdf,)

    return result[0] if which in {"full", "change"} else result


if __name__ == "__main__":
    FULL_VALUE = "LOG_REAL_NFMTTLVL"
    CHNG_VALUE = "LOG_REAL_NFMTTLVL_CHNG"
    
    summarize_change = False
    
    plot_log_value_distributions = False
    plot_log_value_chng_distributions = True

    output_files = True
    #-----------------------------------------------------------------------------------
    print(f"Starting EDA for {FULL_VALUE}...")
    if plot_log_value_distributions:
        panel_gdf = read_panels(which='full')
        print(panel_gdf.head())
    
    if plot_log_value_chng_distributions:
        change_panel_gdf = read_panels(which='change')
        print(change_panel_gdf.head())
    #-----------------------------------------------------------------------------------
    print('-'*150, '\n')
    #-----------------------------------------------------------------------------------
    #EDA plots
    if plot_log_value_distributions:
        print(f"\n\nPlotting {FULL_VALUE} distributions...")
        histogram_path = FULL_PANEL_DIR / f"{FULL_VALUE}_histograms.png"
        trend_path = FULL_PANEL_DIR / f"{FULL_VALUE}_trend.png"

        full_value_data = pivot_panel(panel_gdf, value_field=FULL_VALUE)
        print(f"{full_value_data.shape=}, {full_value_data.columns.tolist()=}")
        timeseries_histogram(full_value_data, value_field=FULL_VALUE,
                         save_path=histogram_path,
                         show_plot=True,
                         n_cols=4, bins=30,
                         plot_trend=True,
                         trend_save_path=trend_path)
    #-----------------------------------------------------------------------------------    
    if plot_log_value_chng_distributions:
        print(f"\n\nPlotting {CHNG_VALUE}_CHNG distributions...")
        change_path = FULL_PANEL_DIR / f"{CHNG_VALUE}_change_histograms.png"
        change_trend_path = FULL_PANEL_DIR / f"{CHNG_VALUE}_change_trend.png"
        timeseries_histogram(change_panel_gdf, value_field=CHNG_VALUE, 
                            save_path=change_path,
                            show_plot=True,
                            n_cols=4, bins=30,
                            plot_trend=True,
                            trend_save_path=change_trend_path)
    #-----------------------------------------------------------------------------------
    if summarize_change:
        print(f"\n\nSummarizing LOG_{VALUE}_CHNG...")
        summary_df = summarize_field(NFMTTLVL_change_gdf, value_field=f"LOG_{VALUE}_CHNG", group_fields=["START_YR", "END_YR"])
        print(summary_df)

        summary_csv_path = FULL_PANEL_DIR / f"{VALUE}_log_change_summary.csv"
        summary_df.to_csv(summary_csv_path, index=False)
        print(f"[Saved] Summary CSV to: {summary_csv_path}")
        
        
        print(f"\n\nSummarizing LOG_{VALUE}_CHNG by GEOGCODE...")
        summary_df = summarize_field(NFMTTLVL_change_gdf, value_field=f"LOG_{VALUE}_CHNG", group_fields=["GEOGCODE", "START_YR", "END_YR"])
        print(summary_df)

        summary_csv_path = FULL_PANEL_DIR / f"{VALUE}_geogcode_log_change_summary.csv"
        summary_df.to_csv(summary_csv_path, index=False)
        print(f"[Saved] Summary CSV to: {summary_csv_path}")
    #-----------------------------------------------------------------------------------


    print("Done.")
