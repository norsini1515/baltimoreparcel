#scripts/eda/nfmtlvl_eda.py
'''Exploratory Data Analysis of NFMTTLVL in the panel data
===============================================================================
'''
from baltimoreparcel.gis_utils import read_vector_layer, write_gpkg_layer, pivot_panel
from baltimoreparcel.directories import get_year_gpkg_dir
from baltimoreparcel.engineer_panel import log_value, calculate_change, summarize_field, enrich_change_gdf
from baltimoreparcel.scripts.eda.plots import timeseries_histogram

import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
import pandas as pd
import numpy as np
from pathlib import Path

FULL_PANEL_GEOPKG = "Baci_full_panel.gpkg"
FULL_PANEL_DIR = get_year_gpkg_dir("full_panel")
LAYER_NAME = "full_panel_subset"

def load_panel_data():
    panel_gdf = read_vector_layer(year='full_panel', name=FULL_PANEL_GEOPKG, 
                                  directory=FULL_PANEL_DIR, layer=LAYER_NAME)
    print(f"Read panel has {panel_gdf.shape[0]} rows, {panel_gdf.shape[1]} columns")
    return panel_gdf


if __name__ == "__main__":
    VALUE = "NFMTTLVL"
    
    read_in_full_panel = False
    create_change_data = False
    output_files = True
    #-----------------------------------------------------------------------------------
    print(f"Starting EDA for {VALUE}...")
    if read_in_full_panel:
        #load full panel data
        panel_gdf = load_panel_data()
        #apply log transform
        panel_gdf = log_value(panel_gdf, value_field=VALUE)
        static_fiels = ["ACCTID", "GEOGCODE"]
        static_cols_df = panel_gdf[static_fiels].drop_duplicates().set_index("ACCTID")
        #pivot to wide format
        NFMTTLVL_gdf = pivot_panel(panel_gdf=panel_gdf, value_field=f"LOG_{VALUE}")
        NFMTTLVL_gdf = NFMTTLVL_gdf.merge(static_cols_df, left_on="ACCTID", right_index=True, how="left")
        print(NFMTTLVL_gdf.columns.to_list())
        
        if output_files:
            print(f"Writing pivoted LOG_{VALUE} to GeoPackage...")
            write_gpkg_layer(NFMTTLVL_gdf, year=f"LOG_{VALUE}_wide_panel", 
                            name=FULL_PANEL_GEOPKG, directory=FULL_PANEL_DIR, 
                            layer="LOG_NFMTTLVL_pivot")
    else:
        NFMTTLVL_gdf = read_vector_layer(year=f"LOG_{VALUE}_wide_panel", name=FULL_PANEL_GEOPKG, 
                                        directory=FULL_PANEL_DIR, layer="LOG_NFMTTLVL_pivot")
        print(f"Read pivoted LOG_{VALUE} has {NFMTTLVL_gdf.shape[0]} rows, {NFMTTLVL_gdf.shape[1]} columns")
    
    print(NFMTTLVL_gdf.head())
    print('-'*150, '\n')
    #-----------------------------------------------------------------------------------
    if create_change_data:
        #Calculate change in NFMTTLVL
        print(f"Calculating change in LOG_{VALUE}...")
        #calculate change- total change per change group
        NFMTTLVL_change_gdf = calculate_change(NFMTTLVL_gdf, value_prefix=f"LOG_{VALUE}", per_year=False)
        #enrich with static fields
        NFMTTLVL_change_gdf = enrich_change_gdf(NFMTTLVL_change_gdf, NFMTTLVL_gdf, ["GEOGCODE"])

        print(NFMTTLVL_change_gdf.columns.to_list())
        print(NFMTTLVL_change_gdf.head())
        print(f"Change GeoDataFrame has {NFMTTLVL_change_gdf.shape[0]} rows, {NFMTTLVL_change_gdf.shape[1]} columns")
        if output_files:
            print(f"Writing change in LOG_{VALUE} to GeoPackage...")
            write_gpkg_layer(NFMTTLVL_change_gdf, year=f"LOG_{VALUE}_change_wide_panel", 
                            name=FULL_PANEL_GEOPKG, directory=FULL_PANEL_DIR, 
                            layer="LOG_NFMTTLVL_change")
    else:
        NFMTTLVL_change_gdf = read_vector_layer(year=f"LOG_{VALUE}_change_wide_panel", name=FULL_PANEL_GEOPKG, 
                                        directory=FULL_PANEL_DIR, layer="LOG_NFMTTLVL_change")
        print(f"Read change in LOG_{VALUE} has {NFMTTLVL_change_gdf.shape[0]} rows, {NFMTTLVL_change_gdf.shape[1]} columns")
    print(NFMTTLVL_change_gdf.head())
    print('-'*150, '\n')
    #-----------------------------------------------------------------------------------
    #EDA plots
    plot_log_value_distributions = False
    if plot_log_value_distributions:
        print(f"\n\nPlotting LOG_{VALUE} distributions...")
        histogram_path = FULL_PANEL_DIR / f"{VALUE}_histograms.png"
        trend_path = FULL_PANEL_DIR / f"{VALUE}_trend.png"

        timeseries_histogram(NFMTTLVL_gdf, value_field=f"LOG_{VALUE}", 
                         save_path=histogram_path,
                         show_plot=True,
                         n_cols=4, bins=30,
                         plot_trend=True,
                         trend_save_path=trend_path)
    #-----------------------------------------------------------------------------------    
    plot_log_value_chng_distributions = False
    if plot_log_value_chng_distributions:
        print(f"\n\nPlotting LOG_{VALUE}_CHNG distributions...")
        change_path = FULL_PANEL_DIR / f"{VALUE}_change_histograms.png"
        change_trend_path = FULL_PANEL_DIR / f"{VALUE}_change_trend.png"
        timeseries_histogram(NFMTTLVL_change_gdf, value_field=f"LOG_{VALUE}_CHNG", 
                            save_path=change_path,
                            show_plot=True,
                            n_cols=4, bins=30,
                            plot_trend=True,
                            trend_save_path=change_trend_path)
    #-----------------------------------------------------------------------------------
    summarize_change = True
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
