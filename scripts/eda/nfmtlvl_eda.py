#scripts/eda/nfmtlvl_eda.py
'''Exploratory Data Analysis of NFMTTLVL in the panel data
===============================================================================
'''
from baltimoreparcel.gis_utils import read_vector_layer, write_gpkg_layer, pivot_panel
from baltimoreparcel.directories import get_year_gpkg_dir
from baltimoreparcel.engineer_panel import log_value

import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
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

def timeseries_histogram(
    value_gdf: gpd.GeoDataFrame, 
    value_field: str = "NFMTTLVL",
    n_cols: int = 4,
    bins: int = 30,
    save_path: Path | None = None,
    show_plot: bool = True,
    plot_trend: bool = True,
    trend_save_path: Path | None = None
):
    """
    Plot a grid of histograms for each year of a pivoted value column.

    Parameters:
        value_gdf   : GeoDataFrame with pivoted year columns (e.g., NFMTTLVL_2003, ..., NFMTTLVL_2024)
        value_field : Field prefix (default = 'NFMTTLVL')
        n_cols      : Number of columns in the subplot grid
        bins        : Number of bins in the histogram
        save_path   : Optional path to save the figure (as PNG or PDF)
        show_plot   : Whether to display the plot interactively
    """
    print(f"Creating histogram grid for {value_field}...")

    years = sorted(
        [col for col in value_gdf.columns if col.startswith(f"{value_field}_")], 
        key=lambda x: int(x.split("_")[-1])
    )
    print(f"Years found for {value_field}: {years}")
    year_nums = [int(col.split("_")[-1]) for col in years]

    n_rows = int(np.ceil(len(years) / n_cols))
    print(f"Creating histogram with {n_rows} rows and {n_cols} columns")

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4.5, n_rows * 3.5), constrained_layout=True)
    axes = axes.flatten()

    medians = []
    means = []

    for i, col in enumerate(years):
        ax = axes[i]
        data = value_gdf[col].dropna()
        ax.hist(data, bins=bins, color='steelblue', edgecolor='black')
        ax.set_title(col.replace(f"{value_field}_", ""), fontsize=10)
        ax.set_yticks([])
        ax.grid(True, linestyle='--', alpha=0.5)

        median = np.median(data)
        mean = np.mean(data)
        medians.append(median)
        means.append(mean)

        ax.axvline(median, color='darkred', linestyle='--', linewidth=1)
        ax.text(median, ax.get_ylim()[1]*0.9, f"{int(median):,}", 
                rotation=90, va='top', ha='right', fontsize=6, color='darkred')

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle(f"{value_field} Distributions by Year", fontsize=16)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[Saved] Histogram grid saved to: {save_path}")
    if show_plot:
        plt.show()

    if plot_trend:
        plt.figure(figsize=(10, 6))
        plt.plot(year_nums, medians, marker='o', label='Median', color='darkred')
        plt.plot(year_nums, means, marker='s', label='Mean', color='darkblue', linestyle='--')
        plt.title(f"{value_field} Trend Over Time")
        plt.xlabel("Year")
        plt.ylabel(value_field)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()

        if trend_save_path:
            plt.savefig(trend_save_path, dpi=150, bbox_inches="tight")
            print(f"[Saved] Trend plot saved to: {trend_save_path}")
        if show_plot:
            plt.show()

def calculate_change(value_gdf: gpd.GeoDataFrame, value_field: str = "NFMTTLVL") -> gpd.GeoDataFrame:
    """
    Calculate absolute and percentage change between the first and last year for a given value field.

    Parameters:
        value_gdf   : GeoDataFrame with pivoted year columns (e.g., NFMTTLVL_2003, ..., NFMTTLVL_2024)
        value_field : Field prefix (default = 'NFMTTLVL')

    Returns:
        GeoDataFrame with additional columns for absolute and percentage change.
    """
    # Step 1: Identify sorted value columns and years
    years = sorted(
        [col for col in value_gdf.columns if col.startswith(f"{value_field}_")], 
        key=lambda x: int(x.split("_")[-1])
    )
    if len(years) < 2:
        raise ValueError("At least two years of data are required to calculate change.")

    first_year_col = years[0]
    last_year_col = years[-1]

    value_gdf = value_gdf.copy()
    value_gdf[f"{value_field}_ABS_CHANGE"] = value_gdf[last_year_col] - value_gdf[first_year_col]
    value_gdf[f"{value_field}_PCT_CHANGE"] = (
        (value_gdf[f"{value_field}_ABS_CHANGE"] / value_gdf[first_year_col].replace(0, np.nan)) * 100
    )

    print(f"Calculated absolute and percentage change from {first_year_col} to {last_year_col}")
    return value_gdf

if __name__ == "__main__":
    VALUE = "NFMTTLVL"
    #load full panel data
    panel_gdf = load_panel_data()
    #apply log transform
    panel_gdf = log_value(panel_gdf, value_field=VALUE)
    #pivot to wide format
    NFMTTLVL_gdf = pivot_panel(panel_gdf=panel_gdf, value_field=f"LOG_{VALUE}")
    print(NFMTTLVL_gdf.columns.to_list())

    histogram_path = FULL_PANEL_DIR / f"{VALUE}_histograms.png"
    trend_path = FULL_PANEL_DIR / f"{VALUE}_trend.png"

    if False:
        timeseries_histogram(NFMTTLVL_gdf, value_field=f"LOG_{VALUE}", 
                         save_path=histogram_path,
                         show_plot=True,
                         n_cols=4, bins=30,
                         plot_trend=True,
                         trend_save_path=trend_path)
        
    print(f"Writing pivoted LOG_{VALUE} to GeoPackage...")
    write_gpkg_layer(NFMTTLVL_gdf, year=f"LOG_{VALUE}_wide_panel", 
                     name=FULL_PANEL_GEOPKG, directory=FULL_PANEL_DIR, 
                     layer="LOG_NFMTTLVL_pivot")

    print("Done.")
