#scripts/eda/nfmtlvl_eda.py
'''Exploratory Data Analysis of NFMTTLVL in the panel data
===============================================================================
'''
from baltimoreparcel.gis_utils import read_vector_layer, write_gpkg_layer, pivot_panel
from baltimoreparcel.directories import get_year_gpkg_dir
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
import numpy as np
from pathlib import Path

def load_panel_data():
    gpkg_filename = f"Baci_full_panel.gpkg"
    output_gpkg_dir = get_year_gpkg_dir("full_panel")
    panel_gdf = read_vector_layer(year='full_panel', name=gpkg_filename, directory=output_gpkg_dir, layer="full_panel_subset")
    print(f"Read panel has {len(panel_gdf):} rows")
    return panel_gdf

def timeseries_histogram(
    value_gdf: gpd.GeoDataFrame, 
    value_field: str = "NFMTTLVL",
    n_cols: int = 4,
    bins: int = 30,
    save_path: Path | None = None,
    show_plot: bool = True
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

    years = sorted([col for col in value_gdf.columns if col.startswith(f"{value_field}_")], 
                   key=lambda x: int(x.split("_")[-1]))
    print(f"Years found for {value_field}: {years}")
    n_rows = int(np.ceil(len(years) / n_cols))
    print(f"Creating histogram with {n_rows} rows and {n_cols} columns")
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4.5, n_rows * 3.5), constrained_layout=True)
    axes = axes.flatten()

    for i, col in enumerate(years):
        ax = axes[i]
        data = value_gdf[col].dropna()
        ax.hist(data, bins=bins, color='steelblue', edgecolor='black')
        ax.set_title(col.replace(f"{value_field}_", ""), fontsize=10)
        ax.set_yticks([])
        ax.grid(True, linestyle='--', alpha=0.5)

        median = np.median(data)
        ax.axvline(median, color='darkred', linestyle='--', linewidth=1)
        ax.text(median, ax.get_ylim()[1]*0.9, f"{int(median):,}", 
                rotation=90, va='top', ha='right', fontsize=6, color='darkred')
        

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle(f"{value_field} Distributions by Year", fontsize=16)

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"[Saved] Histogram grid saved to: {save_path}")
    if show_plot:
        plt.show()
if __name__ == "__main__":
    panel_gdf = load_panel_data()

    unique_panel_gdf = panel_gdf.drop_duplicates(subset=["ACCTID", "YEAR"], keep='first')
    print(f"After dropping duplicate NFMTTLVL rows, data has {len(unique_panel_gdf):} rows")
    unique_panel_gdf['LOG_NFMTTLVL'] = np.log(unique_panel_gdf['NFMTTLVL'])

    NFMTTLVL_gdf = pivot_panel(panel_gdf=unique_panel_gdf, value_field="LOG_NFMTTLVL")
    print(f"Panel data has {len(NFMTTLVL_gdf):} rows, {len(NFMTTLVL_gdf.columns)} columns")
    # print(NFMTTLVL_gdf.head())

    dir = get_year_gpkg_dir("full_panel")
    histogram_path = dir / "NFMTTLVL_histograms.png"
    timeseries_histogram(NFMTTLVL_gdf, value_field="LOG_NFMTTLVL", 
                         save_path=histogram_path,
                         show_plot=True,
                         n_cols=4, bins=30)

