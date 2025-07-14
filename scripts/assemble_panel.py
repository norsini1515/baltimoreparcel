# scripts/assemble_panel.py

import geopandas as gpd
import pandas as pd
from pathlib import Path
import numpy as np

from baltimoreparcel.directories import DATA_DIR, get_year_gpkg_dir, ensure_dir
from baltimoreparcel.gis_utils import read_vector_layer, write_gpkg_layer, pivot_panel

YEARS = range(2003, 2025)  # or subset like range(2010, 2020)
LAYER_NAME = "{year}subset"
ID_FIELD = "ACCTID"  # You said this is most consistent
VALUE_FIELD = "NFMTTLVL"

KEEP_FIELDS = [ID_FIELD, VALUE_FIELD]

# Where to save output (could be CSV or GPKG or feather)
OUTPUT_PATH = ensure_dir(DATA_DIR / "linked_panels")

#TO MOVE TO GIS_UTILS.PY LATER
def load_subset_layer(year):
    try:
        gpkg_dir = get_year_gpkg_dir(year)
        if gpkg_dir is None:
            print(f"[{year}] GPKG folder missing")
            return None
        layer = LAYER_NAME.format(year=year)
        print(f"[{year}] Looking for layer {gpkg_dir} '{layer}' from Baci{year}.gpkg")
        
        gdf = read_vector_layer(
            year=year,
            name=f"Baci{year}.gpkg",
            directory=gpkg_dir,
            layer=layer
        )
        
        gdf["YEAR"] = year
        print(f"[{year}] Loaded {len(gdf)} rows")
        return gdf

    except Exception as e:
        print(f"[{year}] Failed to load: {e}")      

def load_subset_layers(years) -> dict[int, gpd.GeoDataFrame]:
    subset_data = {}
    for year in years:
        gdf = load_subset_layer(year)
        if gdf is not None:
            subset_data[year] = gdf
        print()
        
    return subset_data

def assemble_panel(subset_data: dict[int, gpd.GeoDataFrame]) -> gpd.GeoDataFrame:
    print("Merging all years data")
    long_df = gpd.GeoDataFrame(
        pd.concat(subset_data.values(), ignore_index=True),
        # crs=subset_data[min(subset_data)].crs  # preserve CRS from earliest year
    )
    long_df = long_df.sort_values(by=[ID_FIELD, "YEAR"]).reset_index(drop=True)

    print(f"Final panel has {len(long_df):} rows, {len(long_df.columns)} columns")
    return long_df

# === MAIN RUN ===
if __name__ == "__main__":
    generate_new_panel = True  # Set to False to re-read existing panel
    #---------------------------------------
    if generate_new_panel:
        print(f"Reading in parcel cleaned_raw_subset data...")
        half_baked_parcel_data = load_subset_layers(YEARS)

        print(f"Assembling panel data...")
        panel_gdf = assemble_panel(half_baked_parcel_data)
        
        print(f"Saving full panel data...")
        output_gpkg_dir = get_year_gpkg_dir("full_panel", create=True)
        gpkg_filename = f"Baci_full_panel.gpkg"
        write_gpkg_layer(panel_gdf, year='full_panel', name=gpkg_filename, directory=output_gpkg_dir, layer=f"full_panel_subset")
    else:
        gpkg_filename = f"Baci_full_panel.gpkg"
        output_gpkg_dir = get_year_gpkg_dir("full_panel")
        panel_gdf = read_vector_layer(year='full_panel', name=gpkg_filename, directory=output_gpkg_dir, layer="full_panel_subset")
        print(f"Re-read panel has {len(panel_gdf):} rows")
    #---------------------------------------
    #make a wide format for mapping in ArcGIS
    # Pivot for ArcGIS mapping: one row per ACCTID with one column per year
    print("Creating wide-format pivot for mapping...")
    PIVOT_FIELDS = [VALUE_FIELD]  # Could expand to more fields later
    for value_field in PIVOT_FIELDS:
        print(f"Processing pivot for {value_field}...")
        value_pivot = pivot_panel(panel_gdf=panel_gdf, value_field=value_field)
        write_gpkg_layer(value_pivot, year="wide_panel", name=gpkg_filename, directory=output_gpkg_dir, layer="NFMTTLVL_pivot")
    print("Done.")
