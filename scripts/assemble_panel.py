# scripts/assemble_panel.py
"""
assemble_panel.py

This script constructs the full longitudinal parcel panel and computes parcel-level change metrics 
across time. Outputs are saved to both a GeoPackage and a File Geodatabase for use in ArcGIS Pro 
and other spatial analysis tools.

Main Workflow:
--------------
1. Load parcel subset layers for each year (from GeoPackages).
2. Assemble full long-format panel (merged across years by ACCTID).
3. Optionally apply:
   - CPI-based inflation adjustment to monetary fields (e.g., NFMTTLVL → REAL_NFMTTLVL)
   - Log transformation of selected value fields (e.g., NFMTTLVL → LOG_NFMTTLVL)
4. Construct the change panel:
   - Pivot each selected variable wide by year
   - Compute raw difference: CHANGE = VALUE_t2 - VALUE_t1 or STRING_CHANGE = VALUE_t2 ≠ VALUE_t1
   - Merge calculated changes with most recent geometry
5. Export the final change panel:
   - Save to GeoPackage (`Baci_full_panel.gpkg`)
   - Export to File Geodatabase via ArcPy (`full_change_panel`)
6. Convert temporal fields to DATE fields for ArcGIS visualization:
   - Use ArcPy `ConvertTimeField` to convert START_YR and END_YR to ArcGIS-readable time fields
   - Enables temporal filtering and animation in ArcGIS Pro

Key Outputs:
------------
- `Baci_full_panel.gpkg` with `full_panel_subset` and `full_change_panel` layers
- FileGDB feature class (`full_change_panel`) with ArcGIS time fields

Dependencies:
-------------
- Requires ArcPy (ArcGIS Pro Python environment)
- Assumes processed GeoPackages exist for all years in `YEARS`
- CPI reference file must be present as `Baltimore_CPI.csv` in full panel directory
"""

import sys
import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path
from datetime import datetime

import arcpy

from baltimoreparcel.directories import DATA_DIR, LOGS_DIR, GBD_DIR, get_year_gpkg_dir, ensure_dir
from baltimoreparcel.gis_utils import read_vector_layer, write_gpkg_layer, pivot_panel, export_to_geodb, convert_time_fields
from baltimoreparcel.engineer_panel import enrich_change_gdf, log_value, calculate_change, to_real_data, spatial_join_with_neighborhoods
from baltimoreparcel.utils import Logger, info, warn, error, success

YEARS = range(2003, 2025)  # or subset like range(2010, 2020)
#values to calculate real values for using CPI adjustment
MONETARY_VALUES = ['NFMLNDVL', 'NFMIMPVL', 'NFMTTLVL']
#values to log transform
CALCULATE_LOG_FIELDS = ['NFMLNDVL', 'NFMIMPVL', 'NFMTTLVL', 'REAL_NFMLNDVL', 'REAL_NFMIMPVL', 'REAL_NFMTTLVL']

NUMERIC_FIELD_VALUES = ['ACRES', 'YEARBLT', 'SQFTSTRC', 
                           'NFMLNDVL', 'LOG_NFMTTLVL',
                           'NFMIMPVL', 'LOG_NFMIMPVL',
                           'NFMTTLVL', 'LOG_NFMLNDVL',
                           'REAL_NFMLNDVL', 'LOG_REAL_NFMLNDVL', 
                           'REAL_NFMIMPVL', 'LOG_REAL_NFMIMPVL', 
                           'REAL_NFMTTLVL', 'LOG_REAL_NFMTTLVL'
                        ]

STR_NUMERIC_FIELD_VALUES = ['LU', 'ZONING', 'OWNNAME1']
#values to calculate change for
CALCULATE_CHANGE_VALUES = NUMERIC_FIELD_VALUES + STR_NUMERIC_FIELD_VALUES


LAYER_NAME = "{year}subset"
ID_FIELD = "ACCTID"  # You said this is most consistent

# Where to save output (could be CSV or GPKG or feather)
OUTPUT_PATH = ensure_dir(DATA_DIR / "linked_panels")

FULL_PANEL_GEOPKG = "Baci_full_panel.gpkg"
FULL_PANEL_DIR = get_year_gpkg_dir("full_panel")

def read_full_panel():
    panel_gdf = read_vector_layer(year='full_panel', name=FULL_PANEL_GEOPKG, directory=FULL_PANEL_DIR, layer="full_panel_subset")
    
    print(f"Read panel has {panel_gdf.shape[0]} rows, {panel_gdf.shape[1]} columns")
    return panel_gdf

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

def calculate_real_data(panel_gdf:gpd.GeoDataFrame=None, monetary_values:list[str]=MONETARY_VALUES):
    if panel_gdf is None:
        print("Panel data not loaded, reading existing full panel...")
        panel_gdf = read_full_panel()

    #transform nominal monetary values to real values
    print("Transforming nominal to real values...")
    print('\treading in CPI data...')
    prices = pd.read_csv(FULL_PANEL_DIR / "Baltimore_CPI.csv")
    prices = prices.set_index('YEAR')
    
    for val in monetary_values:
        if val not in panel_gdf.columns:
            print(f"\tSkipping {val}, not in columns")
            continue

        panel_gdf = panel_gdf.set_index(['ACCTID', 'YEAR'])
        print(f"\tCalcualting real {val}...")
        panel_gdf[f'REAL_{val}'] = to_real_data(panel_gdf[val], prices)
        panel_gdf = panel_gdf.reset_index()

    return panel_gdf

def calculate_log_fields(panel_gdf:gpd.GeoDataFrame=None, log_fields:list[str]=CALCULATE_LOG_FIELDS):
    if panel_gdf is None:
        print("Panel data not loaded, reading existing full panel...")
        panel_gdf = read_full_panel()

    #log transform value fields
    print("Creating log-transformed fields...")
    for val in log_fields:
        if val in panel_gdf.columns:
            print(f"\tLogging {val}...")
            panel_gdf= log_value(panel_gdf, value_field=val)
        else:
            print(f"\tSkipping {val}, not in columns")
    return panel_gdf

# === MAIN RUN ===
if __name__ == "__main__":
    generate_new_panel = False  # Set to False to re-read existing panel
    calculate_real_values = False  # Set to False to skip CPI adjustment
    log_value_fields = False  # Set to True to create log-transformed fields
    build_panel_components = [generate_new_panel, calculate_real_values, log_value_fields] #indicate if we should resave the full panel.
    build_change_panel = True #Set to True to create change panel in full panel characteristics
    
    panel_gdf = None # Initialize panel_gdf to None
    
    arcpy.env.workspace = str(GBD_DIR)
    #---------------------------------------
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y%m%d %H%M")
    logger = Logger(LOGS_DIR / f"assemble_panel_{timestamp}.log")

    #---------------------------------------
    #assemble full panel data using cleaned_raw_subset layers
    if generate_new_panel:
        print("Beginning generation of full panel")
        print(f"Reading in parcel cleaned_raw_subset data...")
        half_baked_parcel_data = load_subset_layers(YEARS)

        print(f"Assembling panel data...")
        panel_gdf = assemble_panel(half_baked_parcel_data)
    #---------------------------------------
    #using baltimore-townson-columbia CPI calcualte real values in 2024 dollars
    if calculate_real_values:
        panel_gdf = calculate_real_data(panel_gdf=panel_gdf, monetary_values=MONETARY_VALUES)
    #---------------------------------------
    #take log of log fields
    if log_value_fields:
        panel_gdf = calculate_log_fields(panel_gdf=panel_gdf, log_fields=CALCULATE_LOG_FIELDS)
    #---------------------------------------

    #if panel_gdf is None, read in existing full panel
    if panel_gdf is None:
        print("Panel data not loaded, reading existing full panel...")
        panel_gdf = read_full_panel()

    info(f"Panel gdf columns: {panel_gdf.columns.tolist()}")
    info(f"{panel_gdf.shape=}")
    
    #write updated panel data file out
    if any(build_panel_components):
        print(f"appending neighborhood information. {panel_gdf.shape=}")
        panel_gdf = spatial_join_with_neighborhoods(panel_gdf)
        print(f"resulting shape: {panel_gdf.shape=}")

        full_panel_name = "full_panel_subset"
        print(f"Saving {full_panel_name} data...")
        write_gpkg_layer(panel_gdf, year='full_panel', name=FULL_PANEL_GEOPKG, directory=FULL_PANEL_DIR, layer=full_panel_name)

        print(f'Now writing {full_panel_name} to geodatabase')
        # Step 1: Export to GDB
        exported_fc_path = export_to_geodb(
            input_gpkg_path=FULL_PANEL_DIR / FULL_PANEL_GEOPKG,
            layer_name=full_panel_name,
            gdb_path=GBD_DIR,
            out_feature_name=full_panel_name
        )
        print('covnerting time fields to date')
        # Step 2: Convert START_YR and END_YR to DATE fields (if export succeeded)
        if exported_fc_path:
            convert_time_fields(
            table_path=exported_fc_path.name,  # now just the name of the FC
            field_pairs=[("YEAR", "YEAR_DATE")]
        )
        success(f"exported {full_panel_name}")
    
    #CONSTRUCT CHANGE PANEL
    if build_change_panel:
        #---------------------------------------
        #make a wide format for mapping in ArcGIS
        # Pivot for ArcGIS mapping: one row per ACCTID with one column per year
        print('-'*105)
        info("[build change panel]Creating change panel...")
        geometry_df = (
            panel_gdf
            .sort_values(["ACCTID", "YEAR"])  # Ensure YEAR is sorted
            .drop_duplicates(subset=["ACCTID"], keep="last")[["ACCTID", "geometry"]]
        )
        change_df = pd.DataFrame(panel_gdf[[ID_FIELD]].drop_duplicates())
        change_df = change_df.merge(geometry_df, on=ID_FIELD, how="left")
        change_df = gpd.GeoDataFrame(change_df, geometry="geometry", crs=panel_gdf.crs)
        
        print(f"\tInitial change dataset (geometries): {change_df.shape=}\n\t{change_df.columns=}")
        for i, value_field in enumerate(CALCULATE_CHANGE_VALUES):
            print(f"\n\nProcessing pivot for {value_field}...")
            if value_field not in panel_gdf.columns:
                print(f"\tSkipping {value_field}, not in columns")
                continue
            
            if value_field in NUMERIC_FIELD_VALUES:
                print(f'\tEnsure that {value_field} is numeric...')
                #change variables are numeric
                panel_gdf[value_field] = pd.to_numeric(panel_gdf[value_field], errors='coerce')
            elif value_field in STR_NUMERIC_FIELD_VALUES:
                print(f'\tEnsure that {value_field} is string...')
                panel_gdf[value_field] = panel_gdf[value_field].astype(str).replace('nan', np.nan)
                
            value_pivot = pivot_panel(panel_gdf=panel_gdf, value_field=value_field)
            # print('-'*100)
            # print(value_pivot.head())
            change_value_pivot = calculate_change(value_pivot, value_prefix=value_field, per_year=False, 
                                                field_type='numeric' if value_field in NUMERIC_FIELD_VALUES else 'string')
            # print('-'*100)
            # print(change_value_pivot.head())
            keep_cols = [ID_FIELD, "START_YR", "END_YR", f"{value_field}_CHNG"]
            change_value_pivot = change_value_pivot[keep_cols]
            print(f"\tPivoted dataset: {change_value_pivot.shape=}")
            #merge change data
            if i == 0:
                change_df = change_value_pivot.copy()
            else:
                change_df = change_df.merge(change_value_pivot, on=[ID_FIELD, "START_YR", "END_YR"], how="outer")

            print(f"\tMerger {i}, {change_df.columns.tolist()=}")
            print(f"\tMerger {i}, {change_df.shape=}")
        
        print(f"\nFinal change dataset before geometry merge: {change_df.shape=}")
        # Merge geometry based on most recent available year per ACCTID
        change_df = change_df.merge(geometry_df, on="ACCTID", how="left")

        # Drop any rows that still lack geometry (ideally zero)
        missing_geo = change_df["geometry"].isna().sum()
        print(f"Rows with missing geometry after merge: {missing_geo}")

        print(f"\nFinal change dataset before dropping NA: {change_df.shape=}")
        change_df = change_df[change_df['START_YR'].notna() & change_df['END_YR'].notna()]
        print(f"Final change dataset after dropping NA: {change_df.shape=}")
        change_df = change_df.sort_values(["ACCTID", "START_YR"]).reset_index(drop=True)
        # Convert to GeoDataFrame
        change_df = gpd.GeoDataFrame(change_df, geometry="geometry", crs=panel_gdf.crs)
        
        change_df = enrich_change_gdf(change_df, panel_gdf, enrich_fields=["NEIGHBORHOOD"])
        print(f"Final GeoDataFrame shape: {change_df.shape=}")
        
        change_panel_name = "full_change_panel"
        #output change data
        write_gpkg_layer(change_df, year="change_panel", name=FULL_PANEL_GEOPKG, directory=FULL_PANEL_DIR, layer=change_panel_name)
        
        #-----------------------------------
        print(f'Now writing {change_panel_name} to geodatabase')
        # Step 1: Export to GDB
        exported_fc_path = export_to_geodb(
            input_gpkg_path=FULL_PANEL_DIR / FULL_PANEL_GEOPKG,
            layer_name=change_panel_name,
            gdb_path=GBD_DIR,
            out_feature_name=change_panel_name
        )
        print('covnerting time fields to date')
        # Step 2: Convert START_YR and END_YR to DATE fields (if export succeeded)
        if exported_fc_path:
            convert_time_fields(
            table_path=exported_fc_path.name,  # now just the name of the FC
            field_pairs=[("START_YR", "START_DATE"), ("END_YR", "END_DATE")]
        )

    print("Done.")