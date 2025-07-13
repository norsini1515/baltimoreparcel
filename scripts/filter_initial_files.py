#file: scripts/filter_initial_files.py
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from typing import Iterable
from itertools import repeat
from shapely.geometry import box

from baltimoreparcel.gis_utils import (
    read_vector_layer, ensure_crs, filter_on_field,
    select_columns, write_gpkg_layer
)
from baltimoreparcel.directories import (
    RAW_DIR, FILTERED_DIR, get_year_gpkg_dir
)

from baltimoreparcel.config import PARCEL_FIELDS

# === CONFIGURATION ===
OUTPUT_DIR = FILTERED_DIR #alias for clarity
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def process_year(year: int, column_subset:Iterable[str]=PARCEL_FIELDS):
    try:
        print(f"\nProcessing year: {year}")
        in_name = f"Baci{year}.shp"
        gdf = read_vector_layer(year, name=in_name, directory=RAW_DIR / f"{year}")
        if gdf is None:
            return

        output_gpkg_dir = get_year_gpkg_dir(year, create=True)
        gpkg_filename = f"Baci{year}.gpkg"

        #Ensure correct CRS
        gdf = ensure_crs(gdf, epsg=26985)
        filtered_gdf = filter_on_field(
            gdf,
            fields=["NFMTTLVL", "ADDRESS"],
            filters=[
                5,
                lambda s: s.notna() & (s.str.strip() != "")
            ],
            identifier=year
        )

        # If no valid rows, skip
        if filtered_gdf is None or filtered_gdf.empty:
            print(f"[{year}] Skipped subset – no valid rows")
            return
        
        # Export filtered full file
        write_gpkg_layer(filtered_gdf, year=year, name=gpkg_filename, directory=output_gpkg_dir, layer=f"{year}filtered")
        
        # Subset to relevant fields- easier to work with
        subset_gdf = select_columns(filtered_gdf, columns=column_subset)
        write_gpkg_layer(subset_gdf, year=year, name=gpkg_filename, directory=output_gpkg_dir, layer=f"{year}subset")

        print(f"Completed processing for year: {year}")
    except Exception as e:
        print(f"[{year}] ERROR: {e}")

### === MULTI-PROCESSING MAIN === ###
def main(years:Iterable[int], column_subset:Iterable[str]=PARCEL_FIELDS):
    with ProcessPoolExecutor(max_workers=4) as executor:
        list(tqdm(
            executor.map(process_year, years, repeat(column_subset)),
            total=len(years)
        ))

# === MAIN RUN ===
if __name__ == "__main__":
    #could be set to rerun specific years
    YEARS = range(2003, 2025)
    # YEARS = range(2022, 2023)
    main(years=YEARS, column_subset=PARCEL_FIELDS)


    '''
    Future: could write multiple subsets here
    e.g. core fields only, valuation + modeling fields, vacant lots only, etc.

    # Define different subsets
    SUBSET_CONFIGS = {
        "core": PARCEL_FIELDS,
        "valuation_modeling": PARCEL_FIELDS + ["OWNNAME1", "LU", "ZONING"],
        "vacant_lots": ["ACCTID", "LU", "ACRES", "NFMTTLVL", "geometry"]
    }
    for layer_name, cols in SUBSET_CONFIGS.items():
        gdf_subset = select_columns(filtered_gdf , cols)
        write_gpkg_layer(gdf_subset, year, gpkg_filename, output_gpkg_dir, layer=layer_name)
    '''