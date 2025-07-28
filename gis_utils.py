# baltimoreparcel/gis_utils.py
# Utility functions for GIS data handling

import geopandas as gpd
from pathlib import Path
from .directories import DATA_DIR, GBD_DIR
from .utils import info, success, error, warn
from shapely.ops import transform
import arcpy


##################### SECTION 1 #####################
# Validation + Path Logic
#####################################################
def is_valid_gis_file(name: str) -> bool:
    """Check if a filename is a valid gis-file name."""
    return name.endswith('.shp') or name.endswith('.gpkg')

def parse_gpkg_name(name: str) -> tuple[str, str | None]:
    """Split GPKG filename and layer name if in 'file.gpkg|layer' format."""
    if name.endswith(".gpkg") and "|" in name:
        return name.split("|", maxsplit=1)
    return name, None

def drop_null_geometries(gdf: gpd.GeoDataFrame, year: int = None) -> gpd.GeoDataFrame:
    nulls = gdf.geometry.isnull().sum()
    if nulls > 0:
        if year:
            print(f"[{year}] Dropping {nulls} rows with null geometry")
        gdf = gdf[gdf.geometry.notnull()].copy()
    return gdf

def sanitize_geometry(gdf):
    print("[DEBUG] Sanitizing geometry: forcing 2D...")
    gdf["geometry"] = gdf["geometry"].apply(strip_z)
    return gdf

def strip_z(geom):
    """Force geometry to 2D (drop Z coordinate)."""
    if geom is None:
        return None
    return transform(lambda x, y, *_: (x, y), geom)
##################### SECTION 2 #####################
# IO Functions
#####################################################
def read_gis_file(path: Path, layer: str | None = None) -> gpd.GeoDataFrame:
    if layer:
        return gpd.read_file(path, layer=layer)
    return gpd.read_file(path)

def read_vector_layer(year: int, name: str, directory: Path, layer: str | None = None) -> gpd.GeoDataFrame | None:
    """Read shapefile for a given year."""
    
    if not is_valid_gis_file(name):
        print(f"[{year}] Skipped – invalid file type: {name}")
        return None

    input_path = directory / name
    print(f"[{year}] Reading {input_path} layer '{layer}'")
    
    if not input_path.exists():
        print(f"[{year}] Skipped – file not found: {input_path}")
        return None

    try:
        gdf = gpd.read_file(input_path, layer=layer) if layer else gpd.read_file(input_path)
        print(f"[{year}] Read {len(gdf)} rows from {input_path.name} layer '{layer}'")
        return gdf
    except Exception as e:
        print(f"[{year}] Error reading layer '{layer}' in {name}: {e}")
        return None
    
def write_gpkg_layer(
    gdf: gpd.GeoDataFrame,
    year: int,
    name: str,
    directory: Path,
    drop_nulls: bool = True,
    layer: str = "layer",
) -> None:
    """
    Write GeoDataFrame to GeoPackage.
    Each layer (e.g. 'filtered', 'subset') is written into the same .gpkg file per year.
    """
    output_path = directory / name
    ext = output_path.suffix.lower()

    if ext != ".gpkg":
        print(f"[{year}] Warning – only .gpkg is supported now")
        return

    if gdf is None or gdf.empty:
        print(f"[{year}] Skipped – GeoDataFrame is None or empty")
        return
    
    print("Sanitizing geometry...")
    gdf = sanitize_geometry(gdf)
    print(gdf.head())

    if drop_nulls:
        print("Dropping null geometries before write...")
        gdf = drop_null_geometries(gdf, year=year)

    try:
        gdf.to_file(output_path, layer=layer, driver="GPKG", overwrite=True)
        print(f"[{year}] Wrote {len(gdf)} rows to layer '{layer}' in {output_path.name}")
        
    except Exception as e:
        print(f"[{year}] Error writing layer '{layer}': {e}")

def export_to_geodb(
    input_gpkg_path: Path,
    layer_name: str,
    gdb_path: Path,
    out_feature_name: str
) -> Path:
    """
    Export a layer from a GeoPackage (.gpkg) to a File Geodatabase (.gdb).

    Parameters:
    ----------
    input_gpkg_path : Path
        Path to the .gpkg file containing the source layer.
    layer_name : str
        Name of the layer inside the GPKG.
    gdb_path : Path
        Path to the existing File Geodatabase (.gdb).
    out_feature_name : str
        Name for the output feature class inside the GDB.

    Returns:
    -------
    Path to the exported feature class (str), or None if failed.
    """
    arcpy.env.overwriteOutput = True
    
    input_layer = f"{input_gpkg_path}\\{layer_name}"
    output_fc = gdb_path / out_feature_name

    try:
        if arcpy.Exists(str(output_fc)):
            warn(f"Feature class already exists. Deleting: {output_fc}")
            arcpy.management.Delete(str(output_fc))

        info(f"Exporting {layer_name} from {input_gpkg_path.name} to {output_fc}...")
        arcpy.conversion.ExportFeatures(
            in_features=input_layer,
            out_features=str(output_fc)
        )

        success(f"Export successful: {output_fc}")
        return output_fc

    except Exception as e:
        error(f"export_to_geodb failed: {e}")
        return None

##################### SECTION 3 #####################
# Data Cleaning + Processing Functions
#####################################################
def ensure_crs(gdf: gpd.GeoDataFrame, epsg: int = 2248) -> gpd.GeoDataFrame:
    """
    Ensure GeoDataFrame has a CRS. Set or reproject to specified EPSG if needed.
    EPSG 2248 = NAD83 / Maryland (ftUS), ideal for Baltimore parcel data.
    """
    if gdf.crs is None:
        print("CRS undefined – assigning EPSG:", epsg)
        gdf.set_crs(epsg=epsg, inplace=True)
    elif gdf.crs.to_epsg() != epsg:
        print(f"Reprojecting from {gdf.crs} to EPSG:{epsg}")
        gdf = gdf.to_crs(epsg=epsg)
    return gdf

def filter_on_field(
        gdf: gpd.GeoDataFrame,
        fields: list,
        filters: list,
        identifier: str,
    ) -> gpd.GeoDataFrame:
    
    """
    Apply a list of filters to corresponding fields in a GeoDataFrame.
    
    Each filter can be:
        - A scalar (e.g., 0), which applies 'greater than' filter.
        - A callable, which takes a Series and returns a Boolean mask.
    """
    if len(fields) != len(filters):
        raise ValueError("Fields and filters lists must be the same length.")
    
    if gdf is None:
        print("Input GeoDataFrame is None.")
        return gdf  # return as-is, or None

    filtered_gdf = gdf.copy()
    for field, rule in zip(fields, filters):
        if field not in filtered_gdf.columns:
            print(f"[{identifier}] WARNING: Field '{field}' not in dataframe.")
            return None
        
        if callable(rule):
            mask = rule(filtered_gdf[field])
        else:
            mask = filtered_gdf[field] > rule  # default to 'greater than'
    
        before = len(filtered_gdf)
        filtered_gdf = filtered_gdf[mask]
        after = len(filtered_gdf)
        print(f"[{identifier}] Filtered {field}: {before} → {after}")
    
    return filtered_gdf if not filtered_gdf.empty else None

def select_columns(gdf: gpd.GeoDataFrame, columns: list[str]) -> gpd.GeoDataFrame:
    """Keep only specified columns, warn if any are missing."""
    missing = [col for col in columns if col not in gdf.columns]
    keep = [col for col in columns if col in gdf.columns]
    if missing:
        print(f"Warning – missing columns: {missing}")
    return gdf[keep]

def pivot_panel(
    panel_gdf: gpd.GeoDataFrame,
    value_field: str,
    year_for_geometry: int = 2024,
    id_field: str = "ACCTID",
    aggfunc: str = "first",  # default to safer choice
) -> gpd.GeoDataFrame:
    """
    Pivot a long panel GeoDataFrame to wide format by year for a specified value field.
    Keeps geometry from the reference year.
    """
    print(f"Pivoting '{value_field}' by year...")

    # Pivot
    pivot_gdf = panel_gdf.pivot_table(
        index=id_field,
        columns="YEAR",
        values=value_field,
        aggfunc=aggfunc,
    ).reset_index()

    print(f"{value_field} pivoted data has {len(pivot_gdf):} rows, {len(pivot_gdf.columns)} columns")
    
    # Rename year columns: e.g., NFMTTLVL_2020
    pivot_gdf.columns = [
        f"{value_field}_{col}" if isinstance(col, int) else col
        for col in pivot_gdf.columns
    ]

    # Extract geometry from specified year
    geom_df = (
        panel_gdf
        .dropna(subset=["geometry"])
        .sort_values(["ACCTID", "YEAR"])  # ensure earliest geometry if needed
        .groupby(id_field, as_index=False)
        .first()[[id_field, "geometry"]]
    )
    print(f"[Geometry] Found {len(geom_df)} rows from {year_for_geometry}")
    
    # Merge geometry and pivoted data
    merged = geom_df.merge(pivot_gdf, on=id_field, how="left")

    print(f"[Pivoted Panel] Final shape: {merged.shape}")
    return gpd.GeoDataFrame(merged, geometry="geometry", crs=panel_gdf.crs)

def convert_time_fields(
    table_path: Path | str,
    field_pairs: list[tuple[str, str]],
    input_format: str = "yyyy",
    output_type: str = "DATE"
):
    """
    Convert integer year fields to ArcGIS Date fields using ConvertTimeField.

    Parameters:
    ----------
    table_path : Path or str
        Path to the GDB feature class or table (e.g., '.../BaltimoreParcelProject.gdb/change_panel').
    field_pairs : list of tuples
        Each tuple is (input_year_field, output_date_field).
    input_format : str, optional
        Format of the input year field. Defaults to "yyyy".
    output_type : str, optional
        Output time type. Use "DATE" for standard ArcGIS Date field.

    Returns:
    -------
    None
    """
    try:
        print(f"Converting time fields in {table_path}...")
        for in_field, out_field in field_pairs:
            print(f"{in_field} -> {out_field}...")
            arcpy.management.ConvertTimeField(
                in_table=str(table_path),
                input_time_field=in_field,
                input_time_format=input_format,
                output_time_field=out_field,
                output_time_type=output_type,
                output_time_format="",  # defaults to match input
                timezone_or_field=""
            )
        success(f"Time conversion complete.")
    except Exception as e:
        error(f"convert_time_fields failed: {e}")