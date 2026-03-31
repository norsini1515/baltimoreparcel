# baltimoreparcel/gis/io.py
# GIS data read / write / export functions

import geopandas as gpd
from pathlib import Path
import arcpy

from ..utils import info, success, error, warn
from .validate import is_valid_gis_file, sanitize_geometry, drop_null_geometries


def arcstr(path) -> str:
    """Convert a Path (or string) to a forward-slash string safe for ArcPy."""
    return str(path).replace("\\", "/")

def read_gis_file(path: Path, layer: str | None = None) -> gpd.GeoDataFrame:
    if layer:
        return gpd.read_file(path, layer=layer)
    return gpd.read_file(path)

def read_vector_layer(
    year: int,
    name: str,
    directory: Path,
    layer: str | None = None,
) -> gpd.GeoDataFrame | None:
    """Read a shapefile or GeoPackage layer for a given year."""
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
    """Write a GeoDataFrame to a GeoPackage layer."""
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
    out_feature_name: str,
) -> Path | None:
    """
    Export a GeoPackage layer to a File Geodatabase feature class.

    Returns the Path to the exported feature class, or None on failure.
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
