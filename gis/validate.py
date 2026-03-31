# baltimoreparcel/gis/validate.py
# Geometry validation and sanitization helpers

import geopandas as gpd
from shapely.ops import transform


def is_valid_gis_file(name: str) -> bool:
    """Check if a filename is a valid GIS file (shapefile or GeoPackage)."""
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

def strip_z(geom):
    """Force geometry to 2D (drop Z coordinate)."""
    if geom is None:
        return None
    return transform(lambda x, y, *_: (x, y), geom)

def sanitize_geometry(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Force all geometries in a GeoDataFrame to 2D."""
    print("[DEBUG] Sanitizing geometry: forcing 2D...")
    gdf["geometry"] = gdf["geometry"].apply(strip_z)
    return gdf
