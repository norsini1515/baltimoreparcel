# baltimoreparcel/panel/spatial.py
# Spatial enrichment for panel data

import geopandas as gpd

from ..directories import GBD_DIR


def spatial_join_with_neighborhoods(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Spatially join parcels with Baltimore Neighborhood Statistical Areas (NSAs).
    Appends a 'NEIGHBORHOOD' column from the GDB neighborhoods layer.
    """
    nsa_gdf = (
        gpd.read_file(str(GBD_DIR), layer="neighborhoods")[["Name", "geometry"]]
        .rename(columns={"Name": "NEIGHBORHOOD"})
    )

    if gdf.crs != nsa_gdf.crs:
        nsa_gdf = nsa_gdf.to_crs(gdf.crs)

    joined = gpd.sjoin(gdf, nsa_gdf, how="left", predicate="intersects")
    joined = joined.drop(columns=["index_right"], errors="ignore")
    return joined
