# baltimoreparcel/panel/spatial.py
# Spatial enrichment for panel data

from pathlib import Path
from typing import Optional

import geopandas as gpd

from ..directories import GBD_DIR


def spatial_join_with_neighborhoods(
    gdf: gpd.GeoDataFrame,
    gdb_path: Optional[Path] = None,
    layer: str = "neighborhoods",
    name_field: str = "Name",
    output_field: str = "NEIGHBORHOOD",
) -> gpd.GeoDataFrame:
    """
    Spatially join parcels with a polygon reference layer (e.g. neighborhoods).

    Appends one column to ``gdf`` whose name is ``output_field``, containing
    the value of ``name_field`` from whichever reference polygon each parcel
    intersects.

    Parameters
    ----------
    gdf : GeoDataFrame
        Parcel (or any point/polygon) GeoDataFrame to enrich.
    gdb_path : Path, optional
        Path to the GDB or file containing the reference layer.
        Defaults to the project GDB defined in directories.py.
    layer : str
        Layer name within the GDB/file.  Default: ``"neighborhoods"``.
    name_field : str
        Field in the reference layer whose value is carried over.
        Default: ``"Name"``.
    output_field : str
        Column name written into the result.  Default: ``"NEIGHBORHOOD"``.

    Returns
    -------
    GeoDataFrame
        ``gdf`` with ``output_field`` appended.
    """
    if gdb_path is None:
        gdb_path = GBD_DIR

    ref_gdf = (
        gpd.read_file(str(gdb_path), layer=layer)[[name_field, "geometry"]]
        .rename(columns={name_field: output_field})
    )

    if ref_gdf.crs != gdf.crs:
        ref_gdf = ref_gdf.to_crs(gdf.crs)

    joined = gpd.sjoin(gdf, ref_gdf, how="left", predicate="intersects")
    joined = joined.drop(columns=["index_right"], errors="ignore")
    return joined
