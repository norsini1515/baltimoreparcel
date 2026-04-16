# baltimoreparcel/gis/transform.py
# Data cleaning, filtering, pivoting, and ArcGIS time-field conversion

import geopandas as gpd
from pathlib import Path
import arcpy

from ..utils import success, error


def ensure_crs(gdf: gpd.GeoDataFrame, epsg: int = 2248) -> gpd.GeoDataFrame:
    """
    Ensure a GeoDataFrame has the correct CRS.
    EPSG 2248 = NAD83 / Maryland (ftUS), default for Baltimore parcel data.
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
) -> gpd.GeoDataFrame | None:
    """
    Apply a list of filters to corresponding fields in a GeoDataFrame.

    Each filter can be:
        - A scalar (e.g., 0): applies a 'greater than' comparison.
        - A callable: receives the Series and returns a boolean mask.
    """
    if len(fields) != len(filters):
        raise ValueError("Fields and filters lists must be the same length.")

    if gdf is None:
        print("Input GeoDataFrame is None.")
        return gdf

    filtered_gdf = gdf.copy()
    for field, rule in zip(fields, filters):
        if field not in filtered_gdf.columns:
            print(f"[{identifier}] WARNING: Field '{field}' not in dataframe.")
            return None

        mask = rule(filtered_gdf[field]) if callable(rule) else filtered_gdf[field] > rule

        before = len(filtered_gdf)
        filtered_gdf = filtered_gdf[mask]
        after = len(filtered_gdf)
        print(f"[{identifier}] Filtered {field}: {before} --> {after}")

    return filtered_gdf if not filtered_gdf.empty else None

def select_columns(gdf: gpd.GeoDataFrame, columns: list[str]) -> gpd.GeoDataFrame:
    """Keep only specified columns, warn about any that are missing."""
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
    col_field: str = "YEAR",
    aggfunc: str = "first",
) -> gpd.GeoDataFrame:
    """
    Pivot a long-format panel GeoDataFrame to wide format by year.
    One column per year for the specified value field. Geometry is
    taken from the earliest available row per parcel.
    """
    print(f"Pivoting '{value_field}' by year...")

    pivot_gdf = panel_gdf.pivot_table(
        index=id_field,
        columns=col_field,
        values=value_field,
        aggfunc=aggfunc,
    ).reset_index()

    print(f"{value_field} pivoted data has {len(pivot_gdf):} rows, {len(pivot_gdf.columns)} columns")

    pivot_gdf.columns = [
        f"{value_field}_{col}" if isinstance(col, int) else col
        for col in pivot_gdf.columns
    ]

    geom_df = (
        panel_gdf
        .dropna(subset=["geometry"])
        .sort_values(["ACCTID", col_field])
        .groupby(id_field, as_index=False)
        .first()[[id_field, "geometry"]]
    )
    print(f"[Geometry] Found {len(geom_df)} rows from {year_for_geometry}")

    merged = geom_df.merge(pivot_gdf, on=id_field, how="left")
    print(f"[Pivoted Panel] Final shape: {merged.shape}")
    return gpd.GeoDataFrame(merged, geometry="geometry", crs=panel_gdf.crs)

def convert_time_fields(
    table_path: Path | str,
    field_pairs: list[tuple[str, str]],
    input_format: str = "yyyy",
    output_type: str = "DATE",
) -> None:
    """
    Convert integer year fields to ArcGIS Date fields using ConvertTimeField.

    Parameters:
    ----------
    table_path : Path or str
        GDB feature class path.
    field_pairs : list of (input_field, output_field) tuples
    input_format : str
        Time format of the source field (default: "yyyy").
    output_type : str
        ArcGIS output time type (default: "DATE").
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
                output_time_format="",
                timezone_or_field=""
            )
        success("Time conversion complete.")
    except Exception as e:
        error(f"convert_time_fields failed: {e}")
