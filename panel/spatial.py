# baltimoreparcel/panel/spatial.py
"""
Spatial enrichment steps applied to the panel during assembly.
"""

from pathlib import Path

import geopandas as gpd

from baltimoreparcel.utils import warn


def apply_spatial_joins(
    panel_gdf: gpd.GeoDataFrame,
    joins: list,          # list[SpatialJoinSpec]
    gdb_path: Path,
) -> gpd.GeoDataFrame:
    """
    Apply a list of SpatialJoinSpec enrichments to the panel in sequence.

    Each join reads its reference layer from the GDB.  If the layer is not
    present the step is skipped with a warning so a missing optional layer
    does not abort the whole assembly run.

    attribute joins
        Attach a field value from the overlapping reference polygon.
        One value per parcel — when multiple polygons overlap, the first
        match is kept.

    isin joins
        Add a boolean column: True when the parcel satisfies the spatial
        predicate (``within`` or ``intersects``) against the reference layer.
    """
    for spec in joins:
        # --- Load reference layer -------------------------------------------
        try:
            ref_gdf = gpd.read_file(str(gdb_path), layer=spec.layer)
        except Exception as exc:
            warn(
                f"Spatial join '{spec.output_field}': layer '{spec.layer}' not found "
                f"in GDB — skipping. ({exc})"
            )
            continue

        # Reproject reference to match panel CRS
        if ref_gdf.crs != panel_gdf.crs:
            ref_gdf = ref_gdf.to_crs(panel_gdf.crs)

        # --- Perform join ---------------------------------------------------
        if spec.type == "attribute":
            if not spec.field:
                warn(f"Spatial join '{spec.output_field}': attribute join requires 'field' — skipping")
                continue
            ref_slim = (
                ref_gdf[[spec.field, "geometry"]]
                .rename(columns={spec.field: spec.output_field})
                .reset_index(drop=True)
            )
            joined = gpd.sjoin(panel_gdf, ref_slim, how="left", predicate=spec.how)
            joined = joined[~joined.index.duplicated(keep="first")]
            panel_gdf = panel_gdf.copy()
            panel_gdf[spec.output_field] = joined[spec.output_field]

        elif spec.type == "isin":
            ref_slim = ref_gdf[["geometry"]].reset_index(drop=True)
            joined = gpd.sjoin(
                panel_gdf[["geometry"]],
                ref_slim,
                how="left",
                predicate=spec.how,
            )
            joined = joined[~joined.index.duplicated(keep="first")]
            panel_gdf = panel_gdf.copy()
            panel_gdf[spec.output_field] = joined["index_right"].notna()

        else:
            warn(f"Spatial join '{spec.output_field}': unknown type '{spec.type}' — skipping")
            continue

        matched = panel_gdf[spec.output_field].notna().sum()
        print(f"  {spec.output_field} ({spec.type}, {spec.how}): {matched:,} / {len(panel_gdf):,} matched")

    return panel_gdf
