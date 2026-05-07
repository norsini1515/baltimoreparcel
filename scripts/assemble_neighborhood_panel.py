# baltimoreparcel/scripts/assemble_neighborhood_panel.py
"""
Assembles a neighborhood-year panel.

The panel unit is a neighborhood polygon (from the static 'neighborhoods' GDB
layer), not an individual parcel.  Each row represents one neighborhood for one
year and carries:

  - Neighborhood geometry and attributes (Name, etc.)
  - EZ membership flags (intersects predicate applied once to the static layer)
  - Parcel value aggregates from parcels_{year} within each neighborhood
  - CPI-adjusted real-value columns
  - Log-transformed value columns
  - Year-conditional treatment status derived from EZ flags

Workflow
--------
1. Load the static 'neighborhoods' polygon layer from the GDB.
2. Apply EZ spatial joins once to the neighborhood geometries.
3. For each year:
   a. Load parcels_{year} from the GDB.
   b. Spatial-join parcels to neighborhoods to assign each parcel a neighborhood.
   c. Aggregate parcel value fields by neighborhood.
   d. Merge aggregated values onto a copy of the neighborhood layer.
   e. Tag the frame with YEAR.
4. Stack all year frames into a long panel.
5. Apply CPI inflation adjustment to monetary fields.
6. Log-transform selected value fields.
7. Apply derive rules (treatment status flags).
8. Write to GeoPackage + ArcGIS File Geodatabase.

Usage
-----
    python -m baltimoreparcel.scripts.assemble_neighborhood_panel \
        --config configs/build_neighborhood_panel.yml
"""

import argparse
import datetime

import arcpy
import geopandas as gpd
import pandas as pd

from baltimoreparcel import gis, panel
from baltimoreparcel.run_config import PanelRunConfig, load_config
from baltimoreparcel.utils import Logger, error, info, success, warn


# ---------------------------------------------------------------------------
# Step 1: Load static base layer and apply EZ spatial joins
# ---------------------------------------------------------------------------

def _load_base(cfg: PanelRunConfig) -> gpd.GeoDataFrame:
    layer = cfg.data.base_layer
    if not layer:
        raise ValueError("data.base_layer must be set in the config for the neighborhood panel.")
    print(f"Reading base layer '{layer}' from {cfg.gdb_path.name}")
    try:
        gdf = gpd.read_file(str(cfg.gdb_path), layer=layer)
    except Exception as exc:
        raise RuntimeError(f"Could not read base layer '{layer}' from GDB: {exc}") from exc
    success(f"Loaded {len(gdf):,} base polygons ({layer})")
    return gdf


def _enrich_base_with_ez(base_gdf: gpd.GeoDataFrame, cfg: PanelRunConfig) -> gpd.GeoDataFrame:
    if not cfg.spatial.joins:
        return base_gdf
    print("Applying EZ spatial joins to neighborhood polygons...")
    return panel.apply_spatial_joins(base_gdf, joins=cfg.spatial.joins, gdb_path=cfg.gdb_path)


# ---------------------------------------------------------------------------
# Step 2: Per-year parcel aggregation into neighborhoods
# ---------------------------------------------------------------------------

def _aggregate_parcels_for_year(
    year: int,
    base_gdf: gpd.GeoDataFrame,
    cfg: PanelRunConfig,
) -> gpd.GeoDataFrame | None:
    pa = cfg.parcel_agg
    parcel_layer = cfg.data.parcel_layer_pattern.format(year=year)
    if not parcel_layer:
        warn(f"[{year}] data.parcel_layer_pattern not set — skipping parcel aggregation")
        return None

    print(f"[{year}] Reading '{parcel_layer}' from GDB")
    try:
        parcels = gpd.read_file(str(cfg.gdb_path), layer=parcel_layer)
    except Exception as exc:
        warn(f"[{year}] Could not read '{parcel_layer}': {exc} — skipping")
        return None
    print(f"[{year}] Loaded {len(parcels):,} parcels")

    if parcels.crs != base_gdf.crs:
        parcels = parcels.to_crs(base_gdf.crs)

    # Assign each parcel to a neighborhood via spatial join
    id_field = cfg.data.id_field
    joined = gpd.sjoin(
        parcels,
        base_gdf[[id_field, "geometry"]].reset_index(drop=True),
        how="left",
        predicate=pa.join_predicate,
    )

    matched = joined[id_field].notna().sum()
    print(f"[{year}] {matched:,} / {len(parcels):,} parcels matched to a neighborhood")

    # Aggregate — pandas groupby
    agg_dict = dict(pa.agg)
    if not agg_dict:
        warn(f"[{year}] parcel_agg.agg is empty — no parcel values will be attached")
        agg_result = pd.DataFrame({id_field: base_gdf[id_field].unique()})
    else:
        agg_result = (
            joined.groupby(id_field)
            .agg(agg_dict)
            .reset_index()
        )

    # Rename any "count" column to count_output_field (e.g. ACCTID → PARCEL_COUNT)
    count_cols = [f for f, func in agg_dict.items() if func == "count"]
    for col in count_cols:
        if col in agg_result.columns and col != pa.count_output_field:
            agg_result = agg_result.rename(columns={col: pa.count_output_field})

    # Categorical value counts: one column per unique value in each field
    # e.g. LU → LU_C, LU_R, LU_CC, ... ; STRUGRAD → STRUGRAD_A, STRUGRAD_B, ...
    for cat_field in pa.value_counts:
        if cat_field not in joined.columns:
            warn(f"[{year}] value_counts field '{cat_field}' not in parcels — skipping")
            continue
        ct = pd.crosstab(joined[id_field], joined[cat_field])
        ct.columns = [f"{cat_field}_{str(v).replace(' ', '_')}" for v in ct.columns]
        ct = ct.reset_index()
        agg_result = agg_result.merge(ct, on=id_field, how="left")
        print(f"  {cat_field}: {len(ct.columns) - 1} category columns added")

    # Merge aggregated values onto neighborhood polygons
    nbhd_year = base_gdf.merge(agg_result, on=id_field, how="left")
    nbhd_year[cfg.data.year_field] = year
    return nbhd_year


def _build_panel(base_gdf: gpd.GeoDataFrame, cfg: PanelRunConfig) -> gpd.GeoDataFrame:
    frames = []
    for year in cfg.data.years:
        frame = _aggregate_parcels_for_year(year, base_gdf, cfg)
        if frame is not None:
            frames.append(frame)
        print()

    if not frames:
        raise RuntimeError("No year frames assembled — cannot build panel.")

    print(f"Stacking {len(frames)} year frames...")
    long_df = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True))
    long_df = (
        long_df
        .sort_values([cfg.data.id_field, cfg.data.year_field])
        .reset_index(drop=True)
    )
    success(f"Panel: {len(long_df):,} rows × {len(long_df.columns)} columns")
    return long_df


# ---------------------------------------------------------------------------
# Transformations (same helpers as assemble_panel.py)
# ---------------------------------------------------------------------------

def _apply_cpi(panel_gdf: gpd.GeoDataFrame, cfg: PanelRunConfig) -> gpd.GeoDataFrame:
    cpi_path = cfg.project_dir / cfg.panel.cpi_file
    print(f"Reading CPI from {cpi_path}")
    price_map = pd.read_csv(cpi_path).set_index("year")["price"]
    for val in cfg.fields.monetary:
        if val not in panel_gdf.columns:
            warn(f"Skipping CPI for '{val}' — not in columns")
            continue
        print(f"  {val} --> REAL_{val}")
        panel_gdf[f"REAL_{val}"] = panel_gdf[val] / panel_gdf[cfg.data.year_field].map(price_map)
    return panel_gdf


def _apply_log(panel_gdf: gpd.GeoDataFrame, cfg: PanelRunConfig) -> gpd.GeoDataFrame:
    for val in cfg.fields.log_transform:
        if val not in panel_gdf.columns:
            warn(f"Skipping log for '{val}' — not in columns")
            continue
        print(f"  {val} --> LOG_{val}")
        panel_gdf = panel.log_value(panel_gdf, value_field=val)
    return panel_gdf


def _apply_derive(panel_gdf: gpd.GeoDataFrame, cfg: PanelRunConfig) -> gpd.GeoDataFrame:
    print("Applying derive rules...")
    for rule in cfg.fields.derive:
        print(f"  [{rule.type}] --> {rule.name or rule.output_treated}")
    return panel.apply_derive_rules(panel_gdf, cfg.fields.derive)


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def _write_and_export(
    gdf: gpd.GeoDataFrame,
    layer_name: str,
    cfg: PanelRunConfig,
    time_field_pairs: list,
) -> None:
    gis.write_gpkg_layer(gdf, year=layer_name, name=cfg.panel.output_gpkg,
                         directory=cfg.full_panel_dir, layer=layer_name)
    exported = gis.export_to_geodb(
        input_gpkg_path=cfg.full_panel_dir / cfg.panel.output_gpkg,
        layer_name=layer_name,
        gdb_path=cfg.gdb_path,
        out_feature_name=layer_name,
    )
    if exported and time_field_pairs:
        gis.convert_time_fields(table_path=exported.name,
                                field_pairs=[tuple(p) for p in time_field_pairs])
    success(f"Exported '{layer_name}'")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run(cfg: PanelRunConfig) -> None:
    arcpy.env.workspace = str(cfg.gdb_path)
    cfg.full_panel_dir.mkdir(parents=True, exist_ok=True)
    cfg.logs_path.mkdir(parents=True, exist_ok=True)

    # 1. Load static neighborhood polygons + EZ flags (computed once, replicated per year)
    base_gdf = _load_base(cfg)
    base_gdf = _enrich_base_with_ez(base_gdf, cfg)

    # 2. Stack neighborhood-year frames (parcel values aggregated per year)
    panel_gdf = _build_panel(base_gdf, cfg)

    info(f"Panel shape before transforms: {panel_gdf.shape}")

    # 3. Transforms
    if cfg.toggles.calculate_real_values and cfg.panel.cpi_file:
        print("Applying CPI adjustment...")
        panel_gdf = _apply_cpi(panel_gdf, cfg)

    if cfg.toggles.log_value_fields:
        print("Log-transforming fields...")
        panel_gdf = _apply_log(panel_gdf, cfg)

    if cfg.toggles.derive_fields and cfg.fields.derive:
        panel_gdf = _apply_derive(panel_gdf, cfg)

    info(f"Panel shape after transforms: {panel_gdf.shape}")

    # 4. Write output
    _write_and_export(
        panel_gdf,
        cfg.panel.full_panel_layer,
        cfg,
        cfg.fields.time_fields_full,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Assemble the longitudinal neighborhood panel."
    )
    parser.add_argument("--config", required=True, metavar="PATH",
                        help="Path to a YAML run-config (e.g. configs/build_neighborhood_panel.yml)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    logger = Logger(cfg.logs_path / f"assemble_neighborhood_panel_{timestamp}.log")

    try:
        run(cfg)
        print("Done.")
    except Exception as exc:
        error(str(exc))
        raise
    finally:
        logger.close()
