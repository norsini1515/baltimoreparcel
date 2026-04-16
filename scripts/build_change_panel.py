# baltimoreparcel/scripts/build_change_panel.py
"""
Build the year-over-year change panel from an assembled full panel.

Workflow
--------
1. Read the full panel from the project GDB (assembled by assemble_panel.py).
2. For each numeric change field: pivot to wide format, compute year-over-year diffs.
3. For each string change field: pivot to wide format, compute binary change flags.
4. Merge all change columns into a single change panel GeoDataFrame.
5. Attach enrich fields (e.g. NEIGHBORHOOD, GEOGCODE) from the latest full-panel row.
6. Write the change panel to GeoPackage + ArcGIS File Geodatabase.
7. Convert integer year columns to ArcGIS DATE fields for time animation.

This script is intentionally independent of assemble_panel.py — it reads the
already-assembled full panel and produces its own output layer.

Usage
-----
    python -m baltimoreparcel.scripts.build_change_panel --config configs/ez_change.yml
"""

import argparse
import datetime

import arcpy
import geopandas as gpd

from baltimoreparcel import gis, panel
from baltimoreparcel.run_config import ChangePanelRunConfig, load_change_panel_config
from baltimoreparcel.utils import Logger, error, info, success, warn


# ---------------------------------------------------------------------------
# Read
# ---------------------------------------------------------------------------

def _read_full_panel(cfg: ChangePanelRunConfig) -> gpd.GeoDataFrame:
    print(f"Reading full panel from '{cfg.panel.full_panel_layer}' in {cfg.gdb_path.name}...")
    gdf = gpd.read_file(str(cfg.gdb_path), layer=cfg.panel.full_panel_layer)
    if gdf is None or gdf.empty:
        raise FileNotFoundError(
            f"Full panel layer '{cfg.panel.full_panel_layer}' not found or empty "
            f"in {cfg.gdb_path}. Run assemble_panel.py first."
        )
    info(f"Full panel: {gdf.shape}")
    return gdf


# ---------------------------------------------------------------------------
# Pivot + change
# ---------------------------------------------------------------------------

def _pivot_and_change(
    panel_gdf: gpd.GeoDataFrame,
    fields: list[str],
    field_type: str,
    cfg: ChangePanelRunConfig,
) -> list[gpd.GeoDataFrame]:
    """
    For each field in *fields*, pivot to wide format and compute change.

    Returns a list of long-format change GeoDataFrames (one per field).
    """
    frames = []
    for val in fields:
        if val not in panel_gdf.columns:
            warn(f"Skipping change for '{val}' — not in panel columns")
            continue
        print(f"  {val} ({field_type})")
        wide = gis.pivot_panel(
            panel_gdf,
            value_field=val,
            id_field=cfg.data.id_field,
            col_field=cfg.data.year_field,
        )
        chng = panel.calculate_change(
            wide,
            value_prefix=val,
            acctid_col=cfg.data.id_field,
            field_type=field_type,
        )
        frames.append(chng)
    return frames


# ---------------------------------------------------------------------------
# Merge change frames
# ---------------------------------------------------------------------------

def _merge_change_frames(
    frames: list[gpd.GeoDataFrame],
    id_field: str,
) -> gpd.GeoDataFrame:
    """
    Merge multiple long-format change GeoDataFrames on (id_field, START_YR, END_YR).

    The geometry from the first frame is retained.
    """
    if not frames:
        raise RuntimeError("No change frames to merge — check fields.numeric_change and fields.string_change in your config.")

    merge_keys = [id_field, "START_YR", "END_YR"]
    result = frames[0]
    for frame in frames[1:]:
        # Drop geometry from subsequent frames before merging
        non_geo = frame.drop(columns=["geometry"], errors="ignore")
        result = result.merge(non_geo, on=merge_keys, how="outer")

    return result


# ---------------------------------------------------------------------------
# Write
# ---------------------------------------------------------------------------

def _write_and_export(
    gdf: gpd.GeoDataFrame,
    layer_name: str,
    cfg: ChangePanelRunConfig,
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

def run(cfg: ChangePanelRunConfig) -> None:
    arcpy.env.workspace = str(cfg.gdb_path)
    cfg.full_panel_dir.mkdir(parents=True, exist_ok=True)
    cfg.logs_path.mkdir(parents=True, exist_ok=True)

    panel_gdf = _read_full_panel(cfg)

    all_frames = []

    if cfg.fields.numeric_change:
        print("Computing numeric change fields...")
        all_frames += _pivot_and_change(
            panel_gdf, cfg.fields.numeric_change, "numeric", cfg
        )

    if cfg.fields.string_change:
        print("Computing string change fields...")
        all_frames += _pivot_and_change(
            panel_gdf, cfg.fields.string_change, "string", cfg
        )

    change_gdf = _merge_change_frames(all_frames, cfg.data.id_field)

    if cfg.fields.enrich:
        print("Enriching change panel with metadata fields...")
        change_gdf = panel.enrich_change_gdf(
            change_gdf,
            base_gdf=panel_gdf,
            enrich_fields=cfg.fields.enrich,
        )

    info(f"Change panel shape: {change_gdf.shape}")

    _write_and_export(
        change_gdf,
        cfg.panel.change_panel_layer,
        cfg,
        cfg.fields.time_fields_change,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build the year-over-year change panel.")
    parser.add_argument("--config", required=True, metavar="PATH",
                        help="Path to a YAML run-config (e.g. configs/ez_change.yml)")
    args = parser.parse_args()

    cfg = load_change_panel_config(args.config)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    logger = Logger(cfg.logs_path / f"build_change_panel_{timestamp}.log")

    try:
        run(cfg)
        print("Done.")
    except Exception as exc:
        error(str(exc))
        raise
    finally:
        logger.close()
