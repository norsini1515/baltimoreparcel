# baltimoreparcel/scripts/assemble_panel.py
"""
Assembles the full longitudinal parcel panel.

Workflow
--------
1. Load parcel layers for each year from GDB feature classes or per-year GPKGs.
2. Stack into a long-format panel sorted by (ID, YEAR).
3. Trim to fields.keep column list.
4. Optionally apply CPI inflation adjustment to monetary fields.
5. Optionally log-transform selected value fields.
6. Spatial join with the neighborhoods reference layer.
7. Write the full panel to GeoPackage + ArcGIS File Geodatabase.
8. Convert integer year columns to ArcGIS DATE fields for time animation.

Change panel construction is a separate step — see build_change_panel.py.

Usage
-----
    python -m baltimoreparcel.scripts.assemble_panel --config configs/ez_assemble.yml
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
# Layer loading
# ---------------------------------------------------------------------------

def _load_year_layer(year: int, cfg: PanelRunConfig) -> gpd.GeoDataFrame | None:
    layer_name = cfg.input_layer_for(year)

    if cfg.data.source_type == "gdb":
        print(f"[{year}] Reading '{layer_name}' from {cfg.gdb_path.name}")
        try:
            gdf = gpd.read_file(str(cfg.gdb_path), layer=layer_name)
            print(f"[{year}] Loaded {len(gdf):,} rows")
            return gdf
        except Exception as exc:
            warn(f"[{year}] Could not read from GDB: {exc}")
            return None

    # Default: per-year GPKG
    gpkg_dir = cfg.get_year_gpkg_dir(year)
    if gpkg_dir is None:
        print(f"[{year}] GPKG directory not found — skipping")
        return None

    file_name = cfg.input_file_for(year)
    print(f"[{year}] Reading '{layer_name}' from {file_name}")
    gdf = gis.read_vector_layer(year=year, name=file_name, directory=gpkg_dir, layer=layer_name)
    if gdf is not None:
        gdf[cfg.data.year_field] = year
        print(f"[{year}] Loaded {len(gdf):,} rows")
    return gdf


def _stack_years(years, cfg: PanelRunConfig) -> gpd.GeoDataFrame:
    frames = {}
    for year in years:
        gdf = _load_year_layer(year, cfg)
        if gdf is not None:
            frames[year] = gdf
        print()

    if not frames:
        raise RuntimeError("No year layers loaded — cannot assemble panel.")

    print(f"Merging {len(frames)} year layers...")
    long_df = gpd.GeoDataFrame(pd.concat(frames.values(), ignore_index=True))
    long_df = (
        long_df
        .sort_values([cfg.data.id_field, cfg.data.year_field])
        .reset_index(drop=True)
    )
    print(f"Panel: {len(long_df):,} rows × {len(long_df.columns)} columns")
    return long_df


# ---------------------------------------------------------------------------
# Transformations
# ---------------------------------------------------------------------------

def _apply_column_filter(panel_gdf: gpd.GeoDataFrame, cfg: PanelRunConfig) -> gpd.GeoDataFrame:
    if not cfg.fields.keep:
        return panel_gdf
    must_have = {cfg.data.id_field, cfg.data.year_field, "geometry"}
    cols = list(dict.fromkeys(cfg.fields.keep + list(must_have)))
    return gis.select_columns(panel_gdf, cols)


def _apply_cpi(panel_gdf: gpd.GeoDataFrame, cfg: PanelRunConfig) -> gpd.GeoDataFrame:
    cpi_path = cfg.full_panel_dir / cfg.panel.cpi_file
    print(f"Reading CPI from {cpi_path}")
    prices = pd.read_csv(cpi_path).set_index(cfg.data.year_field)
    id_field, year_field = cfg.data.id_field, cfg.data.year_field

    for val in cfg.fields.monetary:
        if val not in panel_gdf.columns:
            warn(f"Skipping CPI for '{val}' — not in columns")
            continue
        print(f"  {val} → REAL_{val}")
        panel_gdf = panel_gdf.set_index([id_field, year_field])
        panel_gdf[f"REAL_{val}"] = panel.to_real_data(panel_gdf[val], prices)
        panel_gdf = panel_gdf.reset_index()
    return panel_gdf


def _apply_log(panel_gdf: gpd.GeoDataFrame, cfg: PanelRunConfig) -> gpd.GeoDataFrame:
    for val in cfg.fields.log_transform:
        if val not in panel_gdf.columns:
            warn(f"Skipping log for '{val}' — not in columns")
            continue
        print(f"  {val} → LOG_{val}")
        panel_gdf = panel.log_value(panel_gdf, value_field=val)
    return panel_gdf


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def _read_full_panel(cfg: PanelRunConfig) -> gpd.GeoDataFrame:
    print("Reading existing full panel from disk...")
    gdf = gis.read_vector_layer(
        year="full_panel",
        name=cfg.panel.output_gpkg,
        directory=cfg.full_panel_dir,
        layer=cfg.panel.full_panel_layer,
    )
    if gdf is None:
        raise FileNotFoundError(
            f"Full panel not found at {cfg.full_panel_dir / cfg.panel.output_gpkg}. "
            "Set toggles.generate_new_panel: true to build it."
        )
    return gdf


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

    panel_gdf = None
    panel_modified = False

    if cfg.toggles.generate_new_panel:
        panel_gdf = _stack_years(cfg.data.years, cfg)
        panel_gdf = _apply_column_filter(panel_gdf, cfg)
        panel_modified = True

    if cfg.toggles.calculate_real_values:
        if panel_gdf is None:
            panel_gdf = _read_full_panel(cfg)
        print("Applying CPI adjustment...")
        panel_gdf = _apply_cpi(panel_gdf, cfg)
        panel_modified = True

    if cfg.toggles.log_value_fields:
        if panel_gdf is None:
            panel_gdf = _read_full_panel(cfg)
        print("Log-transforming fields...")
        panel_gdf = _apply_log(panel_gdf, cfg)
        panel_modified = True

    if panel_gdf is None:
        panel_gdf = _read_full_panel(cfg)

    info(f"Panel shape: {panel_gdf.shape}")

    if panel_modified:
        print("Spatial join with neighborhoods...")
        panel_gdf = panel.spatial_join_with_neighborhoods(
            panel_gdf,
            gdb_path=cfg.gdb_path,
            layer=cfg.spatial.neighborhoods_layer,
            name_field=cfg.spatial.neighborhoods_name_field,
            output_field=cfg.spatial.neighborhoods_output_field,
        )
        _write_and_export(
            panel_gdf,
            cfg.panel.full_panel_layer,
            cfg,
            cfg.fields.time_fields_full,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Assemble the longitudinal parcel panel.")
    parser.add_argument("--config", required=True, metavar="PATH",
                        help="Path to a YAML run-config (e.g. configs/ez_assemble.yml)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    logger = Logger(cfg.logs_path / f"assemble_panel_{timestamp}.log")

    try:
        run(cfg)
        print("Done.")
    except Exception as exc:
        error(str(exc))
        raise
    finally:
        logger.close()
