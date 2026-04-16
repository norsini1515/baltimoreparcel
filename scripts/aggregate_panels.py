# baltimoreparcel/scripts/aggregate_panels.py
"""
Aggregate and summarize full and change panels.

Reads aggregation rules from a YAML config file, groups the panel data,
and exports one GDB layer per rule.

Usage
-----
    # Run change-panel aggregations only (default):
    python -m baltimoreparcel.scripts.aggregate_panels --config configs/baltimore_default.yml

    # Run full-panel aggregations only:
    python -m baltimoreparcel.scripts.aggregate_panels --config configs/baltimore_default.yml --full

    # Run both:
    python -m baltimoreparcel.scripts.aggregate_panels --config configs/baltimore_default.yml --full --change
"""

import argparse
import datetime
import sys

import geopandas as gpd
import pandas as pd

from baltimoreparcel import gis
from baltimoreparcel.run_config import AggregateRunConfig, AggregationRule, load_aggregate_config
from baltimoreparcel.utils import Logger, error, info, process_step, success, warn


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_panel(which: str, cfg: AggregateRunConfig) -> gpd.GeoDataFrame:
    layer = (
        cfg.panel.full_panel_layer
        if which == "full"
        else cfg.panel.change_panel_layer
    )
    print(f"Reading {which} panel (layer='{layer}')...")
    gdf = gpd.read_file(str(cfg.gdb_path), layer=layer)
    info(f"  {which} panel shape: {gdf.shape}")
    return gdf


def _run_aggregations(
    panel_gdf: gpd.GeoDataFrame,
    rules: list[AggregationRule],
    panel_name: str,
    geom_lookup: dict | None = None,
) -> dict[str, gpd.GeoDataFrame]:
    """
    Apply a list of AggregationRule objects to *panel_gdf*.

    Returns a dict mapping output layer name → aggregated GeoDataFrame.
    """
    process_step(f"Aggregating: {panel_name}")
    results = {}

    for rule in rules:
        group_keys = rule.group_by
        agg_dict = rule.agg
        suffix = "_".join(group_keys).lower()
        out_name = f"{panel_name}_agg_{suffix}"

        info(f"  → {out_name}  (group by {group_keys})")

        # Coerce object columns that are supposed to be numeric
        for col, func in agg_dict.items():
            if (
                func == "mean"
                and col in panel_gdf.columns
                and panel_gdf[col].dtype == "object"
            ):
                warn(f"    '{col}' is object dtype — coercing to numeric")
                panel_gdf[col] = pd.to_numeric(panel_gdf[col], errors="coerce")

        try:
            grouped = (
                panel_gdf
                .groupby(group_keys)
                .agg(agg_dict)
                .reset_index()
            )

            # Attach geometry -----------------------------------------------
            # Prefer an external polygon lookup (e.g. neighborhood boundaries);
            # fall back to sampling one representative row per group.
            geometry_key = next(
                (k for k in group_keys if geom_lookup and k in geom_lookup),
                None,
            )

            if geometry_key:
                geom_df = geom_lookup[geometry_key]
                grouped = grouped.merge(
                    geom_df[[geometry_key, "geometry"]],
                    on=geometry_key,
                    how="left",
                )
            else:
                grouped = grouped.merge(
                    panel_gdf[[*group_keys, "geometry"]].drop_duplicates(
                        subset=group_keys
                    ),
                    on=group_keys,
                    how="left",
                )

            grouped = gpd.GeoDataFrame(
                grouped, geometry="geometry", crs=panel_gdf.crs
            )
            results[out_name] = grouped
            info(f"    Done — {len(grouped):,} rows")

        except Exception as exc:
            error(f"    Aggregation failed for {group_keys}: {exc}")

    if not results:
        raise RuntimeError(
            f"No aggregation rules produced output for '{panel_name}'. "
            "Check that group_by fields exist in the panel."
        )

    success(f"{len(results)} aggregation(s) completed for {panel_name}")
    return results


def _export_layer(
    name: str,
    gdf: gpd.GeoDataFrame,
    cfg: AggregateRunConfig,
    time_field_pairs: list,
) -> None:
    """Write to GPKG, push to GDB, convert time fields."""
    # Nullable int64 → float so ArcPy doesn't choke
    for col in gdf.columns:
        if gdf[col].dtype == "int64" and gdf[col].isna().any():
            gdf[col] = gdf[col].astype("float")

    gis.write_gpkg_layer(
        gdf=gdf,
        year=name,
        name=cfg.panel.output_gpkg,
        directory=cfg.full_panel_dir,
        layer=name,
    )
    layer_path = gis.export_to_geodb(
        input_gpkg_path=cfg.full_panel_dir / cfg.panel.output_gpkg,
        layer_name=name,
        gdb_path=cfg.gdb_path,
        out_feature_name=name,
    )
    if not layer_path:
        error(f"Export to GDB failed for '{name}'")
        return

    if time_field_pairs:
        gis.convert_time_fields(
            table_path=layer_path.name,
            field_pairs=[tuple(p) for p in time_field_pairs],
        )
    success(f"Exported '{name}'")


# ---------------------------------------------------------------------------
# Main run
# ---------------------------------------------------------------------------

def run(cfg: AggregateRunConfig, do_full: bool = False, do_change: bool = True) -> None:
    cfg.logs_path.mkdir(parents=True, exist_ok=True)

    # Load neighborhood geometries for spatial merging
    neigh_gdf = gpd.read_file(
        str(cfg.gdb_path), layer=cfg.spatial.neighborhoods_layer
    )
    neigh_gdf = neigh_gdf.rename(
        columns={cfg.spatial.neighborhoods_name_field: cfg.spatial.neighborhoods_output_field}
    )[[cfg.spatial.neighborhoods_output_field, "geometry"]]

    geom_lookup = {cfg.spatial.neighborhoods_output_field: neigh_gdf}

    all_aggregated: dict[str, gpd.GeoDataFrame] = {}

    if do_full and cfg.aggregations.full_panel:
        full_panel = _read_panel("full", cfg)
        all_aggregated.update(
            _run_aggregations(
                full_panel,
                cfg.aggregations.full_panel,
                panel_name="full_panel",
                geom_lookup=geom_lookup,
            )
        )

    if do_change and cfg.aggregations.change_panel:
        change_panel = _read_panel("change", cfg)
        all_aggregated.update(
            _run_aggregations(
                change_panel,
                cfg.aggregations.change_panel,
                panel_name="change_panel",
                geom_lookup=geom_lookup,
            )
        )

    info(f"Total layers to export: {len(all_aggregated)}")

    for name, gdf in all_aggregated.items():
        is_change_layer = name.startswith("change_panel")
        time_pairs = (
            cfg.time_fields_change
            if is_change_layer
            else cfg.time_fields_full
        )
        _export_layer(name, gdf, cfg, time_pairs)

    success("All aggregations and exports completed.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Aggregate Baltimore parcel panels by neighborhood / GEOGCODE / year."
    )
    parser.add_argument(
        "--config",
        required=True,
        metavar="PATH",
        help="Path to a YAML run-config file (e.g. configs/baltimore_default.yml)",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        default=False,
        help="Run full-panel aggregations",
    )
    parser.add_argument(
        "--change",
        action="store_true",
        default=False,
        help="Run change-panel aggregations",
    )
    args = parser.parse_args()

    # Default: run change panel if neither flag given
    do_full = args.full
    do_change = args.change or (not args.full and not args.change)

    cfg = load_aggregate_config(args.config)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    logger = Logger(cfg.logs_path / f"aggregate_panels_{timestamp}.log")

    try:
        run(cfg, do_full=do_full, do_change=do_change)
    except Exception as exc:
        error(str(exc))
        raise
    finally:
        logger.close()
        print("Script finished. Check logs for details.")
