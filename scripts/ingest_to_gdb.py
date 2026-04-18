# baltimoreparcel/scripts/ingest_to_gdb.py
"""
Generic GIS layer ingest: reads any number of shapefiles / GeoJSONs / GPKGs
and loads each as a named feature class into the project GDB.

Two layer types are supported:

  static       One source file --> one GDB feature class.
               Good for reference layers (neighborhoods, enterprise zones, etc.)

  year_series  One source file per year --> one GDB feature class per year.
               Good for repeated parcel snapshots where the name and path
               contain a ``{year}`` placeholder.

All settings (what to ingest, where to find it, filters, column selection) come
from the ``ingest.layers`` list in the YAML config.  Nothing about the GDB
schema is hard-coded in this script.

Usage
-----
    # Ingest all layers defined in the config:
    python -m baltimoreparcel.scripts.ingest_to_gdb --config configs/ez_ingest.yml

    # Ingest only specific layer specs (matched by name template):
    python -m baltimoreparcel.scripts.ingest_to_gdb --config configs/ez_ingest.yml \\
        --layers "{year}subset" neighborhoods
"""

import argparse
import datetime
from pathlib import Path

import arcpy

from baltimoreparcel import gis
from baltimoreparcel.run_config import IngestRunConfig, LayerSpec, load_ingest_config
from baltimoreparcel.utils import Logger, error, info, success, warn


# ---------------------------------------------------------------------------
# Filter helpers (unchanged from previous version)
# ---------------------------------------------------------------------------

def _filter_callable(spec: dict):
    """
    Convert one YAML filter spec into a scalar or callable accepted by
    gis.filter_on_field.

    Supported types
    ---------------
    gt / gte / lt / lte  — numeric comparison  (value required)
    not_null             — drop rows where field is NaN/None
    not_empty            — drop rows where field is NaN, blank, or whitespace
    """
    ftype = spec.get("type", "")
    value = spec.get("value", None)

    if ftype == "gt":
        return value
    elif ftype == "gte":
        return lambda s: s >= value
    elif ftype == "lt":
        return lambda s: s < value
    elif ftype == "lte":
        return lambda s: s <= value
    elif ftype == "not_null":
        return lambda s: s.notna()
    elif ftype == "not_empty":
        return lambda s: s.notna() & (s.astype(str).str.strip() != "")
    else:
        raise ValueError(
            f"Unknown filter type: {ftype!r}. "
            "Valid types: gt, gte, lt, lte, not_null, not_empty"
        )


# ---------------------------------------------------------------------------
# Core ingest: one resolved file --> one GDB feature class
# ---------------------------------------------------------------------------

def _ingest_single(
    source_path: Path,
    layer_name: str,
    spec: LayerSpec,
    cfg: IngestRunConfig,
    year: int | None = None,
) -> bool:
    """
    Read, clean, and load one file into the GDB.

    Parameters
    ----------
    source_path : resolved absolute path to the source file
    layer_name  : target feature class name in the GDB
    spec        : LayerSpec that governs CRS, filters, column selection
    cfg         : IngestRunConfig (provides gdb_path, data_path, project CRS)
    year        : year being ingested (for year_series layers); None for static

    Returns True on success, False if skipped or failed.
    """
    label = f"[{year}]" if year is not None else f"[{layer_name}]"

    if not source_path.exists():
        warn(f"{label} Not found: {source_path} — skipping")
        return False

    # --- Read ---------------------------------------------------------------
    src_layer = spec.layer.format(year=year) if (spec.layer and year) else spec.layer
    print(f"{label} Reading {source_path.name}" +
          (f" layer='{src_layer}'" if src_layer else ""))
    gdf = gis.read_gis_file(source_path, layer=src_layer)
    if gdf is None or gdf.empty:
        warn(f"{label} Empty or unreadable — skipping")
        return False
    print(f"{label} {len(gdf):,} rows, {len(gdf.columns)} columns")

    # --- CRS ----------------------------------------------------------------
    epsg = spec.crs_epsg if spec.crs_epsg is not None else cfg.project.crs_epsg
    gdf = gis.ensure_crs(gdf, epsg=epsg, source_epsg=spec.source_crs_epsg)

    # --- Bounding box clip --------------------------------------------------
    if spec.bbox:
        xmin, ymin, xmax, ymax = spec.bbox
        before = len(gdf)
        gdf = gdf.cx[xmin:xmax, ymin:ymax]
        dropped = before - len(gdf)
        if dropped:
            warn(f"{label} Dropped {dropped:,} rows outside bbox")
        if gdf.empty:
            warn(f"{label} No rows remain after bbox clip — skipping")
            return False

    # --- Filters ------------------------------------------------------------
    if spec.filters:
        fields    = [f["field"] for f in spec.filters]
        callables = [_filter_callable(f) for f in spec.filters]
        gdf = gis.filter_on_field(gdf, fields=fields, filters=callables, identifier=label)
        if gdf is None or gdf.empty:
            warn(f"{label} No rows survived filtering — skipping")
            return False

    # --- Column selection ---------------------------------------------------
    if spec.keep:
        must_have = {"geometry"}
        cols = list(dict.fromkeys(spec.keep + list(must_have)))
        gdf = gis.select_columns(gdf, cols)

    # --- Inject YEAR column (year_series only) ------------------------------
    if spec.add_year_field and year is not None:
        gdf["YEAR"] = year

    print(f"{label} {len(gdf):,} rows --> {len(gdf.columns)} columns after processing")

    # --- Stage to GPKG, then export to GDB ----------------------------------
    staging_dir = cfg.data_path / "staging"
    staging_dir.mkdir(parents=True, exist_ok=True)
    staging_gpkg = staging_dir / f"{layer_name}_staging.gpkg"

    gis.write_gpkg_layer(
        gdf,
        year=layer_name,
        name=staging_gpkg.name,
        directory=staging_dir,
        layer=layer_name,
    )

    exported = gis.export_to_geodb(
        input_gpkg_path=staging_gpkg,
        layer_name=layer_name,
        gdb_path=cfg.gdb_path,
        out_feature_name=layer_name,
    )

    if not exported:
        error(f"{label} GDB export failed")
        return False

    success(f"{label} --> {cfg.gdb_path.name} / {layer_name}")
    return True


# ---------------------------------------------------------------------------
# Dispatch: static vs year_series
# ---------------------------------------------------------------------------

def ingest_layer(spec: LayerSpec, cfg: IngestRunConfig) -> tuple[int, int]:
    """
    Ingest one LayerSpec.

    Returns (succeeded, attempted) counts.
    """
    if spec.type == "year_series":
        if not spec.years:
            warn(f"Layer '{spec.name}' is type year_series but has no years — skipping")
            return 0, 0
        succeeded = 0
        for year in spec.years:
            source_path = cfg.project_dir / spec.source.format(year=year)
            layer_name  = spec.name.format(year=year)
            ok = _ingest_single(source_path, layer_name, spec, cfg, year=year)
            if ok:
                succeeded += 1
            print()
        return succeeded, len(spec.years)

    else:  # static
        source_path = cfg.project_dir / spec.source
        ok = _ingest_single(source_path, spec.name, spec, cfg)
        print()
        return (1, 1) if ok else (0, 1)


# ---------------------------------------------------------------------------
# Main run
# ---------------------------------------------------------------------------

def run(cfg: IngestRunConfig, layer_names: list[str] | None = None) -> None:
    """
    Ingest all (or a named subset of) layer specs into the GDB.

    Parameters
    ----------
    cfg          : IngestRunConfig
    layer_names  : If provided, only specs whose ``name`` template is in this
                   list are processed.  Pass None to process all specs.
    """
    arcpy.env.workspace = str(cfg.gdb_path)
    cfg.logs_path.mkdir(parents=True, exist_ok=True)

    specs = cfg.layers
    if not specs:
        raise ValueError(
            "No layers defined. Add entries to ingest.layers in your config."
        )

    if layer_names:
        specs = [s for s in specs if s.name in layer_names]
        if not specs:
            raise ValueError(
                f"None of the requested layer names matched config specs: {layer_names}"
            )

    info(f"Ingesting {len(specs)} layer spec(s) into {cfg.gdb_path.name}")

    total_ok  = 0
    total_att = 0
    for spec in specs:
        label = (
            f"{spec.name}  [{len(spec.years)} years]"
            if spec.type == "year_series"
            else spec.name
        )
        info(f"--- {label} ---")
        ok, att = ingest_layer(spec, cfg)
        total_ok  += ok
        total_att += att

    print()
    success(f"Ingested {total_ok}/{total_att} layers/years into {cfg.gdb_path.name}")
    if total_ok < total_att:
        warn(f"{total_att - total_ok} layer(s)/year(s) were skipped or failed")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ingest shapefiles / GeoJSONs / GPKGs into the project GDB."
    )
    parser.add_argument(
        "--config",
        required=True,
        metavar="PATH",
        help="Path to a YAML run-config (e.g. configs/ez_ingest.yml)",
    )
    parser.add_argument(
        "--layers",
        nargs="+",
        metavar="NAME",
        help=(
            "Only ingest the layer spec(s) whose name template matches. "
            "Example: --layers neighborhoods \"{year}subset\""
        ),
    )
    args = parser.parse_args()

    cfg = load_ingest_config(args.config)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    logger = Logger(cfg.logs_path / f"ingest_to_gdb_{timestamp}.log")

    try:
        run(cfg, layer_names=args.layers)
        print("Done.")
    except Exception as exc:
        error(str(exc))
        raise
    finally:
        logger.close()
