# baltimoreparcel

A GIS data pipeline for analyzing Baltimore City parcel data from 2002–2024, with a focus on the impact of Urban Enterprise Zones (EZ) on property values.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Requirements](#requirements)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Pipeline](#pipeline)
  - [Stage 1: Ingest](#stage-1-ingest)
  - [Stage 2: Assemble](#stage-2-assemble)
  - [Stage 3: Change Panel](#stage-3-change-panel)
  - [Stage 4: Aggregate](#stage-4-aggregate)
  - [Running the Full Pipeline](#running-the-full-pipeline)
- [Module Reference](#module-reference)
  - [baltimoreparcel.config](#baltimoreParcelconfig)
  - [baltimoreparcel.directories](#baltimoreParcelDirectories)
  - [baltimoreparcel.run_config](#baltimoreParcelrun_config)
  - [baltimoreparcel.utils](#baltimoreParcelutils)
  - [baltimoreparcel.gis](#baltimoreParcelgis)
  - [baltimoreparcel.panel](#baltimoreParcelPanel)
  - [baltimoreparcel.scripts](#baltimoreParcelScripts)
- [Spatial Analysis](#spatial-analysis)
- [Data Dictionary](#data-dictionary)

---

## Project Overview

This project builds a **longitudinal parcel-level panel** for Baltimore City spanning 2002–2024. The workflow:

1. Ingest raw annual parcel shapefiles into an ArcGIS File Geodatabase (GDB)
2. Stack all years into a long-format panel, applying CPI inflation adjustment and log-transforms
3. Compute year-over-year change fields (value, land use, zoning, ownership)
4. Aggregate the panel to neighborhood or geographic-code level
5. Run spatial autocorrelation (Moran's I) and bivariate analyses on the change data

The primary research application is a **Difference-in-Differences** analysis of Urban Enterprise Zone designation on property values, using `IN_EZ_2012` and `IN_FOCUS_EZ_2012` treatment indicators.

---

## Requirements

- **Python**: ArcGIS Pro's `arcpy_tools` conda environment (includes `arcpy`, `geopandas`, `pandas`, `numpy`)
- **ArcGIS Pro**: Required for GDB writes and time-animation field conversion
- **Additional packages**: `pyyaml>=6.0`, `colorama`, `tqdm`, `esda`, `libpysal`

---

## Installation

```bash
conda activate arcpy_tools
cd path/to/BaltimoreParcelProject
pip install -e .
```

---

## Project Structure

```
BaltimoreParcelProject/
├── configs/                   # YAML run configs for each pipeline stage
│   ├── ez_project.yml         # Shared base config (do not run directly)
│   ├── ez_ingest.yml          # Ingest stage config
│   ├── ez_full_pipeline.yml   # All four stages in sequence
│   ├── ez_change.yml          # Change panel stage config
│   └── ez_aggregate.yml       # Aggregation stage config
├── data/
│   ├── raw/
│   │   └── parcel_data/{year}/Baci{year}.shp   # Annual parcel shapefiles
│   ├── filtered/              # Per-year filtered GeoPackages
│   └── full_panel_gpkg/       # Assembled full panel GeoPackage
├── logs/                      # Timestamped run logs
├── baltimoreparcel/           # Main Python package
│   ├── config.py              # Field lists and constants
│   ├── directories.py         # Path helpers
│   ├── run_config.py          # YAML config loaders and dataclasses
│   ├── utils.py               # Logging and console output utilities
│   ├── gis/                   # GIS I/O, transforms, validation
│   ├── panel/                 # Panel assembly logic
│   └── scripts/               # Runnable pipeline stage scripts
├── UrbanEconomics_EZ_analysis.gdb/   # Output ArcGIS GDB
└── pyproject.toml
```

---

## Configuration

All pipeline behavior is driven by **YAML config files** in `configs/`. Configs support inheritance via an `extends` key:

```yaml
extends: ez_project.yml   # inherits all keys; child keys override parent
```

### Key config sections

| Section | Description |
|---|---|
| `project` | GDB filename, data/logs directories, CRS EPSG code |
| `data` | Years to process, ID field, input file/layer name patterns |
| `panel` | Output GPKG name, full/change panel layer names |
| `fields` | Columns to keep, CPI fields, log-transform fields, time fields |
| `spatial.joins` | Spatial join rules applied during assembly |
| `derive` | Rules for creating binary treatment/membership columns |
| `aggregations` | Group-by and aggregation rules for the aggregate stage |
| `ingest.layers` | Layer specs for the ingest stage |
| `pipeline.stages` | Which stages to run in the pipeline orchestrator |

### Config hierarchy (EZ analysis)

```
ez_project.yml          ← shared base
    └── ez_ingest.yml
    └── ez_assemble.yml (or build_ez_panel.yml)
            └── ez_full_pipeline.yml
    └── ez_change.yml
    └── ez_aggregate.yml
```

---

## Pipeline

The pipeline has four independent stages. Each can be run individually or chained via `pipeline.py`.

### Stage 1: Ingest

**Script**: `baltimoreparcel/scripts/ingest_to_gdb.py`  
**Config**: `configs/ez_ingest.yml`

Reads raw shapefiles (or GeoJSONs / GPKGs) and loads each as a named feature class into the project GDB. Supports two layer types:

- `year_series` — one shapefile per year → one GDB feature class per year (e.g. `parcels_2003`)
- `static` — one file → one GDB layer (e.g. `neighborhoods`, `enterprise_zones_2012`)

Each layer spec supports filters (drop rows below a value threshold, drop nulls/empties), column selection, CRS reprojection, and a bounding-box clip.

```bash
# Ingest all layers defined in the config:
python -m baltimoreparcel.scripts.ingest_to_gdb --config configs/ez_ingest.yml

# Ingest specific layers only:
python -m baltimoreparcel.scripts.ingest_to_gdb --config configs/ez_ingest.yml --layers "parcels_{year}" neighborhoods
```

---

### Stage 2: Assemble

**Script**: `baltimoreparcel/scripts/assemble_panel.py`  
**Config**: `configs/build_ez_panel.yml` (or any config extending `ez_project.yml`)

Builds the full longitudinal panel:

1. Loads parcel layers for each year from the GDB or per-year GPKGs
2. Stacks into a long-format panel sorted by `(ACCTID, YEAR)`
3. Trims to the `fields.keep` column list
4. Applies optional CPI inflation adjustment (produces `REAL_*` columns)
5. Applies optional log-transform (produces `LOG_*` and `LOG_REAL_*` columns)
6. Performs spatial joins (e.g. attach neighborhood name, flag EZ membership)
7. Applies derive rules (e.g. create `IN_EZ_2012`, treatment group flags)
8. Writes the full panel to GeoPackage + ArcGIS GDB
9. Converts integer year columns to ArcGIS `DATE` fields for time animation

```bash
python -m baltimoreparcel.scripts.assemble_panel --config configs/build_ez_panel.yml
```

---

### Stage 3: Change Panel

**Script**: `baltimoreparcel/scripts/build_change_panel.py`  
**Config**: `configs/ez_change.yml`

Reads the assembled full panel and computes year-over-year changes:

- **Numeric fields** (e.g. `LOG_REAL_NFMTTLVL`): absolute change and annualized change per year
- **String fields** (e.g. `LU`, `ZONING`, `OWNNAME1`): binary flag (1 = changed, 0 = no change)

Enrich fields (e.g. `NEIGHBORHOOD`, `GEOGCODE`, `ADDRESS`) are attached from the most recent full-panel row per parcel. Output is written to GeoPackage + GDB.

```bash
python -m baltimoreparcel.scripts.build_change_panel --config configs/ez_change.yml
```

---

### Stage 4: Aggregate

**Script**: `baltimoreparcel/scripts/aggregate_panels.py`  
**Config**: `configs/ez_aggregate.yml`

Reads aggregation rules from the YAML config and exports one GDB layer per rule. Rules specify `group_by` columns and `agg` functions (e.g. `mean`, `sum`).

```bash
# Change-panel aggregations only (default):
python -m baltimoreparcel.scripts.aggregate_panels --config configs/ez_aggregate.yml

# Full-panel aggregations only:
python -m baltimoreparcel.scripts.aggregate_panels --config configs/ez_aggregate.yml --full

# Both:
python -m baltimoreparcel.scripts.aggregate_panels --config configs/ez_aggregate.yml --full --change
```

---

### Running the Full Pipeline

**Script**: `baltimoreparcel/scripts/pipeline.py`  
**Config**: `configs/ez_full_pipeline.yml`

Chains any combination of the four stages in sequence. Stages are defined in the config under `pipeline.stages` and can be overridden at the CLI with `--stages`.

```bash
# Run all four stages:
python -m baltimoreparcel.scripts.pipeline --config configs/ez_full_pipeline.yml

# Run specific stages only:
python -m baltimoreparcel.scripts.pipeline --config configs/ez_full_pipeline.yml --stages assemble change

# Run aggregate stage with full-panel aggregations:
python -m baltimoreparcel.scripts.pipeline --config configs/ez_full_pipeline.yml --stages aggregate --full
```

---

## Module Reference

### `baltimoreparcel.config`

Constants used across the package:

| Name | Description |
|---|---|
| `PARCEL_FIELDS` | Default column list for all parcel GeoDataFrames |
| `VALUE_COLUMNS` | Assessment value columns: `NFMLNDVL`, `NFMIMPVL`, `NFMTTLVL` |
| `IDENTIFIER_COLUMN` | Parcel unique key: `ACCTID` |
| `BALTIMORE_CENTRAL` | `(lon, lat)` center point for the city |
| `ALL_YEARS` | `range(2003, 2025)` |

---

### `baltimoreparcel.directories`

Resolved `Path` objects and directory helpers:

| Name | Description |
|---|---|
| `PROJECT_DIR` | Root of the project (parent of this package) |
| `GBD_DIR` | `BaltimoreParcelProject.gdb` |
| `DATA_DIR` | `data/` |
| `RAW_PARCELS_DIR` | `data/raw/parcel_data/` |
| `FILTERED_DIR` | `data/filtered/` |
| `FIGS_DIR` | `data/figures/` |
| `LOGS_DIR` | `logs/` |
| `ensure_dir(path)` | Creates directory (and parents) if it doesn't exist |
| `get_year_gpkg_dir(year, create)` | Returns `data/{year}_gpkg/`, optionally creating it |

---

### `baltimoreparcel.run_config`

YAML config loaders and typed dataclasses for each pipeline stage.

**Loaders:**

| Function | Returns | Used by |
|---|---|---|
| `load_config(path)` | `PanelRunConfig` | `assemble_panel`, `pipeline` |
| `load_change_panel_config(path)` | `ChangePanelRunConfig` | `build_change_panel` |
| `load_aggregate_config(path)` | `AggregateRunConfig` | `aggregate_panels` |
| `load_ingest_config(path)` | `IngestRunConfig` | `ingest_to_gdb` |

All loaders support YAML inheritance via `extends`.

**Converter helpers** (used by `pipeline.py`):

- `to_change_panel_config(cfg)` — converts `PanelRunConfig` → `ChangePanelRunConfig`
- `to_aggregate_config(cfg)` — converts `PanelRunConfig` → `AggregateRunConfig`
- `to_ingest_config(cfg)` — converts `PanelRunConfig` → `IngestRunConfig`

**Key dataclasses:** `ProjectConfig`, `DataConfig`, `PanelOutputConfig`, `DeriveRule`, `TreatmentPeriod`, `AggregationRule`, `SpatialJoinSpec`, `LayerSpec`

---

### `baltimoreparcel.utils`

Logging and console output utilities.

**`Logger`** — singleton that redirects `stdout`/`stderr` to both the terminal and a timestamped log file under `logs/`.

```python
from baltimoreparcel.utils import Logger
logger = Logger.setup()          # start logging
# ... run pipeline ...
Logger.teardown()                # restore stdout/stderr
```

**Console output helpers** (color-coded prefixes):

| Function | Color | Use |
|---|---|---|
| `process_step(msg)` | Teal | Major pipeline phase |
| `info(msg)` | Cyan | Informational |
| `warn(msg)` | Yellow | Non-fatal issue |
| `error(msg)` | Red | Error condition |
| `success(msg)` | Green | Stage completed |

---

### `baltimoreparcel.gis`

#### `gis.io` — I/O

| Function | Description |
|---|---|
| `read_vector_layer(year, name, directory, layer)` | Read a shapefile or GPKG layer; returns `None` on missing file |
| `write_gpkg_layer(gdf, year, name, directory, layer)` | Write a `GeoDataFrame` to a GPKG layer; sanitizes geometry and drops nulls |
| `export_to_geodb(gdf, gdb_path, layer_name)` | Export a `GeoDataFrame` to an ArcGIS GDB feature class via `arcpy` |
| `arcstr(path)` | Convert a `Path` to a forward-slash string safe for `arcpy` |

#### `gis.transform` — Filtering and projection

| Function | Description |
|---|---|
| `ensure_crs(gdf, epsg, source_epsg)` | Reproject (or assign) CRS; `source_epsg` handles files with missing `.prj` |
| `filter_on_field(gdf, fields, filters, identifier)` | Apply a list of scalar or callable filters to corresponding fields |
| `select_columns(gdf, columns)` | Keep only specified columns; warns about missing ones |
| `pivot_panel(gdf, value_field, id_field, col_field)` | Pivot long-format panel to wide format (one column per year) |

#### `gis.validate` — Geometry

| Function | Description |
|---|---|
| `is_valid_gis_file(name)` | `True` for `.shp` or `.gpkg` |
| `drop_null_geometries(gdf, year)` | Remove rows with null geometry |
| `sanitize_geometry(gdf)` | Force all geometries to 2D (drop Z coordinate) |

---

### `baltimoreparcel.panel`

#### `panel.transform` — Value transformations

| Function | Description |
|---|---|
| `to_real_data(data, price_map, year_series)` | Deflate nominal values using a CPI price map keyed by year |
| `log_value(gdf, value_field)` | Add `LOG_{value_field}` column using `np.log` |

#### `panel.change` — Year-over-year change

| Function | Description |
|---|---|
| `calculate_change(gdf, value_prefix, acctid_col, field_type)` | Pivot wide panel → long-format change; supports `"numeric"` (absolute delta) and `"string"` (binary change flag) |

#### `panel.derive` — Derived columns

| Function | Description |
|---|---|
| `apply_isin_rule(gdf, rule)` | Add a binary column that is `1` when a field value is in a given list |
| `apply_treatment(gdf, rule)` | Add `NOT_TREATED`, `TREATED`, `EXTRA_TREATED` binary columns and a string label, using year-conditional period windows |

#### `panel.spatial` — Spatial enrichment

| Function | Description |
|---|---|
| `apply_spatial_joins(panel_gdf, joins, gdb_path)` | Apply a list of spatial join specs; supports `"attribute"` (attach a field from overlapping polygon) and `"isin"` (boolean flag for spatial predicate) |

---

### `baltimoreparcel.scripts`

| Script | Entry point | Description |
|---|---|---|
| `filter_initial_files.py` | `main(years)` | Pre-pipeline filter: reads raw shapefiles, applies basic value/address filters, writes per-year GPKGs. Uses `ProcessPoolExecutor` for parallel processing. |
| `ingest_to_gdb.py` | `run(cfg)` | Stage 1 — loads source files into the GDB. |
| `assemble_panel.py` | `run(cfg)` | Stage 2 — builds the longitudinal full panel. |
| `build_change_panel.py` | `run(cfg)` | Stage 3 — computes year-over-year changes. |
| `aggregate_panels.py` | `run(cfg, do_full, do_change)` | Stage 4 — aggregates panels by group. |
| `pipeline.py` | CLI | Orchestrates any combination of the four stages. |
| `spatial_analysis_pipeline.py` | `run(cfg)` | Runs Moran's I and bivariate spatial analysis. |
| `merge_time_layers.py` | `merge_layers(...)` | Merges per-year GDB layers into a single time-enabled layer. |

---

## Spatial Analysis

**Script**: `baltimoreparcel/scripts/spatial_analysis_pipeline.py`

Runs spatial autocorrelation and bivariate Moran's I on the assembled change panel. Configured via `AnalysisConfig`:

```python
from baltimoreparcel.scripts.spatial_analysis_pipeline import AnalysisConfig, NeighborhoodSetting

cfg = AnalysisConfig(
    lag_panel_name="base_lag_panel",
    base_var="LOG_REAL_NFMTTLVL_CHNG",
    dependent_vars=["LOG_REAL_NFMLNDVL_CHNG", "ZONING_CHNG"],
    neighborhood_settings=[NeighborhoodSetting("K_NEAREST_NEIGHBORS", 50)],
    num_permutations=199,
)
```

Output reports are saved to `data/spatial_analysis_reports/`.

---

## Data Dictionary

| Field | Description |
|---|---|
| `ACCTID` | Parcel account ID (unique key) |
| `GEOGCODE` | Assessment geography code |
| `ADDRESS` / `STRTNUM` / `STRTNAM` | Parcel address components |
| `OWNNAME1` | Owner name |
| `ZONING` | Zoning classification |
| `LU` / `DESCLU` | Land use code and description |
| `ACRES` / `LANDAREA` | Parcel size |
| `YEARBLT` | Year structure was built |
| `SQFTSTRC` | Structure square footage |
| `NFMLNDVL` | Assessed land value (nominal) |
| `NFMIMPVL` | Assessed improvement value (nominal) |
| `NFMTTLVL` | Assessed total value (nominal) |
| `REAL_*` | CPI-deflated version of the corresponding nominal field |
| `LOG_*` | Natural log of the corresponding field |
| `*_CHNG` | Year-over-year absolute change |
| `*_CHNG_PER_YEAR` | Annualized change |
| `IN_EZ_2012` | `1` if parcel is within an Enterprise Zone boundary (2012 designation) |
| `IN_FOCUS_EZ_2012` | `1` if parcel is within a Focus Area Enterprise Zone (subset of EZ) |
| `NEIGHBORHOOD` | Baltimore NSA neighborhood name (from spatial join) |
| `YEAR` / `YEAR_DATE` | Panel year (integer and ArcGIS DATE formats) |
| `START_YR` / `END_YR` | Change panel period endpoints |

