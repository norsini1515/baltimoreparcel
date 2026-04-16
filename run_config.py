# baltimoreparcel/run_config.py
"""
run_config.py

Config schema and YAML loaders for the parcel pipeline.

Each pipeline stage has its own focused config class and loader:

    load_config(path)              → PanelRunConfig        (ingest + assemble)
    load_change_panel_config(path) → ChangePanelRunConfig  (build_change_panel)
    load_aggregate_config(path)    → AggregateRunConfig    (aggregate_panels)

All three loaders support YAML inheritance via an `extends` key:

    extends: ez_project.yml   # path relative to the child config file

The pipeline orchestrator (pipeline.py) uses PanelRunConfig plus two
converter helpers to hand each stage its own typed config:

    to_change_panel_config(cfg)  → ChangePanelRunConfig
    to_aggregate_config(cfg)     → AggregateRunConfig
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

import yaml


# ---------------------------------------------------------------------------
# Sub-config dataclasses (shared across config types)
# ---------------------------------------------------------------------------

@dataclass
class ProjectConfig:
    """Paths and coordinate-reference settings for the project."""
    name: str = "project"
    gdb: str = ""               # GDB filename, relative to project_dir
    data_dir: str = "data"      # data root, relative to project_dir
    logs_dir: str = "logs"      # logs root, relative to project_dir
    crs_epsg: int = 4326        # default CRS EPSG code


@dataclass
class DataConfig:
    """Source data layout: years, primary key, file/layer name patterns, and source type."""
    years: list = field(default_factory=list)
    id_field: str = "id"
    year_field: str = "YEAR"
    input_layer_pattern: str = "{year}layer"    # GDB feature class name or GPKG layer name
    input_file_pattern: str = "{year}.gpkg"     # GPKG filename (ignored when source_type=gdb)
    source_type: str = "gpkg"                   # "gpkg" or "gdb"


@dataclass
class PanelOutputConfig:
    """Names and paths for panel output files and layers."""
    output_gpkg: str = "panel.gpkg"
    full_panel_layer: str = "full_panel"
    change_panel_layer: str = "change_panel"
    full_panel_subdir: str = "full_panel_gpkg"  # subdir under data_dir
    cpi_file: Optional[str] = None              # CSV filename in full_panel_dir


@dataclass
class FieldsConfig:
    """
    Field lists that drive column selection, transformations, and change calculations.

    Hierarchy
    ---------
    keep            → master column list retained from the source files
                      (empty = keep everything)
    monetary        → subset of keep: apply CPI inflation adjustment
    log_transform   → subset of keep (or derived REAL_* cols): log-transform
    numeric_change  → subset of keep (or derived cols): compute numeric diff
    string_change   → subset of keep: compute binary change flag
    enrich          → fields from the full panel to attach to change panel rows
    """
    keep: list = field(default_factory=list)
    monetary: list = field(default_factory=list)
    log_transform: list = field(default_factory=list)
    numeric_change: list = field(default_factory=list)
    string_change: list = field(default_factory=list)
    enrich: list = field(default_factory=list)
    time_fields_full: list = field(default_factory=list)    # [[src, dst], ...]
    time_fields_change: list = field(default_factory=list)  # [[src, dst], ...]


@dataclass
class SpatialConfig:
    """Reference layer used for spatial enrichment (e.g. neighborhoods)."""
    neighborhoods_layer: str = "neighborhoods"
    neighborhoods_name_field: str = "Name"
    neighborhoods_output_field: str = "NEIGHBORHOOD"


@dataclass
class TogglesConfig:
    """Boolean switches that control which steps run inside assemble_panel."""
    generate_new_panel: bool = False
    calculate_real_values: bool = False
    log_value_fields: bool = False


@dataclass
class LayerSpec:
    """
    Specification for one layer (or year-series of layers) to ingest into the GDB.

    Fields
    ------
    name        Target GDB feature class name.  May contain ``{year}`` for year_series.
    source      Path to the source file, relative to project_dir.  May contain ``{year}``.
    type        ``"static"`` (one file → one GDB layer) or ``"year_series"`` (one file
                per year → one GDB layer per year).  Inferred automatically when ``years``
                is provided; explicit ``type`` overrides inference.
    layer       Sub-layer name inside a multi-layer source (GPKG, GDB).  May contain
                ``{year}`` for year_series.  Leave None for single-layer files.
    crs_epsg    Override the project CRS for this specific layer.  None = use
                ``project.crs_epsg``.
    keep        Column subset to retain.  Empty list = keep all columns.
    filters     Filter specs (same dict format as before: field, type, value).
    years       Year list for year_series layers.  Auto-populated from a ``years:``
                start/end range in the YAML.
    add_year_field
                When True (default) a ``YEAR`` column is injected into year_series
                layers.  Ignored for static layers.
    """
    name: str = ""
    source: str = ""
    type: str = "static"                     # "static" or "year_series"
    layer: Optional[str] = None
    crs_epsg: Optional[int] = None
    keep: list = field(default_factory=list)
    filters: list = field(default_factory=list)
    years: list = field(default_factory=list)
    add_year_field: bool = True


@dataclass
class IngestConfig:
    """
    Top-level ingest settings used by ingest_to_gdb.py.
    Contains a list of LayerSpec entries — one per file (or year-series) to ingest.
    """
    layers: list = field(default_factory=list)   # list[LayerSpec]


@dataclass
class AggregationRule:
    """One groupby-aggregation step."""
    group_by: list = field(default_factory=list)
    agg: dict = field(default_factory=dict)


@dataclass
class AggregationsConfig:
    """Aggregation rules for the full and change panels."""
    full_panel: list = field(default_factory=list)    # list[AggregationRule]
    change_panel: list = field(default_factory=list)  # list[AggregationRule]


@dataclass
class PipelineConfig:
    """
    Stage list for the pipeline orchestrator (pipeline.py).
    Valid stage names: "ingest", "assemble", "change", "aggregate"
    """
    stages: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# Top-level config classes
# ---------------------------------------------------------------------------

@dataclass
class PanelRunConfig:
    """
    Full config for ingest_to_gdb.py and assemble_panel.py.
    Also used as the source by pipeline.py (which converts to stage-specific
    configs as needed via to_change_panel_config / to_aggregate_config).
    """
    project: ProjectConfig = field(default_factory=ProjectConfig)
    data: DataConfig = field(default_factory=DataConfig)
    panel: PanelOutputConfig = field(default_factory=PanelOutputConfig)
    fields: FieldsConfig = field(default_factory=FieldsConfig)
    spatial: SpatialConfig = field(default_factory=SpatialConfig)
    toggles: TogglesConfig = field(default_factory=TogglesConfig)
    aggregations: AggregationsConfig = field(default_factory=AggregationsConfig)
    ingest: IngestConfig = field(default_factory=IngestConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    project_dir_override: Optional[Path] = None

    @property
    def project_dir(self) -> Path:
        if self.project_dir_override is not None:
            return Path(self.project_dir_override)
        return Path(__file__).parent.parent

    @property
    def gdb_path(self) -> Path:
        return self.project_dir / self.project.gdb

    @property
    def data_path(self) -> Path:
        return self.project_dir / self.project.data_dir

    @property
    def logs_path(self) -> Path:
        return self.project_dir / self.project.logs_dir

    @property
    def full_panel_dir(self) -> Path:
        return self.data_path / self.panel.full_panel_subdir

    def get_year_gpkg_dir(self, year) -> Optional[Path]:
        d = self.data_path / f"{year}_gpkg"
        return d if d.exists() else None

    def input_file_for(self, year) -> str:
        return self.data.input_file_pattern.format(year=year)

    def input_layer_for(self, year) -> str:
        return self.data.input_layer_pattern.format(year=year)


@dataclass
class ChangePanelRunConfig:
    """
    Focused config for build_change_panel.py.
    Only the fields that script actually uses.
    """
    project: ProjectConfig = field(default_factory=ProjectConfig)
    data: DataConfig = field(default_factory=DataConfig)
    panel: PanelOutputConfig = field(default_factory=PanelOutputConfig)
    fields: FieldsConfig = field(default_factory=FieldsConfig)
    project_dir_override: Optional[Path] = None

    @property
    def project_dir(self) -> Path:
        if self.project_dir_override is not None:
            return Path(self.project_dir_override)
        return Path(__file__).parent.parent

    @property
    def gdb_path(self) -> Path:
        return self.project_dir / self.project.gdb

    @property
    def data_path(self) -> Path:
        return self.project_dir / self.project.data_dir

    @property
    def logs_path(self) -> Path:
        return self.project_dir / self.project.logs_dir

    @property
    def full_panel_dir(self) -> Path:
        return self.data_path / self.panel.full_panel_subdir


@dataclass
class IngestRunConfig:
    """
    Focused config for ingest_to_gdb.py.
    Defines which files to ingest and where to find them — nothing more.
    """
    project: ProjectConfig = field(default_factory=ProjectConfig)
    layers: list = field(default_factory=list)   # list[LayerSpec]
    project_dir_override: Optional[Path] = None

    @property
    def project_dir(self) -> Path:
        if self.project_dir_override is not None:
            return Path(self.project_dir_override)
        return Path(__file__).parent.parent

    @property
    def gdb_path(self) -> Path:
        return self.project_dir / self.project.gdb

    @property
    def data_path(self) -> Path:
        return self.project_dir / self.project.data_dir

    @property
    def logs_path(self) -> Path:
        return self.project_dir / self.project.logs_dir


@dataclass
class AggregateRunConfig:
    """
    Lightweight config for aggregate_panels.py.
    Enables a ~20-line YAML (extends a base) to drive one aggregation run.

    Note: time_fields_full and time_fields_change are flat here
    (not nested in FieldsConfig) because the agg script only needs those
    two lists — they map from the YAML key fields.time_fields.
    """
    project: ProjectConfig = field(default_factory=ProjectConfig)
    panel: PanelOutputConfig = field(default_factory=PanelOutputConfig)
    spatial: SpatialConfig = field(default_factory=SpatialConfig)
    aggregations: AggregationsConfig = field(default_factory=AggregationsConfig)
    time_fields_full: list = field(default_factory=list)
    time_fields_change: list = field(default_factory=list)
    project_dir_override: Optional[Path] = None

    @property
    def project_dir(self) -> Path:
        if self.project_dir_override is not None:
            return Path(self.project_dir_override)
        return Path(__file__).parent.parent

    @property
    def gdb_path(self) -> Path:
        return self.project_dir / self.project.gdb

    @property
    def data_path(self) -> Path:
        return self.project_dir / self.project.data_dir

    @property
    def logs_path(self) -> Path:
        return self.project_dir / self.project.logs_dir

    @property
    def full_panel_dir(self) -> Path:
        return self.data_path / self.panel.full_panel_subdir


# ---------------------------------------------------------------------------
# YAML helpers
# ---------------------------------------------------------------------------

def _deep_merge(base: dict, override: dict) -> dict:
    """
    Recursively merge *override* into *base*.
    - dicts  → merged key-by-key (override wins on conflict)
    - lists  → override replaces entirely
    - scalars → override wins
    Returns a new dict; neither input is mutated.
    """
    result = dict(base)
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = val
    return result


def _load_raw_yaml(yaml_path: Path) -> dict:
    """
    Load a YAML file, resolving an optional `extends` key by deep-merging
    the parent (recursively) before returning the merged dict.
    The `extends` path is resolved relative to the child config's directory.
    """
    yaml_path = yaml_path.resolve()
    with open(yaml_path) as fh:
        raw = yaml.safe_load(fh) or {}
    if "extends" in raw:
        parent_path = (yaml_path.parent / raw.pop("extends")).resolve()
        raw = _deep_merge(_load_raw_yaml(parent_path), raw)
    return raw


# ---------------------------------------------------------------------------
# Private parse helpers (shared across loaders)
# ---------------------------------------------------------------------------

def _parse_years(years_spec) -> list:
    if isinstance(years_spec, dict):
        return list(range(years_spec["start"], years_spec["end"] + 1))
    if isinstance(years_spec, list):
        return years_spec
    return []


def _parse_project(raw: dict) -> tuple[ProjectConfig, Optional[Path]]:
    p = raw.get("project", {})
    project = ProjectConfig(
        name=p.get("name", "project"),
        gdb=p.get("gdb", ""),
        data_dir=p.get("data_dir", "data"),
        logs_dir=p.get("logs_dir", "logs"),
        crs_epsg=p.get("crs_epsg", 4326),
    )
    override = p.get("project_dir", None)
    return project, Path(override) if override else None


def _parse_panel(raw: dict) -> PanelOutputConfig:
    po = raw.get("panel", {})
    return PanelOutputConfig(
        output_gpkg=po.get("output_gpkg", "panel.gpkg"),
        full_panel_layer=po.get("full_panel_layer", "full_panel"),
        change_panel_layer=po.get("change_panel_layer", "change_panel"),
        full_panel_subdir=po.get("full_panel_subdir", "full_panel_gpkg"),
        cpi_file=po.get("cpi_file", None),
    )


def _parse_spatial(raw: dict) -> SpatialConfig:
    s = raw.get("spatial", {})
    return SpatialConfig(
        neighborhoods_layer=s.get("neighborhoods_layer", "neighborhoods"),
        neighborhoods_name_field=s.get("neighborhoods_name_field", "Name"),
        neighborhoods_output_field=s.get("neighborhoods_output_field", "NEIGHBORHOOD"),
    )


def _parse_aggregations(raw: dict) -> AggregationsConfig:
    a = raw.get("aggregations", {})
    return AggregationsConfig(
        full_panel=[
            AggregationRule(group_by=r["group_by"], agg=r["agg"])
            for r in a.get("full_panel", [])
        ],
        change_panel=[
            AggregationRule(group_by=r["group_by"], agg=r["agg"])
            for r in a.get("change_panel", [])
        ],
    )


def _parse_layer_spec(spec: dict) -> LayerSpec:
    """Parse one entry from the ``ingest.layers`` list."""
    years_raw = spec.get("years", None)
    years = _parse_years(years_raw) if years_raw else []
    # Infer type from presence of years when not explicit
    inferred_type = "year_series" if years else "static"
    return LayerSpec(
        name=spec["name"],
        source=spec["source"],
        type=spec.get("type", inferred_type),
        layer=spec.get("layer", None),
        crs_epsg=spec.get("crs_epsg", None),
        keep=spec.get("keep", []),
        filters=spec.get("filters", []),
        years=years,
        add_year_field=spec.get("add_year_field", True),
    )


def _parse_ingest(raw: dict) -> IngestConfig:
    i = raw.get("ingest", {})
    return IngestConfig(
        layers=[_parse_layer_spec(s) for s in i.get("layers", [])]
    )


def _parse_fields(raw: dict) -> FieldsConfig:
    f = raw.get("fields", {})
    tf = f.get("time_fields", {})
    return FieldsConfig(
        keep=f.get("keep", []),
        monetary=f.get("monetary", []),
        log_transform=f.get("log_transform", []),
        numeric_change=f.get("numeric_change", []),
        string_change=f.get("string_change", []),
        enrich=f.get("enrich", []),
        time_fields_full=tf.get("full_panel", []),
        time_fields_change=tf.get("change_panel", []),
    )


# ---------------------------------------------------------------------------
# Public loaders
# ---------------------------------------------------------------------------

def load_config(yaml_path: Union[str, Path]) -> PanelRunConfig:
    """Load a full PanelRunConfig (ingest + assemble stages)."""
    raw = _load_raw_yaml(Path(yaml_path))

    project, project_dir_override = _parse_project(raw)

    d = raw.get("data", {})
    data = DataConfig(
        years=_parse_years(d.get("years", [])),
        id_field=d.get("id_field", "id"),
        year_field=d.get("year_field", "YEAR"),
        input_layer_pattern=d.get("input_layer_pattern", "{year}layer"),
        input_file_pattern=d.get("input_file_pattern", "{year}.gpkg"),
        source_type=d.get("source_type", "gpkg"),
    )

    t = raw.get("toggles", {})
    toggles = TogglesConfig(
        generate_new_panel=t.get("generate_new_panel", False),
        calculate_real_values=t.get("calculate_real_values", False),
        log_value_fields=t.get("log_value_fields", False),
    )

    ingest = _parse_ingest(raw)

    pl = raw.get("pipeline", {})
    pipeline = PipelineConfig(stages=pl.get("stages", []))

    return PanelRunConfig(
        project=project,
        data=data,
        panel=_parse_panel(raw),
        fields=_parse_fields(raw),
        spatial=_parse_spatial(raw),
        toggles=toggles,
        aggregations=_parse_aggregations(raw),
        ingest=ingest,
        pipeline=pipeline,
        project_dir_override=project_dir_override,
    )


def load_change_panel_config(yaml_path: Union[str, Path]) -> ChangePanelRunConfig:
    """Load a ChangePanelRunConfig (build_change_panel stage)."""
    raw = _load_raw_yaml(Path(yaml_path))
    project, project_dir_override = _parse_project(raw)

    d = raw.get("data", {})
    data = DataConfig(
        id_field=d.get("id_field", "id"),
        year_field=d.get("year_field", "YEAR"),
        source_type=d.get("source_type", "gpkg"),
    )

    return ChangePanelRunConfig(
        project=project,
        data=data,
        panel=_parse_panel(raw),
        fields=_parse_fields(raw),
        project_dir_override=project_dir_override,
    )


def load_aggregate_config(yaml_path: Union[str, Path]) -> AggregateRunConfig:
    """
    Load an AggregateRunConfig (aggregate_panels stage).
    Accepts the same YAML structure as load_config — unknown keys are ignored.
    A full PanelRunConfig YAML is also a valid AggregateRunConfig YAML.
    """
    raw = _load_raw_yaml(Path(yaml_path))
    project, project_dir_override = _parse_project(raw)
    tf = raw.get("fields", {}).get("time_fields", {})

    return AggregateRunConfig(
        project=project,
        panel=_parse_panel(raw),
        spatial=_parse_spatial(raw),
        aggregations=_parse_aggregations(raw),
        time_fields_full=tf.get("full_panel", []),
        time_fields_change=tf.get("change_panel", []),
        project_dir_override=project_dir_override,
    )


# ---------------------------------------------------------------------------
# Converter helpers (used by pipeline.py)
# ---------------------------------------------------------------------------

def to_change_panel_config(cfg: PanelRunConfig) -> ChangePanelRunConfig:
    """Extract a ChangePanelRunConfig from a full PanelRunConfig."""
    return ChangePanelRunConfig(
        project=cfg.project,
        data=cfg.data,
        panel=cfg.panel,
        fields=cfg.fields,
        project_dir_override=cfg.project_dir_override,
    )


def to_aggregate_config(cfg: PanelRunConfig) -> AggregateRunConfig:
    """Extract an AggregateRunConfig from a full PanelRunConfig."""
    return AggregateRunConfig(
        project=cfg.project,
        panel=cfg.panel,
        spatial=cfg.spatial,
        aggregations=cfg.aggregations,
        time_fields_full=cfg.fields.time_fields_full,
        time_fields_change=cfg.fields.time_fields_change,
        project_dir_override=cfg.project_dir_override,
    )


def to_ingest_config(cfg: PanelRunConfig) -> "IngestRunConfig":
    """Extract an IngestRunConfig from a full PanelRunConfig."""
    return IngestRunConfig(
        project=cfg.project,
        layers=cfg.ingest.layers,
        project_dir_override=cfg.project_dir_override,
    )


def load_ingest_config(yaml_path: Union[str, Path]) -> "IngestRunConfig":
    """
    Load an IngestRunConfig.
    Only requires ``project`` and ``ingest.layers`` in the YAML.
    A full PanelRunConfig YAML is also valid — extra keys are ignored.
    """
    raw = _load_raw_yaml(Path(yaml_path))
    project, project_dir_override = _parse_project(raw)
    return IngestRunConfig(
        project=project,
        layers=_parse_ingest(raw).layers,
        project_dir_override=project_dir_override,
    )
