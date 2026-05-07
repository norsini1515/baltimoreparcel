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
    base_layer: str = ""                         # static GDB layer used as panel base (neighborhood panel)
    parcel_layer_pattern: str = ""              # per-year parcel layer to aggregate into base (neighborhood panel)


@dataclass
class PanelOutputConfig:
    """Names and paths for panel output files and layers."""
    output_gpkg: str = "panel.gpkg"
    full_panel_layer: str = "full_panel"
    change_panel_layer: str = "change_panel"
    full_panel_subdir: str = "full_panel_gpkg"  # subdir under data_dir
    cpi_file: Optional[str] = None              # CSV filename in full_panel_dir


@dataclass
class TreatmentPeriod:
    """
    One time-period window for an ``ez_treatment`` derive rule.
    Rows whose year falls in [years_start, years_end] are assigned
    treatment status based on ``ez_field`` and ``focus_field``.
    """
    years_start: int = 0
    years_end: int = 9999
    treated_field: str = ""    # binary column: 1 = in primary treatment group
    extra_field: str = ""      # binary column: 1 = in extra/focus treatment group (subset of treated)


@dataclass
class DeriveRule:
    """
    One derived-column rule applied row-by-row to the panel.

    type = "isin"
        Creates a binary (0/1) column ``name`` that is 1 when ``field``
        is in ``values``.

    type = "ez_treatment"
        Creates three binary columns (``output_treated``,
        ``output_extra_treated``, ``output_untreated``) using year-conditional
        EZ membership.  For each period, a row is classified as:
          extra_treated  if focus_field == 1
          treated        if ez_field == 1 and focus_field == 0
          untreated      if ez_field == 0
    """
    type: str = "isin"
    # isin fields
    name: Optional[str] = None
    source_field: Optional[str] = None    # renamed from 'field' to avoid shadowing dataclasses.field
    values: list = field(default_factory=list)
    # ez_treatment fields
    year_field: str = "YEAR"
    periods: list = field(default_factory=list)   # list[TreatmentPeriod]
    output_treated: str = "TREATED"
    output_extra_treated: str = "EXTRA_TREATED"
    output_not_treated: str = "NOT_TREATED"
    output_label: str = "TREATMENT_GROUP"    # string label: "treated" / "extra treated" / "not treated"


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
    derive          → computed columns added to the panel (isin flags, treatment status)
    numeric_change  → subset of keep (or derived cols): compute numeric diff
    string_change   → subset of keep: compute binary change flag
    enrich          → fields from the full panel to attach to change panel rows
    """
    keep: list = field(default_factory=list)
    monetary: list = field(default_factory=list)
    log_transform: list = field(default_factory=list)
    derive: list = field(default_factory=list)       # list[DeriveRule]
    numeric_change: list = field(default_factory=list)
    string_change: list = field(default_factory=list)
    enrich: list = field(default_factory=list)
    time_fields_full: list = field(default_factory=list)    # [[src, dst], ...]
    time_fields_change: list = field(default_factory=list)  # [[src, dst], ...]


@dataclass
class SpatialJoinSpec:
    """
    One spatial enrichment step applied during panel assembly.

    type
    ----
    attribute   Spatial join that attaches a field value from the overlapping
                reference polygon (e.g. neighbourhood name).  Requires ``field``.
    isin        Creates a boolean column: True if the parcel geometry satisfies
                the spatial predicate against the reference layer.

    how
    ---
    within      Strict containment — the parcel geometry must lie fully inside
                the reference polygon.
    intersects  Any overlap counts.  More permissive; useful for parcels that
                straddle boundaries.
    """
    layer: str = ""                      # GDB feature class name
    type: str = "attribute"              # "attribute" or "isin"
    field: Optional[str] = None          # source field (attribute type only)
    output_field: str = ""               # column name added to the panel
    how: str = "within"                  # spatial predicate: "within" | "intersects"


@dataclass
class SpatialConfig:
    """Spatial enrichment steps applied during panel assembly."""
    joins: list = field(default_factory=list)   # list[SpatialJoinSpec]


@dataclass
class ParcelAggConfig:
    """
    How to aggregate per-year parcel layers into the neighborhood base polygon.
    Only used by assemble_neighborhood_panel.py — ignored by assemble_panel.py.

    join_predicate   Spatial predicate used to assign each parcel to a neighborhood:
                     "within" (parcel fully inside polygon) or "intersects".
    count_output_field
                     Name given to the parcel-count column produced by agg.
    agg              Dict of {source_field: pandas_agg_func} applied per neighborhood.
                     Use "count" for any field you want to count (e.g. ACCTID: count).
    """
    join_predicate: str = "within"
    count_output_field: str = "PARCEL_COUNT"
    agg: dict = field(default_factory=dict)
    value_counts: list = field(default_factory=list)  # categorical fields to pivot into per-value count columns


@dataclass
class TogglesConfig:
    """Boolean switches that control which steps run inside assemble_panel."""
    generate_new_panel: bool = False
    calculate_real_values: bool = False
    log_value_fields: bool = False
    derive_fields: bool = False    # run derive rules (requires spatial joins to run first)


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
    crs_epsg    Target CRS for this layer.  None = use ``project.crs_epsg``.
    source_crs_epsg
                The CRS the source file's coordinates are *actually in*, for
                use when the file has no .prj / CRS metadata.  When set,
                ``ensure_crs`` assigns this as the source CRS and then
                reprojects to ``crs_epsg``.  Leave None when the file either
                has its CRS embedded or is already in the target CRS.
    bbox        ``[xmin, ymin, xmax, ymax]`` in the *target* CRS.  Applied
                after reprojection — rows with centroids outside this envelope
                are dropped.  Use to remove stray/sentinel-coordinate features.
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
    source_crs_epsg: Optional[int] = None
    bbox: Optional[list] = None              # [xmin, ymin, xmax, ymax] in target CRS; drops anything outside
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
    name: Optional[str] = None    # output GDB layer name; auto-generated from group_by if omitted


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
    parcel_agg: ParcelAggConfig = field(default_factory=ParcelAggConfig)
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
    joins = [
        SpatialJoinSpec(
            layer=j["layer"],
            type=j.get("type", "attribute"),
            field=j.get("field", None),
            output_field=j["output_field"],
            how=j.get("how", "within"),
        )
        for j in s.get("joins", [])
    ]
    return SpatialConfig(joins=joins)


def _parse_parcel_agg(raw: dict) -> ParcelAggConfig:
    pa = raw.get("parcel_agg", {})
    return ParcelAggConfig(
        join_predicate=pa.get("join_predicate", "within"),
        count_output_field=pa.get("count_output_field", "PARCEL_COUNT"),
        agg=pa.get("agg", {}),
        value_counts=pa.get("value_counts", []),
    )


def _parse_aggregations(raw: dict) -> AggregationsConfig:
    a = raw.get("aggregations", {})
    return AggregationsConfig(
        full_panel=[
            AggregationRule(group_by=r["group_by"], agg=r["agg"], name=r.get("name"))
            for r in a.get("full_panel", [])
        ],
        change_panel=[
            AggregationRule(group_by=r["group_by"], agg=r["agg"], name=r.get("name"))
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
        source_crs_epsg=spec.get("source_crs_epsg", None),
        bbox=spec.get("bbox", None),
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


def _parse_derive_rules(rules_raw: list) -> list:
    """Parse the fields.derive list into DeriveRule objects."""
    rules = []
    for r in rules_raw:
        rule_type = r.get("type", "isin")
        if rule_type == "isin":
            rules.append(DeriveRule(
                type="isin",
                name=r["name"],
                source_field=r["field"],
                values=r.get("values", []),
            ))
        elif rule_type == "treatment":
            periods = [
                TreatmentPeriod(
                    years_start=p["years"]["start"],
                    years_end=p["years"]["end"],
                    treated_field=p["treated_field"],
                    extra_field=p["extra_field"],
                )
                for p in r.get("periods", [])
            ]
            rules.append(DeriveRule(
                type="treatment",
                year_field=r.get("year_field", "YEAR"),
                periods=periods,
                output_treated=r.get("output_treated", "TREATED"),
                output_extra_treated=r.get("output_extra_treated", "EXTRA_TREATED"),
                output_not_treated=r.get("output_not_treated", "NOT_TREATED"),
                output_label=r.get("output_label", "TREATMENT_GROUP"),
            ))
        else:
            raise ValueError(f"Unknown derive rule type: {rule_type!r}. Valid types: isin, treatment")
    return rules


def _parse_fields(raw: dict) -> FieldsConfig:
    f = raw.get("fields", {})
    tf = f.get("time_fields", {})
    # Note: derive rules are parsed from top-level 'derive:' key (after 'spatial:'),
    # not from 'fields.derive', so that key order in the YAML reflects execution order.
    return FieldsConfig(
        keep=f.get("keep", []),
        monetary=f.get("monetary", []),
        log_transform=f.get("log_transform", []),
        derive=[],   # populated by load_config from top-level 'derive:' key
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
        base_layer=d.get("base_layer", ""),
        parcel_layer_pattern=d.get("parcel_layer_pattern", ""),
    )

    t = raw.get("toggles", {})
    toggles = TogglesConfig(
        generate_new_panel=t.get("generate_new_panel", False),
        calculate_real_values=t.get("calculate_real_values", False),
        log_value_fields=t.get("log_value_fields", False),
        derive_fields=t.get("derive_fields", False),
    )

    ingest = _parse_ingest(raw)

    fields = _parse_fields(raw)
    fields.derive = _parse_derive_rules(raw.get("derive", []))

    pl = raw.get("pipeline", {})
    pipeline = PipelineConfig(stages=pl.get("stages", []))

    return PanelRunConfig(
        project=project,
        data=data,
        panel=_parse_panel(raw),
        fields=fields,
        spatial=_parse_spatial(raw),
        parcel_agg=_parse_parcel_agg(raw),
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
