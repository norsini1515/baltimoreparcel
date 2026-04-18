# baltimoreparcel/panel/derive.py
"""Row-level derived column rules applied during panel assembly."""

import numpy as np
import geopandas as gpd
import pandas as pd

from baltimoreparcel.run_config import DeriveRule


def apply_isin_rule(gdf: gpd.GeoDataFrame, rule: DeriveRule) -> gpd.GeoDataFrame:
    """
    Add a binary (0/1) column ``rule.name`` that is 1 when ``rule.field``
    is in ``rule.values``.
    """
    if rule.source_field not in gdf.columns:
        from baltimoreparcel.utils import warn
        warn(f"  isin derive: field '{rule.source_field}' not in panel — skipping '{rule.name}'")
        return gdf
    gdf = gdf.copy()
    gdf[rule.name] = gdf[rule.source_field].isin(rule.values).astype(int)
    return gdf


def apply_treatment(gdf: gpd.GeoDataFrame, rule: DeriveRule) -> gpd.GeoDataFrame:
    """
    Add four columns for treatment status using year-conditional binary membership flags.

    For each period [years_start, years_end], classifies rows using two input
    binary columns (``treated_field`` and ``extra_field``):

    Binary (0/1):
      NOT_TREATED    = 1  when treated_field == 0
      TREATED        = 1  when treated_field == 1 and extra_field == 0
      EXTRA_TREATED  = 1  when extra_field == 1

    String label:
      "extra treated" | "treated" | "not treated"
      Rows whose year falls in no defined period receive pd.NA in all columns.
    """
    gdf = gdf.copy()
    treated     = pd.Series(0, index=gdf.index, dtype=int)
    extra       = pd.Series(0, index=gdf.index, dtype=int)
    not_treated = pd.Series(0, index=gdf.index, dtype=int)
    in_any_period = pd.Series(False, index=gdf.index)

    for period in rule.periods:
        missing = [
            c for c in (rule.year_field, period.treated_field, period.extra_field)
            if c not in gdf.columns
        ]
        if missing:
            from baltimoreparcel.utils import warn
            warn(f"  treatment derive: missing columns {missing} — skipping period "
                 f"{period.years_start}–{period.years_end}")
            continue

        in_period   = (
            (gdf[rule.year_field] >= period.years_start) &
            (gdf[rule.year_field] <= period.years_end)
        )
        is_extra    = in_period & (gdf[period.extra_field] == 1)
        is_treated  = in_period & (gdf[period.treated_field] == 1) & (gdf[period.extra_field] == 0)
        is_control  = in_period & (gdf[period.treated_field] == 0)

        extra[is_extra]           = 1
        treated[is_treated]       = 1
        not_treated[is_control]   = 1
        in_any_period             = in_any_period | in_period

    gdf[rule.output_extra_treated] = extra
    gdf[rule.output_treated]       = treated
    gdf[rule.output_not_treated]   = not_treated

    # String label — NA for rows outside all defined periods
    label = np.select(
        [extra == 1, treated == 1, not_treated == 1],
        ["extra treated", "treated", "not treated"],
        default=None,
    )
    label = pd.array(label, dtype=pd.StringDtype())
    label[~in_any_period] = pd.NA
    gdf[rule.output_label] = label

    return gdf


def apply_derive_rules(gdf: gpd.GeoDataFrame, rules: list) -> gpd.GeoDataFrame:
    """Dispatch and apply each DeriveRule in *rules* to *gdf* in order."""
    for rule in rules:
        if rule.type == "isin":
            gdf = apply_isin_rule(gdf, rule)
        elif rule.type == "treatment":
            gdf = apply_treatment(gdf, rule)
        else:
            raise ValueError(f"Unknown derive rule type: {rule.type!r}")
    return gdf
