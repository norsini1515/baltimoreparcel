# baltimoreparcel/panel/change.py
# Year-over-year change calculation, panel enrichment, and summary statistics

import numpy as np
import geopandas as gpd
import pandas as pd


def calculate_change(
    gdf: gpd.GeoDataFrame,
    value_prefix: str = "LOG_NFMTTLVL",
    acctid_col: str = "ACCTID",
    geom_col: str = "geometry",
    per_year: bool = True,
    dropna: bool = True,
    field_type: str = "numeric",
) -> gpd.GeoDataFrame:
    """
    Vectorized long-form change calculator.

    Parameters:
        gdf          : Wide-format GeoDataFrame with year columns
                       e.g. LOG_NFMTTLVL_2003 … LOG_NFMTTLVL_2024
        value_prefix : Column prefix (e.g. 'LOG_NFMTTLVL')
        acctid_col   : Parcel ID column name
        geom_col     : Geometry column name
        per_year     : Include annualized change column
        dropna       : Drop rows where change is NA or zero
        field_type   : 'numeric' or 'string'

    Returns:
        Long-form GeoDataFrame with one row per ACCTID-period.
    """
    value_cols = sorted(
        [col for col in gdf.columns if col.startswith(f"{value_prefix}_")],
        key=lambda x: int(x.split("_")[-1])
    )
    years = [int(col.split("_")[-1]) for col in value_cols]

    result_frames = []

    for i in range(1, len(years)):
        y0, y1 = years[i - 1], years[i]
        col0, col1 = f"{value_prefix}_{y0}", f"{value_prefix}_{y1}"

        if field_type == "numeric":
            delta = gdf[col1] - gdf[col0]
            out = pd.DataFrame({
                acctid_col: gdf[acctid_col],
                geom_col: gdf[geom_col],
                "START_YR": y0,
                "END_YR": y1,
                f"{value_prefix}_CHNG": delta,
            })
            if per_year:
                out[f"{value_prefix}_CHNG_PER_YEAR"] = delta / (y1 - y0)
            if dropna:
                out = out[delta.notna() & (delta != 0)]

        elif field_type == "string":
            s0 = gdf[col0].astype(str).str.strip().str.lower()
            s1 = gdf[col1].astype(str).str.strip().str.lower()
            delta = (s0 != s1).astype(int)
            out = pd.DataFrame({
                acctid_col: gdf[acctid_col],
                geom_col: gdf[geom_col],
                "START_YR": y0,
                "END_YR": y1,
                f"{value_prefix}_CHNG": delta,
            })
            if dropna:
                out = out[~gdf[col0].isna() & ~gdf[col1].isna()]

        else:
            raise ValueError(f"Unsupported field_type: {field_type}")

        result_frames.append(out)

    result = pd.concat(result_frames, ignore_index=True)
    return gpd.GeoDataFrame(result, geometry=geom_col, crs=gdf.crs)


def enrich_change_gdf(
    change_gdf: gpd.GeoDataFrame,
    base_gdf: gpd.GeoDataFrame,
    enrich_fields: list[str],
) -> gpd.GeoDataFrame:
    """Attach metadata fields from the base panel to a change GeoDataFrame."""
    enrich_df = base_gdf[["ACCTID"] + enrich_fields].drop_duplicates()
    return change_gdf.merge(enrich_df, on="ACCTID", how="left")


def summarize_field(
    change_gdf: gpd.GeoDataFrame,
    value_field: str = "LOG_NFMTTLVL_CHNG",
    group_fields: list[str] = ["START_YR", "END_YR"],
) -> pd.DataFrame:
    """Aggregate change statistics (mean, median, std, pos/neg counts) by group."""
    df = (
        change_gdf
        .groupby(group_fields)
        .agg(
            n=(        "ACCTID",     "count"),
            mean=(     value_field,  "mean"),
            median=(   value_field,  "median"),
            std=(      value_field,  "std"),
            pos=(      value_field,  lambda x: (x > 0).sum()),
            neg=(      value_field,  lambda x: (x < 0).sum()),
        )
        .reset_index()
    )
    df["net_growth"] = df["pos"] - df["neg"]
    df["span_years"] = df["END_YR"] - df["START_YR"]
    df["mean_ann"]   = df["mean"]   / df["span_years"]
    df["median_ann"] = df["median"] / df["span_years"]
    return df
