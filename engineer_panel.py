"""
baltimoreparcel/engineer_panel.py
"""
import numpy as np
import geopandas as gpd
import pandas as pd


def log_value(gdf: gpd.GeoDataFrame, value_field:str) -> gpd.GeoDataFrame:
    # Example: create a log-transformed value column
    gdf = gdf.copy()
    gdf[f"LOG_{value_field}"] = np.log(gdf[value_field])
    return gdf

def calculate_change(
    gdf: gpd.GeoDataFrame,
    value_prefix: str = "LOG_NFMTTLVL",
    acctid_col: str = "ACCTID",
    geom_col: str = "geometry",
    per_year: bool = True,
    dropna: bool = True
) -> gpd.GeoDataFrame:
    """
    Vectorized long-form change calculator for log value fields.

    Parameters:
        gdf         : Wide-format GeoDataFrame | LOG_NFMTTLVL_2003, ..., _2024
        value_prefix: Prefix for columns (e.g., 'LOG_NFMTTLVL')
        acctid_col  : Name of parcel ID column
        geom_col    : Name of geometry column
        per_year    : Whether to include per-year log change
        dropna      : Whether to exclude rows with NA or no change

    Returns:
        Long-form GeoDataFrame
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
        
        delta = gdf[col1] - gdf[col0]
        out = pd.DataFrame({
            acctid_col: gdf[acctid_col],
            geom_col: gdf[geom_col],
            "START_YR": y0,
            "END_YR": y1,
            f"{value_prefix}_CHNG": delta
        })

        if per_year:
            out[f"{value_prefix}_CHNG_PER_YEAR"] = delta / (y1 - y0)

        if dropna:
            out = out[delta.notna() & (delta != 0)]

        result_frames.append(out)

    result = pd.concat(result_frames, ignore_index=True)
    return gpd.GeoDataFrame(result, geometry=geom_col, crs=gdf.crs)

def enrich_change_gdf(change_gdf: gpd.GeoDataFrame, base_gdf: gpd.GeoDataFrame, enrich_fields: list[str]) -> gpd.GeoDataFrame:
    enrich_df = base_gdf[["ACCTID"] + enrich_fields].drop_duplicates()
    return change_gdf.merge(enrich_df, on="ACCTID", how="left")



def summarize_field(change_gdf: gpd.GeoDataFrame, 
                    value_field: str = "LOG_NFMTTLVL_CHNG"):
    df = (
        change_gdf
        .groupby(["START_YR", "END_YR"])
        .agg(
            n=("ACCTID", "count"),
            mean=(value_field, "mean"),
            median=(value_field, "median"),
            std=(value_field, "std"),
            pos=(value_field, lambda x: (x > 0).sum()),
            neg=(value_field, lambda x: (x < 0).sum()),
        )
        .assign(net_growth=lambda df: df["pos"] - df["neg"])
        .reset_index()
    )
    df["net_growth"] = df["pos"] - df["neg"]
    df["span_years"] = df["END_YR"] - df["START_YR"]

    df["mean_ann"] = df["mean"] / df["span_years"]
    df["median_ann"] = df["median"] / df["span_years"]

    return df