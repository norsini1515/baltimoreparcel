# baltimoreparcel/panel/transform.py
# CPI inflation adjustment and log-transform utilities

import numpy as np
import geopandas as gpd
import pandas as pd


def to_real_data(data: pd.Series, price_map: pd.Series, year_series: pd.Series) -> pd.Series:
    """
    Deflate a nominal value series to real dollars.

    Parameters
    ----------
    data        : nominal values (e.g. panel_gdf['NFMTTLVL'])
    price_map   : price index keyed by year (e.g. prices.set_index('year')['price'])
    year_series : year for each row  (e.g. panel_gdf['YEAR'])
    """
    return data / year_series.map(price_map)

def log_value(gdf: gpd.GeoDataFrame, value_field: str) -> gpd.GeoDataFrame:
    """Add a log-transformed column LOG_{value_field} to the GeoDataFrame."""
    gdf = gdf.copy()
    gdf[f"LOG_{value_field}"] = np.log(gdf[value_field].replace(0, np.nan))
    return gdf
