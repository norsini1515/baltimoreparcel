# baltimoreparcel/panel/transform.py
# CPI inflation adjustment and log-transform utilities

import numpy as np
import geopandas as gpd
import pandas as pd


def to_real_data(data: pd.Series, prices: pd.DataFrame) -> pd.Series:
    """Deflate a nominal value series to real (2024) dollars using a CPI price index."""
    return data / prices['Price']

def log_value(gdf: gpd.GeoDataFrame, value_field: str) -> gpd.GeoDataFrame:
    """Add a log-transformed column LOG_{value_field} to the GeoDataFrame."""
    gdf = gdf.copy()
    gdf[f"LOG_{value_field}"] = np.log(gdf[value_field].replace(0, np.nan))
    return gdf
