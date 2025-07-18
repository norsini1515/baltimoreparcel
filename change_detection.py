import geopandas as gpd
import pandas as pd

def detect_zoning_change(wide_gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    """Returns a DataFrame of zoning changes per parcel over time."""
    # logic: compare wide-form columns like ZONING_2003 to ZONING_2024
    ...

def detect_owner_change(wide_gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    """Detects if parcel changed owner at any point."""
    ...