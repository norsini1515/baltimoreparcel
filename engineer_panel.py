import numpy as np
import geopandas as gpd

def log_value(gdf: gpd.GeoDataFrame, value_field:str) -> gpd.GeoDataFrame:
    # Example: create a log-transformed value column
    gdf = gdf.copy()
    gdf[f"LOG_{value_field}"] = np.log(gdf[value_field])
    return gdf
