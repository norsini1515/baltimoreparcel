def engineer_panel(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    # Example: create a log-transformed value column
    gdf = gdf.copy()
    gdf["LOG_" + VALUE_FIELD] = np.log(gdf[VALUE_FIELD])
    return gdf
