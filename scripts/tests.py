import arcpy
from baltimoreparcel.gis_utils import arcstr
from baltimoreparcel.directories import GBD_DIR

# print(help(arcpy.stats.BivariateSpatialAssociation))
# from multiprocessing import Pool, cpu_count
# print(f"Number of CPU cores available: {cpu_count()}")

lag_panel_name = "base_lag_panel"
lag_panel_gpkg_path = arcstr(GBD_DIR / lag_panel_name)


# out_path = arcstr(GBD_DIR / 'test_bivariate')
out_path = arcstr(GBD_DIR / 'biv_TTLCHG_ZCHG_knn50_2005_2004')
# arcpy.stats.BivariateSpatialAssociation(
#             in_features=lag_panel_gpkg_path,
#             analysis_field1='LOG_REAL_NFMTTLVL_CHNG_2021',
#             analysis_field2='LOG_REAL_NFMIMPVL_CHNG_2022',
#             out_features=out_path,
#             neighborhood_type="K_NEAREST_NEIGHBORS",
#             num_neighbors=20,
#             local_weighting_scheme="UNWEIGHTED",
#             num_permutations=199
#         )

# Quick field inspection
fields = [f.name for f in arcpy.ListFields(out_path)]
print("Field names:")
for field in fields:
    print(f"  {field}")
    
print(f"\nTotal fields: {len(fields)}")