import arcpy

# print(help(arcpy.stats.BivariateSpatialAssociation))
from multiprocessing import Pool, cpu_count
print(f"Number of CPU cores available: {cpu_count()}")

from sklearn.metrics import r2_score, mean_squared_error