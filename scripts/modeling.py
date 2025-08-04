# baltimoreparcel/scripts/modeling.py
# -*- coding: utf-8 -*-
"""
Run modeling pipeline after spatially joining bivariate layers
"""
import sys
import arcpy
import geopandas as gpd
import pandas as pd
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

from baltimoreparcel.utils import Logger, info, warn, error, success, process_step
from baltimoreparcel.directories import LOGS_DIR, GBD_DIR
from baltimoreparcel.gis_utils import arcstr

arcpy.env.overwriteOutput = True
arcpy.env.workspace = arcstr(GBD_DIR)

# Set input layer names
biv_knn50_lnd = "biv_TTLCHG_LNDCHG_knn50_merged"
biv_knn50_imp = "biv_TTLCHG_IMPCHG_knn50_merged"
biv_knn50_zch = "biv_TTLCHG_ZCHG_knn50_merged"

# Output layers
joined_lnd_imp = "joined_lnd_imp"
final_merged = "final_merged_bivar"

if __name__ == "__main__":
    # Init logger
    now = datetime.now()
    timestamp = now.strftime("%Y%m%d %H%M")
    log_file = LOGS_DIR / f"modeling_{timestamp}.log"
    logger = Logger(log_file)

    process_step("Step 1: SpatialJoin LND + IMP")
    arcpy.analysis.SpatialJoin(
        target_features=arcstr(GBD_DIR / biv_knn50_lnd),
        join_features=arcstr(GBD_DIR / biv_knn50_imp),
        out_feature_class=arcstr(GBD_DIR / joined_lnd_imp),
        join_type="KEEP_ALL",
        join_operation="JOIN_ONE_TO_ONE",
        match_option="INTERSECT"
    )
    success("LND + IMP spatial join complete.")

    process_step("Step 2: SpatialJoin result + ZCH")
    arcpy.analysis.SpatialJoin(
        target_features=arcstr(GBD_DIR / joined_lnd_imp),
        join_features=arcstr(GBD_DIR / biv_knn50_zch),
        out_feature_class=arcstr(GBD_DIR / final_merged),
        join_type="KEEP_ALL",
        join_operation="JOIN_ONE_TO_ONE",
        match_option="INTERSECT"
    )
    success("Final spatial join (LND+IMP + ZCH) complete.")

    process_step("Step 3: Read merged data")
    merged = gpd.read_file(str(GBD_DIR), layer=final_merged)
    info(f"Final merged shape: {merged.shape}")

    # Step 4: Create exposure flags
    merged["NEAR_HOT_LND"] = (merged["ASSOC_CAT"] != 0).astype(int)
    merged["NEAR_HOT_IMP"] = (merged["ASSOC_CAT_1"] != 0).astype(int)
    merged["NEAR_HOT_ZCH"] = (merged["ASSOC_CAT_12"] != 0).astype(int)

    process_step("Step 5: Prepare model inputs")
    target_col = "LOG_REAL_NFMTTLVL_CHNG_2021"
    predictors = [
        "NEAR_HOT_LND", "NEAR_HOT_IMP", "NEAR_HOT_ZCH",
        "LOG_REAL_NFMIMPVL_CHNG_2017",
        "LOG_REAL_NFMLNDVL_CHNG_2017",
        "ZONING_CHNG_2017",
        "OWNNAME1_CHNG_2017"
    ]

    model_df = merged[predictors + [target_col]].copy().dropna()
    X = model_df[predictors]
    y = model_df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    process_step("Step 6: Fit Linear Regression")
    lr = LinearRegression().fit(X_train, y_train)
    lr_preds = lr.predict(X_test)
    print("Linear Model:")
    print("R²:", r2_score(y_test, lr_preds))
    print("RMSE:", mean_squared_error(y_test, lr_preds, squared=False))
    print(pd.Series(lr.coef_, index=X.columns))

    process_step("Step 7: Fit Random Forest")
    rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42).fit(X_train, y_train)
    rf_preds = rf.predict(X_test)
    print("\nRandom Forest:")
    print("R²:", r2_score(y_test, rf_preds))
    print("RMSE:", mean_squared_error(y_test, rf_preds, squared=False))
