# baltimoreparcel/scripts/merge_time_layers.py
# -*- coding: utf-8 -*-
"""
Merge Time Series Layers Script for Baltimore Parcel Project
"""

import arcpy
from datetime import datetime
from pathlib import Path

from baltimoreparcel.directories import LOGS_DIR, GBD_DIR
from baltimoreparcel.utils import Logger, info, warn, error, success, process_step
from baltimoreparcel.gis_utils import convert_time_fields, arcstr

arcpy.env.overwriteOutput = True
arcpy.env.workspace = arcstr(GBD_DIR)

def merge_layers(
    prefix: str,
    output_fc: str,
    convert_time: bool = True,
    time_pairs: list[tuple[str, str]] = [("START_YR", "START_DATE"), ("END_YR", "END_DATE")]
):
    """
    Merges feature classes with a common prefix into a single layer with time info.
    """
    info(f"Merging layers with prefix '{prefix}' into '{output_fc}'...")

    all_fcs = [
        fc for fc in arcpy.ListFeatureClasses(f"{prefix}_*")
        if "_with_time" not in fc and fc != output_fc
    ]
    if not all_fcs:
        error(f"No layers found with prefix '{prefix}'.")
        return

    info(f"Found {len(all_fcs)} layers with prefix '{prefix}'")

    temp_fcs = []
    for fc in all_fcs:
        # Parse start and end year
        start_yr, end_yr = fc.split("_")[-2:]
        if int(end_yr) > int(start_yr):
            end_yr, start_yr = start_yr, end_yr  # Ensure start is always less than end

        print(f"Processing {fc} for years {start_yr} to {end_yr}")

        year_id = fc.replace(prefix, "")
        temp_fc = f"{fc}_with_time"

        process_step(f"Processing {fc} --> {temp_fc}")
        arcpy.management.CopyFeatures(fc, temp_fc)

        # Add START_DATE and END_DATE fields
        arcpy.management.AddField(temp_fc, "START_YR", "LONG")
        arcpy.management.AddField(temp_fc, "END_YR", "LONG")
        arcpy.management.CalculateField(temp_fc, "START_YR", str(start_yr), "PYTHON3")
        arcpy.management.CalculateField(temp_fc, "END_YR", str(end_yr), "PYTHON3")
        temp_fcs.append(temp_fc)

    info(f"Merging {len(temp_fcs)} layers into {output_fc}")
    arcpy.management.Merge(temp_fcs, output_fc)
    print('merged.')
    # arcpy.management.DeleteFeatures(temp_fcs)
    if convert_time:
        fields = [f.name for f in arcpy.ListFields(output_fc)]
        info(f"Fields in {output_fc}: {fields}")

        process_step("Converting time fields to proper DATE type...")
        convert_time_fields(output_fc, field_pairs=time_pairs)

    success(f"Output written to: {output_fc} in {GBD_DIR}")


if __name__ == "__main__":
    # Initialize logging
    now = datetime.now().strftime("%Y%m%d_%H%M")
    log_file = LOGS_DIR / f"merge_time_layers_{now}.log"
    logger = Logger(log_file)

    try:
        # merge_layers(
        #     prefix="biv_TTLCHG_IMPCHG_knn50",
        #     output_fc="biv_TTLCHG_IMPCHG_knn50_merged",
        #     convert_time=True,
        # )
        merge_layers(
            prefix="biv_TTLCHG_ZCHG_knn50",
            output_fc="biv_TTLCHG_ZCHG_knn50_merged",
            convert_time=True,
        )
        merge_layers(
            prefix="biv_TTLCHG_LNDCHG_knn50",
            output_fc="biv_TTLCHG_LNDCHG_knn50_merged",
            convert_time=True,
        )
    except Exception as e:
        error(f"Failed to complete merge: {e}")
    
    success("Merge completed successfully.")