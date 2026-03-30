# baltimoreparcel/__init__.py
# Public API — safe to import without arcpy/geopandas being present

from baltimoreparcel.config import (
    ALL_YEARS,
    PARCEL_FIELDS,
    VALUE_COLUMNS,
    PARCEL_SPECIFIC_COLUMNS,
    LOCATION_COLUMNS,
    IDENTIFIER_COLUMN,
    BALTIMORE_CENTRAL,
)
from baltimoreparcel.directories import (
    PROJECT_DIR,
    DATA_DIR,
    RAW_DIR,
    FILTERED_DIR,
    FIGS_DIR,
    LOGS_DIR,
    ensure_dir,
    get_year_gpkg_dir,
)
from baltimoreparcel.utils import (
    Logger,
    info,
    warn,
    error,
    success,
    process_step,
    color_text,
    extract_base_var,
)
