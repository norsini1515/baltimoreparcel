from baltimoreparcel.gis_utils import (
    read_vector_layer, ensure_crs, filter_on_field,
    select_columns, write_gpkg_layer
)
from baltimoreparcel.directories import (
    GBD_DIR, FILTERED_DIR, get_year_gpkg_dir
)


if  __name__ == "__main__":
    print("Testing moved files...", GBD_DIR)