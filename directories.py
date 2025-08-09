from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent

GBD_DIR = PROJECT_DIR / "BaltimoreParcelProject.gdb"
DATA_DIR = PROJECT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
FILTERED_DIR = DATA_DIR / "filtered"
FIGS_DIR = DATA_DIR / "figures"
LOGS_DIR = PROJECT_DIR / "logs"

def ensure_dir(path: Path) -> Path:
    """Ensure a directory exists before use."""
    path.mkdir(parents=True, exist_ok=True)
    return path

def get_year_gpkg_dir(year: int, create: bool=False) -> Path | None:
    """Return directory for a given year's GeoPackage outputs."""
    dir_path = DATA_DIR / f"{year}_gpkg"
    if create:
        return ensure_dir(dir_path)
    elif dir_path.exists():
        return dir_path
    else:
        print(f"[{year}] WARNING: GPKG directory does not exist. Use create=True to generate.")
        return None  

if __name__ == "__main__":
    print(f"Project directory: {PROJECT_DIR}")