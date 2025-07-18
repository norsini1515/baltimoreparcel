import os
import geopandas as gpd
import matplotlib.pyplot as plt
from baltimoreparcel.gis_utils import read_vector_layer
from baltimoreparcel.directories import get_year_gpkg_dir

# === Load GeoData ===
FULL_PANEL_GEOPKG = "Baci_full_panel.gpkg"
FULL_PANEL_DIR = get_year_gpkg_dir("full_panel")
VALUE = "NFMTTLVL"

NFMTTLVL_change_gdf = read_vector_layer(
    year=f"LOG_{VALUE}_change_wide_panel",
    name=FULL_PANEL_GEOPKG,
    directory=FULL_PANEL_DIR,
    layer="LOG_NFMTTLVL_change"
)
print(f"Read change in LOG_{VALUE} has {NFMTTLVL_change_gdf.shape[0]} rows, {NFMTTLVL_change_gdf.shape[1]} columns")

# === Ensure START_YR is integer ===
NFMTTLVL_change_gdf["START_YR"] = NFMTTLVL_change_gdf["START_YR"].dt.year.astype(int)

# === Define plot grid ===
years = sorted(NFMTTLVL_change_gdf["START_YR"].unique())
n_cols = 4
n_rows = -(-len(years) // n_cols)  # Ceiling division

# === Shared color scale ===
vmin = NFMTTLVL_change_gdf["LOG_NFMTTLVL_CHNG"].min()
vmax = NFMTTLVL_change_gdf["LOG_NFMTTLVL_CHNG"].max()

fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))

for i, year in enumerate(years):
    ax = axes.flat[i]
    subset = NFMTTLVL_change_gdf[NFMTTLVL_change_gdf["START_YR"] == year]
    subset.plot(
        column="LOG_NFMTTLVL_CHNG",
        ax=ax,
        cmap="RdYlGn",
        vmin=vmin,
        vmax=vmax,
        legend=False
    )
    ax.set_title(f"{year} → {year + 1}", fontsize=12)
    ax.axis("off")

# === Hide unused subplots ===
for j in range(i + 1, n_rows * n_cols):
    axes.flat[j].axis("off")

# === Add shared colorbar ===
fig.subplots_adjust(right=0.9)
cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
sm = plt.cm.ScalarMappable(cmap="RdYlGn", norm=plt.Normalize(vmin=vmin, vmax=vmax))
sm._A = []  # Required for matplotlib < 3.1
fig.colorbar(sm, cax=cbar_ax, label="Log Change in NFMTTLVL")

plt.suptitle("Parcel-Level Change in Appraised Value (Log-Transformed)", fontsize=16, y=1.02)
plt.tight_layout()
plt.show()
