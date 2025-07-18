import geopandas as gpd
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

def timeseries_histogram(
    value_gdf: gpd.GeoDataFrame, 
    value_field: str = "NFMTTLVL",
    n_cols: int = 4,
    bins: int = 30,
    save_path: Path | None = None,
    show_plot: bool = True,
    plot_trend: bool = True,
    trend_save_path: Path | None = None
):
    """
    Plot a grid of histograms for each year or year-pair depending on data format.

    Supports:
    - Wide format: Columns like LOG_NFMTTLVL_2003, ..., _2024
    - Long format: Columns START_YR, END_YR, LOG_NFMTTLVL_CHNG
    """
    print(f"Creating histogram grid for {value_field}...")

    is_long_form = {"START_YR", "END_YR", f"{value_field}"}.issubset(value_gdf.columns)

    if is_long_form:
        # --- LONG FORMAT HANDLING ---
        print("Detected long-form input (START_YR + END_YR).")
        value_col = f"{value_field}"
        value_gdf["PERIOD"] = (
            value_gdf["START_YR"].astype(str) + "–" + value_gdf["END_YR"].astype(str)
        )
        periods = sorted(value_gdf["PERIOD"].unique())
        n_rows = int(np.ceil(len(periods) / n_cols))

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4.5, n_rows * 3.5), constrained_layout=True)
        axes = axes.flatten()

        means, medians = [], []

        for i, period in enumerate(periods):
            ax = axes[i]
            data = value_gdf[value_gdf["PERIOD"] == period][value_col].dropna()
            ax.hist(data, bins=bins, color='skyblue', edgecolor='black')
            ax.set_title(f"Δ {period}", fontsize=10)
            ax.set_yticks([])
            ax.grid(True, linestyle='--', alpha=0.5)

            median = np.median(data)
            mean = np.mean(data)
            medians.append(median)
            means.append(mean)
            ax.axvline(median, color='darkred', linestyle='--', linewidth=1)
            ax.text(median, ax.get_ylim()[1]*0.9, f"{median:.2f}", 
                    rotation=90, va='top', ha='right', fontsize=6, color='darkred')

        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        fig.suptitle(f"{value_field} Change Distributions by Period", fontsize=16)

        if plot_trend:
            years = [int(p.split("–")[0]) for p in periods]
            plt.figure(figsize=(10, 6))
            plt.plot(years, medians, marker='o', label='Median', color='darkred')
            plt.plot(years, means, marker='s', label='Mean', color='darkblue', linestyle='--')
            plt.title(f"{value_field} Change Trend")
            plt.xlabel("Start Year")
            plt.ylabel("Log Change")
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.legend()
            if trend_save_path:
                plt.savefig(trend_save_path, dpi=150, bbox_inches="tight")
            if show_plot:
                plt.show()

    else:
        # --- WIDE FORMAT HANDLING (existing logic) ---
        print("Detected wide-format input (year columns with prefix)...")
        years = sorted(
            [col for col in value_gdf.columns if col.startswith(f"{value_field}_")], 
            key=lambda x: int(x.split("_")[-1])
        )
        year_nums = [int(col.split("_")[-1]) for col in years]
        n_rows = int(np.ceil(len(years) / n_cols))

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4.5, n_rows * 3.5), constrained_layout=True)
        axes = axes.flatten()

        medians, means = [], []

        for i, col in enumerate(years):
            ax = axes[i]
            data = value_gdf[col].dropna()
            ax.hist(data, bins=bins, color='steelblue', edgecolor='black')
            ax.set_title(col.replace(f"{value_field}_", ""), fontsize=10)
            ax.set_yticks([])
            ax.grid(True, linestyle='--', alpha=0.5)

            median = np.median(data)
            mean = np.mean(data)
            medians.append(median)
            means.append(mean)
            ax.axvline(median, color='darkred', linestyle='--', linewidth=1)
            ax.text(median, ax.get_ylim()[1]*0.9, f"{int(median):,}", 
                    rotation=90, va='top', ha='right', fontsize=6, color='darkred')

        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        fig.suptitle(f"{value_field} Distributions by Year", fontsize=16)

        if plot_trend:
            plt.figure(figsize=(10, 6))
            plt.plot(year_nums, medians, marker='o', label='Median', color='darkred')
            plt.plot(year_nums, means, marker='s', label='Mean', color='darkblue', linestyle='--')
            plt.title(f"{value_field} Trend Over Time")
            plt.xlabel("Year")
            plt.ylabel(value_field)
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.legend()
            if trend_save_path:
                plt.savefig(trend_save_path, dpi=150, bbox_inches="tight")
            if show_plot:
                plt.show()

    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[Saved] Histogram grid to: {save_path}")
    if show_plot:
        plt.show()
