import re
import math
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from baltimoreparcel.directories import DATA_DIR, FIGS_DIR, LOGS_DIR
from baltimoreparcel.scripts.eda.plots import plot_line
from baltimoreparcel.utils import Logger, info, warn, error, success, extract_base_var


if __name__ == '__main__':
    # Initialize logger
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    log_file = LOGS_DIR / f"morans_i_plots_{timestamp}.log"
    logger = Logger(log_file)

    # data_paths = [DATA_DIR / "morans_i_results_LOG_REAL_NFMIMPVL_20250809_1106.csv",
    #         DATA_DIR / "morans_i_results_LOG_REAL_NFMTTLVL_20250809_0249.csv",
    #         ]
    data_paths = [DATA_DIR / "morans_i_results_LOG_REAL_NFMTTLVL_CHNG_20250809_1543.csv",
            DATA_DIR / "morans_i_results_LOG_REAL_NFMIMPVL_CHNG_20250809_1549.csv",
            ]
    base_name = "morans_i_results_"
    FILENAME_RE = re.compile(
        rf"^{re.escape(base_name)}(.+?)_(\d{{8}}_\d{{4}})$"
    )

    n = len(data_paths)
    rows = n
    cols = 1
    fig, axes = plt.subplots(rows, cols, figsize=(8, 4 * rows), squeeze=False)
    axes = axes.ravel()

    for i, path in enumerate(data_paths):
        var = extract_base_var(Path(path), FILENAME_RE)
        info(f"Processing {var} from {path.name}")
        
        
        df = pd.read_csv(path)
        plot_df = df.dropna(subset=["morans_i", "z_score"]).copy()
        plot_df = plot_df.sort_values("year")

        info(f"Loaded data from {path.name}\n{df.shape=}")

        ax = axes[i]
        plot_line(
            df=plot_df,
            x="year",
            y="morans_i",
            y2="z_score",
            title=f"Moran's I and z-score over time • {var}",
            xlabel="Year",
            ylabel="Moran's I",
            y2label="z-score",
            markers=True,
            ax=ax,         # draw into this subplot
            ax2=None,      # will be created by plot_line via twinx()
            save_path=None # do not save per-panel here
        )
    
    # hide any unused axes
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    fig.tight_layout()
    out_path = FIGS_DIR / f"morans_grid_{timestamp}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved grid to {out_path}")
