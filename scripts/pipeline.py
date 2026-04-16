# baltimoreparcel/scripts/pipeline.py
"""
Monolithic pipeline orchestrator.

Chains any combination of the four pipeline stages in sequence.
Stages are defined in the config (pipeline.stages) and can be overridden
at the CLI with --stages.

Stages
------
  ingest     Read raw shapefiles → GDB feature classes   (ingest_to_gdb.py)
  assemble   Stack years + CPI + log → full panel        (assemble_panel.py)
  change     Full panel → change panel                   (build_change_panel.py)
  aggregate  Panel → aggregated layers                   (aggregate_panels.py)

Usage
-----
    # Run all stages defined in the config:
    python -m baltimoreparcel.scripts.pipeline --config configs/ez_full_pipeline.yml

    # Override which stages run:
    python -m baltimoreparcel.scripts.pipeline --config configs/ez_project.yml --stages assemble aggregate

    # Aggregate full panel only (no change):
    python -m baltimoreparcel.scripts.pipeline --config configs/ez_project.yml --stages aggregate --full
"""

import argparse
import datetime

from baltimoreparcel.run_config import (
    load_config,
    to_aggregate_config,
    to_change_panel_config,
    to_ingest_config,
)
from baltimoreparcel.utils import Logger, error, info, success

import baltimoreparcel.scripts.ingest_to_gdb as ingest_to_gdb
import baltimoreparcel.scripts.assemble_panel as assemble_panel
import baltimoreparcel.scripts.build_change_panel as build_change_panel
import baltimoreparcel.scripts.aggregate_panels as aggregate_panels

# ---------------------------------------------------------------------------
# Stage registry
# ---------------------------------------------------------------------------

VALID_STAGES = ("ingest", "assemble", "change", "aggregate")


def _run_ingest(cfg, **_kwargs):
    ingest_to_gdb.run(to_ingest_config(cfg))


def _run_assemble(cfg, **_kwargs):
    assemble_panel.run(cfg)


def _run_change(cfg, **_kwargs):
    build_change_panel.run(to_change_panel_config(cfg))


def _run_aggregate(cfg, do_full=False, do_change=True, **_kwargs):
    aggregate_panels.run(to_aggregate_config(cfg), do_full=do_full, do_change=do_change)


STAGES = {
    "ingest":     _run_ingest,
    "assemble":   _run_assemble,
    "change":     _run_change,
    "aggregate":  _run_aggregate,
}


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run(cfg, stages: list[str], do_full: bool = False, do_change: bool = True) -> None:
    info(f"Pipeline stages: {stages}")
    for stage in stages:
        if stage not in STAGES:
            raise ValueError(
                f"Unknown stage '{stage}'. Valid stages: {VALID_STAGES}"
            )
        info(f"--- Stage: {stage} ---")
        STAGES[stage](cfg, do_full=do_full, do_change=do_change)
        success(f"Stage '{stage}' complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the parcel pipeline: ingest → assemble → change → aggregate."
    )
    parser.add_argument(
        "--config",
        required=True,
        metavar="PATH",
        help="Path to a YAML run-config (e.g. configs/ez_full_pipeline.yml)",
    )
    parser.add_argument(
        "--stages",
        nargs="+",
        metavar="STAGE",
        choices=VALID_STAGES,
        help=(
            "Override which stages to run. If omitted, uses pipeline.stages from config. "
            "Choices: ingest, assemble, change, aggregate"
        ),
    )
    parser.add_argument(
        "--full",
        action="store_true",
        default=False,
        help="(aggregate stage) Run full-panel aggregations",
    )
    parser.add_argument(
        "--change",
        action="store_true",
        default=False,
        help="(aggregate stage) Run change-panel aggregations",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)

    # Resolve stages: CLI flag > config > error
    stages = args.stages or cfg.pipeline.stages
    if not stages:
        parser.error(
            "No stages specified. Pass --stages or set pipeline.stages in the config."
        )

    # Aggregate flags: default to change-panel only if neither flag given
    do_full = args.full
    do_change = args.change or (not args.full and not args.change)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    logger = Logger(cfg.logs_path / f"pipeline_{timestamp}.log")

    try:
        run(cfg, stages=stages, do_full=do_full, do_change=do_change)
        print("Done.")
    except Exception as exc:
        error(str(exc))
        raise
    finally:
        logger.close()
