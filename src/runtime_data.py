"""Load compact runtime artifacts for the deployed Streamlit app."""

from pathlib import Path
import json
import pandas as pd

from .north_sea_mask import apply_mask


def runtime_root(project_root: Path | None = None) -> Path:
    return (project_root or Path(__file__).resolve().parents[1]) / "data" / "runtime"


def has_runtime_package(root: Path | None = None, month: str = "january") -> bool:
    path = (root or runtime_root()) / month
    required = [
        "spatial_period_means.parquet",
        "site_daily_profiles.parquet",
        "daily_period_summaries.parquet",
        "period_summaries.json",
    ]
    return all((path / name).exists() for name in required)


def load_runtime_package(root: Path | None = None, month: str = "january") -> dict:
    path = (root or runtime_root()) / month
    with open(path / "period_summaries.json", encoding="utf-8") as stream:
        summaries = json.load(stream)
    spatial = apply_mask(pd.read_parquet(path / "spatial_period_means.parquet"))
    return {
        "root": path,
        "month": month,
        "spatial": spatial,
        "sites": pd.read_parquet(path / "site_daily_profiles.parquet"),
        "daily": pd.read_parquet(path / "daily_period_summaries.parquet"),
        "summaries": summaries,
    }
