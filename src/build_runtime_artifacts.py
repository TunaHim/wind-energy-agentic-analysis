"""Create compact runtime artifacts for the Streamlit deployment.

Full EERIE-derived Parquet files stay in largeData. The public app consumes
only the small outputs written under data/runtime.
"""

from pathlib import Path
import json
import os

import numpy as np
import pandas as pd

from .january_analysis import add_wind_metrics, summarize_window
from .north_sea_mask import apply_mask

SITES = {
    "Dogger Bank": (55.0, 2.5),
    "Hornsea": (54.0, 1.5),
    "East Anglia": (52.4, 2.0),
    "Moray Firth": (58.0, -2.5),
    "Borssele": (51.8, 3.5),
    "German Bight": (55.0, 5.0),
    "Kriegers Flak": (54.8, 13.0),
}
PERIODS = {
    "January": ("January 2020-2024", "January 2036-2040"),
    "July": ("July 2020-2024", "July 2036-2040"),
}


def load_full(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    frame["time"] = pd.to_datetime(frame["time"])
    frame = add_wind_metrics(frame)
    # Apply a preliminary North Sea polygon mask instead of relying on SST as an
    # ocean proxy. The mask is a conservative hand-drawn polygon; a hydrographic
    # North Sea polygon should replace it for bankable work.
    frame = apply_mask(frame)
    return frame


def site_profiles(frame: pd.DataFrame, period: str) -> pd.DataFrame:
    rows = []
    points = frame[["lat", "lon"]].drop_duplicates()
    for name, (lat, lon) in SITES.items():
        nearest = points.iloc[((points["lat"] - lat) ** 2 + (points["lon"] - lon) ** 2).argmin()]
        site = frame[(frame["lat"] == nearest["lat"]) & (frame["lon"] == nearest["lon"])].copy()
        site["site"] = name
        site["period"] = period
        site["requested_lat"] = lat
        site["requested_lon"] = lon
        site["grid_lat"] = nearest["lat"]
        site["grid_lon"] = nearest["lon"]
        rows.append(site[["site", "period", "time", "requested_lat", "requested_lon", "grid_lat", "grid_lon", "wind_speed_10m_ms", "wind_direction_10m_deg", "air_density_kgm3", "wind_power_density_10m_wm2", "msst"]])
    return pd.concat(rows, ignore_index=True)


def period_daily(frame: pd.DataFrame, period: str) -> pd.DataFrame:
    return frame.groupby("time", as_index=False).agg(
        period=("time", lambda _: period),
        mean_wind_speed_10m_ms=("wind_speed_10m_ms", "mean"),
        median_wind_speed_10m_ms=("wind_speed_10m_ms", "median"),
        p10_wind_speed_10m_ms=("wind_speed_10m_ms", lambda x: np.percentile(x, 10)),
        p90_wind_speed_10m_ms=("wind_speed_10m_ms", lambda x: np.percentile(x, 90)),
        mean_wpd_wm2=("wind_power_density_10m_wm2", "mean"),
        mean_sst_k=("msst", "mean"),
    )


def build_month(month_name: str, source_root: Path, output_root: Path) -> dict:
    period_labels = PERIODS[month_name]
    frames = {}
    site_frames = []
    daily_frames = []
    summaries = {}
    for label in period_labels:
        filename = label.replace(" ", "_").replace("–", "-") + ".parquet"
        frame = load_full(source_root / filename)
        frames[label] = frame
        site_frames.append(site_profiles(frame, label))
        daily_frames.append(period_daily(frame, label))
        summaries[label] = summarize_window(frame, label)

    early_label, late_label = period_labels
    early = frames[early_label].groupby(["lat", "lon"], as_index=False).agg(
        early_wind=("wind_speed_10m_ms", "mean"),
        early_wpd=("wind_power_density_10m_wm2", "mean"),
        early_sst=("msst", "mean"),
    )
    late = frames[late_label].groupby(["lat", "lon"], as_index=False).agg(
        late_wind=("wind_speed_10m_ms", "mean"),
        late_wpd=("wind_power_density_10m_wm2", "mean"),
        late_sst=("msst", "mean"),
    )
    spatial = early.merge(late, on=["lat", "lon"])
    spatial["wind_change"] = spatial["late_wind"] - spatial["early_wind"]
    spatial["wpd_change"] = spatial["late_wpd"] - spatial["early_wpd"]
    spatial["sst_change"] = spatial["late_sst"] - spatial["early_sst"]

    snapshot_date = pd.Timestamp(f"2021-{1 if month_name == 'January' else 7:02d}-01")
    snapshot = frames[early_label].loc[frames[early_label]["time"] == snapshot_date, ["time", "lat", "lon", "m10u", "m10v", "mean2t", "msp", "msst", "wind_speed_10m_ms", "wind_direction_10m_deg", "wind_power_density_10m_wm2"]]
    month_output = output_root / month_name.lower()
    month_output.mkdir(parents=True, exist_ok=True)
    spatial.to_parquet(month_output / "spatial_period_means.parquet", index=False)
    pd.concat(site_frames, ignore_index=True).to_parquet(month_output / "site_daily_profiles.parquet", index=False)
    pd.concat(daily_frames, ignore_index=True).to_parquet(month_output / "daily_period_summaries.parquet", index=False)
    snapshot.to_parquet(month_output / "variable_snapshot.parquet", index=False)
    with open(month_output / "period_summaries.json", "w", encoding="utf-8") as stream:
        json.dump(summaries, stream, indent=2, default=str)
    manifest = {
        "month": month_name,
        "periods": list(period_labels),
        "source": "EERIE IFS-FESOM2-SR highres-future-ssp245 via DKRZ km-scale cloud",
        "region_bbox": {"longitude": [-5, 13], "latitude": [50, 62]},
        "runtime_mask": "Preliminary North Sea polygon mask (hand-drawn); replaces the finite-SST ocean proxy. A hydrographic North Sea polygon should be used for bankable work.",
        "files": ["spatial_period_means.parquet", "site_daily_profiles.parquet", "daily_period_summaries.parquet", "variable_snapshot.parquet", "period_summaries.json"],
    }
    with open(month_output / "manifest.json", "w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2)
    return manifest


def main():
    repo_large = Path(__file__).resolve().parents[1] / "largeData" / "eerie_north_sea_expanded_corrected"
    home_large = Path.home() / "largeData" / "eerie_north_sea_expanded_corrected"
    env_root = os.getenv("EERIE_CACHE_ROOT")
    if env_root:
        source_root = Path(env_root)
    elif repo_large.exists():
        source_root = repo_large
    else:
        source_root = home_large
    output_root = Path(__file__).resolve().parents[1] / "data" / "runtime"
    output_root.mkdir(parents=True, exist_ok=True)
    result = {
        "January": build_month("January", source_root / "January", output_root),
        "July": build_month("July", source_root / "July", output_root),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
