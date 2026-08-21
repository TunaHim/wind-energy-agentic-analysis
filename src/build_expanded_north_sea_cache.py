"""Build resumable expanded North Sea January and July caches from EERIE.

The original 0-10E cache is preserved. This builder writes only regional
Parquet extracts under largeData/eerie_north_sea_expanded and never persists
the global source chunks.
"""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import calendar
import json
import time

import numpy as np
import pandas as pd
import requests
from numcodecs import Blosc

from .january_analysis import (
    EARLY_WINDOW,
    LATE_WINDOW,
    NATIVE_POINTS,
    VARIABLE_CHUNK_DAYS,
    load_native_region,
    native_daily_zarr_url,
)

VARIABLES = {
    "m10u": "<f4",
    "m10v": "<f4",
    "mean2t": "<f4",
    "msp": "<f4",
    "msst": "<f4",
}
DATASET_START = pd.Timestamp("2015-01-01")
LON_BOUNDS = (-5.0, 13.0)
LAT_BOUNDS = (50.0, 62.0)
DEFAULT_ROOT = Path.home() / "largeData" / "eerie_north_sea_expanded_corrected"


class MonthWindow:
    def __init__(self, label: str, first_year: int, last_year: int, month: int):
        self.label = label
        self.first_year = first_year
        self.last_year = last_year
        self.month = month

    @property
    def years(self):
        return tuple(range(self.first_year, self.last_year + 1))

    @property
    def dates(self):
        values = []
        for year in self.years:
            days = calendar.monthrange(year, self.month)[1]
            values.extend(pd.date_range(f"{year}-{self.month:02d}-01", periods=days, freq="D"))
        return pd.DatetimeIndex(values)


WINDOWS = {
    "January": (
        MonthWindow("January 2020-2024", 2020, 2024, 1),
        MonthWindow("January 2036-2040", 2036, 2040, 1),
    ),
    "July": (
        MonthWindow("July 2020-2024", 2020, 2024, 7),
        MonthWindow("July 2036-2040", 2036, 2040, 7),
    ),
}


def _download_decode(base_url: str, variable: str, chunk_index: int, retries: int = 6) -> np.ndarray:
    url = f"{base_url}/{variable}/{chunk_index}.0"
    last_error = None
    for attempt in range(retries):
        try:
            response = requests.get(url, timeout=900)
            response.raise_for_status()
            decoded = Blosc().decode(response.content)
            return np.frombuffer(decoded, dtype=np.dtype(VARIABLES[variable])).reshape(VARIABLE_CHUNK_DAYS, NATIVE_POINTS)
        except (requests.RequestException, RuntimeError, ValueError) as exc:
            last_error = exc
            if attempt + 1 < retries:
                time.sleep(2 ** min(attempt, 5))
    raise RuntimeError(f"Failed to retrieve {variable} chunk {chunk_index} after {retries} attempts: {last_error}")


def chunk_indices(window: MonthWindow) -> list[int]:
    return sorted({(timestamp - DATASET_START).days // VARIABLE_CHUNK_DAYS for timestamp in window.dates})


def extract_window(window, lat, lon, region_indices, month_root: Path, base_url: str) -> pd.DataFrame:
    period_root = month_root / window.label.replace(" ", "_").replace("–", "-")
    period_root.mkdir(parents=True, exist_ok=True)
    wanted_dates = set(window.dates.date)
    outputs = []

    for chunk_index in chunk_indices(window):
        output_path = period_root / f"chunk_{chunk_index:05d}.parquet"
        if output_path.exists():
            outputs.append(pd.read_parquet(output_path))
            print(f"Cached {output_path.name}")
            continue

        started = time.time()
        with ThreadPoolExecutor(max_workers=len(VARIABLES)) as executor:
            futures = {variable: executor.submit(_download_decode, base_url, variable, chunk_index) for variable in VARIABLES}
            arrays = {variable: future.result()[:, region_indices] for variable, future in futures.items()}

        chunk_start = DATASET_START + pd.Timedelta(days=chunk_index * VARIABLE_CHUNK_DAYS)
        timestamps = pd.date_range(chunk_start, periods=VARIABLE_CHUNK_DAYS, freq="D")
        rows = []
        for row, timestamp in enumerate(timestamps):
            if timestamp.date() in wanted_dates:
                rows.append(pd.DataFrame({
                    "time": timestamp,
                    "lat": lat[region_indices],
                    "lon": lon[region_indices],
                    "m10u": arrays["m10u"][row],
                    "m10v": arrays["m10v"][row],
                    "mean2t": arrays["mean2t"][row],
                    "msp": arrays["msp"][row],
                    "msst": arrays["msst"][row],
                }))
        frame = pd.concat(rows, ignore_index=True)
        frame.to_parquet(output_path, index=False)
        outputs.append(frame)
        print(f"Downloaded {output_path.name} in {time.time() - started:.1f}s")

    return pd.concat(outputs, ignore_index=True).sort_values(["time", "lat", "lon"])


def build_month(month_name: str, root: Path, base_url: str) -> dict:
    month_root = root / month_name
    month_root.mkdir(parents=True, exist_ok=True)
    lat, lon, region_indices = load_native_region(base_url, LON_BOUNDS, LAT_BOUNDS)
    print(f"{month_name}: expanded-box points={len(region_indices)}")
    frames = {}
    for window in WINDOWS[month_name]:
        frames[window.label] = extract_window(window, lat, lon, region_indices, month_root, base_url)

    manifest = {
        "dataset_url": base_url,
        "month": month_name,
        "region_bbox": {"longitude": list(LON_BOUNDS), "latitude": list(LAT_BOUNDS)},
        "longitude_convention": "normalized to -180..180 before masking",
        "mask_status": "expanded bounding box; land/sea and hydrographic North Sea masks not yet applied",
        "variables": list(VARIABLES),
        "wind_reference_height_m": 10,
        "raw_global_chunks_persisted": False,
        "periods": {},
    }
    for label, frame in frames.items():
        filename = label.replace(" ", "_").replace("–", "-") + ".parquet"
        path = month_root / filename
        frame.to_parquet(path, index=False)
        manifest["periods"][label] = {
            "path": str(path),
            "rows": len(frame),
            "start": str(frame.time.min().date()),
            "end": str(frame.time.max().date()),
            "points_per_timestep": int(frame.groupby("time").size().iloc[0]),
        }
    with open(month_root / "manifest.json", "w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2)
    return manifest


def build_all(root: Path = DEFAULT_ROOT) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    base_url = native_daily_zarr_url()
    return {month: build_month(month, root, base_url) for month in WINDOWS}


if __name__ == "__main__":
    print(json.dumps(build_all(), indent=2))
