"""Build a resumable local North Sea January cache from EERIE.

The source chunks are global native-grid chunks. Only the decoded North Sea
points are persisted under largeData; raw global chunks are never written.
"""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
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
    native_daily_zarr_url,
    load_native_region,
)


VARIABLES = {
    "m10u": "<f4",
    "m10v": "<f4",
    "mean2t": "<f4",
    "msp": "<f4",
}
DATASET_START = pd.Timestamp("2015-01-01")
DEFAULT_CACHE_ROOT = Path.home() / "largeData" / "eerie_north_sea_january"


def _download_decode(base_url: str, variable: str, chunk_index: int, timeout: int = 900) -> np.ndarray:
    url = f"{base_url}/{variable}/{chunk_index}.0"
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    decoded = Blosc().decode(response.content)
    return np.frombuffer(decoded, dtype=np.dtype(VARIABLES[variable])).reshape(
        VARIABLE_CHUNK_DAYS, NATIVE_POINTS
    )


def chunk_indices(window) -> list[int]:
    indices = set()
    for timestamp in window.dates:
        day = (timestamp - DATASET_START).days
        indices.add(day // VARIABLE_CHUNK_DAYS)
    return sorted(indices)


def extract_period(window, lat, lon, region_indices, cache_root: Path, base_url: str) -> pd.DataFrame:
    period_root = cache_root / window.label.replace(" ", "_").replace("–", "-")
    period_root.mkdir(parents=True, exist_ok=True)
    outputs = []
    wanted_dates = set(window.dates.date)

    for chunk_index in chunk_indices(window):
        output_path = period_root / f"chunk_{chunk_index:05d}.parquet"
        if output_path.exists():
            outputs.append(pd.read_parquet(output_path))
            print(f"Cached {output_path.name}")
            continue

        started = time.time()
        with ThreadPoolExecutor(max_workers=len(VARIABLES)) as executor:
            futures = {
                variable: executor.submit(_download_decode, base_url, variable, chunk_index)
                for variable in VARIABLES
            }
            arrays = {variable: future.result()[:, region_indices] for variable, future in futures.items()}

        chunk_start = DATASET_START + pd.Timedelta(days=chunk_index * VARIABLE_CHUNK_DAYS)
        times = pd.date_range(chunk_start, periods=VARIABLE_CHUNK_DAYS, freq="D")
        rows = []
        for row, timestamp in enumerate(times):
            if timestamp.date() not in wanted_dates:
                continue
            rows.append(pd.DataFrame({
                "time": timestamp,
                "lat": lat[region_indices],
                "lon": lon[region_indices],
                "m10u": arrays["m10u"][row],
                "m10v": arrays["m10v"][row],
                "mean2t": arrays["mean2t"][row],
                "msp": arrays["msp"][row],
            }))
        frame = pd.concat(rows, ignore_index=True)
        frame.to_parquet(output_path, index=False)
        outputs.append(frame)
        print(f"Downloaded {output_path.name} in {time.time() - started:.1f}s")

    return pd.concat(outputs, ignore_index=True).sort_values(["time", "lat", "lon"])


def build_cache(cache_root: Path = DEFAULT_CACHE_ROOT) -> dict:
    cache_root.mkdir(parents=True, exist_ok=True)
    base_url = native_daily_zarr_url()
    lat, lon, region_indices = load_native_region(base_url)
    print(f"North Sea points: {len(region_indices)}")

    frames = {
        window.label: extract_period(window, lat, lon, region_indices, cache_root, base_url)
        for window in (EARLY_WINDOW, LATE_WINDOW)
    }
    manifest = {
        "dataset_url": base_url,
        "region": {"longitude": [0, 10], "latitude": [50, 62]},
        "periods": {},
        "variables": list(VARIABLES),
        "wind_height_m": 10,
        "raw_global_chunks_persisted": False,
        "warning": "Daily January data supports exploratory resource analysis, not bankable annual energy estimates.",
    }
    for label, frame in frames.items():
        filename = label.replace(" ", "_").replace("–", "-") + ".parquet"
        path = cache_root / filename
        frame.to_parquet(path, index=False)
        manifest["periods"][label] = {
            "path": str(path),
            "rows": len(frame),
            "start": str(frame.time.min().date()),
            "end": str(frame.time.max().date()),
        }
    with open(cache_root / "manifest.json", "w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2)
    return manifest


if __name__ == "__main__":
    print(json.dumps(build_cache(), indent=2))
