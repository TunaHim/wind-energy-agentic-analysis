"""Planning and analysis helpers for the EERIE January prototype.

The prototype compares January 2020-2024 with January 2036-2040 from the
IFS-FESOM SSP2-4.5 simulation. Native-grid retrieval is deliberately explicit:
source chunks span the global reduced Gaussian grid, so North Sea masking does
not imply North-Sea-sized network transfers.
"""

from dataclasses import dataclass
from datetime import date
from typing import Iterable, Mapping

import numpy as np
import pandas as pd
import requests
from numcodecs import Blosc

from .wind_energy import (
    compute_air_density,
    compute_capacity_factor,
    compute_wind_direction,
    compute_wind_power_density,
    compute_wind_speed,
    extrapolate_wind_speed_power_law,
    fit_weibull,
)


DATASET_ROOT = "https://km-scale-cloud.dkrz.de/datasets"
MODEL = "ifs-fesom2-sr"
VERSION = "v20240304"
EXPERIMENT = "highres-future-ssp245"
NATIVE_DAILY_SUFFIX = "atmos.native.2D_daily_avg"
NATIVE_POINTS = 6_599_680
VARIABLE_CHUNK_DAYS = 2
BYTES_PER_VALUE = 4


@dataclass(frozen=True)
class JanuaryWindow:
    """A named January comparison window."""

    label: str
    first_year: int
    last_year: int

    @property
    def years(self) -> tuple[int, ...]:
        return tuple(range(self.first_year, self.last_year + 1))

    @property
    def dates(self) -> pd.DatetimeIndex:
        dates = []
        for year in self.years:
            dates.extend(pd.date_range(f"{year}-01-01", f"{year}-01-31", freq="D"))
        return pd.DatetimeIndex(dates)


EARLY_WINDOW = JanuaryWindow("January 2020-2024", 2020, 2024)
LATE_WINDOW = JanuaryWindow("January 2036-2040", 2036, 2040)


def native_daily_dataset_id(experiment: str = EXPERIMENT) -> str:
    return f"{MODEL}.{experiment}.{VERSION}.{NATIVE_DAILY_SUFFIX}"


def native_daily_zarr_url(experiment: str = EXPERIMENT) -> str:
    return f"{DATASET_ROOT}/{native_daily_dataset_id(experiment)}/zarr"


def estimate_native_transfer_gb(
    windows: Iterable[JanuaryWindow] = (EARLY_WINDOW, LATE_WINDOW),
    variables: int = 4,
    compressed_factor: float = 1.0,
) -> float:
    """Estimate source-chunk transfer volume in decimal GB.

    This is an upper-bound-style uncompressed estimate. ``compressed_factor``
    can be set below one when measured compressed chunk sizes are available.
    """
    total_days = sum(len(window.dates) for window in windows)
    chunks = int(np.ceil(total_days / VARIABLE_CHUNK_DAYS))
    bytes_total = chunks * VARIABLE_CHUNK_DAYS * NATIVE_POINTS * BYTES_PER_VALUE * variables
    return float(bytes_total * compressed_factor / 1_000_000_000)


def period_label(year: int) -> str:
    return f"January {year}"


def add_wind_metrics(
    frame: pd.DataFrame,
    u_column: str = "m10u",
    v_column: str = "m10v",
    temperature_column: str = "mean2t",
    pressure_column: str = "msp",
    target_height_m: float | None = None,
    shear_exponent: float = 0.12,
) -> pd.DataFrame:
    """Add transparent 10 m and optional extrapolated wind metrics."""
    result = frame.copy()
    result["wind_speed_10m_ms"] = compute_wind_speed(result[u_column], result[v_column])
    result["wind_direction_10m_deg"] = compute_wind_direction(result[u_column], result[v_column])
    result["air_density_kgm3"] = compute_air_density(
        result[temperature_column], result[pressure_column]
    )
    result["wind_power_density_10m_wm2"] = compute_wind_power_density(
        result["wind_speed_10m_ms"], result["air_density_kgm3"]
    )
    if target_height_m is not None:
        result["wind_speed_target_ms"] = extrapolate_wind_speed_power_law(
            result["wind_speed_10m_ms"], target_height_m, shear_exponent
        )
    return result


def summarize_window(
    frame: pd.DataFrame,
    period: str,
    wind_column: str = "wind_speed_10m_ms",
    power_density_column: str = "wind_power_density_10m_wm2",
) -> dict:
    """Summarize one January window without implying annual bankability."""
    wind = frame[wind_column].to_numpy(dtype=float)
    k, scale, mean_speed = fit_weibull(wind)
    return {
        "period": period,
        "days": int(frame["time"].dt.normalize().nunique()) if "time" in frame else None,
        "mean_wind_speed_ms": float(np.nanmean(wind)),
        "median_wind_speed_ms": float(np.nanmedian(wind)),
        "p10_wind_speed_ms": float(np.nanpercentile(wind, 10)),
        "p90_wind_speed_ms": float(np.nanpercentile(wind, 90)),
        "mean_wpd_wm2": float(np.nanmean(frame[power_density_column])),
        "weibull_shape_k": k,
        "weibull_scale_a_ms": scale,
        "note": "January daily screening result; not an annual energy estimate",
    }


def compare_window_summaries(early: Mapping, late: Mapping) -> dict:
    """Compare two summary dictionaries and identify metric changes."""
    metrics = [
        "mean_wind_speed_ms",
        "median_wind_speed_ms",
        "mean_wpd_wm2",
        "weibull_shape_k",
        "weibull_scale_a_ms",
    ]
    changes = {}
    for metric in metrics:
        first = early.get(metric)
        second = late.get(metric)
        if first is None or second is None:
            continue
        absolute = float(second - first)
        relative = float(absolute / first * 100) if first else np.nan
        changes[metric] = {
            "early": first,
            "late": second,
            "absolute_change": absolute,
            "relative_change_percent": relative,
        }
    return {
        "early_period": early.get("period"),
        "late_period": late.get("period"),
        "changes": changes,
        "interpretation": (
            "Exploratory comparison of two five-January windows from one SSP2-4.5 "
            "simulation; not a definitive climate trend or bankable energy forecast."
        ),
    }


def scientific_guardrails(data_frequency: str = "daily", wind_height_m: float = 10.0) -> list[str]:
    """Return caveats the agent should include for a requested analysis."""
    caveats = [
        "EERIE is one climate-model realization and does not represent the full model ensemble uncertainty.",
        "The January windows contain five years each, so results are exploratory rather than robust trend estimates.",
        "The model grid is approximately 9 km and does not resolve turbine-scale flow, wakes, or local marine conditions.",
    ]
    if data_frequency == "monthly":
        caveats.append("Monthly means cannot reproduce wind-speed distributions or turbine power-curve integration reliably.")
    if data_frequency == "daily":
        caveats.append("Daily means lose intraday variability and are not sufficient for bankable capacity-factor estimates.")
    if wind_height_m == 10:
        caveats.append("Wind is directly available at 10 m; hub-height values require an explicit vertical-profile assumption or model-level workflow.")
    return caveats


def agent_analysis_contract() -> dict:
    """Describe the bounded analysis the agent is allowed to perform."""
    return {
        "question": "How does January North Sea wind resource differ between two future windows?",
        "windows": [
            {"label": EARLY_WINDOW.label, "years": EARLY_WINDOW.years},
            {"label": LATE_WINDOW.label, "years": LATE_WINDOW.years},
        ],
        "region": {"longitude": [-5, 13], "latitude": [50, 62]},
        "region_mask": "Expanded bounding box; a finite-SST ocean proxy can exclude many land points, but a hydrographic North Sea polygon is still required for final marine-only analysis.",
        "data_source": "EERIE IFS-FESOM2-SR highres-future-ssp245 via the DKRZ km-scale cloud Zarr endpoint",
        "prepared_variables": ["m10u", "m10v", "mean2t", "msp", "msst"],
        "direct_wind_height_m": 10,
        "allowed_outputs": [
            "daily wind-speed distributions",
            "mean and percentile wind speed",
            "wind power density",
            "January spatial differences",
            "10 m to 100 m sensitivity under explicit shear assumptions",
        ],
        "disallowed_claims": [
            "bankable annual energy production",
            "definitive climate trend from five years per window",
            "direct 100 m wind without a documented method",
        ],
    }


def _get_chunk(base_url: str, key: str, timeout_s: int = 600) -> bytes:
    response = requests.get(f"{base_url}/{key}", timeout=timeout_s)
    response.raise_for_status()
    return response.content


def _decode_chunk(base_url: str, key: str, dtype: str, shape: tuple[int, ...]) -> np.ndarray:
    decoded = Blosc().decode(_get_chunk(base_url, key))
    return np.frombuffer(decoded, dtype=np.dtype(dtype)).reshape(shape)


def load_native_region(base_url: str = None, lon_bounds=(0.0, 10.0), lat_bounds=(50.0, 62.0)) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load native coordinates and return only indices inside a region."""
    base_url = base_url or native_daily_zarr_url()
    metadata = requests.get(f"{base_url}/.zmetadata", timeout=180).json()["metadata"]
    npoints = int(metadata["lat/.zarray"]["shape"][0])
    lat = _decode_chunk(base_url, "lat/0", "<f8", (npoints,))
    lon_raw = _decode_chunk(base_url, "lon/0", "<f8", (npoints,))
    # EERIE stores native longitudes on 0–360°. Normalize to −180–180° so
    # western North Sea bounds such as −5°E work as expected.
    lon = ((lon_raw + 180.0) % 360.0) - 180.0
    mask = (lon >= lon_bounds[0]) & (lon <= lon_bounds[1]) & (lat >= lat_bounds[0]) & (lat <= lat_bounds[1])
    return lat, lon, np.flatnonzero(mask)


def fetch_native_daily_chunk(
    chunk_index: int,
    region_indices: np.ndarray,
    base_url: str = None,
    variables=("m10u", "m10v", "mean2t", "msp"),
) -> dict[str, np.ndarray]:
    """Fetch one two-day native chunk and retain only selected region points.

    This intentionally exposes the global-chunk cost instead of pretending the
    request transfers only the North Sea bytes.
    """
    base_url = base_url or native_daily_zarr_url()
    shape = (VARIABLE_CHUNK_DAYS, NATIVE_POINTS)
    return {
        variable: _decode_chunk(base_url, f"{variable}/{chunk_index}.0", "<f4", shape)[:, region_indices]
        for variable in variables
    }
