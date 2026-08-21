"""Bounded agent tools for the EERIE North Sea January prototype."""

from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from .january_analysis import (
    EARLY_WINDOW,
    LATE_WINDOW,
    add_wind_metrics,
    agent_analysis_contract,
    compare_window_summaries,
    scientific_guardrails,
    summarize_window,
)
from .wind_energy import compute_wind_direction, height_sensitivity


REFERENCE_SITES = {
    "Dogger Bank": (55.0, 2.5),
    "Hornsea": (54.0, 1.5),
    "East Anglia": (52.4, 2.0),
    "Moray Firth": (58.0, -2.5),
    "Borssele": (51.8, 3.5),
    "German Bight": (55.0, 5.0),
    "Kriegers Flak": (54.8, 13.0),
}


class JanuaryAgentToolkit:
    """Scientific tools operating only on prepared, validated local data."""

    def __init__(self, period_frames: Mapping[str, pd.DataFrame] | None = None):
        self.period_frames = {
            label: frame if "wind_speed_10m_ms" in frame.columns else add_wind_metrics(frame)
            for label, frame in (period_frames or {}).items()
        }

    @classmethod
    def from_cache(cls, cache_root: Path | None = None):
        root = cache_root or (Path.home() / "largeData" / "eerie_north_sea_january")
        frames = {}
        for label in (EARLY_WINDOW.label, LATE_WINDOW.label):
            filename = label.replace(" ", "_").replace("–", "-") + ".parquet"
            path = root / filename
            if path.exists():
                frames[label] = add_wind_metrics(pd.read_parquet(path))
        return cls(frames)

    def analysis_contract(self) -> dict:
        contract = agent_analysis_contract()
        contract["workflow"] = [
            "check_data_availability",
            "calculate_spatial_difference",
            "identify_top_regions",
            "compare_reference_sites",
            "test_hub_height_sensitivity",
            "explain_analysis_limitations",
        ]
        return contract

    def check_data_availability(self, requested_period: str | None = None) -> dict:
        available = list(self.period_frames)
        result = {
            "available_periods": available,
            "available_variables": ["m10u", "m10v", "mean2t", "msp", "derived wind speed", "derived WPD"],
            "spatial_domain": {"longitude": [-5, 13], "latitude": [50, 62]},
            "mask_status": "Prepared expanded bounding box; finite SST is used as an ocean proxy by the app, but no hydrographic North Sea polygon has been applied yet.",
            "frequency": "daily January records",
            "wind_reference_height_m": 10,
        }
        if requested_period and requested_period not in available:
            result["status"] = "unavailable"
            result["message"] = f"The requested period '{requested_period}' is not in the prepared prototype cache."
        else:
            result["status"] = "available"
        return result

    def compare_periods(self) -> dict:
        required = [EARLY_WINDOW.label, LATE_WINDOW.label]
        missing = [label for label in required if label not in self.period_frames]
        if missing:
            return {"error": "Required January period data is not loaded.", "missing_periods": missing, "required_periods": required}
        early = summarize_window(self.period_frames[EARLY_WINDOW.label], EARLY_WINDOW.label)
        late = summarize_window(self.period_frames[LATE_WINDOW.label], LATE_WINDOW.label)
        return compare_window_summaries(early, late)

    def period_summary(self, period: str) -> dict:
        if period not in self.period_frames:
            return {"error": f"Period '{period}' is not loaded.", "available_periods": list(self.period_frames)}
        return summarize_window(self.period_frames[period], period)

    def calculate_spatial_difference(self, metric: str = "wind_power_density_10m_wm2", top_n: int = 10) -> dict:
        if EARLY_WINDOW.label not in self.period_frames or LATE_WINDOW.label not in self.period_frames:
            return {"error": "Both prepared periods are required for a spatial difference."}
        early = self.period_frames[EARLY_WINDOW.label].groupby(["lat", "lon"], as_index=False)[metric].mean().rename(columns={metric: "early"})
        late = self.period_frames[LATE_WINDOW.label].groupby(["lat", "lon"], as_index=False)[metric].mean().rename(columns={metric: "late"})
        result = early.merge(late, on=["lat", "lon"])
        result["change"] = result["late"] - result["early"]
        result["relative_change_percent"] = np.where(result["early"] != 0, result["change"] / result["early"] * 100, np.nan)
        top = result.reindex(result["change"].abs().sort_values(ascending=False).index).head(top_n)
        return {"metric": metric, "top_regions": top.round(4).to_dict(orient="records"), "point_count": len(result)}

    def identify_top_regions(self, metric: str = "wind_power_density_10m_wm2", top_n: int = 10) -> dict:
        if LATE_WINDOW.label not in self.period_frames:
            return {"error": "The later period is not loaded."}
        frame = self.period_frames[LATE_WINDOW.label]
        top = frame.groupby(["lat", "lon"], as_index=False)[metric].mean().nlargest(top_n, metric)
        return {"period": LATE_WINDOW.label, "metric": metric, "top_regions": top.round(4).to_dict(orient="records")}

    def site_summary(self, site_name: str, period: str = EARLY_WINDOW.label) -> dict:
        if site_name not in REFERENCE_SITES:
            return {"error": f"Unknown site '{site_name}'.", "available_sites": list(REFERENCE_SITES)}
        if period not in self.period_frames:
            return {"error": f"Period '{period}' is not loaded.", "available_periods": list(self.period_frames)}
        target_lat, target_lon = REFERENCE_SITES[site_name]
        frame = self.period_frames[period]
        points = frame[["lat", "lon"]].drop_duplicates()
        nearest = points.iloc[((points["lat"] - target_lat) ** 2 + (points["lon"] - target_lon) ** 2).argmin()]
        site = frame[(frame["lat"] == nearest["lat"]) & (frame["lon"] == nearest["lon"])]
        wind = site["wind_speed_10m_ms"].to_numpy()
        return {
            "site": site_name,
            "period": period,
            "requested_coordinates": {"lat": target_lat, "lon": target_lon},
            "nearest_grid_coordinates": {"lat": float(nearest["lat"]), "lon": float(nearest["lon"])},
            "mean_wind_speed_10m_ms": float(np.mean(wind)),
            "mean_wpd_wm2": float(site["wind_power_density_10m_wm2"].mean()),
            "p90_wind_speed_10m_ms": float(np.percentile(wind, 90)),
            "daily_records": int(site["time"].nunique()),
        }

    def compare_reference_sites(self, period: str = LATE_WINDOW.label, metric: str = "mean_wpd_wm2") -> dict:
        rows = []
        for site_name in REFERENCE_SITES:
            summary = self.site_summary(site_name, period)
            if "error" not in summary:
                summary["ranking_metric"] = summary[metric]
                rows.append(summary)
        rows.sort(key=lambda row: row["ranking_metric"], reverse=True)
        return {"period": period, "metric": metric, "ranking": rows}

    def wind_distribution(self, period: str, site_name: str | None = None, bins: int = 12) -> dict:
        if period not in self.period_frames:
            return {"error": f"Period '{period}' is not loaded.", "available_periods": list(self.period_frames)}
        if site_name:
            summary = self.site_summary(site_name, period)
            if "error" in summary:
                return summary
            lat = summary["nearest_grid_coordinates"]["lat"]
            lon = summary["nearest_grid_coordinates"]["lon"]
            values = self.period_frames[period].query("lat == @lat and lon == @lon")["wind_speed_10m_ms"].to_numpy()
        else:
            values = self.period_frames[period]["wind_speed_10m_ms"].to_numpy()
        counts, edges = np.histogram(values, bins=bins)
        return {"period": period, "site": site_name or "North Sea grid", "bins_ms": edges.round(3).tolist(), "counts": counts.tolist(), "sample_size": int(len(values))}

    def height_sensitivity(self, period: str, heights=(100.0,), shear_exponents=(0.08, 0.12, 0.16)) -> dict:
        if period not in self.period_frames:
            return {"error": f"Period '{period}' is not loaded.", "available_periods": list(self.period_frames)}
        frame = self.period_frames[period]
        table = height_sensitivity(frame["wind_speed_10m_ms"].to_numpy(), heights, shear_exponents)
        return {"period": period, "reference_height_m": 10, "results": table.to_dict(orient="records"), "warning": "Sensitivity estimates, not model-level 100 m winds or bankable capacity factors."}

    def recommend_region(self) -> dict:
        """Run the complete bounded multi-step recommendation workflow."""
        trace = []
        trace.append({"step": 1, "tool": "check_data_availability", "result": self.check_data_availability()})
        spatial = self.calculate_spatial_difference(top_n=5)
        trace.append({"step": 2, "tool": "calculate_spatial_difference", "result": spatial})
        regions = self.identify_top_regions(top_n=5)
        trace.append({"step": 3, "tool": "identify_top_regions", "result": regions})
        sites = self.compare_reference_sites()
        trace.append({"step": 4, "tool": "compare_reference_sites", "result": sites})
        sensitivity = self.height_sensitivity(EARLY_WINDOW.label)
        trace.append({"step": 5, "tool": "test_hub_height_sensitivity", "result": sensitivity})
        limitations = self.limitations()
        trace.append({"step": 6, "tool": "explain_analysis_limitations", "result": limitations})
        recommendation = sites.get("ranking", [])[:3]
        return {"recommendation": recommendation, "trace": trace, "limitations": limitations["caveats"]}

    def limitations(self, data_frequency: str = "daily") -> dict:
        return {"caveats": scientific_guardrails(data_frequency=data_frequency, wind_height_m=10), "recommended_next_step": "Use model-level or validated hub-height wind and a longer analysis window before engineering decisions."}

    def tool_definitions(self) -> list[dict]:
        enum_periods = [EARLY_WINDOW.label, LATE_WINDOW.label]
        return [
            self._tool("check_data_availability", "Check whether a requested period and variables are available.", {"requested_period": {"type": "string"}}),
            self._tool("compare_january_periods", "Compare January 2020-2024 and January 2036-2040 wind-resource summaries.", {}),
            self._tool("calculate_spatial_difference", "Find the strongest spatial changes between the two January periods.", {"metric": {"type": "string", "enum": ["wind_power_density_10m_wm2", "wind_speed_10m_ms"]}}),
            self._tool("identify_top_regions", "Identify the highest-resource North Sea grid regions in the later period.", {"metric": {"type": "string", "enum": ["wind_power_density_10m_wm2", "wind_speed_10m_ms"]}}),
            self._tool("compare_reference_sites", "Rank recognizable North Sea reference sites by a wind-resource metric.", {"period": {"type": "string", "enum": enum_periods}, "metric": {"type": "string", "enum": ["mean_wpd_wm2", "mean_wind_speed_10m_ms"]}}),
            self._tool("get_site_summary", "Summarize one named reference site for a loaded period.", {"site_name": {"type": "string", "enum": list(REFERENCE_SITES)}, "period": {"type": "string", "enum": enum_periods}}),
            self._tool("get_wind_distribution", "Return histogram counts for North Sea or site-level wind speed.", {"period": {"type": "string", "enum": enum_periods}, "site_name": {"type": "string", "enum": list(REFERENCE_SITES)}}),
            self._tool("test_hub_height_sensitivity", "Estimate sensitivity to 10 m to 100 m power-law assumptions.", {"period": {"type": "string", "enum": enum_periods}, "shear_exponents": {"type": "array", "items": {"type": "number"}}}),
            self._tool("recommend_region", "Run a complete multi-step North Sea screening workflow and return its trace.", {}),
            self._tool("explain_analysis_limitations", "Explain caveats for daily January climate-model wind analysis.", {"data_frequency": {"type": "string", "enum": ["daily", "monthly"]}}),
        ]

    @staticmethod
    def _tool(name: str, description: str, properties: dict) -> dict:
        return {"type": "function", "function": {"name": name, "description": description, "parameters": {"type": "object", "properties": properties}}}
