"""Bounded agent tools for the EERIE North Sea January prototype."""

from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from .january_analysis import (
    DATASET_ROOT,
    EARLY_WINDOW,
    LATE_WINDOW,
    EXPERIMENT,
    MODEL,
    VERSION,
    add_wind_metrics,
    agent_analysis_contract,
    compare_window_summaries,
    native_daily_dataset_id,
    scientific_guardrails,
    summarize_window,
)
from .wind_energy import (
    compute_aep,
    compute_capacity_factor,
    compute_wind_direction,
    extrapolate_wind_speed_power_law,
    fit_weibull,
    height_sensitivity,
)


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

    def __init__(self, period_frames: Mapping[str, pd.DataFrame] | None = None, runtime: dict | None = None):
        self.period_frames = {
            label: frame if "wind_speed_10m_ms" in frame.columns else add_wind_metrics(frame)
            for label, frame in (period_frames or {}).items()
        }
        self.runtime = runtime or {}
        self.spatial = self.runtime.get("spatial")
        self.spatial_by_month: dict[str, pd.DataFrame] = {}
        if self.spatial is not None:
            self.spatial_by_month["January"] = self.spatial
        self.site_profiles = self.runtime.get("sites")
        self.daily_profiles = self.runtime.get("daily")
        self.summaries = self.runtime.get("summaries", {})
        self.seasonal = self.runtime.get("seasonal", {})
        self.seasonal_summaries = {}
        self.seasonal_sites = None
        for month, package in self.seasonal.items():
            self.seasonal_summaries.update(package.get("summaries", {}))
            if package.get("spatial") is not None:
                self.spatial_by_month[month] = package["spatial"]
            site_frames = [package["sites"] for package in self.seasonal.values() if package.get("sites") is not None]
            if site_frames:
                self.seasonal_sites = pd.concat(site_frames, ignore_index=True)

    @classmethod
    def from_runtime(cls, runtime: dict):
        return cls(runtime=runtime)

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

    def provenance(self) -> dict:
        return {
            "model": f"EERIE {MODEL} {EXPERIMENT} {VERSION}",
            "data_url": f"{DATASET_ROOT}/{native_daily_dataset_id()}/zarr",
            "region": {"longitude": [-5, 13], "latitude": [50, 62]},
            "mask": "Preliminary North Sea polygon mask; not an EEZ or hydrographic boundary",
            "periods": [EARLY_WINDOW.label, LATE_WINDOW.label],
            "seasonal_windows": ["January", "July"],
            "wind_reference_height_m": 10,
            "hub_height_demo_m": 120,
            "prepared_variables": ["m10u", "m10v", "mean2t", "msp", "msst"],
        }

    def validate_screening_readiness(self, question: str = "") -> dict:
        """Return active guardrails and a go/no-go decision for the request."""
        question_lower = question.lower()
        violations = []
        disallowed_engineering = ["bankable", "investment", "finance", "capex", "opex", "lcoe", "turbine selection"]
        if any(term in question_lower for term in disallowed_engineering):
            violations.append("This prototype cannot support bankable, investment, or cost-of-energy decisions.")
        if any(term in question_lower for term in ["definitive trend", "climate trend", "long-term trend", "conclusive"]):
            violations.append("A 5-year January window from one realization is not a definitive climate trend.")
        if any(term in question_lower for term in ["100 m", "hub height", "direct hub"]) and "10 m" not in question_lower and "sensitivity" not in question_lower:
            violations.append("Direct hub-height wind is not in the data; only 10 m-to-120 m sensitivity under assumptions is available.")
        contract = self.analysis_contract()
        for claim in contract.get("disallowed_claims", []):
            if claim in question_lower:
                violations.append(f"Request contained a disallowed claim: '{claim}'.")
        if "july" in question_lower and ("compare" not in question_lower and "january" not in question_lower):
            violations.append("July is only prepared as a seasonal comparison with January; it is not a primary analysis window.")
        can_answer = len(violations) == 0
        if can_answer:
            violations.append("Request is within the prototype contract; answer will include uncertainty and caveats.")
        return {"can_answer": can_answer, "violations": violations}

    def get_data_provenance(self) -> dict:
        return self.provenance()

    def analysis_contract(self) -> dict:
        contract = agent_analysis_contract()
        contract["workflow"] = [
            "check_data_availability",
            "calculate_spatial_difference",
            "identify_top_regions",
            "compare_reference_sites",
            "test_hub_height_sensitivity",
            "compare_january_july",
            "explain_analysis_limitations",
        ]
        return contract

    def check_data_availability(self, requested_period: str | None = None) -> dict:
        available = list(self.period_frames) or list(self.summaries) + [period for period in self.seasonal_summaries if period not in self.summaries]
        result = {
            "available_periods": available,
            "available_variables": ["m10u", "m10v", "mean2t", "msp", "derived wind speed", "derived WPD", "derived 120 m wind", "capacity factor", "AEP"],
            "spatial_domain": {"longitude": [-5, 13], "latitude": [50, 62]},
            "mask_status": "Preliminary North Sea polygon mask; a hydrographic or EEZ polygon should be used for bankable work.",
            "frequency": "daily January and July records",
            "wind_reference_height_m": 10,
            "hub_height_demo_m": 120,
        }
        if requested_period and requested_period not in available:
            result["status"] = "unavailable"
            result["message"] = f"The requested period '{requested_period}' is not in the prepared prototype cache."
        else:
            result["status"] = "available"
        return result

    def compare_periods(self) -> dict:
        required = [EARLY_WINDOW.label, LATE_WINDOW.label]
        if self.summaries:
            missing = [label for label in required if label not in self.summaries]
            if missing:
                return {"error": "Required January period summary is not loaded.", "missing_periods": missing, "required_periods": required}
            return compare_window_summaries(self.summaries[EARLY_WINDOW.label], self.summaries[LATE_WINDOW.label])
        missing = [label for label in required if label not in self.period_frames]
        if missing:
            return {"error": "Required January period data is not loaded.", "missing_periods": missing, "required_periods": required}
        return compare_window_summaries(summarize_window(self.period_frames[EARLY_WINDOW.label], EARLY_WINDOW.label), summarize_window(self.period_frames[LATE_WINDOW.label], LATE_WINDOW.label))

    def period_summary(self, period: str) -> dict:
        if period in self.summaries:
            return self.summaries[period]
        if period in self.seasonal_summaries:
            return self.seasonal_summaries[period]
        if period not in self.period_frames:
            return {"error": f"Period '{period}' is not loaded.", "available_periods": list(self.period_frames) or list(self.summaries) + list(self.seasonal_summaries)}
        return summarize_window(self.period_frames[period], period)

    def _get_spatial(self, month: str) -> pd.DataFrame | None:
        if month in self.spatial_by_month:
            return self.spatial_by_month[month]
        if month == "January" and self.spatial is not None:
            return self.spatial
        return None

    def calculate_spatial_difference(self, metric: str = "wind_power_density_10m_wm2", month: str = "January", top_n: int = 10) -> dict:
        spatial = self._get_spatial(month)
        if spatial is not None:
            column_map = {"wind_power_density_10m_wm2": ("early_wpd", "late_wpd"), "wind_speed_10m_ms": ("early_wind", "late_wind")}
            early_name, late_name = column_map.get(metric, ("early_wpd", "late_wpd"))
            result = spatial[["lat", "lon", early_name, late_name]].rename(columns={early_name: "early", late_name: "late"}).copy()
        else:
            period = f"{month} 2020-2024"
            late_period = f"{month} 2036-2040"
            if period not in self.period_frames or late_period not in self.period_frames:
                return {"error": f"Both {month} periods are required for a spatial difference."}
            early = self.period_frames[period].groupby(["lat", "lon"], as_index=False)[metric].mean().rename(columns={metric: "early"})
            late = self.period_frames[late_period].groupby(["lat", "lon"], as_index=False)[metric].mean().rename(columns={metric: "late"})
            result = early.merge(late, on=["lat", "lon"])
        result["change"] = result["late"] - result["early"]
        result["relative_change_percent"] = np.where(result["early"] != 0, result["change"] / result["early"] * 100, np.nan)
        top = result.reindex(result["change"].abs().sort_values(ascending=False).index).head(top_n)
        change = result["change"].dropna()
        return {
            "metric": metric,
            "month": month,
            "top_regions": top.round(4).to_dict(orient="records"),
            "point_count": len(result),
            "change_summary": {
                "mean": round(float(change.mean()), 4),
                "std": round(float(change.std()), 4),
                "p10": round(float(np.percentile(change, 10)), 4),
                "p90": round(float(np.percentile(change, 90)), 4),
                "interpretation": "Spatial change distribution across all grid points; p10 and p90 describe the spread.",
            },
        }

    def identify_top_regions(self, metric: str = "wind_power_density_10m_wm2", month: str = "January", top_n: int = 10) -> dict:
        spatial = self._get_spatial(month)
        if spatial is not None:
            column = "late_wpd" if metric == "wind_power_density_10m_wm2" else "late_wind"
            top = spatial[["lat", "lon", column]].nlargest(top_n, column)
            return {"month": month, "period": f"{month} 2036-2040", "metric": metric, "top_regions": top.round(4).to_dict(orient="records")}
        late_period = f"{month} 2036-2040"
        if late_period not in self.period_frames:
            return {"error": f"The later {month} period is not loaded."}
        frame = self.period_frames[late_period]
        top = frame.groupby(["lat", "lon"], as_index=False)[metric].mean().nlargest(top_n, metric)
        return {"month": month, "period": late_period, "metric": metric, "top_regions": top.round(4).to_dict(orient="records")}

    def _get_profile_data(self, period: str) -> pd.DataFrame | None:
        profile_data = self.site_profiles
        if profile_data is not None and not (profile_data["period"] == period).any() and self.seasonal_sites is not None:
            profile_data = self.seasonal_sites
        return profile_data

    def site_summary(self, site_name: str, period: str = EARLY_WINDOW.label, hub_height_m: float = 120.0, shear_exponent: float = 0.12) -> dict:
        if site_name not in REFERENCE_SITES:
            return {"error": f"Unknown site '{site_name}'.", "available_sites": list(REFERENCE_SITES)}
        profile_data = self._get_profile_data(period)
        if profile_data is not None:
            site = profile_data[(profile_data["site"] == site_name) & (profile_data["period"] == period)]
            if site.empty:
                return {"error": f"Site '{site_name}' is not available for period '{period}'."}
            return self._summarise_site(site, site_name, period, hub_height_m, shear_exponent)
        if period not in self.period_frames:
            return {"error": f"Period '{period}' is not loaded.", "available_periods": list(self.period_frames)}
        target_lat, target_lon = REFERENCE_SITES[site_name]
        frame = self.period_frames[period]
        points = frame[["lat", "lon"]].drop_duplicates()
        nearest = points.iloc[((points["lat"] - target_lat) ** 2 + (points["lon"] - target_lon) ** 2).argmin()]
        site = frame[(frame["lat"] == nearest["lat"]) & (frame["lon"] == nearest["lon"])]
        return self._summarise_site(site, site_name, period, hub_height_m, shear_exponent)

    def _summarise_site(self, site: pd.DataFrame, site_name: str, period: str, hub_height_m: float, shear_exponent: float) -> dict:
        wind_10m = site["wind_speed_10m_ms"].to_numpy()
        wind_hub = extrapolate_wind_speed_power_law(wind_10m, hub_height_m, shear_exponent)
        k, a, _ = fit_weibull(wind_10m)
        return {
            "site": site_name,
            "period": period,
            "requested_coordinates": {"lat": float(site["lat"].iloc[0]) if "requested_lat" not in site.columns else float(site["requested_lat"].iloc[0]),
                                      "lon": float(site["lon"].iloc[0]) if "requested_lon" not in site.columns else float(site["requested_lon"].iloc[0])},
            "nearest_grid_coordinates": {"lat": float(site["lat"].iloc[0]) if "grid_lat" not in site.columns else float(site["grid_lat"].iloc[0]),
                                         "lon": float(site["lon"].iloc[0]) if "grid_lon" not in site.columns else float(site["grid_lon"].iloc[0])},
            "hub_height_m": hub_height_m,
            "shear_exponent": shear_exponent,
            "mean_wind_speed_10m_ms": float(np.mean(wind_10m)),
            "p10_wind_speed_10m_ms": float(np.percentile(wind_10m, 10)),
            "p90_wind_speed_10m_ms": float(np.percentile(wind_10m, 90)),
            "std_wind_speed_10m_ms": float(np.std(wind_10m)),
            "mean_wind_speed_120m_ms": float(np.mean(wind_hub)),
            "p10_wind_speed_120m_ms": float(np.percentile(wind_hub, 10)),
            "p90_wind_speed_120m_ms": float(np.percentile(wind_hub, 90)),
            "std_wind_speed_120m_ms": float(np.std(wind_hub)),
            "mean_wpd_wm2": float(site["wind_power_density_10m_wm2"].mean()),
            "p10_wpd_wm2": float(np.percentile(site["wind_power_density_10m_wm2"], 10)),
            "p90_wpd_wm2": float(np.percentile(site["wind_power_density_10m_wm2"], 90)),
            "std_wpd_wm2": float(np.std(site["wind_power_density_10m_wm2"])),
            "weibull_shape_k": float(k) if not np.isnan(k) else None,
            "weibull_scale_a_ms": float(a) if not np.isnan(a) else None,
            "capacity_factor_120m": float(compute_capacity_factor(wind_hub)),
            "annual_energy_production_mwh": float(compute_aep(wind_hub)),
            "daily_records": int(site["time"].nunique()),
            "caveat": "Capacity factor and AEP use daily-mean wind and are biased downward; not bankable.",
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
        profile_data = self._get_profile_data(period)
        if profile_data is not None:
            values = profile_data.loc[profile_data["period"] == period, "wind_speed_10m_ms"]
            if site_name:
                values = profile_data.loc[(profile_data["period"] == period) & (profile_data["site"] == site_name), "wind_speed_10m_ms"]
        elif period in self.period_frames:
            if site_name:
                summary = self.site_summary(site_name, period)
                if "error" in summary:
                    return summary
                lat = summary["nearest_grid_coordinates"]["lat"]
                lon = summary["nearest_grid_coordinates"]["lon"]
                values = self.period_frames[period].query("lat == @lat and lon == @lon")["wind_speed_10m_ms"]
            else:
                values = self.period_frames[period]["wind_speed_10m_ms"]
        else:
            return {"error": f"Period '{period}' is not loaded.", "available_periods": list(self.period_frames) or list(self.summaries)}
        values = values.to_numpy()
        counts, edges = np.histogram(values, bins=bins)
        return {"period": period, "site": site_name or "North Sea grid", "bins_ms": edges.round(3).tolist(), "counts": counts.tolist(), "sample_size": int(len(values))}

    def compare_january_july(self) -> dict:
        if not self.seasonal or "January" not in self.seasonal or "July" not in self.seasonal:
            return {"error": "Both January and July compact runtime packages are required."}
        result = {}
        for month, package in self.seasonal.items():
            summaries = package.get("summaries", {})
            early = summaries.get(f"{month} 2020-2024", {})
            late = summaries.get(f"{month} 2036-2040", {})
            result[month] = {
                "early_mean_wind_ms": early.get("mean_wind_speed_ms"),
                "late_mean_wind_ms": late.get("mean_wind_speed_ms"),
                "wind_change_ms": (late.get("mean_wind_speed_ms", 0) - early.get("mean_wind_speed_ms", 0)),
                "early_wpd_wm2": early.get("mean_wpd_wm2"),
                "late_wpd_wm2": late.get("mean_wpd_wm2"),
                "wpd_change_wm2": (late.get("mean_wpd_wm2", 0) - early.get("mean_wpd_wm2", 0)),
            }
        return {"months": result, "interpretation": "Seasonal comparison of two January and July windows from one SSP2-4.5 simulation."}

    def height_sensitivity(self, period: str, heights=(120.0,), shear_exponents=(0.08, 0.12, 0.16)) -> dict:
        profile_data = self._get_profile_data(period)
        if profile_data is not None:
            values = profile_data.loc[profile_data["period"] == period, "wind_speed_10m_ms"].to_numpy()
        elif period in self.period_frames:
            values = self.period_frames[period]["wind_speed_10m_ms"].to_numpy()
        else:
            return {"error": f"Period '{period}' is not loaded.", "available_periods": list(self.period_frames) or list(self.summaries)}
        table = height_sensitivity(values, heights, shear_exponents)
        return {"period": period, "reference_height_m": 10, "results": table.to_dict(orient="records"), "warning": "Sensitivity estimates, not model-level 120 m winds or bankable capacity factors."}

    def recommend_region(self) -> dict:
        """Run the complete bounded multi-step recommendation workflow."""
        trace = []
        trace.append({"step": 1, "tool": "check_data_availability", "result": self.check_data_availability()})
        spatial = self.calculate_spatial_difference(month="January", top_n=5)
        trace.append({"step": 2, "tool": "calculate_spatial_difference", "result": spatial})
        regions = self.identify_top_regions(month="January", top_n=5)
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
        enum_periods.extend(period for period in self.seasonal_summaries if period not in enum_periods)
        return [
            self._tool("check_data_availability", "Check whether a requested period and variables are available.", {"requested_period": {"type": "string"}}),
            self._tool("compare_january_periods", "Compare January 2020-2024 and January 2036-2040 wind-resource summaries.", {}),
            self._tool("compare_january_july", "Compare the seasonal January and July period changes in wind speed and WPD.", {}),
            self._tool("get_january_period_summary", "Return a summary for one loaded January or July period.", {"period": {"type": "string", "enum": enum_periods}}),
            self._tool("calculate_spatial_difference", "Find the strongest spatial changes between the two periods for a selected month.", {
                "metric": {"type": "string", "enum": ["wind_power_density_10m_wm2", "wind_speed_10m_ms"]},
                "month": {"type": "string", "enum": ["January", "July"]},
            }),
            self._tool("identify_top_regions", "Identify the highest-resource North Sea grid regions in the later period for a selected month.", {
                "metric": {"type": "string", "enum": ["wind_power_density_10m_wm2", "wind_speed_10m_ms"]},
                "month": {"type": "string", "enum": ["January", "July"]},
            }),
            self._tool("compare_reference_sites", "Rank recognizable North Sea reference sites by a wind-resource metric.", {
                "period": {"type": "string", "enum": enum_periods},
                "metric": {"type": "string", "enum": ["mean_wpd_wm2", "mean_wind_speed_10m_ms", "capacity_factor_120m"]},
            }),
            self._tool("get_site_summary", "Summarise one named reference site for a loaded period, including Weibull, CF and AEP.", {
                "site_name": {"type": "string", "enum": list(REFERENCE_SITES)},
                "period": {"type": "string", "enum": enum_periods},
            }),
            self._tool("get_wind_distribution", "Return histogram counts for North Sea or site-level wind speed.", {
                "period": {"type": "string", "enum": enum_periods},
                "site_name": {"type": "string", "enum": list(REFERENCE_SITES)},
            }),
            self._tool("test_hub_height_sensitivity", "Estimate sensitivity to 10 m to 120 m power-law assumptions.", {
                "period": {"type": "string", "enum": enum_periods},
                "shear_exponents": {"type": "array", "items": {"type": "number"}},
            }),
            self._tool("validate_screening_readiness", "Check whether the question is within the prototype contract before answering.", {"question": {"type": "string"}}),
            self._tool("get_data_provenance", "Return the data source, model, periods, and spatial mask for any answer.", {}),
            self._tool("recommend_region", "Run a complete multi-step North Sea screening workflow and return its trace.", {}),
            self._tool("explain_analysis_limitations", "Explain caveats for daily January climate-model wind analysis.", {"data_frequency": {"type": "string", "enum": ["daily", "monthly"]}}),
        ]

    @staticmethod
    def _tool(name: str, description: str, properties: dict) -> dict:
        return {"type": "function", "function": {"name": name, "description": description, "parameters": {"type": "object", "properties": properties}}}
