"""Streamlit prototype for agentic North Sea climate and wind analysis."""

from io import BytesIO
from pathlib import Path
import json
from typing import Any

import numpy as np
import pandas as pd
import requests
import streamlit as st
import xarray as xr

try:
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri
except ImportError:  # pragma: no cover
    plt = None
    mtri = None

from src.january_analysis import (
    EARLY_WINDOW,
    LATE_WINDOW,
    scientific_guardrails,
)
from src.january_agent_tools import JanuaryAgentToolkit
from src.llm_providers import (
    PROVIDERS,
    get_configured_key,
    run_tool_calling_agent,
)
from src.runtime_data import has_runtime_package, load_runtime_package, runtime_root
from src.wind_energy import compute_aep, compute_capacity_factor, fit_weibull


try:
    import plotly.express as px
except ImportError:  # pragma: no cover
    px = None


st.set_page_config(
    page_title="North Sea Wind Intelligence",
    page_icon="🌬️",
    layout="wide",
    initial_sidebar_state="expanded",
)

EARLY_LABEL = EARLY_WINDOW.label
LATE_LABEL = LATE_WINDOW.label

SITES = {
    "Dogger Bank": {"lat": 55.0, "lon": 2.5, "country": "United Kingdom", "reference": "North Sea offshore wind zone"},
    "Hornsea": {"lat": 54.0, "lon": 1.5, "country": "United Kingdom", "reference": "East coast of England"},
    "East Anglia": {"lat": 52.4, "lon": 2.0, "country": "United Kingdom", "reference": "East coast of England"},
    "Moray Firth": {"lat": 58.0, "lon": -2.5, "country": "United Kingdom", "reference": "Northern Scotland"},
    "Borssele": {"lat": 51.8, "lon": 3.5, "country": "Netherlands", "reference": "Dutch North Sea sector"},
    "German Bight": {"lat": 55.0, "lon": 5.0, "country": "Germany", "reference": "German North Sea sector"},
    "Kriegers Flak": {"lat": 54.8, "lon": 13.0, "country": "Denmark/Germany", "reference": "Southern Baltic reference site"},
}

COUNTRY_GEOJSON_URL = "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_50m_admin_0_countries.geojson"
BATHYMETRY_URL = "https://erddap.emodnet.eu/erddap/griddap/bathymetry_dtm_2024.nc?elevation%5B(50):100:(62)%5D%5B(-5):100:(13)%5D"
EEZ_MAP_URL = "https://upload.wikimedia.org/wikipedia/commons/7/7f/North_Sea_map-en.png"


@st.cache_data(show_spinner="Loading compact runtime artifacts...")
def load_compact_runtime(month: str = "january") -> dict:
    return load_runtime_package(runtime_root(), month)


@st.cache_data(show_spinner="Loading political boundaries...")
def load_country_geojson() -> dict:
    response = requests.get(COUNTRY_GEOJSON_URL, timeout=120)
    response.raise_for_status()
    return response.json()


@st.cache_data(show_spinner="Loading EMODnet bathymetry subset...")
def load_bathymetry_bytes() -> bytes:
    response = requests.get(BATHYMETRY_URL, timeout=180)
    response.raise_for_status()
    return response.content


@st.cache_data(show_spinner="Loading North Sea EEZ map...")
def load_eez_map_image() -> bytes:
    response = requests.get(EEZ_MAP_URL, timeout=60)
    response.raise_for_status()
    return response.content


def geometry_lines(geometry: dict):
    coordinates = geometry.get("coordinates", [])
    if geometry.get("type") == "Polygon":
        polygons = [coordinates]
    elif geometry.get("type") == "MultiPolygon":
        polygons = coordinates
    else:
        polygons = []
    for polygon in polygons:
        for ring in polygon:
            if len(ring) > 1:
                yield ring


def plot_geojson_lines(ax, geojson: dict, color: str, linewidth: float, alpha: float = 1.0):
    for feature in geojson.get("features", []):
        for ring in geometry_lines(feature.get("geometry") or {}):
            visible = [point for point in ring if -5 <= point[0] <= 13 and 50 <= point[1] <= 62]
            if len(visible) > 1:
                stride = max(1, len(visible) // 1500)
                points = visible[::stride]
                ax.plot([p[0] for p in points], [p[1] for p in points], color=color, linewidth=linewidth, alpha=alpha)


def continuous_field(ax, frame: pd.DataFrame, column: str, cmap: str, vmin: float, vmax: float, triangulation=None):
    """Render irregular native points as a continuous triangular field.

    If `triangulation` is supplied, it is reused; otherwise one is computed from the frame.
    Reusing the triangulation saves a lot of time when many subplots share the same grid.
    """
    if mtri is None:
        return ax.scatter(frame["lon"], frame["lat"], c=frame[column], s=1.0, cmap=cmap, vmin=vmin, vmax=vmax)
    if triangulation is None:
        triangulation = mtri.Triangulation(frame["lon"].to_numpy(), frame["lat"].to_numpy())
    return ax.tripcolor(triangulation, frame[column].to_numpy(), shading="gouraud", cmap=cmap, vmin=vmin, vmax=vmax)


def render_static_context_figures() -> None:
    """Render the lightweight political-boundary and reference-site context figure."""
    if plt is None:
        st.warning("Matplotlib is not installed; static context figures are unavailable.")
        return
    try:
        countries = load_country_geojson()
    except Exception as exc:
        st.warning(f"Geographic context figure could not be loaded: {exc}")
        return
    fig, ax = plt.subplots(figsize=(10, 6), dpi=120)
    ax.set_facecolor("#dcecf7")
    plot_geojson_lines(ax, countries, "#172033", 0.9, 0.95)
    for name, site in SITES.items():
        ax.scatter(site["lon"], site["lat"], s=45, color="#f59e0b", edgecolor="#172033", linewidth=0.5, zorder=5)
        ax.annotate(name, (site["lon"], site["lat"]), xytext=(3, 3), textcoords="offset points", fontsize=11)
    ax.set_xlim(-5, 13); ax.set_ylim(50, 62)
    ax.set_xlabel("Longitude (°E)", fontsize=14); ax.set_ylabel("Latitude (°N)", fontsize=14)
    ax.set_title("North Sea political geography and reference sites", fontsize=14)
    ax.tick_params(axis="both", which="major", labelsize=11)
    ax.grid(True, alpha=0.25); fig.tight_layout()
    st.pyplot(fig, use_container_width=True); plt.close(fig)
    st.caption("Dark lines: political boundaries and coastlines. Amber points: reference sites.")


def render_bathymetry_figure(selected_site: str) -> None:
    if plt is None:
        st.warning("Matplotlib is not installed; bathymetry figure is unavailable.")
        return
    try:
        countries = load_country_geojson()
        bathymetry = xr.open_dataset(BytesIO(load_bathymetry_bytes()))
    except Exception as exc:
        st.warning(f"Bathymetry figure could not be loaded: {exc}")
        return
    fig, ax = plt.subplots(figsize=(8, 6), dpi=120)
    elevation = bathymetry["elevation"]
    lon_name = next(name for name in elevation.dims if "lon" in name.lower())
    lat_name = next(name for name in elevation.dims if "lat" in name.lower())
    mesh = ax.pcolormesh(elevation[lon_name].values, elevation[lat_name].values, elevation.values, shading="auto", cmap="Blues_r", vmin=-300, vmax=50)
    plot_geojson_lines(ax, countries, "#172033", 0.9, 1.0)
    for name, site in SITES.items():
        size = 55 if name == selected_site else 22
        color = "#dc2626" if name == selected_site else "#f59e0b"
        ax.scatter(site["lon"], site["lat"], s=size, color=color, edgecolor="#172033", linewidth=0.5, zorder=5)
        ax.annotate(name, (site["lon"], site["lat"]), xytext=(3, 3), textcoords="offset points", fontsize=11, fontweight="bold" if name == selected_site else "normal")
    ax.set_xlim(-5, 13); ax.set_ylim(50, 62)
    ax.set_xlabel("Longitude (°E)", fontsize=14); ax.set_ylabel("Latitude (°N)", fontsize=14)
    ax.set_title(f"North Sea bathymetry: {selected_site}", fontsize=14)
    ax.tick_params(axis="both", which="major", labelsize=11)
    ax.grid(True, alpha=0.25); fig.colorbar(mesh, ax=ax, label="Elevation / depth (m)")
    fig.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close(fig)
    site = SITES[selected_site]
    st.caption(f"Selected site: {selected_site} ({site['lat']:.2f}°N, {site['lon']:.2f}°E). EMODnet DTM 2024; negative values indicate depth.")


def render_north_sea_eez_figure(selected_site: str) -> None:
    """Display a public-domain North Sea EEZ map from Wikimedia Commons."""
    try:
        img_bytes = load_eez_map_image()
    except Exception as exc:
        st.warning(f"EEZ map could not be loaded: {exc}")
        return
    st.image(BytesIO(img_bytes), use_container_width=True)
    st.caption("Exclusive economic zones for the North Sea region (Wikimedia Commons: North_Sea_map-en.png). Site shown for context: " + selected_site)


def format_change(value: float, unit: str = "") -> str:
    return f"{'+' if value >= 0 else ''}{value:.2f}{unit}"


def check_dependencies() -> bool:
    if px is None or plt is None:
        st.error("Plotly and Matplotlib are required. Install the project dependencies before running the app.")
        return False
    return True


def render_intro() -> None:
    st.markdown("## North Sea Wind Intelligence")
    st.write(
        "This prototype combines high-resolution EERIE climate-model output, wind-resource analysis, "
        "and agentic AI. It compares January and July wind conditions from two future SSP2-4.5 windows, "
        "helps explore representative North Sea sites, and explains what the data can—and cannot—support."
    )
    cols = st.columns(3)
    cols[0].markdown("**Climate analysis**\n\nSpatial differences, daily distributions, percentiles, and model-period comparisons.")
    cols[1].markdown("**Wind screening**\n\nWind speed, wind-power density, site context, and hub-height sensitivity.")
    cols[2].markdown("**Agentic AI**\n\nThe agent plans an analysis, calls bounded scientific tools, shows its trace, and reports caveats.")
    st.info("Prototype scope: January and July 2020–2024 versus 2036–2040, daily 10 m wind, approximately 9 km native atmospheric grid. Results are exploratory, not bankable.")
    st.caption("Source: EERIE IFS-FESOM2-SR highres-future-ssp245, retrieved from the DKRZ km-scale cloud Zarr endpoint. The compact runtime package contains January and July m10u, m10v, mean2t, msp, and msst; the build applies a preliminary North Sea polygon mask.")





def render_static_seasonal_maps(january: dict, july: dict, metric_kind: str) -> None:
    if plt is None:
        st.warning("Matplotlib is required for static seasonal maps.")
        return
    countries = load_country_geojson()
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), dpi=110, constrained_layout=True)
    rows = [("January", january, "January 2020–2024", "January 2036–2040"), ("July", july, "July 2020–2024", "July 2036–2040")]
    if metric_kind == "wind":
        specs = [("early_wind", "Early mean", "viridis", 0, 10), ("late_wind", "Later mean", "viridis", 0, 10), ("wind_change", "Difference", "RdBu_r", -2, 2)]
        unit = "m/s"
    else:
        specs = [("early_wpd", "Early mean", "viridis", 0, 1300), ("late_wpd", "Later mean", "viridis", 0, 1300), ("wpd_change", "Difference", "RdBu_r", -200, 200)]
        unit = "W/m²"
    for row_index, (month, package, early_label, late_label) in enumerate(rows):
        spatial = package["spatial"].copy()
        spatial["wind_change"] = spatial["late_wind"] - spatial["early_wind"]
        spatial["wpd_change"] = spatial["late_wpd"] - spatial["early_wpd"]
        triangulation = mtri.Triangulation(spatial["lon"].to_numpy(), spatial["lat"].to_numpy())
        for col_index, (metric, label, cmap, vmin, vmax) in enumerate(specs):
            ax = axes[row_index, col_index]
            plot = continuous_field(ax, spatial, metric, cmap, vmin, vmax, triangulation=triangulation)
            plot_geojson_lines(ax, countries, "#172033", 0.35, 0.7)
            ax.set_xlim(-5, 13); ax.set_ylim(50, 62)
            ax.set_xlabel("Longitude (°E)", fontsize=12); ax.set_ylabel("Latitude (°N)", fontsize=12)
            ax.set_title(f"{month}: {label}", fontsize=12)
            ax.tick_params(axis="both", which="major", labelsize=10)
            ax.grid(True, alpha=0.2)
            fig.colorbar(plot, ax=ax, shrink=0.78, pad=0.02, label=unit)
    fig.suptitle("January and July mean wind" if metric_kind == "wind" else "January and July mean wind-power density", fontsize=16)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def render_runtime_seasonal_overview(january: dict, july: dict) -> None:
    """Render the complete lightweight January/July overview."""
    render_intro()
    st.header("North Sea overview")
    st.caption("January is the current prototype focus; July is included as a prepared seasonal comparison. All maps use compact period means, while the agent operates on compact site and summary artifacts.")
    st.subheader("Geography and reference sites")
    render_static_context_figures()

    st.subheader("Mean wind: January and July")
    st.caption("Static 3×2 figure: January is the top row and July is the bottom row. Each row is early mean, later mean, and later-minus-early difference. Absolute mean-wind colors are fixed at 0–10 m/s; differences at −2 to +2 m/s.")
    render_static_seasonal_maps(january, july, "wind")
    st.info("Mean wind-speed fields are non-negative. Negative values can occur in the signed U/V wind components, or in the difference panels when the later period is lower than the early period; they are not negative wind speeds.")
    st.subheader("Mean WPD: January and July")
    st.caption("Static 3×2 figure: absolute WPD colors are fixed at 0–1300 W/m²; differences at −200 to +200 W/m².")
    render_static_seasonal_maps(january, july, "wpd")

    st.subheader("Combined January and July daily distributions")
    jan_daily = january["daily"].assign(month="January")
    jul_daily = july["daily"].assign(month="July")
    daily = pd.concat([jan_daily, jul_daily], ignore_index=True)
    metric = st.selectbox("Distribution metric", ["mean_wind_speed_10m_ms", "mean_wpd_wm2", "p90_wind_speed_10m_ms"], key="seasonal_distribution_metric")
    st.plotly_chart(px.box(daily, x="period", y=metric, color="month", points="outliers", title="Daily distributions by month and period"), use_container_width=True)
    st.warning("January and July are seasonal windows, not annual production estimates. The prepared runtime data uses daily 10 m wind and requires explicit hub-height assumptions for turbine screening.")


def render_runtime_site_explorer(january: dict, july: dict) -> None:
    st.header("Site explorer")
    site_name = st.selectbox("Choose a reference site", list(SITES))
    site_info = SITES[site_name]
    jan_sites = january["sites"]
    jul_sites = july["sites"]
    jan_early = jan_sites.query("site == @site_name and period == 'January 2020-2024'").copy()
    jan_late = jan_sites.query("site == @site_name and period == 'January 2036-2040'").copy()
    jul_early = jul_sites.query("site == @site_name and period == 'July 2020-2024'").copy()
    jul_late = jul_sites.query("site == @site_name and period == 'July 2036-2040'").copy()
    left, right = st.columns([1.65, 1.0])
    with left:
        st.markdown(f"### {site_name}")
        st.write(f"**Reference:** {site_info['reference']} ({site_info['country']})")
        st.caption(f"Nearest expanded-cache grid point: {jan_early.grid_lat.iloc[0]:.3f}°N, {jan_early.grid_lon.iloc[0]:.3f}°E")
        st.markdown("**Mean 10 m wind speed (m/s)**")
        wind_cards = st.columns(4)
        wind_cards[0].metric("Jan 2020–2024", f"{jan_early.wind_speed_10m_ms.mean():.2f}", help="Mean 10 m wind speed for all January days in 2020–2024.")
        wind_cards[1].metric("Jul 2020–2024", f"{jul_early.wind_speed_10m_ms.mean():.2f}", help="Mean 10 m wind speed for all July days in 2020–2024.")
        wind_cards[2].metric("Jan 2036–2040", f"{jan_late.wind_speed_10m_ms.mean():.2f}", help="Mean 10 m wind speed for all January days in 2036–2040.")
        wind_cards[3].metric("Jul 2036–2040", f"{jul_late.wind_speed_10m_ms.mean():.2f}", help="Mean 10 m wind speed for all July days in 2036–2040.")
        st.markdown("**Mean 10 m wind-power density (W/m²)**")
        wpd_cards = st.columns(4)
        wpd_cards[0].metric("Jan 2020–2024", f"{jan_early.wind_power_density_10m_wm2.mean():.1f}", help="Mean 10 m WPD for all January days in 2020–2024.")
        wpd_cards[1].metric("Jul 2020–2024", f"{jul_early.wind_power_density_10m_wm2.mean():.1f}", help="Mean 10 m WPD for all July days in 2020–2024.")
        wpd_cards[2].metric("Jan 2036–2040", f"{jan_late.wind_power_density_10m_wm2.mean():.1f}", help="Mean 10 m WPD for all January days in 2036–2040.")
        wpd_cards[3].metric("Jul 2036–2040", f"{jul_late.wind_power_density_10m_wm2.mean():.1f}", help="Mean 10 m WPD for all July days in 2036–2040.")
        def _render_daily_figure(y_column: str, y_label: str, title: str) -> None:
            st.subheader(title)
            st.caption("Each cluster is one month (31 daily points). Red = January, blue = July. The x-axis is one tick per month and year, in chronological order.")
            fig, ax = plt.subplots(figsize=(22, 5.5), dpi=110)
            all_months = []
            for period, color, month_name in [
                (jan_early, "#dc2626", "Jan"),
                (jul_early, "#2563eb", "Jul"),
                (jan_late, "#dc2626", "Jan"),
                (jul_late, "#2563eb", "Jul"),
            ]:
                if period.empty:
                    continue
                for year, group in period.groupby(period["time"].dt.year):
                    all_months.append((year, 1 if month_name == "Jan" else 7, month_name, color, group))
            all_months.sort(key=lambda x: (x[0], x[1]))
            tick_positions = []
            tick_labels = []
            x = 0
            for year, _, month_name, color, group in all_months:
                label = "January" if (month_name == "Jan" and year == 2020) else ("July" if (month_name == "Jul" and year == 2020) else None)
                ax.scatter([x] * len(group), group[y_column], color=color, s=14, alpha=0.65, label=label)
                tick_positions.append(x)
                tick_labels.append(f"{month_name} {int(year)}")
                x += 1
            ax.set_xticks(tick_positions)
            ax.set_xticklabels(tick_labels, rotation=90, ha="center", fontsize=13)
            ax.set_ylabel(y_label, fontsize=16)
            ax.set_xlabel("Month and year", fontsize=16)
            ax.tick_params(axis="both", which="major", labelsize=13)
            ax.grid(True, alpha=0.25)
            ax.legend(title="Month", loc="upper right")
            st.pyplot(fig, use_container_width=True); plt.close(fig)

        _render_daily_figure("wind_speed_10m_ms", "10 m wind speed (m/s)", "Daily 10 m wind: January and July, 2020–2024 and 2036–2040")
        _render_daily_figure("wind_power_density_10m_wm2", "10 m WPD (W/m²)", "Daily 10 m WPD: January and July, 2020–2024 and 2036–2040")
        st.subheader("Hub-height sensitivity")
        st.latex(r"v(z)=v(10)\left(\frac{z}{10}\right)^\alpha")
        st.info(
            "This extrapolates 10 m wind to 120 m, the hub height of the NREL 15 MW IEA turbine. "
            "α is the wind-shear exponent. A value near 0.12 is a common offshore default; smaller α means less shear, "
            "larger α means more shear. It is an assumption, not a measured or model-level profile."
        )
        alpha = st.slider("Power-law shear exponent α", 0.05, 0.20, 0.12, 0.01, key=f"runtime_shear_{site_name}", help="Larger α raises the 120 m wind more steeply.")
        st.write(f"Estimated mean 120 m January 2020–2024 wind: **{(jan_early.wind_speed_10m_ms * (120 / 10) ** alpha).mean():.2f} m/s**")
        hub_wind = jan_early.wind_speed_10m_ms * (120 / 10) ** alpha
        k, a, _ = fit_weibull(jan_early.wind_speed_10m_ms)
        cf = compute_capacity_factor(hub_wind)
        aep = compute_aep(hub_wind)
        metric_cards = st.columns(3)
        metric_cards[0].metric("Weibull shape k", f"{k:.2f}" if not np.isnan(k) else "N/A", help="Maximum-likelihood Weibull fit (scipy.stats.weibull_min) to the daily 10 m wind speed time series. Higher k = more consistent wind.")
        metric_cards[1].metric("120 m capacity factor", f"{cf:.1%}", help="Mean power from the NREL 15 MW IEA power curve fed with the 120 m extrapolated daily wind, divided by 15 MW.")
        metric_cards[2].metric("AEP (15 MW)", f"{aep:,.0f} MWh", help="Capacity factor × 15 MW × 8760 h. Uses 120 m daily-mean wind, so it is biased downward and not bankable.")
        st.warning("These metrics are for the early period (2020–2024), January. The power curve is applied to daily-mean wind, which smooths out the high-wind hours that produce most energy, so the capacity factor and AEP are biased downward and are not bankable.")
    with right:
        st.subheader("Bathymetry and selected site")
        render_bathymetry_figure(site_name)
        st.subheader("North Sea / EEZ context")
        render_north_sea_eez_figure(site_name)


def _raw_tool_result(toolkit: JanuaryAgentToolkit, name: str, arguments: dict) -> Any:
    if name == "check_data_availability":
        return toolkit.check_data_availability(arguments.get("requested_period"))
    if name == "compare_january_periods":
        return toolkit.compare_periods()
    if name == "compare_january_july":
        return toolkit.compare_january_july()
    if name == "get_january_period_summary":
        return toolkit.period_summary(arguments["period"])
    if name == "calculate_spatial_difference":
        return toolkit.calculate_spatial_difference(
            arguments.get("metric", "wind_power_density_10m_wm2"),
            month=arguments.get("month", "January"),
        )
    if name == "identify_top_regions":
        return toolkit.identify_top_regions(
            arguments.get("metric", "wind_power_density_10m_wm2"),
            month=arguments.get("month", "January"),
        )
    if name == "compare_reference_sites":
        return toolkit.compare_reference_sites(
            arguments.get("period", LATE_LABEL),
            arguments.get("metric", "mean_wpd_wm2"),
        )
    if name == "get_site_summary":
        return toolkit.site_summary(arguments["site_name"], arguments.get("period", EARLY_LABEL))
    if name == "get_wind_distribution":
        return toolkit.wind_distribution(arguments["period"], arguments.get("site_name"))
    if name == "test_hub_height_sensitivity":
        exponents = tuple(arguments.get("shear_exponents", [0.08, 0.12, 0.16]))
        return toolkit.height_sensitivity(arguments["period"], shear_exponents=exponents)
    if name == "recommend_region":
        return toolkit.recommend_region()
    if name == "explain_analysis_limitations":
        return toolkit.limitations(arguments.get("data_frequency", "daily"))
    if name == "validate_screening_readiness":
        return toolkit.validate_screening_readiness(arguments.get("question", ""))
    if name == "get_data_provenance":
        return toolkit.get_data_provenance()
    return {"error": f"Unknown tool: {name}"}


def dispatch_tool(toolkit: JanuaryAgentToolkit, name: str, arguments: dict) -> dict:
    result = _raw_tool_result(toolkit, name, arguments)
    if isinstance(result, dict) and "_provenance" not in result and "provenance" not in result:
        result = dict(result)
        result["_provenance"] = toolkit.provenance()
    return result


def determine_intent(question: str) -> str:
    lower = question.lower()
    climate_terms = ["climate", "trend", "spatial", "pattern", "variability", "season", "july", "winter", "summer", "period", "difference", "change over"]
    engineering_terms = ["site", "rank", "turbine", "capacity", "aep", "hub", "siting", "screening", "production", "energy", "recommend", "best region", "most suitable"]
    c = sum(1 for t in climate_terms if t in lower)
    e = sum(1 for t in engineering_terms if t in lower)
    if e > c:
        return "engineering"
    return "climate"


def deterministic_agent(toolkit: JanuaryAgentToolkit, question: str) -> tuple[str, list[dict[str, Any]]]:
    lower = question.lower()
    intent = determine_intent(question)
    trace = [{"step": 1, "action": "Interpret user goal", "detail": question, "intent": intent}]

    readiness = toolkit.validate_screening_readiness(question)
    trace.append({"step": len(trace) + 1, "action": "Call bounded tool", "tool": "validate_screening_readiness", "arguments": {"question": question}, "result": readiness})
    if not readiness["can_answer"]:
        violations = [v for v in readiness["violations"] if "within" not in v.lower()]
        answer = "**I cannot answer that as a screening result.**\n\n" + "\n".join(f"- {v}" for v in violations)
        trace.append({"step": len(trace) + 1, "action": "Refuse request due to active guardrails"})
        return answer, trace

    def _provenance_footer() -> str:
        prov = toolkit.provenance()
        return (
            f"\n\n---\n"
            f"**Provenance:** {prov['model']}; periods {prov['periods'][0]} and {prov['periods'][1]}; "
            f"{prov['wind_reference_height_m']} m wind (120 m via power-law sensitivity); "
            f"region lon {prov['region']['longitude']}°E, lat {prov['region']['latitude']}°N; {prov['mask']}."
        )

    if any(word in lower for word in ["provenance", "where did", "source", "ssp2-4.5", "eerie", "variables", "prepared"]):
        tool_name = "get_data_provenance"
        trace.append({"step": len(trace) + 1, "action": "Call bounded tool", "tool": tool_name})
        result = dispatch_tool(toolkit, tool_name, {})
        prov = result.get("provenance", result)
        answer = (
            "**Data provenance**\n\n"
            f"- Model: {prov['model']}\n"
            f"- Zarr endpoint: {prov['data_url']}\n"
            f"- Periods: {prov['periods'][0]} and {prov['periods'][1]}; also {prov['seasonal_windows']}\n"
            f"- Region: lon {prov['region']['longitude']}°E, lat {prov['region']['latitude']}°N\n"
            f"- Mask: {prov['mask']}\n"
            f"- Reference wind height: {prov['wind_reference_height_m']} m\n"
            f"- Prepared variables: {', '.join(prov['prepared_variables'])}"
        )
        answer += _provenance_footer()
        return answer, trace

    if any(word in lower for word in ["july", "season", "summer"]):
        tool_name = "compare_january_july"
        trace.append({"step": len(trace) + 1, "action": "Call bounded tool", "tool": tool_name})
        result = dispatch_tool(toolkit, tool_name, {})
        if "months" in result:
            answer = "**Seasonal comparison (climate intent)**\n\n" + "\n".join(
                f"- **{month}**: wind {values['early_mean_wind_ms']:.2f} → {values['late_mean_wind_ms']:.2f} m/s; "
                f"WPD {values['early_wpd_wm2']:.1f} → {values['late_wpd_wm2']:.1f} W/m² "
                f"(changes: {values['wind_change_ms']:+.2f} m/s, {values['wpd_change_wm2']:+.1f} W/m²)"
                for month, values in result["months"].items()
            )
            answer += "\n\nThis is a seasonal-window comparison, not an annual production estimate."
        else:
            answer = result.get("error", "The seasonal comparison is unavailable.")
        answer += _provenance_footer()
        return answer, trace

    if any(word in lower for word in ["rank", "dogger", "hornsea", "moray", "borssele", "german bight", "kriegers", "reference sites", "compare dogger"]):
        tool_name = "compare_reference_sites"
        args = {"period": LATE_LABEL, "metric": "mean_wpd_wm2"}
        trace.append({"step": len(trace) + 1, "action": "Call bounded tool", "tool": tool_name, "arguments": args})
        result = dispatch_tool(toolkit, tool_name, args)
        ranking = result.get("ranking", [])
        answer = "**Reference-site ranking (engineering intent)**\n\n"
        if ranking:
            for row in ranking:
                answer += (
                    f"- **{row['site']}**: {row['mean_wpd_wm2']:.1f} W/m² at 10 m, "
                    f"{row['mean_wind_speed_120m_ms']:.2f} m/s at 120 m (10–90 range "
                    f"{row['p10_wind_speed_120m_ms']:.2f}–{row['p90_wind_speed_120m_ms']:.2f}), "
                    f"CF {row['capacity_factor_120m']:.1%}, AEP {row['annual_energy_production_mwh']:,.0f} MWh\n"
                )
        answer += "\nThis ranks 10 m-to-120 m sensitivity estimates. It is not a bankable siting recommendation."
        answer += _provenance_footer()
        return answer, trace

    if any(word in lower for word in ["largest", "biggest", "spatial change", "change in wpd", "region has the largest"]):
        tool_name = "calculate_spatial_difference"
        args = {"metric": "wind_power_density_10m_wm2", "month": "January"}
        trace.append({"step": len(trace) + 1, "action": "Call bounded tool", "tool": tool_name, "arguments": args})
        result = dispatch_tool(toolkit, tool_name, args)
        summary = result.get("change_summary", {})
        top = result.get("top_regions", [])[:3]
        answer = "**Largest January WPD spatial changes (climate intent)**\n\n"
        if summary:
            answer += (
                f"Across the domain, WPD changes average {summary.get('mean', 0):.1f} W/m² "
                f"(spread: {summary.get('p10', 0):.1f} to {summary.get('p90', 0):.1f}, std {summary.get('std', 0):.1f}).\n\n"
            )
        if top:
            answer += "Top change regions:\n" + "\n".join(
                f"- ({row['lat']:.2f}°N, {row['lon']:.2f}°E): {row['change']:+.1f} W/m² ({row['relative_change_percent']:+.1f}%)"
                for row in top
            )
        answer += "\n\nA large change is more useful for climate-process interpretation than for direct siting."
        answer += _provenance_footer()
        return answer, trace

    if any(word in lower for word in ["promising", "recommend", "best region", "which region", "most suitable"]):
        workflow = toolkit.recommend_region()
        for item in workflow["trace"]:
            trace.append({"step": len(trace) + 1, "action": "Call bounded tool", "tool": item["tool"]})
        ranking = workflow["recommendation"]
        answer = "**Multi-step screening recommendation (engineering intent)**\n\n"
        if ranking:
            answer += "The highest-ranked reference sites in the later period by mean WPD are:\n\n"
            answer += "\n".join(f"- **{row['site']}**: {row['mean_wpd_wm2']:.1f} W/m² at 10 m" for row in ranking)
        answer += "\n\nThis is a screening result. It is not a turbine-siting or investment recommendation."
        answer += "\n\n**Limitations**\n" + "\n".join(f"- {c}" for c in workflow.get("limitations", []))
        answer += _provenance_footer()
        return answer, trace

    if any(word in lower for word in ["height", "100 m", "hub", "120 m", "shear"]):
        tool_name = "test_hub_height_sensitivity"
        args = {"period": EARLY_LABEL, "shear_exponents": [0.08, 0.12, 0.16]}
        trace.append({"step": len(trace) + 1, "action": "Call bounded tool", "tool": tool_name, "arguments": args})
        result = dispatch_tool(toolkit, tool_name, args)
        rows = result.get("results", [])
        if rows:
            cfs = [row["capacity_factor"] for row in rows]
            answer = (
                "**Hub-height sensitivity (engineering intent)**\n\n"
                + "\n".join(f"At shear {row['shear_exponent']:.2f}, estimated 120 m CF is {row['capacity_factor']:.1%}." for row in rows)
                + f"\n\nThis gives a CF range of {min(cfs):.1%}–{max(cfs):.1%} across the tested shear exponents. "
                "It is a sensitivity estimate, not a direct model-level 120 m wind or a bankable capacity factor."
            )
        else:
            answer = result.get("error", "The hub-height sensitivity analysis is unavailable.")
        answer += _provenance_footer()
        return answer, trace

    if any(word in lower for word in ["limit", "valid", "bank", "caveat", "missing", "engineering inputs"]):
        tool_name = "explain_analysis_limitations"
        args = {"data_frequency": "daily"}
        trace.append({"step": len(trace) + 1, "action": "Call bounded tool", "tool": tool_name, "arguments": args})
        result = dispatch_tool(toolkit, tool_name, args)
        answer = "**Analysis limitations**\n\n" + "\n".join(f"- {item}" for item in result["caveats"])
        answer += _provenance_footer()
        return answer, trace

    tool_name = "compare_january_periods"
    trace.append({"step": len(trace) + 1, "action": "Call bounded tool", "tool": tool_name})
    result = dispatch_tool(toolkit, tool_name, {})
    changes = result.get("changes", {})
    if changes:
        w = changes["mean_wind_speed_ms"]
        wpd = changes["mean_wpd_wm2"]
        answer = (
            f"**January period comparison ({intent} intent)**\n\n"
            f"Mean 10 m wind: **{w['early']:.2f} → {w['late']:.2f} m/s** ({w['relative_change_percent']:+.2f}%).\n"
            f"10 m wind 10–90 percentile range: {changes['p10_wind_speed_ms']['early']:.2f}–{changes['p90_wind_speed_ms']['early']:.2f} (early) "
            f"vs {changes['p10_wind_speed_ms']['late']:.2f}–{changes['p90_wind_speed_ms']['late']:.2f} (later).\n\n"
            f"Mean WPD: **{wpd['early']:.1f} → {wpd['late']:.1f} W/m²** ({wpd['relative_change_percent']:+.2f}%).\n"
            f"WPD 10–90 percentile range: {changes['p10_wpd_wm2']['early']:.1f}–{changes['p90_wpd_wm2']['early']:.1f} (early) "
            f"vs {changes['p10_wpd_wm2']['late']:.1f}–{changes['p90_wpd_wm2']['late']:.1f} (later)."
        )
    else:
        answer = result.get("error", "The requested comparison is unavailable.")
    answer += _provenance_footer()
    trace.append({"step": len(trace) + 1, "action": "Return evidence-based answer with provenance"})
    return answer, trace


def provider_agent(toolkit: JanuaryAgentToolkit, provider: str, model: str, question: str, api_key: str | None = None) -> tuple[str, list[dict[str, Any]]]:
    return run_tool_calling_agent(
        provider=provider,
        model=model,
        question=question,
        tool_definitions=toolkit.tool_definitions(),
        dispatch=lambda name, arguments: dispatch_tool(toolkit, name, arguments),
        api_key=api_key,
    )


def render_agent(january: dict, july: dict) -> None:
    st.header("Agent workspace")
    st.caption("Converse with the North Sea wind and climate agent. Pick a quick question or type your own below.")

    runtime = dict(january)
    runtime["seasonal"] = {"January": january, "July": july}
    toolkit = JanuaryAgentToolkit({}, runtime=runtime)

    provider_choice = st.session_state.get("agent_provider", "Built-in scientific agent")
    provider = provider_choice if provider_choice in PROVIDERS else None
    model = st.session_state.get(f"model_{provider}") if provider else None
    api_key = (st.session_state.get(f"key_{provider}") or None) if provider else None

    if "agent_messages" not in st.session_state:
        st.session_state["agent_messages"] = [
            {"role": "assistant", "content": "Hi. I'm the North Sea wind intelligence agent. Ask me about January/July changes, regional screening, site ranking, hub-height sensitivity, or the limits of this data. I'll tell you which tool I used and where the data came from."}
        ]
    if "pending_agent_question" not in st.session_state:
        st.session_state["pending_agent_question"] = None

    pending = st.session_state.get("pending_agent_question")
    if pending:
        st.session_state["pending_agent_question"] = None
        question = pending.strip()
        if question:
            with st.spinner("Agent is thinking..."):
                try:
                    if provider and (api_key or get_configured_key(provider)):
                        answer, trace = provider_agent(toolkit, provider, model, question, api_key=api_key)
                    else:
                        answer, trace = deterministic_agent(toolkit, question)
                        if provider:
                            trace.insert(1, {"step": 2, "action": "Provider key unavailable; used deterministic fallback", "provider": provider})
                except Exception as exc:
                    answer = f"Agent execution failed: {exc}"
                    trace = [{"step": 1, "action": "Error", "detail": str(exc)}]
            st.session_state["agent_messages"].append({"role": "user", "content": question})
            st.session_state["agent_messages"].append({"role": "assistant", "content": answer, "trace": trace})

    chat_col, side_col = st.columns([2.0, 1.0])
    with chat_col:
        st.markdown("### Conversation")
        st.info("Select an agent provider below **Workspace** in the sidebar. The Built-in scientific agent runs offline; Groq and Gemini need a configured API key.")
        for msg in st.session_state["agent_messages"]:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if "trace" in msg:
                    with st.expander("Agent execution trace", expanded=False):
                        for item in msg["trace"]:
                            step = item.get("step", "?")
                            action = item.get("action", "")
                            st.markdown(f"**Step {step}:** {action}")
                            if item.get("tool"):
                                st.markdown(f"- Tool: `{item['tool']}`")
                            if item.get("detail"):
                                st.markdown(f"- Detail: {item['detail']}")
                            if item.get("provider"):
                                st.markdown(f"- Provider: {item['provider']}" + (f" / {item.get('model', '')}" if item.get("model") else ""))
                            if "arguments" in item:
                                st.markdown(f"- Arguments: `{item['arguments']}`")
                            if item.get("result") and "can_answer" in item.get("result", {}):
                                st.markdown(f"- Readiness: `{item['result']['can_answer']}`")

    with side_col:
        with st.expander("Scientific guardrails", expanded=False):
            st.markdown("**Agent contract**")
            st.json(toolkit.analysis_contract())
            st.markdown("**Tool inventory**")
            st.write([tool["function"]["name"] for tool in toolkit.tool_definitions()])
            st.markdown("**Guardrails**")
            guardrails = scientific_guardrails() + [
                "Expanded bounding box; finite SST is only a preliminary ocean proxy.",
                "No final hydrographic North Sea polygon has been applied yet.",
                "SST supports climate interpretation but is not direct turbine-production input.",
                "July is included for seasonal comparison; January remains the primary prototype focus.",
            ]
            for caveat in guardrails:
                st.caption(f"• {caveat}")

        st.markdown("#### Quick questions")
        quick = [
            ("Compare January periods", "Compare January 2020–2024 and January 2036–2040, then explain the caveats."),
            ("Recommend a region", "Run a full North Sea screening recommendation, then explain the limitations."),
            ("Rank reference sites", "Compare Dogger Bank, Hornsea, German Bight, and Moray Firth in the later period. Rank them by WPD and explain the caveats."),
            ("Largest WPD change", "Which region has the largest January change in wind-power density, and is that more useful for climate analysis or wind-project screening?"),
            ("Hub-height sensitivity", "How sensitive is the estimated 120 m wind and illustrative capacity factor to the shear exponent?"),
            ("Bankability audit", "Can this result support a bankable wind-farm decision? Audit the temporal resolution, height, model, mask, and missing engineering inputs."),
            ("Data provenance", "Where did the EERIE data come from, what is SSP2-4.5, and what variables were prepared for this analysis?"),
        ]
        for i, (label, text) in enumerate(quick):
            if st.button(label, key=f"quick_{i}", use_container_width=True):
                st.session_state["pending_agent_question"] = text
                st.rerun()

    st.divider()
    st.markdown("**What would you like to know?**")
    st.caption("Type or paste a question and press Enter. The agent will call a bounded scientific tool, show its trace, and cite the data source.")
    question = st.chat_input("Ask or paste a wind / climate question...", key="agent_chat_input")
    if question:
        st.session_state["pending_agent_question"] = question
        st.rerun()

    st.sidebar.divider()
    st.sidebar.markdown("**Agent workspace tools**")
    st.sidebar.write("• Check data availability\n• Find spatial changes\n• Rank reference sites\n• Test hub-height assumptions\n• Explain limitations")


def render_sidebar() -> str:
    st.sidebar.markdown("# North Sea Wind Intelligence")
    st.sidebar.caption("EERIE climate model + wind-resource analytics + agentic AI")
    st.sidebar.divider()

    page = st.sidebar.radio(
        "Workspace",
        ["North Sea overview", "Site explorer", "Agent workspace"],
        key="workspace_page",
    )

    st.sidebar.divider()
    st.sidebar.markdown("**Agent provider**")
    provider_options = ["Groq", "Gemini", "Built-in scientific agent"]
    st.sidebar.selectbox(
        "Agent provider",
        provider_options,
        index=2,
        key="agent_provider",
        help="Groq and Gemini are cloud LLMs; the built-in agent is a deterministic rule engine on the prepared data.",
    )
    provider = st.session_state.get("agent_provider", "Built-in scientific agent")
    if provider in PROVIDERS:
        st.sidebar.selectbox("Model", PROVIDERS[provider]["models"], key=f"model_{provider}")
        configured = bool(get_configured_key(provider))
        st.sidebar.caption(f"{provider} key: {'configured' if configured else 'not configured'}")
        if not configured:
            st.sidebar.text_input(f"{provider} API key (session-only)", type="password", key=f"key_{provider}")

    st.sidebar.divider()
    st.sidebar.markdown("**Prototype scope**")
    st.sidebar.info(
        "Compare two future January windows (2020–2024 vs 2036–2040) from one EERIE SSP2-4.5 realization, "
        "plus a July seasonal comparison. Explore wind, WPD, site metrics, and the agent's bounded answers."
    )
    st.sidebar.markdown(
        "• North Sea: −5–13°E, 50–62°N\n"
        "• Variable: daily native-grid wind at 10 m\n"
        "• Grid: ~9 km IFS-FESOM2-SR"
    )
    st.sidebar.divider()
    st.sidebar.markdown("**Data status**")
    st.sidebar.success("January & July compact packages are loaded")
    st.sidebar.caption("All plots and the agent use pre-built, masked artifacts. No live DKRZ chunk downloads.")
    return page


def render_back_to_top() -> None:
    st.markdown(
        '<div style="position: fixed; bottom: 1.5rem; right: 1.5rem; z-index: 1000;">'
        '<a href="#top" style="background-color: #0e1117; color: #ffffff; padding: 0.5rem 0.75rem; border-radius: 0.5rem; text-decoration: none; font-size: 0.85rem; font-family: sans-serif;">Back to top</a>'
        '</div>',
        unsafe_allow_html=True,
    )


def main() -> None:
    if not check_dependencies():
        return
    page = render_sidebar()
    st.markdown('<a id="top"></a>', unsafe_allow_html=True)
    if not has_runtime_package(runtime_root(), "january") or not has_runtime_package(runtime_root(), "july"):
        st.error("The prepared North Sea runtime package is not available. Build it with `python -m src.build_runtime_artifacts`.")
        return
    january = load_compact_runtime("january")
    july = load_compact_runtime("july")
    if page == "North Sea overview":
        render_runtime_seasonal_overview(january, july)
    elif page == "Site explorer":
        render_runtime_site_explorer(january, july)
    else:
        render_agent(january, july)
    render_back_to_top()


if __name__ == "__main__":
    main()
