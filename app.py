"""Streamlit prototype for agentic North Sea climate and wind analysis."""

from io import BytesIO
from pathlib import Path
import json
import os
from typing import Any

import numpy as np
import pandas as pd
import requests
import streamlit as st
import xarray as xr

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover
    plt = None

from src.january_analysis import (
    EARLY_WINDOW,
    LATE_WINDOW,
    add_wind_metrics,
    scientific_guardrails,
)
from src.january_agent_tools import JanuaryAgentToolkit
from src.llm_providers import PROVIDERS, get_configured_key, provider_status, run_tool_calling_agent

try:
    import plotly.express as px
    import plotly.graph_objects as go
except ImportError:  # pragma: no cover
    px = None
    go = None


st.set_page_config(
    page_title="North Sea Wind Intelligence",
    page_icon="🌬️",
    layout="wide",
    initial_sidebar_state="expanded",
)

HOME_CACHE_ROOT = Path.home() / "largeData" / "eerie_north_sea_expanded_corrected" / "January"
REPO_LARGE_CACHE_ROOT = Path(__file__).resolve().parent / "largeData" / "eerie_north_sea_expanded_corrected" / "January"
REPO_CACHE_ROOT = Path(__file__).resolve().parent / "data" / "january"
if os.getenv("EERIE_CACHE_ROOT"):
    CACHE_ROOT = Path(os.environ["EERIE_CACHE_ROOT"])
elif (REPO_LARGE_CACHE_ROOT / "January_2020-2024.parquet").exists() and (REPO_LARGE_CACHE_ROOT / "January_2036-2040.parquet").exists():
    CACHE_ROOT = REPO_LARGE_CACHE_ROOT
elif (HOME_CACHE_ROOT / "January_2020-2024.parquet").exists() and (HOME_CACHE_ROOT / "January_2036-2040.parquet").exists():
    CACHE_ROOT = HOME_CACHE_ROOT
else:
    CACHE_ROOT = REPO_CACHE_ROOT
EARLY_LABEL = EARLY_WINDOW.label
LATE_LABEL = LATE_WINDOW.label
PERIOD_FILES = {
    EARLY_LABEL: CACHE_ROOT / "January_2020-2024.parquet",
    LATE_LABEL: CACHE_ROOT / "January_2036-2040.parquet",
}

SITES = {
    "Dogger Bank": {"lat": 55.0, "lon": 2.5, "country": "United Kingdom", "reference": "North Sea offshore wind zone"},
    "Hornsea": {"lat": 54.0, "lon": 1.5, "country": "United Kingdom", "reference": "East coast of England"},
    "East Anglia": {"lat": 52.4, "lon": 2.0, "country": "United Kingdom", "reference": "East coast of England"},
    "Moray Firth": {"lat": 58.0, "lon": -2.5, "country": "United Kingdom", "reference": "Northern Scotland"},
    "Borssele": {"lat": 51.8, "lon": 3.5, "country": "Netherlands", "reference": "Dutch North Sea sector"},
    "German Bight": {"lat": 55.0, "lon": 5.0, "country": "Germany", "reference": "German North Sea sector"},
    "Kriegers Flak": {"lat": 54.8, "lon": 13.0, "country": "Denmark/Germany", "reference": "Southern Baltic reference site"},
}

EEZ_WFS_URL = (
    "https://geo.vliz.be/geoserver/MarineRegions/wfs?service=WFS&version=1.0.0"
    "&request=GetFeature&typeNames=eez&outputFormat=application%2Fjson"
    "&bbox=-5,50,13,62,EPSG:4326&maxFeatures=25"
)
COUNTRY_GEOJSON_URL = "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_110m_admin_0_countries.geojson"
BATHYMETRY_URL = "https://erddap.emodnet.eu/erddap/griddap/bathymetry_dtm_2024.nc?elevation%5B(50):100:(62)%5D%5B(-5):100:(13)%5D"


@st.cache_data(show_spinner="Loading prepared EERIE North Sea data...")
def load_period_data(path_string: str) -> pd.DataFrame:
    frame = pd.read_parquet(path_string)
    frame["time"] = pd.to_datetime(frame["time"])
    # Preliminary ocean proxy: the expanded EERIE cache has missing SST over
    # many land points. This is not a replacement for a hydrographic mask.
    if "msst" in frame.columns:
        frame = frame.loc[frame["msst"].notna()].copy()
    return add_wind_metrics(frame)


@st.cache_data(show_spinner="Preparing daily summaries...")
def daily_summary(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        frame.groupby("time", as_index=False)
        .agg(
            mean_wind_speed_10m_ms=("wind_speed_10m_ms", "mean"),
            median_wind_speed_10m_ms=("wind_speed_10m_ms", "median"),
            p90_wind_speed_10m_ms=("wind_speed_10m_ms", lambda value: np.percentile(value, 90)),
            mean_wpd_wm2=("wind_power_density_10m_wm2", "mean"),
        )
        .sort_values("time")
    )


@st.cache_data(show_spinner="Loading official EEZ boundaries...")
def load_eez_geojson() -> dict:
    response = requests.get(EEZ_WFS_URL, timeout=120)
    response.raise_for_status()
    return response.json()


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
                stride = max(1, len(visible) // 500)
                points = visible[::stride]
                ax.plot([p[0] for p in points], [p[1] for p in points], color=color, linewidth=linewidth, alpha=alpha)


def render_static_context_figures() -> None:
    """Render two cached raster-style figures with explicit longitude/latitude axes."""
    if plt is None:
        st.warning("Matplotlib is not installed; static context figures are unavailable.")
        return
    try:
        eez = load_eez_geojson()
        countries = load_country_geojson()
        bathymetry = xr.open_dataset(BytesIO(load_bathymetry_bytes()))
    except Exception as exc:
        st.warning(f"Context figures could not be loaded: {exc}")
        return

    left, right = st.columns(2)
    with left:
        fig, ax = plt.subplots(figsize=(8, 5.5), dpi=120)
        ax.set_facecolor("#dcecf7")
        plot_geojson_lines(ax, countries, "#172033", 0.9, 0.95)
        plot_geojson_lines(ax, eez, "#1769aa", 0.7, 0.85)
        for name, site in SITES.items():
            ax.scatter(site["lon"], site["lat"], s=24, color="#f59e0b", edgecolor="#172033", linewidth=0.5, zorder=5)
            ax.annotate(name, (site["lon"], site["lat"]), xytext=(3, 3), textcoords="offset points", fontsize=6)
        ax.set_xlim(-5, 13); ax.set_ylim(50, 62)
        ax.set_xlabel("Longitude (°E)"); ax.set_ylabel("Latitude (°N)")
        ax.set_title("North Sea EEZ and reference sites")
        ax.grid(True, alpha=0.25); fig.tight_layout()
        st.pyplot(fig, use_container_width=True); plt.close(fig)
        st.caption("Blue dotted lines: Marine Regions EEZ boundaries. Dark lines: political coastlines/boundaries. Amber points: reference sites.")

    with right:
        fig, ax = plt.subplots(figsize=(8, 5.5), dpi=120)
        elevation = bathymetry["elevation"]
        lon_name = next(name for name in elevation.dims if "lon" in name.lower())
        lat_name = next(name for name in elevation.dims if "lat" in name.lower())
        lon = elevation[lon_name].values
        lat = elevation[lat_name].values
        values = elevation.values
        mesh = ax.pcolormesh(lon, lat, values, shading="auto", cmap="Blues_r", vmin=-300, vmax=50)
        plot_geojson_lines(ax, countries, "#172033", 0.9, 1.0)
        for name, site in SITES.items():
            ax.scatter(site["lon"], site["lat"], s=18, color="#f59e0b", edgecolor="#172033", linewidth=0.4, zorder=5)
        ax.set_xlim(-5, 13); ax.set_ylim(50, 62)
        ax.set_xlabel("Longitude (°E)"); ax.set_ylabel("Latitude (°N)")
        ax.set_title("North Sea bathymetry context")
        ax.grid(True, alpha=0.25); fig.colorbar(mesh, ax=ax, label="Elevation / depth (m)")
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True); plt.close(fig)
        st.caption("EMODnet DTM 2024 bathymetry subset. Negative values indicate depth; political boundaries are drawn on top.")


def format_change(value: float, unit: str = "") -> str:
    return f"{'+' if value >= 0 else ''}{value:.2f}{unit}"


def check_dependencies() -> bool:
    if px is None or go is None or plt is None:
        st.error("Plotly and Matplotlib are required. Install the project dependencies before running the app.")
        return False
    return True


def period_files_available() -> bool:
    if all(path.exists() for path in PERIOD_FILES.values()):
        return True
    st.error("The prepared North Sea analysis package is not available in this environment.")
    st.info("The public deployment should package or host the compact prepared Parquet files. Raw EERIE data is never fetched during an interview session.")
    return False


def add_eez_traces(fig, geojson: dict) -> None:
    """Add clipped, simplified EEZ outlines as one browser-friendly trace."""
    lons, lats = [], []
    for feature in geojson.get("features", []):
        geometry = feature.get("geometry") or {}
        geometry_type = geometry.get("type")
        coordinates = geometry.get("coordinates", [])
        polygons = coordinates if geometry_type == "MultiPolygon" else [coordinates]
        for polygon in polygons:
            for ring in polygon:
                visible = [point for point in ring if -6 <= point[0] <= 15 and 48 <= point[1] <= 64]
                if len(visible) < 2:
                    continue
                stride = max(1, len(visible) // 300)
                visible = visible[::stride]
                lons.extend([point[0] for point in visible] + [None])
                lats.extend([point[1] for point in visible] + [None])
    if lons:
        fig.add_trace(go.Scattergeo(
            lon=lons,
            lat=lats,
            mode="lines",
            line={"color": "rgba(20, 85, 145, 0.85)", "width": 1.2, "dash": "dot"},
            name="EEZ boundaries",
            showlegend=True,
            hoverinfo="skip",
        ))


def geographic_context_figure(show_eez: bool = False):
    fig = go.Figure()
    site_frame = pd.DataFrame(
        [{"site": name, **values} for name, values in SITES.items()]
    )
    fig.add_trace(go.Scattergeo(
        lon=site_frame["lon"],
        lat=site_frame["lat"],
        text=site_frame["site"],
        customdata=site_frame[["country", "reference"]],
        mode="markers+text",
        textposition="top center",
        marker={"size": 8, "color": "#f59e0b", "line": {"color": "#172033", "width": 1}},
        name="Reference sites",
        hovertemplate="<b>%{text}</b><br>%{customdata[0]}<br>%{customdata[1]}<br>%{lat:.2f}°N, %{lon:.2f}°E<extra></extra>",
    ))
    if show_eez:
        try:
            add_eez_traces(fig, load_eez_geojson())
        except Exception as exc:
            st.warning(f"EEZ layer could not be loaded: {exc}")
    fig.update_geos(
        projection_type="mercator",
        lonaxis_range=[-6, 15],
        lataxis_range=[48, 64],
        showcountries=True,
        countrycolor="rgba(20, 30, 45, 0.75)",
        showcoastlines=True,
        coastlinecolor="#172033",
        showland=True,
        landcolor="#eef1e8",
        showocean=True,
        oceancolor="#dcecf7",
        showlakes=True,
        lakecolor="#dcecf7",
        resolution=50,
    )
    fig.update_layout(
        height=600,
        margin=dict(l=0, r=0, t=25, b=0),
        legend=dict(orientation="h", y=1.02, x=0),
    )
    return fig


def render_intro() -> None:
    st.markdown("## Agentic North Sea Wind Intelligence")
    st.write(
        "This prototype combines high-resolution EERIE climate-model output, wind-resource analysis, "
        "and agentic AI. It compares January wind conditions from two future SSP2-4.5 windows, "
        "helps explore representative North Sea sites, and explains what the data can—and cannot—support."
    )
    cols = st.columns(3)
    cols[0].markdown("**Climate analysis**\n\nSpatial differences, daily distributions, percentiles, and model-period comparisons.")
    cols[1].markdown("**Wind screening**\n\nWind speed, wind-power density, site context, and hub-height sensitivity.")
    cols[2].markdown("**Agentic AI**\n\nThe agent plans an analysis, calls bounded scientific tools, shows its trace, and reports caveats.")
    st.info("Prototype scope: January 2020–2024 versus January 2036–2040, daily 10 m wind, approximately 9 km native atmospheric grid. Results are exploratory, not bankable.")
    st.caption("Source: EERIE IFS-FESOM2-SR highres-future-ssp245, retrieved from the DKRZ km-scale cloud Zarr endpoint. The prepared expanded cache contains m10u, m10v, mean2t, msp, and msst; the current interface applies finite SST as a preliminary ocean proxy.")


def render_variable_snapshot(frames: dict[str, pd.DataFrame]) -> None:
    st.subheader("What was downloaded? January 1, 2021 snapshot")
    st.caption("These panels expose the source and derived fields available to the prototype on the first day of January 2021.")
    snapshot = frames[EARLY_LABEL]
    snapshot = snapshot[snapshot["time"] == pd.Timestamp("2021-01-01")]
    variables = {
        "m10u": ("10 m U wind", "m/s", "RdBu_r"),
        "m10v": ("10 m V wind", "m/s", "RdBu_r"),
        "wind_speed_10m_ms": ("Derived 10 m wind speed", "m/s", "Viridis"),
        "mean2t": ("Mean 2 m temperature", "K", "Turbo"),
        "msp": ("Mean sea-level pressure", "Pa", "Viridis"),
        "msst": ("Daily mean sea-surface temperature", "K", "Turbo"),
        "wind_power_density_10m_wm2": ("Derived 10 m WPD", "W/m²", "Viridis"),
    }
    choices = list(variables)
    left, right = st.columns(2)
    selected_left = left.selectbox("Panel 1", choices, format_func=lambda value: variables[value][0])
    selected_right = right.selectbox("Panel 2", choices, index=2, format_func=lambda value: variables[value][0])
    for column, selected in [(left, selected_left), (right, selected_right)]:
        title, unit, colors = variables[selected]
        fig = px.scatter_geo(
            snapshot,
            lat="lat",
            lon="lon",
            color=selected,
            hover_data={"lat": ":.3f", "lon": ":.3f", selected: ":.3f"},
            color_continuous_scale=colors,
            labels={selected: unit},
            projection="mercator",
            title=f"{title} ({unit})",
        )
        fig.update_geos(lonaxis_range=[-6, 15], lataxis_range=[48, 64], showcountries=True, showcoastlines=True, showland=True, landcolor="#eef1e8", showocean=True, oceancolor="#dcecf7")
        fig.update_layout(height=420, margin=dict(l=0, r=0, t=45, b=0))
        column.plotly_chart(fig, use_container_width=True)


def render_overview(frames: dict[str, pd.DataFrame]) -> None:
    render_intro()
    early = frames[EARLY_LABEL]
    late = frames[LATE_LABEL]
    early_mean = float(early["wind_speed_10m_ms"].mean())
    late_mean = float(late["wind_speed_10m_ms"].mean())
    early_wpd = float(early["wind_power_density_10m_wm2"].mean())
    late_wpd = float(late["wind_power_density_10m_wm2"].mean())
    section = st.sidebar.radio(
        "Overview section",
        ["Geography and sites", "Downloaded variables", "Integrated comparison"],
        key="overview_section",
    )
    if section == "Geography and sites":
        st.subheader("North Sea geography and reference sites")
        st.caption("Two lightweight raster-style figures: official EEZ/reference-site context and EMODnet bathymetry. Political boundaries are drawn on top and both figures use longitude/latitude axes.")
        render_static_context_figures()
        return

    if section == "Downloaded variables":
        render_variable_snapshot(frames)
        return

    comparison = early.groupby(["lat", "lon"], as_index=False).agg(
        early_wpd=("wind_power_density_10m_wm2", "mean"),
        early_wind=("wind_speed_10m_ms", "mean"),
    )
    later_map = late.groupby(["lat", "lon"], as_index=False).agg(
        late_wpd=("wind_power_density_10m_wm2", "mean"),
        late_wind=("wind_speed_10m_ms", "mean"),
    )
    comparison = comparison.merge(later_map, on=["lat", "lon"])
    comparison["wpd_change"] = comparison["late_wpd"] - comparison["early_wpd"]
    comparison["wind_change"] = comparison["late_wind"] - comparison["early_wind"]
    st.subheader("Integrated period comparison")
    st.caption("The corrected expanded EERIE bounding box is −5° to 13°E and 50° to 62°N. The app uses finite SST as a preliminary ocean proxy; a hydrographic North Sea mask is still required for final marine-only analysis.")
    cards = st.columns(4)
    cards[0].metric("2020–2024 mean wind", f"{early_mean:.2f} m/s")
    cards[1].metric("2036–2040 mean wind", f"{late_mean:.2f} m/s", format_change(late_mean - early_mean, " m/s"))
    cards[2].metric("2020–2024 mean WPD", f"{early_wpd:.1f} W/m²")
    cards[3].metric("2036–2040 mean WPD", f"{late_wpd:.1f} W/m²", format_change(late_wpd - early_wpd, " W/m²"))
    with st.expander("How is WPD calculated?", expanded=False):
        st.latex(r"\mathrm{WPD}=\frac{1}{2}\rho v^3")
        st.write("Here, v = sqrt(m10u² + m10v²) is the 10 m wind speed, and rho = p/(R_d T) is air density from pressure and temperature. WPD is calculated at each grid point and day before the period mean is taken.")
    st.markdown("**Three-panel spatial comparison: early mean, later mean, and later minus early**")
    st.caption("Panel 1 is the January 2020–2024 mean WPD; Panel 2 is the January 2036–2040 mean WPD; Panel 3 is Panel 2 minus Panel 1. The third panel shows change, not absolute wind resource.")
    panel_specs = [("early_wpd", "Mean WPD: 2020–2024", "Viridis", [0, 1300]), ("late_wpd", "Mean WPD: 2036–2040", "Viridis", [0, 1300]), ("wpd_change", "WPD change: later minus early", "RdBu_r", [-200, 200])]
    panel_columns = st.columns(3)
    for column, (metric, title, colors, value_range) in zip(panel_columns, panel_specs):
        fig = px.scatter_geo(comparison, lat="lat", lon="lon", color=metric, hover_data={"lat": ":.2f", "lon": ":.2f", "early_wpd": ":.1f", "late_wpd": ":.1f", "wpd_change": ":.1f"}, color_continuous_scale=colors, range_color=value_range, labels={metric: "W/m²"}, projection="mercator", title=title)
        fig.update_geos(lonaxis_range=[-5, 13], lataxis_range=[50, 62], showcountries=True, countrycolor="rgba(20, 30, 45, 0.75)", showcoastlines=True, showland=True, landcolor="#eef1e8", showocean=True, oceancolor="#dcecf7")
        fig.update_layout(height=430, margin=dict(l=0, r=0, t=55, b=0))
        column.plotly_chart(fig, use_container_width=True)

    st.markdown("**Three-panel mean-wind comparison**")
    st.caption("Mean 10 m wind speed uses the same early, later, and difference structure as the WPD row. Mean-wind scales are fixed at 0–15 m/s and −5 to +5 m/s for the difference.")
    wind_specs = [("early_wind", "Mean wind: 2020–2024", "Viridis", [0, 15]), ("late_wind", "Mean wind: 2036–2040", "Viridis", [0, 15]), ("wind_change", "Wind change: later minus early", "RdBu_r", [-5, 5])]
    wind_columns = st.columns(3)
    for column, (metric, title, colors, value_range) in zip(wind_columns, wind_specs):
        fig = px.scatter_geo(comparison, lat="lat", lon="lon", color=metric, hover_data={"lat": ":.2f", "lon": ":.2f", "early_wind": ":.2f", "late_wind": ":.2f", "wind_change": ":.2f"}, color_continuous_scale=colors, range_color=value_range, labels={metric: "m/s"}, projection="mercator", title=title)
        fig.update_geos(lonaxis_range=[-5, 13], lataxis_range=[50, 62], showcountries=True, countrycolor="rgba(20, 30, 45, 0.75)", showcoastlines=True, showland=True, landcolor="#eef1e8", showocean=True, oceancolor="#dcecf7")
        fig.update_layout(height=390, margin=dict(l=0, r=0, t=55, b=0))
        column.plotly_chart(fig, use_container_width=True)

    early_daily = daily_summary(early).assign(period=EARLY_LABEL)
    late_daily = daily_summary(late).assign(period=LATE_LABEL)
    daily = pd.concat([early_daily, late_daily], ignore_index=True)
    comparison_metric = st.selectbox("Daily distribution", ["mean_wind_speed_10m_ms", "mean_wpd_wm2", "p90_wind_speed_10m_ms"], format_func=lambda value: {"mean_wind_speed_10m_ms": "Mean daily wind speed", "mean_wpd_wm2": "Mean daily WPD", "p90_wind_speed_10m_ms": "Daily P90 wind speed"}[value])
    st.plotly_chart(px.box(daily, x="period", y=comparison_metric, color="period", points="outliers", title="January daily distribution"), use_container_width=True)
    st.warning("The two windows contain five Januarys each. This is an exploratory model comparison, not a definitive climate trend or bankable energy forecast.")


def render_site_explorer(frames: dict[str, pd.DataFrame]) -> None:
    st.header("Site explorer")
    site_name = st.selectbox("Choose a reference site", list(SITES))
    selected_period = st.radio("Period", [EARLY_LABEL, LATE_LABEL], horizontal=True)
    site_info = SITES[site_name]
    frame = frames[selected_period]
    points = frame[["lat", "lon"]].drop_duplicates()
    nearest = points.iloc[((points["lat"] - site_info["lat"]) ** 2 + (points["lon"] - site_info["lon"]) ** 2).argmin()]
    site = frame[(frame["lat"] == nearest["lat"]) & (frame["lon"] == nearest["lon"])].copy()
    site_daily = site.groupby("time", as_index=False).agg(
        wind_speed_10m_ms=("wind_speed_10m_ms", "mean"),
        wpd_wm2=("wind_power_density_10m_wm2", "mean"),
    )
    site_daily["year"] = site_daily["time"].dt.year
    site_daily["january_day"] = site_daily["time"].dt.day

    st.markdown(f"### {site_name}")
    st.write(f"**Reference:** {site_info['reference']} ({site_info['country']})")
    st.caption(f"Nearest EERIE native grid point: {nearest['lat']:.3f}°N, {nearest['lon']:.3f}°E")
    cards = st.columns(3)
    cards[0].metric("Mean 10 m wind", f"{site_daily.wind_speed_10m_ms.mean():.2f} m/s")
    cards[1].metric("Mean 10 m WPD", f"{site_daily.wpd_wm2.mean():.1f} W/m²")
    cards[2].metric("Daily records", f"{len(site_daily)}")

    st.subheader("January day-of-month profile")
    st.caption("The x-axis is day of January, not continuous calendar time. Each colored line is a separate January year; the plot does not connect January 31 to the following January 1.")
    variable = st.selectbox("Site metric", ["wind_speed_10m_ms", "wpd_wm2"], format_func=lambda value: "10 m wind speed" if value == "wind_speed_10m_ms" else "10 m wind-power density")
    fig = px.line(site_daily, x="january_day", y=variable, color="year", markers=True, labels={"january_day": "Day of January", variable: "Value"})
    fig.update_xaxes(dtick=5, range=[1, 31])
    fig.update_layout(height=470, legend_title="Year", margin=dict(l=0, r=0, t=25, b=0))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Hub-height sensitivity")
    st.latex(r"v(z)=v(10)\left(\frac{z}{10}\right)^\alpha")
    st.caption("The power-law shear exponent α describes how wind speed increases with height. It matters because turbines operate around 100–150 m, while this dataset directly provides 10 m wind. The exponent is an assumption, not a direct model-level observation.")
    shear = st.slider("Power-law shear exponent α", 0.05, 0.20, 0.12, 0.01, key=f"shear_{site_name}_{selected_period}")
    estimated = site_daily.wind_speed_10m_ms * (100 / 10) ** shear
    st.write(f"Estimated mean 100 m wind: **{estimated.mean():.2f} m/s**")
    st.warning("This is an assumption-based extrapolation from 10 m wind, not direct model-level 100 m wind.")


def dispatch_tool(toolkit: JanuaryAgentToolkit, name: str, arguments: dict) -> dict:
    if name == "check_data_availability":
        return toolkit.check_data_availability(arguments.get("requested_period"))
    if name == "compare_january_periods":
        return toolkit.compare_periods()
    if name == "get_january_period_summary":
        return toolkit.period_summary(arguments["period"])
    if name == "calculate_spatial_difference":
        return toolkit.calculate_spatial_difference(arguments.get("metric", "wind_power_density_10m_wm2"))
    if name == "identify_top_regions":
        return toolkit.identify_top_regions(arguments.get("metric", "wind_power_density_10m_wm2"))
    if name == "compare_reference_sites":
        return toolkit.compare_reference_sites(arguments.get("period", LATE_LABEL), arguments.get("metric", "mean_wpd_wm2"))
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
    return {"error": f"Unknown tool: {name}"}


def deterministic_agent(toolkit: JanuaryAgentToolkit, question: str) -> tuple[str, list[dict[str, Any]]]:
    lower = question.lower()
    trace = [{"step": 1, "action": "Interpret user goal", "detail": question}]
    if any(word in lower for word in ["promising", "recommend", "best region", "which region", "most suitable"]):
        workflow = toolkit.recommend_region()
        for item in workflow["trace"]:
            trace.append({"step": len(trace) + 1, "action": "Call bounded tool", "tool": item["tool"]})
        ranking = workflow["recommendation"]
        answer = "**Multi-step screening recommendation**\n\n"
        if ranking:
            answer += "The highest-ranked reference sites in the later period by mean WPD are:\n\n"
            answer += "\n".join(f"- **{row['site']}**: {row['mean_wpd_wm2']:.1f} W/m² at 10 m" for row in ranking)
        answer += "\n\nThis is a screening result. It is not a turbine-siting or investment recommendation."
    elif any(word in lower for word in ["height", "100 m", "hub"]):
        tool_name = "test_hub_height_sensitivity"
        args = {"period": EARLY_LABEL}
        trace.append({"step": 2, "action": "Call bounded tool", "tool": tool_name, "arguments": args})
        result = dispatch_tool(toolkit, tool_name, args)
        rows = result.get("results", [])
        answer = "\n".join(f"At shear {row['shear_exponent']:.2f}, estimated 100 m CF is {row['capacity_factor']:.1%}." for row in rows)
        answer += "\n\nThese are sensitivity estimates, not direct 100 m winds or bankable capacity factors."
    elif any(word in lower for word in ["limit", "valid", "bank"]):
        tool_name = "explain_analysis_limitations"
        args = {"data_frequency": "daily"}
        trace.append({"step": 2, "action": "Call bounded tool", "tool": tool_name, "arguments": args})
        result = dispatch_tool(toolkit, tool_name, args)
        answer = "\n".join(f"- {item}" for item in result["caveats"])
    else:
        tool_name = "compare_january_periods"
        args = {}
        trace.append({"step": 2, "action": "Call bounded tool", "tool": tool_name, "arguments": args})
        result = dispatch_tool(toolkit, tool_name, args)
        changes = result.get("changes", {})
        if changes:
            answer = (
                f"Mean 10 m wind changes from {changes['mean_wind_speed_ms']['early']:.2f} to "
                f"{changes['mean_wind_speed_ms']['late']:.2f} m/s "
                f"({changes['mean_wind_speed_ms']['relative_change_percent']:+.2f}%).\n\n"
                f"Mean WPD changes from {changes['mean_wpd_wm2']['early']:.1f} to "
                f"{changes['mean_wpd_wm2']['late']:.1f} W/m² "
                f"({changes['mean_wpd_wm2']['relative_change_percent']:+.2f}%)."
            )
        else:
            answer = result.get("error", "The requested comparison is unavailable.")
    trace.append({"step": len(trace) + 1, "action": "Apply scientific guardrails", "detail": "Daily January, five years per window, 10 m wind, one model realization."})
    trace.append({"step": len(trace) + 1, "action": "Return evidence-based answer"})
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


def render_agent(frames: dict[str, pd.DataFrame]) -> None:
    st.header("Agent workspace")
    st.write("Ask a scientific or wind-energy question. The agent interprets the goal, calls bounded analysis tools, shows the execution trace, and states what the data can support.")
    st.success("The LLM is an orchestrator of validated climate and wind-analysis tools, not a free-form answer generator.")
    st.info("Type a climate/wind question OR copy a sample question. Choose Gemini, Groq, or the deterministic tool demo in the sidebar.")
    toolkit = JanuaryAgentToolkit(frames)

    sample_questions = [
        ("Period comparison", "Compare January 2020–2024 and January 2036–2040, then explain whether the WPD change is consistent with the wind-speed change."),
        ("Regional recommendation [LLM]", "Which North Sea region is most promising for preliminary offshore wind screening? Use spatial change, site ranking, hub-height sensitivity, and limitations."),
        ("Site ranking [LLM]", "Compare Dogger Bank, Hornsea, German Bight, and Moray Firth in the later period. Rank them by WPD and explain the caveats."),
        ("Largest change [LLM]", "Which region has the largest January change in wind-power density, and is that more useful for climate analysis or wind-project screening?"),
        ("Hub-height sensitivity", "How sensitive is the estimated 100 m wind and illustrative capacity factor to the shear exponent?"),
        ("Bankability audit [LLM]", "Can this result support a bankable wind-farm decision? Audit the temporal resolution, height, model, mask, and missing engineering inputs."),
        ("Climate scientist audit [LLM]", "Which variables and assumptions should a climate scientist audit before interpreting the January period difference?"),
        ("Unsupported request [LLM]", "Can you compare July 2020–2024 with July 2036–2040 and explain what is currently prepared versus what is exposed in the prototype?"),
        ("Data provenance", "Where did the EERIE data come from, what is SSP2-4.5, and what variables were prepared for this analysis?"),
        ("Metric selection [LLM]", "Which wind-energy metrics can be calculated from this dataset, which require hub-height assumptions, and which cannot be trusted from daily means?"),
    ]

    provider_choice = st.sidebar.selectbox("Agent provider", ["Deterministic tool demo", *PROVIDERS.keys()], key="agent_provider", help="Gemini and Groq keys are read from Streamlit secrets or environment variables.")
    provider = provider_choice if provider_choice in PROVIDERS else None
    model = None
    api_key = None
    if provider:
        model = st.sidebar.selectbox("Model", PROVIDERS[provider]["models"], key=f"model_{provider}")
        configured = bool(get_configured_key(provider))
        st.sidebar.caption(f"{provider} key: {'configured' if configured else 'not configured'}")
        if not configured:
            api_key = st.sidebar.text_input(f"{provider} API key (session-only)", type="password", key=f"key_{provider}") or None

    agent_column, right_column = st.columns([1.65, 1.35])
    with agent_column:
        question = st.text_area("Ask the North Sea agent", key="agent_question", height=110, placeholder="Type a climate/wind question OR copy a sample question...")
        run_agent = st.button("Run agent", type="primary")
        with st.expander("Agent contract and available tools", expanded=False):
            st.json(toolkit.analysis_contract())
            st.write([tool["function"]["name"] for tool in toolkit.tool_definitions()])
        if run_agent and question.strip():
            st.chat_message("user").write(question)
            try:
                if provider and (api_key or get_configured_key(provider)):
                    answer, trace = provider_agent(toolkit, provider, model, question, api_key=api_key)
                else:
                    answer, trace = deterministic_agent(toolkit, question)
                    if provider:
                        trace.insert(1, {"step": 2, "action": "Provider key unavailable; used deterministic fallback", "provider": provider})
                with st.expander("Agent execution trace", expanded=True):
                    for item in trace:
                        st.write(item)
                st.chat_message("assistant").markdown(answer)
            except Exception as exc:
                st.error(f"Agent execution failed: {exc}")
        with st.expander("Wind-energy metric coverage", expanded=False):
            st.markdown("**Available from the prepared data**")
            st.write("Mean wind speed, wind direction, air density, WPD, percentiles, wind distributions, Weibull diagnostics, SST context, and 10 m-to-100 m sensitivity.")
            st.markdown("**Not supported as bankable outputs**")
            st.write("Direct 100 m wind, hourly capacity factor, gust loads, wake losses, availability, curtailment, and electrical losses.")

    with right_column:
        st.markdown("#### Sample questions")
        st.caption("Copy any complete prompt below. Prompts marked [LLM] are especially useful for Gemini/Groq multi-step reasoning.")
        st.markdown("<style>div[data-testid='stCodeBlock'] pre {font-size: 0.80rem !important; line-height: 1.25 !important; padding: 0.65rem !important;}</style>", unsafe_allow_html=True)
        for title, sample in sample_questions:
            st.markdown(f"**{title}**")
            st.code(sample, language=None)
        with st.container(border=True):
            st.markdown("#### Scientific guardrails")
            guardrails = scientific_guardrails() + [
                "Expanded bounding box; finite SST is only a preliminary ocean proxy.",
                "No final hydrographic North Sea polygon has been applied yet.",
                "SST supports climate interpretation but is not direct turbine-production input.",
                "July is prepared for future seasonal expansion, not used in the January prototype.",
            ]
            for caveat in guardrails:
                st.caption(f"• {caveat}")
    st.sidebar.divider()
    st.sidebar.markdown("**Agent workspace tools**")
    st.sidebar.write("• Check data availability\n• Find spatial changes\n• Rank reference sites\n• Test hub-height assumptions\n• Explain limitations")


def render_sidebar() -> None:
    st.sidebar.markdown("# North Sea Wind Intelligence")
    st.sidebar.caption("EERIE climate model + wind-resource analytics + agentic AI")
    st.sidebar.divider()
    st.sidebar.markdown("**Prototype scope**")
    st.sidebar.write("January 2020–2024 versus January 2036–2040")
    st.sidebar.write("North Sea: −5–13°E, 50–62°N")
    st.sidebar.write("Daily native-grid wind at 10 m")
    st.sidebar.divider()
    st.sidebar.markdown("**Data status**")
    st.sidebar.success("Prepared January package loaded")
    st.sidebar.caption("The app uses prepared data and does not download raw EERIE chunks during a session.")


def main() -> None:
    if not check_dependencies() or not period_files_available():
        return
    render_sidebar()
    frames = {label: load_period_data(str(path)) for label, path in PERIOD_FILES.items()}
    page = st.sidebar.radio("Workspace", ["Agent workspace", "Overview & comparison", "Site explorer"], key="workspace_page")
    if page == "Agent workspace":
        render_agent(frames)
    elif page == "Overview & comparison":
        render_overview(frames)
    else:
        render_site_explorer(frames)


if __name__ == "__main__":
    main()
