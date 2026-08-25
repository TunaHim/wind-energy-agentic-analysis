"""
Wind energy calculations and metrics.

This module provides functions for computing wind energy resource metrics
from climate model data, including:
- Wind power density (WPD)
- Weibull distribution fitting
- Capacity factor (CF)
- Annual energy production (AEP)
- Extreme wind statistics
"""

import numpy as np
import pandas as pd
from scipy.stats import weibull_min
from scipy.optimize import curve_fit
import warnings

warnings.filterwarnings('ignore')


# NREL 15 MW IEA Reference Turbine Power Curve
# Source: https://www.nrel.gov/docs/fy20osti/77014.pdf
REFERENCE_TURBINE = {
    'name': 'NREL 15 MW IEA',
    'rated_power': 15.0,  # MW
    'hub_height': 120,  # m
    'rotor_diameter': 178,  # m
    'power_curve': {
        'wind_speed': np.array([0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 20, 25, 30]),
        'power': np.array([0, 0, 0.5, 1.5, 3, 5, 7, 9, 11, 12.5, 14, 14.8, 15, 15, 15, 15, 0])
    }
}


def compute_wind_speed(u, v):
    """
    Compute wind speed magnitude from u/v components.
    
    Parameters
    ----------
    u : array-like
        U-component of wind (m/s)
    v : array-like
        V-component of wind (m/s)
    
    Returns
    -------
    wind_speed : array-like
        Wind speed magnitude (m/s)
    """
    return np.sqrt(u**2 + v**2)


def compute_wind_direction(u, v):
    """
    Compute wind direction from u/v components (meteorological convention).
    
    Meteorological convention: 0° = wind from North, 90° = wind from East
    
    Parameters
    ----------
    u : array-like
        U-component of wind (m/s)
    v : array-like
        V-component of wind (m/s)
    
    Returns
    -------
    direction : array-like
        Wind direction (degrees, 0-360)
    """
    direction = np.arctan2(u, v) * 180 / np.pi
    direction = (direction + 360) % 360
    return direction


def compute_air_density(T, P):
    """
    Compute air density from temperature and pressure using ideal gas law.
    
    ρ = P / (R_specific * T)
    
    Parameters
    ----------
    T : array-like
        Temperature (K)
    P : array-like
        Pressure (Pa)
    
    Returns
    -------
    rho : array-like
        Air density (kg/m³)
    """
    R_specific = 287  # J/(kg·K) for dry air
    rho = P / (R_specific * T)
    return rho


def compute_wind_power_density(wind_speed, air_density):
    """
    Compute wind power density.
    
    WPD = 0.5 * ρ * v³ (W/m²)
    
    Parameters
    ----------
    wind_speed : array-like
        Wind speed (m/s)
    air_density : array-like
        Air density (kg/m³)
    
    Returns
    -------
    wpd : array-like
        Wind power density (W/m²)
    """
    return 0.5 * air_density * wind_speed**3


def fit_weibull(wind_speeds):
    """
    Fit Weibull distribution to wind speed data.
    
    Uses maximum likelihood estimation via scipy.stats.weibull_min.
    
    Parameters
    ----------
    wind_speeds : array-like
        Wind speed time series (m/s)
    
    Returns
    -------
    k : float
        Weibull shape parameter
    A : float
        Weibull scale parameter
    mean_speed : float
        Mean wind speed from data (m/s)
    """
    # Remove NaN and zero values
    ws = np.asarray(wind_speeds)
    ws = ws[~np.isnan(ws)]
    ws = ws[ws > 0]
    
    if len(ws) < 10:
        return np.nan, np.nan, np.nan
    
    try:
        # weibull_min.fit returns (k, loc, scale)
        # We use loc=0 (shape, scale) parameterization
        k, loc, A = weibull_min.fit(ws, floc=0)
        mean_speed = np.mean(ws)
        return float(k), float(A), float(mean_speed)
    except Exception as e:
        print(f"Warning: Weibull fitting failed: {e}")
        return np.nan, np.nan, np.nan


def get_power_output(wind_speed, turbine=REFERENCE_TURBINE):
    """
    Get power output from wind speed using turbine power curve.
    
    Parameters
    ----------
    wind_speed : array-like
        Wind speed (m/s)
    turbine : dict
        Turbine specification with 'power_curve' key
    
    Returns
    -------
    power : array-like
        Power output (MW)
    """
    pc = turbine['power_curve']
    ws = pc['wind_speed']
    p = pc['power']
    
    # Interpolate (linear)
    power = np.interp(wind_speed, ws, p, left=0, right=0)
    return power


def compute_capacity_factor(wind_speeds, turbine=REFERENCE_TURBINE):
    """
    Compute capacity factor from wind speed time series.
    
    CF = mean(power) / rated_power
    
    Parameters
    ----------
    wind_speeds : array-like
        Wind speed time series (m/s)
    turbine : dict
        Turbine specification
    
    Returns
    -------
    cf : float
        Capacity factor (0-1)
    """
    power = get_power_output(wind_speeds, turbine)
    rated_power = turbine['rated_power']
    cf = np.nanmean(power) / rated_power
    return float(cf)


def compute_aep(wind_speeds, installed_capacity_mw=15.0, turbine=REFERENCE_TURBINE):
    """
    Compute annual energy production (AEP).
    
    AEP = CF * P_rated * 8760 hours
    
    Parameters
    ----------
    wind_speeds : array-like
        Wind speed time series (m/s)
    installed_capacity_mw : float
        Installed capacity (MW), default 15 MW
    turbine : dict
        Turbine specification
    
    Returns
    -------
    aep : float
        Annual energy production (MWh/year)
    """
    cf = compute_capacity_factor(wind_speeds, turbine)
    aep = cf * installed_capacity_mw * 8760
    return float(aep)


def compute_extreme_wind_return_period(wind_speeds, return_period_years=50, samples_per_year=None):
    """
    Estimate extreme wind speed at a given return period.

    Uses Weibull distribution to extrapolate to rare events.

    Parameters
    ----------
    wind_speeds : array-like
        Wind speed time series (m/s)
    return_period_years : int
        Return period (years), default 50
    samples_per_year : int or None
        Number of samples per year. If None, it is inferred from a one-year
        time series when available, otherwise defaults to 365 (daily).

    Returns
    -------
    extreme_wind : float
        Estimated wind speed at return period (m/s)
    """
    k, A, _ = fit_weibull(wind_speeds)

    if np.isnan(k) or np.isnan(A):
        return np.nan

    # For Weibull distribution:
    # Probability of exceeding v: P(V > v) = exp(-(v/A)^k)
    # Return period T (years) corresponds to probability p = 1 / (T * n_samples_per_year)
    n_samples_per_year = samples_per_year if samples_per_year is not None else 365
    p = 1 / (return_period_years * n_samples_per_year)

    # Solve for v: exp(-(v/A)^k) = p
    # (v/A)^k = -ln(p)
    # v = A * (-ln(p))^(1/k)
    extreme_wind = A * (-np.log(p))**(1/k)

    return float(extreme_wind)


def compute_wind_rose(wind_speeds, wind_directions, n_bins=16):
    """
    Compute wind rose statistics (frequency by direction).
    
    Parameters
    ----------
    wind_speeds : array-like
        Wind speed time series (m/s)
    wind_directions : array-like
        Wind direction time series (degrees)
    n_bins : int
        Number of direction bins (default 16 = 22.5° bins)
    
    Returns
    -------
    wind_rose : dict
        Dictionary with 'directions', 'frequencies', 'mean_speeds'
    """
    ws = np.asarray(wind_speeds)
    wd = np.asarray(wind_directions)
    
    # Remove NaN
    mask = ~(np.isnan(ws) | np.isnan(wd))
    ws = ws[mask]
    wd = wd[mask]
    
    # Bin directions
    bin_edges = np.linspace(0, 360, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    frequencies = np.zeros(n_bins)
    mean_speeds = np.zeros(n_bins)
    
    for i in range(n_bins):
        # Handle wrap-around at 360°
        if i == 0:
            mask_bin = (wd >= bin_edges[i]) | (wd < bin_edges[1])
        else:
            mask_bin = (wd >= bin_edges[i]) & (wd < bin_edges[i+1])
        
        frequencies[i] = np.sum(mask_bin) / len(wd)
        if np.sum(mask_bin) > 0:
            mean_speeds[i] = np.mean(ws[mask_bin])
    
    return {
        'directions': bin_centers,
        'frequencies': frequencies,
        'mean_speeds': mean_speeds
    }


def compute_seasonal_statistics(wind_speeds, times):
    """
    Compute seasonal wind statistics.
    
    Parameters
    ----------
    wind_speeds : array-like
        Wind speed time series (m/s)
    times : array-like
        Time indices (datetime or similar)
    
    Returns
    -------
    seasonal_stats : dict
        Dictionary with seasonal means, WPD, CF
    """
    df = pd.DataFrame({
        'time': times,
        'wind_speed': wind_speeds
    })
    df['time'] = pd.to_datetime(df['time'])
    df['month'] = df['time'].dt.month
    
    # Define seasons (Northern Hemisphere)
    def get_season(month):
        if month in [12, 1, 2]:
            return 'Winter'
        elif month in [3, 4, 5]:
            return 'Spring'
        elif month in [6, 7, 8]:
            return 'Summer'
        else:
            return 'Fall'
    
    df['season'] = df['month'].apply(get_season)
    
    seasonal_stats = {}
    for season in ['Winter', 'Spring', 'Summer', 'Fall']:
        season_data = df[df['season'] == season]['wind_speed']
        if len(season_data) > 0:
            seasonal_stats[season] = {
                'mean_wind_speed': float(np.mean(season_data)),
                'std_wind_speed': float(np.std(season_data)),
                'min_wind_speed': float(np.min(season_data)),
                'max_wind_speed': float(np.max(season_data))
            }
    
    return seasonal_stats


def extrapolate_wind_speed_power_law(wind_speed_10m, target_height_m=100.0, shear_exponent=0.12):
    """Extrapolate wind speed from 10 m to a target height using a power law.

    This is an assumption-sensitive screening method, not a replacement for
    model-level interpolation or site-specific measurement campaign data.
    """
    if target_height_m <= 10 or shear_exponent < 0:
        raise ValueError("target_height_m must exceed 10 m and shear_exponent must be non-negative")
    return np.asarray(wind_speed_10m) * (target_height_m / 10.0) ** shear_exponent


def height_sensitivity(wind_speed_10m, heights=(120.0,), shear_exponents=(0.08, 0.12, 0.16), turbine=REFERENCE_TURBINE):
    """Calculate capacity-factor sensitivity to hub-height assumptions."""
    rows = []
    for height in heights:
        for exponent in shear_exponents:
            wind_at_height = extrapolate_wind_speed_power_law(wind_speed_10m, height, exponent)
            rows.append({
                'height_m': float(height),
                'shear_exponent': float(exponent),
                'mean_wind_speed_ms': float(np.nanmean(wind_at_height)),
                'capacity_factor': compute_capacity_factor(wind_at_height, turbine),
            })
    return pd.DataFrame(rows)


def compute_interannual_trend(wind_speeds, times):
    """
    Compute interannual trend in wind speed.
    
    Uses linear regression to estimate trend (m/s per year).
    
    Parameters
    ----------
    wind_speeds : array-like
        Wind speed time series (m/s)
    times : array-like
        Time indices (datetime or similar)
    
    Returns
    -------
    trend_dict : dict
        Dictionary with 'slope' (m/s/year), 'intercept', 'r_squared'
    """
    df = pd.DataFrame({
        'time': times,
        'wind_speed': wind_speeds
    })
    df['time'] = pd.to_datetime(df['time'])
    df['year'] = df['time'].dt.year
    
    # Annual means
    annual_means = df.groupby('year')['wind_speed'].mean()
    
    if len(annual_means) < 3:
        return {'slope': np.nan, 'intercept': np.nan, 'r_squared': np.nan}
    
    # Linear regression
    x = np.arange(len(annual_means))
    y = annual_means.values
    
    # Remove NaN
    mask = ~np.isnan(y)
    x = x[mask]
    y = y[mask]
    
    if len(x) < 3:
        return {'slope': np.nan, 'intercept': np.nan, 'r_squared': np.nan}
    
    # Fit line
    coeffs = np.polyfit(x, y, 1)
    slope = coeffs[0]
    intercept = coeffs[1]
    
    # R-squared
    y_pred = np.polyval(coeffs, x)
    ss_res = np.sum((y - y_pred)**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else np.nan
    
    return {
        'slope': float(slope),
        'intercept': float(intercept),
        'r_squared': float(r_squared)
    }
