"""
Utility functions for data loading, formatting, and visualization.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')


def load_precomputed_stats(data_dir: Path = Path('data')) -> Dict:
    """
    Load precomputed statistics from JSON file.
    
    Parameters
    ----------
    data_dir : Path
        Directory containing precomputed data
    
    Returns
    -------
    stats : dict
        Precomputed statistics for all sites
    """
    stats_file = Path(data_dir) / 'precomputed_stats.json'
    
    if not stats_file.exists():
        print(f"Warning: {stats_file} not found.")
        return {}
    
    try:
        with open(stats_file, 'r') as f:
            stats = json.load(f)
        return stats
    except Exception as e:
        print(f"Error loading stats: {e}")
        return {}


def load_climatology(data_dir: Path = Path('data')):
    """
    Load climatology NetCDF file.
    
    Parameters
    ----------
    data_dir : Path
        Directory containing precomputed data
    
    Returns
    -------
    ds : xarray.Dataset or None
        Climatology dataset
    """
    try:
        import xarray as xr
    except ImportError:
        print("xarray not installed. Cannot load climatology.")
        return None
    
    clim_file = Path(data_dir) / 'north_sea_climatology.nc'
    
    if not clim_file.exists():
        print(f"Warning: {clim_file} not found.")
        return None
    
    try:
        ds = xr.open_dataset(clim_file)
        return ds
    except Exception as e:
        print(f"Error loading climatology: {e}")
        return None


def load_timeseries(data_dir: Path = Path('data')):
    """
    Load time series Parquet file.
    
    Parameters
    ----------
    data_dir : Path
        Directory containing precomputed data
    
    Returns
    -------
    df : pandas.DataFrame or None
        Time series data
    """
    ts_file = Path(data_dir) / 'site_timeseries.parquet'
    
    if not ts_file.exists():
        print(f"Warning: {ts_file} not found.")
        return None
    
    try:
        df = pd.read_parquet(ts_file)
        return df
    except Exception as e:
        print(f"Error loading time series: {e}")
        return None


def format_metric(value: float, unit: str = '', decimals: int = 2) -> str:
    """
    Format a metric for display.
    
    Parameters
    ----------
    value : float
        Metric value
    unit : str
        Unit string
    decimals : int
        Number of decimal places
    
    Returns
    -------
    formatted : str
        Formatted string
    """
    if np.isnan(value):
        return 'N/A'
    
    formatted = f"{value:.{decimals}f}"
    if unit:
        formatted += f" {unit}"
    
    return formatted


def create_site_summary(site_name: str, stats: Dict) -> str:
    """
    Create a human-readable summary of site statistics.
    
    Parameters
    ----------
    site_name : str
        Site name
    stats : dict
        Site statistics
    
    Returns
    -------
    summary : str
        Formatted summary
    """
    summary = f"""
    **{site_name}**
    
    Location: {stats.get('lon', 'N/A')}°E, {stats.get('lat', 'N/A')}°N ({stats.get('country', 'N/A')})
    
    **Wind Resource:**
    - Mean wind speed: {format_metric(stats.get('mean_wind_speed', 0), 'm/s')}
    - Wind power density: {format_metric(stats.get('wpd', 0), 'W/m²', 0)}
    - Capacity factor: {format_metric(stats.get('cf', 0) * 100, '%', 1)}
    
    **Weibull Distribution:**
    - Shape parameter (k): {format_metric(stats.get('weibull_k', 0), '', 2)}
    - Scale parameter (A): {format_metric(stats.get('weibull_a', 0), 'm/s', 2)}
    
    **Energy Production (15 MW turbine):**
    - Annual energy production: {format_metric(stats.get('aep', 0), 'MWh/year', 0)}
    
    **Extreme Wind (50-year return period):**
    - Estimated wind speed: {format_metric(stats.get('extreme_wind_50yr', 0), 'm/s', 1)}
    
    **Trend (1995-2014):**
    - Interannual trend: {format_metric(stats.get('trend', 0), 'm/s per year', 4)}
    """
    
    return summary.strip()


def get_color_for_cf(cf: float) -> str:
    """
    Get color for capacity factor visualization.
    
    Parameters
    ----------
    cf : float
        Capacity factor (0-1)
    
    Returns
    -------
    color : str
        Hex color code
    """
    if cf < 0.25:
        return '#d73027'  # Red
    elif cf < 0.35:
        return '#fc8d59'  # Orange
    elif cf < 0.45:
        return '#fee090'  # Yellow
    elif cf < 0.50:
        return '#91bfdb'  # Light blue
    else:
        return '#4575b4'  # Dark blue


def get_color_for_wpd(wpd: float) -> str:
    """
    Get color for wind power density visualization.
    
    Parameters
    ----------
    wpd : float
        Wind power density (W/m²)
    
    Returns
    -------
    color : str
        Hex color code
    """
    if wpd < 100:
        return '#d73027'  # Red
    elif wpd < 200:
        return '#fc8d59'  # Orange
    elif wpd < 300:
        return '#fee090'  # Yellow
    elif wpd < 400:
        return '#91bfdb'  # Light blue
    else:
        return '#4575b4'  # Dark blue


def create_comparison_table(site1_stats: Dict, site2_stats: Dict) -> pd.DataFrame:
    """
    Create a comparison table for two sites.
    
    Parameters
    ----------
    site1_stats : dict
        Statistics for site 1
    site2_stats : dict
        Statistics for site 2
    
    Returns
    -------
    df : pandas.DataFrame
        Comparison table
    """
    metrics = [
        ('Mean wind speed (m/s)', 'mean_wind_speed'),
        ('Wind power density (W/m²)', 'wpd'),
        ('Capacity factor (%)', 'cf', lambda x: x * 100),
        ('Weibull shape (k)', 'weibull_k'),
        ('Weibull scale (A)', 'weibull_a'),
        ('Annual energy (MWh)', 'aep'),
        ('Extreme wind 50yr (m/s)', 'extreme_wind_50yr'),
        ('Trend (m/s/year)', 'trend')
    ]
    
    data = []
    for metric_info in metrics:
        metric_name = metric_info[0]
        key = metric_info[1]
        transform = metric_info[2] if len(metric_info) > 2 else lambda x: x
        
        val1 = site1_stats.get(key, np.nan)
        val2 = site2_stats.get(key, np.nan)
        
        if not np.isnan(val1):
            val1 = transform(val1)
        if not np.isnan(val2):
            val2 = transform(val2)
        
        data.append({
            'Metric': metric_name,
            'Site 1': f"{val1:.2f}" if not np.isnan(val1) else 'N/A',
            'Site 2': f"{val2:.2f}" if not np.isnan(val2) else 'N/A'
        })
    
    return pd.DataFrame(data)


def validate_api_key(api_key: str, provider: str = 'openai') -> Tuple[bool, str]:
    """
    Validate API key format (basic check).
    
    Parameters
    ----------
    api_key : str
        API key to validate
    provider : str
        Provider ('openai' or 'anthropic')
    
    Returns
    -------
    is_valid : bool
        Whether key format is valid
    message : str
        Validation message
    """
    if not api_key or len(api_key) < 10:
        return False, "API key is too short."
    
    if provider == 'openai':
        if api_key.startswith('sk-'):
            return True, "OpenAI API key format looks valid."
        else:
            return False, "OpenAI API key should start with 'sk-'."
    
    elif provider == 'anthropic':
        if api_key.startswith('sk-ant-'):
            return True, "Anthropic API key format looks valid."
        else:
            return False, "Anthropic API key should start with 'sk-ant-'."
    
    return False, f"Unknown provider: {provider}"


def get_site_coordinates(site_name: str, stats: Dict) -> Optional[Tuple[float, float]]:
    """
    Get latitude/longitude for a site.
    
    Parameters
    ----------
    site_name : str
        Site name
    stats : dict
        Precomputed statistics
    
    Returns
    -------
    coords : tuple or None
        (latitude, longitude) or None if not found
    """
    if site_name not in stats:
        return None
    
    site_data = stats[site_name]
    lat = site_data.get('lat')
    lon = site_data.get('lon')
    
    if lat is not None and lon is not None:
        return (lat, lon)
    
    return None


def create_wind_rose_data(directions: List[float], frequencies: List[float], 
                         mean_speeds: List[float]) -> Dict:
    """
    Create wind rose data for visualization.
    
    Parameters
    ----------
    directions : list
        Wind directions (degrees)
    frequencies : list
        Frequency for each direction
    mean_speeds : list
        Mean wind speed for each direction
    
    Returns
    -------
    data : dict
        Wind rose data for plotting
    """
    return {
        'directions': directions,
        'frequencies': frequencies,
        'mean_speeds': mean_speeds
    }
