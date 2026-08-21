"""
Bounded tools for agentic AI.

This module defines a set of tools that the LLM agent can call to analyze
wind energy resources. Tools operate on precomputed data and are designed
to be reliable and fast for a live Streamlit demo.

Tools:
- get_site_stats: Retrieve statistics for a specific site
- compare_sites: Compare two sites side-by-side
- compute_trend: Analyze long-term trends
- estimate_extreme_return_period: Estimate extreme wind speeds
- wind_rose_plot: Generate wind rose visualization
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings

from .january_analysis import (
    EARLY_WINDOW,
    LATE_WINDOW,
    agent_analysis_contract,
    scientific_guardrails,
    summarize_window,
    compare_window_summaries,
)
from .wind_energy import height_sensitivity

warnings.filterwarnings('ignore')


class AgentToolkit:
    """
    Toolkit of bounded tools for wind energy analysis.
    
    All tools operate on precomputed data loaded at initialization.
    """
    
    def __init__(self, data_dir: Path = Path('data')):
        """
        Initialize toolkit by loading precomputed data.
        
        Parameters
        ----------
        data_dir : Path
            Directory containing precomputed data files
        """
        self.data_dir = Path(data_dir)
        self.stats = {}
        self.climatology = None
        self.timeseries = None
        
        self._load_data()
    
    def _load_data(self):
        """Load precomputed data files."""
        try:
            # Load precomputed statistics
            stats_file = self.data_dir / 'precomputed_stats.json'
            if stats_file.exists():
                with open(stats_file, 'r') as f:
                    self.stats = json.load(f)
            else:
                print(f"Warning: {stats_file} not found. Some tools may not work.")
        except Exception as e:
            print(f"Warning: Could not load data: {e}")
    
    def get_site_stats(self, site_name: str) -> Dict:
        """
        Retrieve statistics for a specific site.
        
        Parameters
        ----------
        site_name : str
            Name of the site (e.g., 'Dogger Bank', 'Hornsea')
        
        Returns
        -------
        stats : dict
            Dictionary with wind speed, WPD, CF, Weibull parameters, etc.
        """
        if site_name not in self.stats:
            return {
                'error': f'Site "{site_name}" not found.',
                'available_sites': list(self.stats.keys())
            }
        
        site_data = self.stats[site_name]
        
        # Format for readability
        formatted = {
            'site_name': site_name,
            'location': f"{site_data.get('lon', 'N/A')}°E, {site_data.get('lat', 'N/A')}°N",
            'country': site_data.get('country', 'N/A'),
            'mean_wind_speed_ms': round(site_data.get('mean_wind_speed', 0), 2),
            'wind_power_density_wm2': round(site_data.get('wpd', 0), 1),
            'capacity_factor': round(site_data.get('cf', 0), 3),
            'capacity_factor_percent': round(site_data.get('cf', 0) * 100, 1),
            'weibull_shape_k': round(site_data.get('weibull_k', 0), 2),
            'weibull_scale_a': round(site_data.get('weibull_a', 0), 2),
            'annual_energy_production_mwh': round(site_data.get('aep', 0), 0),
            'extreme_wind_50yr_ms': round(site_data.get('extreme_wind_50yr', 0), 1),
            'interannual_trend_ms_per_year': round(site_data.get('trend', 0), 3)
        }
        
        return formatted
    
    def compare_sites(self, site1: str, site2: str) -> Dict:
        """
        Compare two sites side-by-side.
        
        Parameters
        ----------
        site1 : str
            Name of first site
        site2 : str
            Name of second site
        
        Returns
        -------
        comparison : dict
            Comparison table with key metrics
        """
        stats1 = self.get_site_stats(site1)
        stats2 = self.get_site_stats(site2)
        
        if 'error' in stats1 or 'error' in stats2:
            return {
                'error': 'One or both sites not found.',
                'available_sites': list(self.stats.keys())
            }
        
        # Create comparison
        comparison = {
            'site1': site1,
            'site2': site2,
            'metrics': {
                'mean_wind_speed_ms': {
                    site1: stats1['mean_wind_speed_ms'],
                    site2: stats2['mean_wind_speed_ms'],
                    'difference': round(stats1['mean_wind_speed_ms'] - stats2['mean_wind_speed_ms'], 2),
                    'winner': site1 if stats1['mean_wind_speed_ms'] > stats2['mean_wind_speed_ms'] else site2
                },
                'capacity_factor_percent': {
                    site1: stats1['capacity_factor_percent'],
                    site2: stats2['capacity_factor_percent'],
                    'difference': round(stats1['capacity_factor_percent'] - stats2['capacity_factor_percent'], 1),
                    'winner': site1 if stats1['capacity_factor_percent'] > stats2['capacity_factor_percent'] else site2
                },
                'wind_power_density_wm2': {
                    site1: stats1['wind_power_density_wm2'],
                    site2: stats2['wind_power_density_wm2'],
                    'difference': round(stats1['wind_power_density_wm2'] - stats2['wind_power_density_wm2'], 1),
                    'winner': site1 if stats1['wind_power_density_wm2'] > stats2['wind_power_density_wm2'] else site2
                },
                'weibull_shape_k': {
                    site1: stats1['weibull_shape_k'],
                    site2: stats2['weibull_shape_k'],
                    'note': 'Higher k = more consistent wind (less variability)'
                },
                'annual_energy_production_mwh': {
                    site1: stats1['annual_energy_production_mwh'],
                    site2: stats2['annual_energy_production_mwh'],
                    'difference': round(stats1['annual_energy_production_mwh'] - stats2['annual_energy_production_mwh'], 0),
                    'winner': site1 if stats1['annual_energy_production_mwh'] > stats2['annual_energy_production_mwh'] else site2
                }
            }
        }
        
        return comparison
    
    def compute_trend(self, site_name: str) -> Dict:
        """
        Analyze long-term trend in wind resource.
        
        Parameters
        ----------
        site_name : str
            Name of the site
        
        Returns
        -------
        trend_analysis : dict
            Trend slope (m/s per year), interpretation
        """
        if site_name not in self.stats:
            return {
                'error': f'Site "{site_name}" not found.',
                'available_sites': list(self.stats.keys())
            }
        
        site_data = self.stats[site_name]
        trend = site_data.get('trend', 0)
        
        # Interpret trend
        if trend > 0.01:
            interpretation = 'Increasing (wind resource improving)'
        elif trend < -0.01:
            interpretation = 'Decreasing (wind resource declining)'
        else:
            interpretation = 'Stable (no significant trend)'
        
        return {
            'site_name': site_name,
            'trend_ms_per_year': round(trend, 4),
            'interpretation': interpretation,
            'period': '1995-2014 (20 years)',
            'total_change_ms': round(trend * 20, 2),
            'note': 'Positive trend indicates strengthening wind resource (good for future investment)'
        }
    
    def estimate_extreme_return_period(self, site_name: str, return_period_years: int = 50) -> Dict:
        """
        Estimate extreme wind speed at a given return period.
        
        Parameters
        ----------
        site_name : str
            Name of the site
        return_period_years : int
            Return period (years), default 50
        
        Returns
        -------
        extreme_wind : dict
            Estimated extreme wind speed and interpretation
        """
        if site_name not in self.stats:
            return {
                'error': f'Site "{site_name}" not found.',
                'available_sites': list(self.stats.keys())
            }
        
        site_data = self.stats[site_name]
        extreme_wind = site_data.get('extreme_wind_50yr', 0)
        mean_wind = site_data.get('mean_wind_speed', 0)
        
        return {
            'site_name': site_name,
            'return_period_years': return_period_years,
            'estimated_wind_speed_ms': round(extreme_wind, 1),
            'mean_wind_speed_ms': round(mean_wind, 2),
            'ratio_to_mean': round(extreme_wind / mean_wind if mean_wind > 0 else 0, 2),
            'note': 'Used for turbine design loads and extreme event planning'
        }
    
    def wind_rose_plot(self, site_name: str) -> Dict:
        """
        Get wind rose data for visualization.
        
        Parameters
        ----------
        site_name : str
            Name of the site
        
        Returns
        -------
        wind_rose : dict
            Wind rose data (directions, frequencies, mean speeds)
        """
        if site_name not in self.stats:
            return {
                'error': f'Site "{site_name}" not found.',
                'available_sites': list(self.stats.keys())
            }
        
        site_data = self.stats[site_name]
        wind_rose = site_data.get('wind_rose', {})
        
        if not wind_rose:
            return {
                'error': 'Wind rose data not available for this site.',
                'note': 'Wind rose will be generated during preprocessing.'
            }
        
        return {
            'site_name': site_name,
            'directions': wind_rose.get('directions', []),
            'frequencies': wind_rose.get('frequencies', []),
            'mean_speeds': wind_rose.get('mean_speeds', []),
            'note': 'Directions in degrees (0=N, 90=E, 180=S, 270=W)'
        }
    
    def list_available_sites(self) -> Dict:
        """
        List all available sites.
        
        Returns
        -------
        sites : dict
            Dictionary of available sites with locations
        """
        sites = {}
        for site_name, data in self.stats.items():
            sites[site_name] = {
                'location': f"{data.get('lon', 'N/A')}°E, {data.get('lat', 'N/A')}°N",
                'country': data.get('country', 'N/A'),
                'capacity_factor': round(data.get('cf', 0), 3)
            }
        
        return {
            'available_sites': sites,
            'total_sites': len(sites)
        }
    
    def get_tool_definitions(self) -> List[Dict]:
        """
        Get tool definitions for LLM function calling.
        
        Returns
        -------
        tools : list
            List of tool definitions in OpenAI format
        """
        tools = [
            {
                'type': 'function',
                'function': {
                    'name': 'get_site_stats',
                    'description': 'Retrieve wind energy statistics for a specific North Sea site',
                    'parameters': {
                        'type': 'object',
                        'properties': {
                            'site_name': {
                                'type': 'string',
                                'description': 'Name of the site (e.g., "Dogger Bank", "Hornsea")'
                            }
                        },
                        'required': ['site_name']
                    }
                }
            },
            {
                'type': 'function',
                'function': {
                    'name': 'compare_sites',
                    'description': 'Compare wind resources between two sites',
                    'parameters': {
                        'type': 'object',
                        'properties': {
                            'site1': {
                                'type': 'string',
                                'description': 'Name of first site'
                            },
                            'site2': {
                                'type': 'string',
                                'description': 'Name of second site'
                            }
                        },
                        'required': ['site1', 'site2']
                    }
                }
            },
            {
                'type': 'function',
                'function': {
                    'name': 'compute_trend',
                    'description': 'Analyze long-term wind resource trend (1995-2014)',
                    'parameters': {
                        'type': 'object',
                        'properties': {
                            'site_name': {
                                'type': 'string',
                                'description': 'Name of the site'
                            }
                        },
                        'required': ['site_name']
                    }
                }
            },
            {
                'type': 'function',
                'function': {
                    'name': 'estimate_extreme_return_period',
                    'description': 'Estimate extreme wind speed for turbine design',
                    'parameters': {
                        'type': 'object',
                        'properties': {
                            'site_name': {
                                'type': 'string',
                                'description': 'Name of the site'
                            },
                            'return_period_years': {
                                'type': 'integer',
                                'description': 'Return period in years (default 50)',
                                'default': 50
                            }
                        },
                        'required': ['site_name']
                    }
                }
            },
            {
                'type': 'function',
                'function': {
                    'name': 'wind_rose_plot',
                    'description': 'Get wind rose data (direction distribution)',
                    'parameters': {
                        'type': 'object',
                        'properties': {
                            'site_name': {
                                'type': 'string',
                                'description': 'Name of the site'
                            }
                        },
                        'required': ['site_name']
                    }
                }
            },
            {
                'type': 'function',
                'function': {
                    'name': 'list_available_sites',
                    'description': 'List all available North Sea wind farm sites',
                    'parameters': {
                        'type': 'object',
                        'properties': {}
                    }
                }
            }
        ]
        
        return tools
