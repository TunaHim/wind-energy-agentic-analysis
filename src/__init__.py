"""
Wind Energy Agentic Analysis Platform

A Streamlit application for wind energy resource assessment using high-resolution
climate model data (EERIE IFS-FESOM) with agentic AI capabilities.
"""

__version__ = "0.1.0"
__author__ = "Your Name"
__license__ = "MIT"

from . import wind_energy
from . import january_analysis
from . import january_agent_tools
from . import agent_tools
from . import utils

__all__ = ['wind_energy', 'january_analysis', 'january_agent_tools', 'agent_tools', 'utils']
