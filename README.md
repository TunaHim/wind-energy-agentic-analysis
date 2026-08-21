# Wind Energy Agentic Analysis Platform

An AI-powered Streamlit application for wind energy resource assessment using high-resolution climate model data from the EERIE project (European Eddy-Rich Earth System Models). The platform combines climate data analysis with agentic AI capabilities to provide renewable energy companies with actionable insights for offshore wind farm site selection and climate-risk assessment.

## Overview

This project demonstrates:
- **High-resolution climate modeling:** IFS-FESOM coupled model output (9 km atmosphere, 5 km ocean) from the EERIE project
- **Wind energy resource characterization:** Capacity factor, power density, Weibull distribution, AEP, and extreme-wind statistics
- **Agentic AI analysis:** LLM-powered tool-calling agent that autonomously analyzes wind resources and compares sites
- **Interactive visualization:** Streamlit app with climatology maps, time-series analysis, and agent-assisted exploration

## Data Source

**EERIE IFS-FESOM2-SR Historical Simulation (1950–2014)**

- **Model:** Integrated Forecasting System (IFS) coupled to FESOM2.5 ocean model
- **Resolution:** ~9 km atmosphere (Tco1279), ~5 km ocean (NG5 unstructured grid)
- **Period:** 1950–2014 (65-year historical run following HighResMIP protocol)
- **Variables available in the retrieved native daily dataset:** 10 m wind components (`m10u`, `m10v`), 2 m temperature (`mean2t`), mean sea-level pressure (`msp`), and additional surface diagnostics
- **Access:** DKRZ km-scale cloud through its public Zarr chunk endpoint; the native grid is a reduced Gaussian grid, not a regular latitude/longitude raster
- **Data catalogue:** https://km-scale-cloud.dkrz.de/datasets
- **Dataset metadata endpoint:** https://km-scale-cloud.dkrz.de/datasets/ifs-fesom2-sr.hist-1950.v20240304.atmos.native.2D_daily_avg/
- **Zarr access endpoint:** https://km-scale-cloud.dkrz.de/datasets/ifs-fesom2-sr.hist-1950.v20240304.atmos.native.2D_daily_avg/zarr
- **January prototype future dataset metadata:** https://km-scale-cloud.dkrz.de/datasets/ifs-fesom2-sr.highres-future-ssp245.v20240304.atmos.native.2D_daily_avg/
- **January prototype future Zarr endpoint:** https://km-scale-cloud.dkrz.de/datasets/ifs-fesom2-sr.highres-future-ssp245.v20240304.atmos.native.2D_daily_avg/zarr
- **EERIE project:** https://eerie-project.eu/
- **EERIE data-access documentation:** https://easy.gems.dkrz.de/simulations/EERIE/eerie_data-access_online.html
- **EERIE Phase 1 documentation:** https://easy.gems.dkrz.de/simulations/EERIE/eerie_phase1.html
- **Long-term archive / dataset record:** https://www.wdc-climate.de/ui/entry?acronym=EERIE_FESOM_hist_HRdayva19
- **Dataset DOI:** https://doi.org/10.26050/WDCC/EERIE_FESOM_hist_v1
- **License shown by the archive record:** CC BY-NC-SA 4.0; check the record and access terms before redistribution
- **Citation:** Ghosh et al. (2025), EERIE IFS-FESOM historical simulation, https://doi.org/10.26050/WDCC/EERIE_FESOM_hist_v1

**Expanded prepared subset:** longitude −5–13°E, latitude 50–62°N, using the EERIE `highres-future-ssp245` simulation for January and July 2020–2024 and 2036–2040. It contains approximately 16,694 native atmospheric grid points per timestep.

**Masking status:** the expanded cache is a bounding-box extraction. The app uses finite daily SST as a preliminary ocean proxy to exclude many land points, but no final hydrographic North Sea polygon or official marine-only mask has yet been applied. SST itself is missing over many land points.

**Important height limitation:** the retrieved EERIE atmospheric dataset provides 10 m wind, not 100 m wind. Capacity-factor results are therefore illustrative at 10 m and should not be presented as bankable hub-height estimates without a documented vertical-extrapolation or model-level workflow.

## January Prototype Scope

The first agentic workflow compares two future SSP2-4.5 January windows:

- January 2020–2024
- January 2036–2040

The agent operates on a North Sea subset ( −5–13°E, 50–62°N) and can calculate daily wind-speed distributions, wind-power density, spatial differences, and sensitivity to 10 m-to-100 m power-law assumptions. Five Januarys per window are suitable for an exploratory prototype, not a definitive climate trend or bankable energy assessment. The agent is required to report these caveats.

## Local EERIE Cache

The extraction workflow stores only the decoded regional records under `C:myfolder\January` and `...\\July`; it does not persist the global EERIE source chunks. The cache is resumable at two-day chunk boundaries and is intended to be created once during preprocessing. The Streamlit app loads the compact January Parquet files from the corrected cache and does not download raw EERIE chunks during an interviewer session.

For the January-only deployment, the two required Parquet files are approximately 19 MB each (about 39 MB total), plus a small manifest. The chunk directories, July files, raw EERIE data, and API keys do not need to be uploaded to GitHub. The current `.gitignore` excludes Parquet files by default, so deployment packaging must explicitly include only the two prepared January artifacts.

## Project Structure

```
wind-energy-agentic-analysis/
├── README.md                              # This file
├── requirements.txt                       # Python dependencies
├── .gitignore                             # Git ignore rules
├── notebooks/
│   └── 01_eerie_download_preprocess.ipynb # EERIE data fetch & wind-energy preprocessing
├── data/
│   ├── north_sea_highres_sample_points.parquet # Retrieved native-grid sample
│   ├── north_sea_highres_sample_points.csv     # Portable copy of the sample
│   └── highres_sample_provenance.json          # Dataset and limitation metadata
├── src/
│   ├── __init__.py
│   ├── wind_energy.py                     # Wind energy calculations (WPD, Weibull, CF)
│   ├── january_analysis.py                # January windows and EERIE analysis contract
│   ├── january_agent_tools.py             # Bounded January comparison tools
│   ├── llm_providers.py                   # Gemini/Groq provider abstraction
│   ├── agent_tools.py                     # Existing bounded tools for agentic AI
│   └── utils.py                           # Utility functions
├── tests/
│   └── test_wind_energy.py                # Unit tests
└── app.py                                 # Streamlit main application
```

## Installation

### Prerequisites
- Python 3.9+
- Git

### Setup

```bash
git clone https://github.com/yourusername/wind-energy-agentic-analysis.git
cd wind-energy-agentic-analysis

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### 1. Download & Preprocess EERIE Data

Run the Jupyter notebook to fetch EERIE IFS-FESOM data from the DKRZ km-scale cloud and precompute wind-energy metrics:

```bash
jupyter notebook notebooks/01_eerie_download_preprocess.ipynb
```

This notebook:
- Connects to EERIE intake catalog
- Subsets North Sea region (1995–2014)
- Computes wind speed, direction, air density
- Calculates WPD, Weibull parameters, capacity factor
- Saves precomputed outputs to `data/`

**Note:** This step requires internet access to DKRZ servers. Estimated runtime: 30–60 minutes depending on network and compute resources.

### 2. Configure Gemini or Groq (optional)

```toml
# .streamlit/secrets.toml — never commit this file
GEMINI_API_KEY = "your-gemini-key"
GROQ_API_KEY = "your-groq-key"
```

The app reads these values through `st.secrets` and never displays or logs them. For Streamlit Community Cloud, add the same names under the app's **Settings → Secrets** panel. The interviewer can then select Gemini or Groq from the Agent workspace without entering a key.

Both providers use their OpenAI-compatible tool-calling endpoints, so the same bounded climate-analysis tools work with either model. If no key is configured, the deterministic tool-calling demo remains available.

### 3. Run the Streamlit App

```bash
streamlit run app.py
```

## Limitations

1. **Climate model uncertainty:** EERIE is a single model; ensemble uncertainty not quantified here
2. **Spatial resolution:** 5–9 km is high but still coarser than microscale site-specific assessment
3. **Historical period:** 1995–2014 may not capture all decadal variability
4. **Offshore-only focus:** This analysis is for offshore wind; onshore would require different surface roughness treatment

## Future Work

- Add ensemble uncertainty from multiple HighResMIP models (AWI-CM, EC-Earth3P-HR, etc.)
- Integrate NextGEMS future projections (2020–2049) for climate-change impact assessment
- Add solar resource analysis (GHI, DNI from CAMS if available)
- Extend to wave energy (significant wave height, period)
- Implement cost-benefit analysis (LCOE, NPV) for site selection

## Contributing

Contributions welcome! Please open an issue or submit a pull request.

## License

This project is licensed under the MIT License. See LICENSE file for details.

EERIE data is licensed under CC-BY 4.0. See https://creativecommons.org/licenses/by/4.0/ for details.

## Citation

If you use this project in research, please cite:

```bibtex
@software{wind_energy_agentic_2024,
  title={Wind Energy Agentic Analysis Platform},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/wind-energy-agentic-analysis}
}

@article{ghosh2024eerie,
  title={EERIE: Ocean Eddy-rich Kilometer-scale Climate Simulation with IFS-FESOM},
  author={Ghosh, Rohit and others},
  journal={Geoscientific Model Development},
  year={2024},
  doi={10.26050/wdcc/eerie_fesom_hist_HRday}
}
```

## Contact & Support

For questions about the project, open an issue on GitHub.

For questions about EERIE data, see: https://eerie-project.eu/

For questions about the km-scale cloud, see: https://easy.gems.dkrz.de/
