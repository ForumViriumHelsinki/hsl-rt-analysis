# hsl-rt-analysis

Analysis of High Frequency Positioning (HFP) data from public transport vehicles.

## Project Objectives

The main goal is to analyze high-frequency GPS positioning data from public transport vehicles to identify and study patterns in vehicle behavior, particularly focusing on:

1. Anomaly Detection in Vehicle Movement
   - Detect sudden deceleration events
   - Identify unusual stopping patterns
   - Analyze speed variations in critical zones

2. Spatial Analysis
   - Study vehicle behavior at intersections
   - Analyze movement patterns at roundabouts
   - Identify potential conflict points with other traffic

3. Statistical Analysis
   - Develop statistical methods for pattern recognition
   - Perform time series analysis on speed and acceleration data
   - Create baseline models for normal vehicle behavior

4. Data Visualization
   - Create interactive map visualizations
   - Generate time series plots of vehicle movements
   - Develop dashboards for pattern analysis

## Data Source

The project uses High Frequency Positioning (HFP) data from public transport vehicles. The data is stored in text files captured directly from MQTT topics and is available at:
- https://bri3.fvh.io/opendata/hsl-rt/

Download one or more of the files and place them in the `data/raw/` directory.

## Setting up the environment

### Virtual environment

This project uses Python 3.12 and `uv` for dependency management. Follow these steps to set up your development environment:

Install `uv` if you haven't already. You can use brew:

```bash
brew install uv
```

or download the install script:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Create a new virtual environment and install dependencies:
```bash
uv venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
uv pip install -e ".[dev,test]"
```

This will install:
- Core dependencies for data analysis and visualization
- Development tools (ruff, pre-commit, jupyter)
- Testing frameworks (pytest with coverage)

### Pre-commit hooks

This project uses pre-commit hooks for linting and code quality checks.

Install the hooks to your local repository:
```bash
pre-commit install
```

## Data Processing Pipeline

1. Data Collection
   - Download HFP data files from the server
   - Parse MQTT message format
   - Convert to structured data format

2. Data Preprocessing
   - Clean GPS coordinates
   - Calculate derived metrics (speed, acceleration)
   - Filter relevant geographic areas
   - Handle missing or erroneous data

3. Analysis
   - Time series analysis of vehicle movements
   - Statistical pattern recognition
   - Anomaly detection in speed/acceleration profiles
   - Spatial clustering of events

4. Visualization
   - Interactive maps showing vehicle paths
   - Time series plots of movement patterns
   - Statistical distribution visualizations
   - Dashboard for pattern analysis

## Required Libraries

Core Analysis:
- pandas: Data manipulation and analysis
- numpy: Numerical computations
- scipy: Scientific computing and statistics

Geospatial Analysis:
- geopandas: Spatial data operations
- shapely: Geometric operations
- folium: Interactive maps

Visualization:
- matplotlib: Basic plotting
- seaborn: Statistical visualizations
- plotly: Interactive plots
- dash: Web-based dashboards

Machine Learning:
- scikit-learn: Statistical analysis and machine learning
- statsmodels: Time series analysis

## Getting Started

1. Clone the repository
2. Set up the virtual environment as described above
3. Download the HFP data using the provided scripts
4. Run the analysis notebooks in the `notebooks/` directory

## Project Structure

```
├── data/              # Data storage
│   ├── raw/           # Original HFP data files
│   ├── processed/     # Cleaned and preprocessed data
│   └── interim/       # Intermediate processing results
├── notebooks/         # Jupyter notebooks for analysis
├── src/               # Source code
│   ├── data/          # Data processing scripts
│   ├── features/      # Feature calculation
│   └── visualization/ # Visualization tools
├── tests/             # Unit tests
└── reports/           # Generated analysis reports
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
