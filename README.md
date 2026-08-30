# social-exposure

## Social Exposure through Urban Mobility Reallocation

The code is used to construct origin–destination mobility matrices, quantify social exposure, implement mobility-reallocation models, evaluate the resulting changes in exposure and travel distance, and reproduce the figures reported in the manuscript.

### Repository structure

The repository currently contains:

```text
social-exposure/
│
├── dynamic_integer.py
├── dynamic_integer_allcore_weight.py
├── static_integer.py
│
├── processing/
│   ├── regular_month_flow_nmf.py
│   ├── pre_figure1a.py
│   ├── pre_figure1c.py
│   ├── pre_figure1d.py
│   └── pre_figure4a.py
│
├── figure1/
│   ├── figure1a.py
│   ├── figure1b.py
│   ├── figure1c.py
│   ├── figure1d.py
│   └── figure1e.py
│
├── figure2/
│   ├── figure2a.py
│   ├── figure2b.py
│   ├── figure2c.py
│   ├── figure2d.py
│   └── figure2e.py
│
├── figure3/
│   ├── figure3a.py
│   ├── figure3b.py
│   ├── figure3c.py
│   ├── figure3d.py
│   ├── figure3e.py
│   └── figure3f.py
│
├── figure4/
│   ├── figure4a.py
│   ├── figure4b.py
│   ├── figure4c.py
│   ├── figure4d.py
│   ├── figure4e.py
│   └── figure4f.py
│
└── geo_data/
    └── geographic boundary files
```

### Important: working-directory requirement

> **All Python scripts must be placed in and executed from the main project directory.**

The subdirectories `processing/`, `figure1/`, `figure2/`, `figure3/`, and `figure4/` are used only to organize the code in this GitHub repository.

After downloading or cloning the repository, please **move or copy all `.py` files from these subdirectories into the repository root directory before running them**.

### Requirements

The analysis is implemented in Python. The main external packages include:

```text
numpy
pandas
scipy
matplotlib
seaborn
ortools
geopandas
shapely
pyproj
scikit-learn
networkx
```

They can be installed using `pip`, for example:

```bash
pip install numpy pandas scipy matplotlib seaborn ortools geopandas shapely pyproj scikit-learn networkx
```

A standard Python 3 environment is recommended.

### Path configuration

Before running the scripts, please inspect the path settings near the beginning of each file.

Most scripts use paths relative to the project root. Some plotting or preprocessing scripts may additionally contain a user-specific absolute path such as:

```python
PROJECT_ROOT = r"..."
```

If such a variable is present, change it to the location of the downloaded `social-exposure` directory on your computer.

For example:

```python
PROJECT_ROOT = r"D:\social-exposure"
```

or modify the script to use the current working directory:

```python
PROJECT_ROOT = os.getcwd()
```

All scripts should then be executed with `social-exposure/` as the current working directory.

### Input data

The analysis uses origin–destination mobility matrices, travel-distance matrices, CBG-level income distributions, POI information, and geographic boundary files.

The main processed input directories used by the scripts include paths of the following form:

```text
matrices_A_D_S_Distribution/
matrices_A_D_S_Distribution_<city>_core/
geo_data/
```

For each POI category, the analysis generally uses matrices such as:

```text
flow_matrix.csv
distance_matrix.csv
social_exposure_matrix.csv
```

together with CBG-level income-distribution files.

The `geo_data/` directory contains geographic boundary files used in the spatial analyses and map visualizations.

Some source mobility data may be subject to third-party access or redistribution restrictions. If an input dataset required by a script is not included in this repository, please obtain it according to the **Data Availability** statement of the accompanying manuscript and place the processed files in the directory expected by the corresponding script.
