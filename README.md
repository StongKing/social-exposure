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
