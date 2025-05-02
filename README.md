# Flux-footprints

> **FluxFootprint** is a fast, fully‑featured Python implementation of the
> Kljun et al. (2015) flux‑footprint parameterisation for eddy‑covariance research.
> It provides vectorised and xarray‑enabled utilities to compute per‑timestamp
> footprints, aggregate footprint climatologies, extract source‑area contours,
> and visualise results—scaling seamlessly from single towers to multi‑year
> datasets.

---

### Table of Contents

1. [Key Features](#key-features)
2. [Installation](#installation)
3. [Quick‑start Example](#quick-start-example)
4. [Command‑line Interface](#command-line-interface)
5. [Documentation](#documentation)
6. [Input Requirements](#input-requirements)
7. [Citing & Referencing](#citing--referencing)
8. [Contributing](#contributing)
9. [Development Road‑map](#development-road-map)
10. [License](#license)

---

### Key Features

| Category                | Highlights                                                                                                                                                                                                   |
| ----------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Core model**          | • Implements Eq. 14 & 17 of Kljun et al. (2015) with stability‑specific coefficients<br>• Optional roughness‑sublayer corrections<br>• Supports per‑footprint filtering based on theoretical validity limits |
| **Performance**         | • Pure NumPy + xarray for vectorised calculations<br>• Lazy computation and Dask compatibility for large archives                                                                                            |
| **I/O & preprocessing** | • Pandas helpers to map tower log fields automatically<br>• Quality‑control filters for *u\* ≥ 0.1 m s⁻¹*, finite σᵥ, etc.                                                                                   |
| **Analysis tools**      | • Aggregate footprint climatologies<br>• Compute *r%* source‑area contours (10–90 %)<br>• Functions to derive transects, footprint peak statistics, and 80 % area coverage                                   |
| **Visualisation**       | • Matplotlib helpers for heat‑maps & contour overlays<br>• Geospatial export to GeoTIFF / shapefile (EPSG aware)                                                                                             |
| **Extensibility**       | • Modular class design (`FFPModel`, `FFPClimatology`)<br>• Plug‑in hooks for alternative parameter sets or dispersion schemes                                                                                |
| **Reproducibility**     | • 100 % test coverage with `pytest`<br>• Pre‑commit hooks (black, ruff, isort, doc8)                                                                                                                         |

---

### Installation

```bash
# Stable release (PyPI)
pip install fluxfootprint

# Development version (GitHub)
pip install git+https://github.com/YourOrg/fluxfootprint.git
```

> Minimum Python 3.9.  Core dependencies: `numpy`, `pandas`, `xarray`,
> `scipy`, `matplotlib`.  Optional: `dask`, `rioxarray`, `pyproj` for advanced
> geospatial export.

---

### Quick‑start Example

```python
import pandas as pd
from fluxfootprint import FFPModel

# 1. Load half‑hourly tower data ----------------------------
df = pd.read_csv("AMF_US-CRT_BASE_HH_3-5_abb.csv", parse_dates=["TIMESTAMP"])

# Minimal renaming if your columns differ
df = df.rename(
    columns={
        "V_SIGMA": "V_SIGMA",
        "USTAR": "USTAR",
        "MO_LENGTH": "MO_LENGTH",
        "WD": "WD",
        "WS": "WS",
    }
)

# 2. Instantiate the model ----------------------------------
model = FFPModel(
    df,
    domain=[-1_000, 1_000, -1_000, 1_000],
    dx=2,
    dy=2,
    inst_height=3.5,
    crop_height=0.2,
    smooth_data=True,
)

# 3. Run & retrieve results ---------------------------------
result = model.run()            # dict with x_2d, y_2d, fclim_2d …

# 4. Plot 80 % source area ----------------------------------
model.plot_contour(r=0.8)
```

---

### Command‑line Interface

```bash
# Compute footprint climatology for a CSV file
fluxfootprint run \
  --input  met_data.csv \
  --dz     3.5 \
  --domain -1000 1000 -1000 1000 \
  --dx     2 \
  --out    footprint.nc
```

Run `fluxfootprint --help` for all options.

---

### Documentation

Full API docs, tutorials, and example notebooks are hosted at **Read the Docs**:

```
https://fluxfootprint.readthedocs.io
```

To build locally:

```bash
pip install -r docs/requirements.txt
sphinx-build -M html docs/ docs/_build
```

---

### Input Requirements

| Column                          | Units | Description                    |
| ------------------------------- | ----- | ------------------------------ |
| `USTAR`                         | m s⁻¹ | Friction velocity, u\*         |
| `V_SIGMA`                       | m s⁻¹ | Lateral velocity std. dev., σᵥ |
| `MO_LENGTH`                     | m     | Monin–Obukhov length, L        |
| `WD`                            | °     | Wind direction (0–360)         |
| `WS`                            | m s⁻¹ | Mean wind speed at *zₘ*        |
| *(optional)* `crop_height`      | m     | Canopy height, h\_c            |
| *(optional)* `atm_bound_height` | m     | Boundary‑layer height, h       |

Any additional columns are ignored unless you plug in custom routines.

---

### Citing & Referencing

If you use *Flux-Footprints* in a publication, please cite the original
parameterisation:

> Kljun, N., Calanca, P., Rotach, M.W., & Schmid, H.P. (2015).
> **A simple two‑dimensional parameterisation for flux footprint prediction (FFP)**.
> *Geoscientific Model Development*, 8(11), 3695–3713.
> [https://doi.org/10.5194/gmd-8-3695-2015](https://doi.org/10.5194/gmd-8-3695-2015)

You may also cite the software directly (see `CITATION.cff`).

---

### Contributing

1. **Fork** → 2. **Create branch** → 3. **Commit changes**
2. **Run tests** (`pytest`) & linters (`pre‑commit run --all-files`)
3. **Open a pull‑request**

All contributions—bug reports, suggestions, or code—are welcome!

---

### Development Road‑map

* [ ] Footprint uncertainty quantification via Monte‑Carlo resampling
* [ ] GPU acceleration using CuPy
* [ ] Footprint aggregation across different time periods
* [ ] QGIS plug‑in for in‑map footprint visualisation

---

### License

This project is licensed under the **MIT License** – see the
[`LICENSE`](LICENSE) file for details.

---

*Happy footprinting!*

