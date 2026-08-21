# fluxfootprints

[![DOI](https://zenodo.org/badge/976380813.svg)](https://doi.org/10.5281/zenodo.22048260)
[![Read the Docs](https://img.shields.io/readthedocs/fluxfootprints)](https://fluxfootprints.readthedocs.io/en/latest/)
[![PyPI - Version](https://img.shields.io/pypi/v/fluxfootprints)](https://pypi.org/project/fluxfootprints/)
[![Conda Version](https://img.shields.io/conda/vn/conda-forge/fluxfootprints.svg)](https://anaconda.org/conda-forge/fluxfootprints)
[![codecov](https://codecov.io/github/inkenbrandt/footprints/branch/master/graph/badge.svg?token=27J6ExMY9A)](https://codecov.io/github/inkenbrandt/footprints)



> **fluxfootprints** is a Python package for micrometeorological flux-footprint
> analysis. It bundles several footprint models — the Kljun et al. (2015)
> parameterisation, the Kormann & Meixner (2001) and Wang et al. (2006)
> analytical models, and a backward Lagrangian stochastic particle model — behind one
> consistent interface, together with tooling to build footprint climatologies
> from eddy-covariance tower data, summarise them by day and month, and export
> the results as GeoPackages, GeoTIFFs, and CSV statistics.



---

### Table of Contents

- [fluxfootprints](#fluxfootprints)
    - [Table of Contents](#table-of-contents)
    - [Installation](#installation)
    - [Documentation](#documentation)
    - [Quick-start Example](#quick-start-example)
    - [Available Models](#available-models)
    - [Input Requirements](#input-requirements)
    - [Beyond the Climatology](#beyond-the-climatology)
    - [Citing \& Referencing](#citing--referencing)
    - [Contributing](#contributing)
    - [Development Road-map](#development-road-map)
    - [License](#license)

---

### Installation

```bash
# Stable release (PyPI)
pip install fluxfootprints

# Development version (GitHub)
pip install git+https://github.com/inkenbrandt/footprints.git
```

Requires **Python 3.10+**. Because the export tooling is geospatial, the
required dependencies include a full GIS stack: `numpy`, `pandas`, `xarray`,
`scipy`, `matplotlib`, `shapely`, `fiona`, `geopandas`, `rasterio`, `pyproj`,
`affine`, `requests`, and `netcdf4`.

Optional extras:

| Extra      | Installs                                          | For                                        |
| ---------- | ------------------------------------------------- | ------------------------------------------ |
| `contours` | `scikit-image`, `scikit-learn`                    | the marching-squares contour export path   |
| `examples` | `jupyter`, `ipykernel`, `seaborn`, `plotly`       | running the notebooks in `docs/notebooks/` |
| `docs`     | `sphinx`, `nbsphinx`, `numpydoc`, …               | building the documentation                 |
| `test`     | `pytest`, `pytest-cov`, `tox`, `black`, `ruff`    | running the test suite and linters         |

```bash
pip install "fluxfootprints[contours,examples]"
```

---

### Documentation

Full API docs, background material, and example notebooks are hosted at
**Read the Docs**: <https://fluxfootprints.readthedocs.io/en/latest/>

To build locally:

```bash
pip install -r docs/requirements.txt
sphinx-build -M html docs/ docs/_build
```

---

### Quick-start Example

`build_climatology` is the main entry point. It maps your column names onto the
package's internal names, constructs the requested model, runs it, and returns
the model object.

```python
import numpy as np
import pandas as pd
from fluxfootprints import build_climatology

# Half-hourly tower data indexed by timestamp
idx = pd.date_range("2024-06-01", periods=48, freq="30min")
rng = np.random.default_rng(0)
df = pd.DataFrame(
    {
        "USTAR":     rng.uniform(0.2, 0.6, 48),   # friction velocity  [m s-1]
        "MO_LENGTH": rng.uniform(-200, -50, 48),  # Obukhov length     [m]
        "WS":        rng.uniform(2.0, 5.0, 48),   # wind speed at zm   [m s-1]
        "V_SIGMA":   rng.uniform(0.3, 0.9, 48),   # lateral sigma_v    [m s-1]
        "WD":        rng.uniform(0, 360, 48),     # wind direction     [deg]
    },
    index=idx,
)

model = build_climatology(
    df,
    model_type="ffp",
    # Each of these takes a column name, a scalar, or a pandas Series
    ustar="USTAR", ol="MO_LENGTH", umean="WS", sigmav="V_SIGMA", wind_dir="WD",
    zm=3.0,        # measurement height       [m]
    z0=0.05,       # roughness length         [m]
    h=1500.0,      # boundary-layer height    [m]
    dx=10.0, dy=10.0,
    domain=(-500, 500, -500, 500),
)

fclim = model.get_footprint_climatology()   # xarray.DataArray, dims (x, y)
print(fclim.sizes)                          # {'x': 101, 'y': 101}
```

Every model exposes the same interface, so switching models is a one-word
change (`model_type="km"`, `"wang"`, `"ls"`, `"ffp_xr"`):

```python
model.get_footprint_climatology()   # 2-D climatology,      dims (x, y)
model.get_footprint_timeseries()    # per-timestamp 3-D,    dims (time, x, y)
model.get_coordinates()             # (x, y) arrays in metres
model.get_results()                 # full xarray.Dataset
model.to_netcdf("footprint.nc")
```

---

### Available Models

All models subclass `BaseFootprintModel` and share the API shown above.

| `model_type`         | Class                       | Type                  | Reference                  |
| -------------------- | --------------------------- | --------------------- | -------------------------- |
| `"ffp"`              | `FFPModel`                  | Parameterisation      | Kljun et al. (2015)        |
| `"ffp_xr"`           | `ffp_climatology_new`       | Parameterisation      | Kljun et al. (2015), xarray implementation |
| `"km"`, `"kormann-meixner"` | `KormannMeixnerModel` | Analytical            | Kormann & Meixner (2001)   |
| `"wang"`, `"wang2006"` | `WangFootprintModel`      | Analytical (convective BL) | Wang et al. (2006)    |
| `"ls"`, `"lagrangian"` | `LSFootprintModelAdapter` | Backward Lagrangian stochastic | —                 |

`wang` is not a general-purpose model: it is a semi-empirical parameterisation
for the **daytime convective boundary layer** and requires an Obukhov length
below zero. Its published validity range is −L/h ≈ 0.01–0.1 and
0.1 h ≤ zₘ ≤ 0.6 h; results outside that range should be treated with caution.

The models can also be instantiated directly, and the underlying analytical
functions (`footprint_2d`, `wang2006_fy`, `crosswind_integrated_footprint`, …)
are exported for use on their own. `fluxfootprints.ep_footprint` additionally
provides one-dimensional, EddyPro-style estimates (Kljun 2004,
Kormann & Meixner 2001, Hsieh et al. 2000), and `fluxfootprints.compare`
runs several models side by side and reports RMSE, peak-location bias, and
80 % source-area overlap.

---

### Input Requirements

Internally the models all work with these standardised, **lower-case** column
names:

| Column     | Units | Description                    |
| ---------- | ----- | ------------------------------ |
| `ustar`    | m s⁻¹ | Friction velocity, u\*         |
| `sigmav`   | m s⁻¹ | Lateral velocity std. dev., σᵥ |
| `ol`       | m     | Obukhov length, L              |
| `wind_dir` | °     | Wind direction (0–360)         |
| `umean`    | m s⁻¹ | Mean wind speed at *zₘ*        |
| `zm`       | m     | Measurement height             |
| `z0`       | m     | Roughness length               |
| `h`        | m     | Boundary-layer height          |

You do not normally have to rename anything yourself: `build_climatology`
accepts a **column name, a scalar, or a `pandas.Series`** for each of these and
assembles the standardised frame for you, so AmeriFlux-style names
(`USTAR`, `MO_LENGTH`, `V_SIGMA`, `WD`, `WS`) work by passing them as arguments.
Note that the defaults are the lower-case AmeriFlux-style variants
(`ustar_1_1_1`, `mo_length_1_1_1`, `ws_1_1_1`, `v_sigma_1_1_1`, `wd_1_1_1`).

In practice, supply all eight. `build_climatology` fills every one of them
(`h` defaults to `2000.0`), and the individual models differ in which they
consume: `km` and `ls` ignore `h`, while `ffp`, `ffp_xr`, and `wang` use it.
One special case is worth knowing: in `ffp_xr`, `z0` and `umean` are
interchangeable — supply either one per timestep, and a missing `z0` is
derived from the log-wind profile.

Rows failing basic physical checks (u\* ≤ 0.1 m s⁻¹, non-finite σᵥ, `zm` above
the boundary layer, out-of-range wind direction) are dropped automatically.

If you have canopy height rather than roughness length,
`compute_aerodynamic_params` derives `z0` and displacement height from
instrument and crop height.

---

### Beyond the Climatology

```python
from fluxfootprints import (
    summarize_periods, export_contours_gpkg,
    export_rasters_geotiff, export_contour_stats_csv,
)

# Daily and monthly means, optionally weighted by ET derived from latent heat
summaries = summarize_periods(model, df, et_source="LE", is_le=True)

# Georeferenced output, projected to an auto-selected UTM zone
export_rasters_geotiff(model, summaries, station_lat, station_lon, "out/")
export_contours_gpkg(model, summaries, df, station_lat, station_lon,
                     "footprints.gpkg", levels=(0.8,))
```

The package also includes helpers to pull forcing data
(`fetch_nldas_forcing_dataset`, `call_nldas_time_series`) and to turn per-row
fetch geometry into daily centroids and density rasters
(`polar_to_cartesian_dataframe`, `aggregate_to_daily_centroid`,
`generate_density_raster`, `concat_fetch_gdf`).

---

### Citing & Referencing

If you use *fluxfootprints* in a publication, please cite the parameterisation
you actually used. For the default `ffp` model:

> Kljun, N., Calanca, P., Rotach, M.W., & Schmid, H.P. (2015).
> **A simple two-dimensional parameterisation for flux footprint prediction (FFP)**.
> *Geoscientific Model Development*, 8(11), 3695–3713.
> [https://doi.org/10.5194/gmd-8-3695-2015](https://doi.org/10.5194/gmd-8-3695-2015)

For the other models, cite the corresponding paper — Kormann & Meixner (2001)
for `km` (entry `Kormann2001Analytical` in `docs/refs.bib`), and for `wang`:

> Wang, W., Davis, K.J., Ricciuto, D.M., & Butler, M.P. (2006).
> **An Approximate Footprint Model for Flux Measurements in the Convective Boundary Layer**.
> *Journal of Atmospheric and Oceanic Technology*, 23(10), 1384–1394.
> [https://doi.org/10.1175/JTECH1911.1](https://doi.org/10.1175/JTECH1911.1)

Note that `wang` implements this 2006 paper (`Wang2006Approximate` in
`docs/refs.bib`), **not** the separate Wang & Davis (2008) convective
boundary-layer model listed there as `Wang2008Analytical`.

You may also cite the software directly (see `CITATION.cff`).

---

### Contributing

1. **Fork** → **create a branch** → **commit your changes**
2. Run the tests (`pytest`) and linters (`ruff check .`, `black .`)
3. **Open a pull request**

All contributions — bug reports, suggestions, or code — are welcome!

---

### Development Road-map

* [ ] Footprint uncertainty quantification via Monte-Carlo resampling
* [ ] Broader OpenET API integration and comparison
* [ ] Consolidate `tools.py` into `by_row_fetch_tools.py`
* [ ] QGIS plug-in for in-map footprint visualisation

---

### License

This project is licensed under the **GNU General Public License v3.0** – see
the [`LICENSE`](LICENSE) file for details.

---

*Happy footprinting!*
