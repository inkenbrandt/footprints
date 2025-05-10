Flux Footprints: Key Concepts
=================================

What is a flux footprint and how is it represented mathematically?
------------------------------------------------------------------

A **flux footprint** is the up-wind surface area that contributes to the flux
measured at a given sensor position.  In two dimensions it is described by a
source-weighting density :math:`f(x, y, z_m)` that varies with stream-wise
distance **x**, cross-wind distance **y**, and measurement height :math:`z_m`.

A common representation splits the footprint into a **cross‑wind–integrated
component** :math:`f_y(x, z_m)` and a **cross‑wind distribution**
:math:`D_y(x, y)`.

The cross‑wind–integrated footprint is obtained by integrating the
two‑dimensional footprint over the **y**‑direction:

.. math::
    f_y(x, z_m\bigr) = \int_{-\infty}^{+\infty} f\bigl(x, y, z_m\bigr)\mathrm{d}y

where:

* :math:`f(x, y, z_m)` is the two‑dimensional footprint density (m :sup:`–2`),
* :math:`x` is the along‑wind distance from the sensor (m),
* :math:`y` is the cross‑wind distance (m), and
* :math:`z_m` is the measurement height (m).

How are footprint values calculated in modelling?
-------------------------------------------------

Several numerical approaches exist.  In **Lagrangian stochastic models** a
large ensemble of virtual particles is released at the sensor height and
followed backwards in time until they reach the surface.
The footprint value assigned to a surface element is proportional to the
number of particle “touch-downs” within that element—optionally weighted by the
particles' properties (e.g. vertical velocity).

Two common implementations are:

1. **Grid‑counting**
The up-wind area is discretised into grid cells; all touchdown events in a
cell are counted and normalised to give the footprint density on that cell.

2. **Kernel density estimation (KDE)**
Touchdown locations are treated as sample points of an unknown probability
density.  A kernel function (e.g. Gaussian, bi‑weight) is centred on each
touchdown and the kernels are summed to form a smooth, continuous footprint
field.

What meteorological parameters are crucial for footprint prediction models?
---------------------------------------------------------------------------

The following variables exert primary control on footprint size, shape and
location:

* :math:`z_m` — measurement height
* :math:`z_0` — aerodynamic roughness length
* :math:`u_{\text{mean}}` — mean wind speed at :math:`z_m`
* :math:`h` — boundary‑layer height
* :math:`L` — Obukhov length (stability)
* :math:`\sigma_v` — standard deviation of lateral velocity fluctuations
* :math:`u_*` — friction velocity

Three non‑dimensional groups appear repeatedly:

* :math:`z_m/L` (stability),
* :math:`z_m/h` (relative sensor height),
* the wind‑speed profile :math:`u(z)`.

How does atmospheric stability affect the footprint?
----------------------------------------------------

Atmospheric stability modulates turbulent mixing and therefore alters both the
extent and the peak position of the footprint:

* **Unstable / convective** (:math:`L < 0`)
  Strong vertical mixing broadens the footprint and shifts its peak **closer**
  to the sensor, while the tail extends further down‑wind.

* **Neutral** (:math:`|L| \to \infty`)
  Footprint size and shape lie between unstable and stable limits.

* **Stable** (:math:`L > 0`)
  Suppressed turbulence produces a **narrow, elongated** footprint whose peak
  lies further up-wind of the sensor.

These regime-dependent differences give rise to distinct *footprint
climatologies* when long time-series are stratified by stability class.

.. bibliography:: refs.bib