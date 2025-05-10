
Outline of Flux Measurement Footprint Estimation Approaches
=======================================================

The concept of the flux footprint is used to estimate the **location and relative importance of passive scalar sources** that influence flux measurements taken at a specific point :cite:`Kljun2015SimpleGMD,Kljun2004Simple`. Footprint information is vital for connecting atmospheric observations to their surface sources and is especially important for designing field experiments and interpreting flux measurements over heterogeneous areas :cite:`Rannik2012Footprint`.

Here are the major types of approaches discussed in the sources for estimating flux measurement footprints:

1. Analytical Models and (Semi-)Empirical Parameterizations
-------------------------------------------------------------

*   **Description:** These models often derive footprint estimates based on analytical solutions to simplified diffusion equations, sometimes applying K-theory, or they use semi-empirical relationships and parameterizations of results from more complex models or theoretical analyses (:cite:`Kormann2001Analytical`; :cite:`Kljun2004Simple`; :cite:`Rannik2012Footprint`; :cite:`Wang2008Analytical`). They often assume horizontally homogeneous turbulence :cite:`Kljun2004Simple`. Parameterizations aim to provide quick and precise algebraic estimations, simplifying more complex algorithms for practical use :cite:`Kljun2004Simple`. Some analytical models adjust idealized solutions to match features from Lagrangian stochastic models (:cite:`Wang2008Analytical`).

*   **Characteristics:**
    *   Generally computationally less intensive than other methods :cite:`Kljun2004Simple`.
    *   Simple enough for routine analysis and real-time evaluation in long-term measurements :cite:`Rannik2012Footprint`.

*   **Limitations:**
    *   Often assume horizontally homogeneous turbulence, which may not be accurate in complex conditions :cite:`Kljun2004Simple`.
    *   Fast models based on surface layer theory may have restricted validity for many real-world applications :cite:`Rannik2012Footprint`.
    *   Some parameterizations are limited to specific turbulence scaling domains or ranges of stratifications :cite:`Kljun2004Simple`.
    *   Analytical formulations fundamentally struggle to describe footprints in strongly inhomogeneous turbulence :cite:`Rannik2012Footprint`.

*   **Significant Contributors and Models:**
    *   **Pasquill:** Proposed the first concept for estimating a two-dimensional source weight distribution using a simple Gaussian model :cite:`pasquill1972some,Rannik2012Footprint`.
    *   **Van Ulden:** Contributed analytical solutions to the diffusion equation based on Monin-Obukhov similarity theory that were used in later models :cite:`van1978simple,Kormann2001Analytical,Rannik2012Footprint`.
    *   **Gash:** Presented an early simple analytical model for neutral stratification using a constant velocity profile (:cite:`Gash1986`; :cite:`Rannik2012Footprint`).
    *   **Schuepp et al.:** Adapted Gash's approach and established the concept of "flux footprint," defining it as the relative contribution from surface area elements (:cite:`Schuepp1990Footprint`; :cite:`Rannik2012Footprint`).
    *   **Horst and Weil:** Developed one-dimensional analytical footprint models and contributed to parameterizations based on diffusion models (:cite:`HorstWeil1992`; :cite:`HorstWeil1994`; :cite:`Rannik2012Footprint`; :cite:`Kljun2004Simple`; :cite:`Kormann2001Analytical`).
    *   **Schmid:** Overcame analytical difficulties with numerical modeling followed by parameterization (:cite:`Schmid1994Footprint`; :cite:`Kormann2001Analytical`). Developed analytical models like FSAM and clarified the separation of footprints for scalars and fluxes (:cite:`Schmid1994Footprint`; :cite:`Schmid1997Footprint`; :cite:`Rannik2012Footprint`; :cite:`Kljun2003Validation`).
    *   **Kormann and Meixner:** Developed analytical models (e.g., KM) accounting for thermal stability, based on modifications of analytical solutions to the advection-diffusion equation (:cite:`Kormann2001Analytical`; :cite:`Kljun2003Validation`; :cite:`Rannik2012Footprint`). Their model is widely used for interpreting flux measurements over spatially limited sources :cite:`Rannik2012Footprint`.
    *   **Hsieh et al.:** Developed approximate analytical models (e.g., HS) and parameterizations for footprint estimation (:cite:`Hsieh2000Footprint`; :cite:`Wang2008Analytical`). Their original model is one-dimensional (:cite:`Hsieh2000Footprint`).
    *   **Kljun et al.:** Developed simple parameterizations for flux footprint predictions (e.g., FFP), with versions accounting for the two-dimensional shape and surface roughness effects, based on Lagrangian stochastic model simulations (:cite:`Kljun2004Simple`; :cite:`Kljun2015SimpleGMD`). Their parameterization scales footprint estimates to collapse them into similar curves across a range of stratifications and receptor heights (:cite:`Kljun2004Simple`; :cite:`Kljun2015SimpleGMD`).
    *   **Wang and Davis:** Developed an analytical model for the lower convective boundary layer (CBL) by adjusting analytical solutions to results from a Lagrangian stochastic model (:cite:`Wang2008Analytical`).
    *   **Kumar and Sharan:** Developed an analytical model for dispersion from a continuous source in the atmospheric boundary layer, comparing their techniques to other analytical models (Kumar & Sharan, 2010).

2. Lagrangian Particle Models / Lagrangian Stochastic (LS) Models
------------------------------------------------------------------------

*   **Description:** These models describe scalar diffusion using stochastic differential equations :cite:`Rannik2012Footprint`. Particle trajectories are calculated to simulate the dispersion process. The **backward time frame approach**, initiated at the measurement point and tracked back to surface sources, is common as it focuses calculations on trajectories influencing the receptor (:cite:`Kljun2002LSmodel`; :cite:`Kljun2003Validation`; :cite:`Rannik2012Footprint`). The forward approach involves releasing particles at the source and tracking them past the receptor (:cite:`Kljun2002LSmodel`; :cite:`Kljun2003Validation`; :cite:`Rannik2012Footprint`). LS models can satisfy the well-mixed condition in inhomogeneous turbulence (:cite:`Kljun2002LSmodel`).

*   **Characteristics:**
    *   Capable of accounting for three-dimensional turbulent diffusion and non-Gaussian inhomogeneous turbulence :cite:`Rannik2012Footprint`.
    *   The backward approach is specific to a given measurement height but can consider sources at arbitrary levels or geometries with one simulation (:cite:`Kljun2002LSmodel`).

*   **Limitations:**
    *   Computationally expensive, often not suitable for long-term observational programs :cite:`Kljun2004Simple`.
    *   Can suffer from numerical errors near the surface or violate the well-mixed condition if not using a suitable numerical scheme (Cai & Leclerc, 2007; :cite:`Rannik2012Footprint`).

*   **Significant Contributors and Models:**
    *   **Thomson:** Developed a Lagrangian stochastic trajectory simulation (Thomson, 1987; :cite:`Rannik2012Footprint`).
    *   **Leclerc and Thurtell:** Developed Lagrangian footprint models and contributed to LS model comparisons (Leclerc & Thurtell, 1990; :cite:`Rannik2012Footprint`).
    *   **Flesch et al. / Flesch:** Contributed to the development and application of backward Lagrangian stochastic models for footprint estimation, describing the footprint from backward models (:cite:`Flesch1995`; :cite:`Flesch1996`; :cite:`Rannik2012Footprint`).
    *   **Rotach et al.:** Developed the core three-dimensional Lagrangian stochastic particle dispersion model (LPDM) upon which models like LPDM-B are based (:cite:`Rotach1996LPDM`; :cite:`Kljun2003Validation`).
    *   **de Haan and Rotach:** Developed the Puff-Particle Model (PPM) which has LPDM at its core, and contributed the density kernel method for evaluating touchdown locations in Lagrangian models (:cite:`deHaanRotach1998`; :cite:`Kljun2003Validation`).
    *   **Rannik et al.:** Developed Lagrangian models, including some for forests (:cite:`Rannik2000Forest`; :cite:`Rannik2003Forest`; :cite:`Rannik2012Footprint`).
    *   **Kljun et al.:** Developed three-dimensional backward Lagrangian footprint models (e.g., LPDM-B) valid for a wide range of boundary layer stratifications and receptor heights, incorporating a spin-up procedure and density kernel method for efficiency (:cite:`Kljun2002LSmodel`; :cite:`Kljun2003Validation`; :cite:`Rannik2012Footprint`). Their FFP parameterization was developed and evaluated using LPDM-B simulations (:cite:`Kljun2004Simple`; :cite:`Kljun2015SimpleGMD`).
    *   **Kurbanmuradov and Sabelfeld:** Developed Lagrangian stochastic models (:cite:`Kurbanmuradov2000`; :cite:`Rannik2012Footprint`).
    *   **Cai et al.:** Used Lagrangian stochastic modeling, sometimes coupled with LES fields, to derive flux footprints, including using forward LS simulations with the inverse plume assumption (:cite:`Cai2010LES`). They also developed adjusted numerical schemes to address issues with backward simulations near the surface (Cai & Leclerc, 2007; :cite:`Cai2008LS`; :cite:`Rannik2012Footprint`).
    *   **Finn et al.:** Performed tracer experiments against which Lagrangian simulations were tested (:cite:`Finn1996Tracer`; :cite:`Rannik2012Footprint`).

3. Large Eddy Simulations (LES)
---------------------------------

*   **Description:** LES models numerically simulate the dispersion process and are capable of addressing spatial heterogeneity and complex topography explicitly :cite:`Rannik2012Footprint`. They can be coupled with Lagrangian models (:cite:`Cai2010LES`).

*   **Characteristics:**
    *   Can simulate dispersion in heterogeneous conditions.

*   **Limitations:**
    *   Highly CPU-intensive :cite:`Rannik2012Footprint`.

*   **Significant Contributors:**
    *   **Leclerc et al.:** Developed LES models for footprints (:cite:`Leclerc1997LES`; :cite:`Rannik2012Footprint`).
    *   **Cai et al.:** Used LES coupled with LS modeling for flux footprint calculations (:cite:`Cai2010LES`).
    *   **Steinfeld et al.:** Conducted LES studies that have been used for comparison and evaluation of footprint models (:cite:`Steinfeld2008LES`; :cite:`Rannik2012Footprint`).

4. Ensemble-Averaged Closure Models / Eulerian Models
-----------------------------------------------------

*   **Description:** These models use closure schemes to simulate flow fields that account for inhomogeneity :cite:`Rannik2012Footprint`. They can estimate the contribution of surface areas by excluding sources/sinks in specific cells or excluding sources/sinks everywhere except the cell of interest :cite:`Rannik2012Footprint`.

*   **Characteristics:**
    *   Capable of simulating flow fields over complex terrain and spatially varying vegetation :cite:`Rannik2012Footprint`.
    *   Can be used for tasks like sensor placement or interpreting data over complex surfaces :cite:`Rannik2012Footprint`.

*   **Limitations:**
    *   Difficult to predefine equal source strength in all grid cells, especially over complex terrain :cite:`Rannik2012Footprint`.
    *   The calculated "footprint function" may represent a normalized contribution function where variations in horizontal flux distributions affect the function :cite:`Rannik2012Footprint`.

*   **Significant Contributors and Models:**
    *   **Sogachev and Lloyd / Sogachev et al.:** Developed Eulerian models of higher-order turbulence closure, including the SCADIS model, and applied this approach to estimate footprints for real sites (:cite:`Sogachev2005a`; :cite:`SogachevEtAl2005a`; :cite:`Sogachev2006`; :cite:`Rannik2012Footprint`). They introduced fractional flux functions for data interpretation :cite:`Rannik2012Footprint`.
    *   **Rannik et al.:** Authors of the chapter describing this approach in detail, including its validation by comparison with other models :cite:`Rannik2012Footprint`.

Validation of footprint models often involves comparing different models or evaluating them against experimental tracer release data (Foken & Leclerc, 2004; :cite:`Rannik2012Footprint`). While LS dispersion models have been tested against numerous dispersion experiments, fewer experimental datasets are available specifically for validating footprint *functions* :cite:`Rannik2012Footprint`. The choice of an appropriate model for a given application remains a challenge :cite:`Rannik2012Footprint`.

.. bibliography:: refs.bib
