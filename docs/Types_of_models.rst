
Outline of Flux Measurement Footprint Estimation Approaches
=======================================================

The concept of the flux footprint is used to estimate the **location and relative importance of passive scalar sources** that influence flux measurements taken at a specific point (Kljun et al., 2004; Kljun et al., 2015). Footprint information is vital for connecting atmospheric observations to their surface sources and is especially important for designing field experiments and interpreting flux measurements over heterogeneous areas (Rannik et al., 2012).

Here are the major types of approaches discussed in the sources for estimating flux measurement footprints:

1. Analytical Models and (Semi-)Empirical Parameterizations
-------------------------------------------------------------

*   **Description:** These models often derive footprint estimates based on analytical solutions to simplified diffusion equations, sometimes applying K-theory, or they use semi-empirical relationships and parameterizations of results from more complex models or theoretical analyses (Kormann & Meixner, 2001; Kljun et al., 2004; Rannik et al., 2012; Wang & Davis, 2008). They often assume horizontally homogeneous turbulence (Kljun et al., 2004). Parameterizations aim to provide quick and precise algebraic estimations, simplifying more complex algorithms for practical use (Kljun et al., 2004). Some analytical models adjust idealized solutions to match features from Lagrangian stochastic models (Wang & Davis, 2008).

*   **Characteristics:**
    *   Generally computationally less intensive than other methods (Kljun et al., 2004).
    *   Simple enough for routine analysis and real-time evaluation in long-term measurements (Rannik et al., 2012).

*   **Limitations:**
    *   Often assume horizontally homogeneous turbulence, which may not be accurate in complex conditions (Kljun et al., 2004).
    *   Fast models based on surface layer theory may have restricted validity for many real-world applications (Rannik et al., 2012).
    *   Some parameterizations are limited to specific turbulence scaling domains or ranges of stratifications (Kljun et al., 2004).
    *   Analytical formulations fundamentally struggle to describe footprints in strongly inhomogeneous turbulence (Rannik et al., 2012).

*   **Significant Contributors and Models:**
    *   **Pasquill:** Proposed the first concept for estimating a two-dimensional source weight distribution using a simple Gaussian model (Pasquill, 1972; Rannik et al., 2012).
    *   **Van Ulden:** Contributed analytical solutions to the diffusion equation based on Monin-Obukhov similarity theory that were used in later models (Van Ulden, 1978; Kormann & Meixner, 2001; Rannik et al., 2012).
    *   **Gash:** Presented an early simple analytical model for neutral stratification using a constant velocity profile (Gash, 1986; Rannik et al., 2012).
    *   **Schuepp et al.:** Adapted Gash's approach and established the concept of "flux footprint," defining it as the relative contribution from surface area elements (Schuepp et al., 1990; Rannik et al., 2012).
    *   **Horst and Weil:** Developed one-dimensional analytical footprint models and contributed to parameterizations based on diffusion models (Horst & Weil, 1992; Horst & Weil, 1994; Rannik et al., 2012; Kljun et al., 2004; Kormann & Meixner, 2001).
    *   **Schmid:** Overcame analytical difficulties with numerical modeling followed by parameterization (Schmid, 1994; Kormann & Meixner, 2001). Developed analytical models like FSAM and clarified the separation of footprints for scalars and fluxes (Schmid, 1994; Schmid, 1997; Rannik et al., 2012; Kljun et al., 2003).
    *   **Kormann and Meixner:** Developed analytical models (e.g., KM) accounting for thermal stability, based on modifications of analytical solutions to the advection-diffusion equation (Kormann & Meixner, 2001; Kljun et al., 2003; Rannik et al., 2012). Their model is widely used for interpreting flux measurements over spatially limited sources (Rannik et al., 2012).
    *   **Hsieh et al.:** Developed approximate analytical models (e.g., HS) and parameterizations for footprint estimation (Hsieh et al., 2000; Wang & Davis, 2008). Their original model is one-dimensional (Hsieh et al., 2000).
    *   **Kljun et al.:** Developed simple parameterizations for flux footprint predictions (e.g., FFP), with versions accounting for the two-dimensional shape and surface roughness effects, based on Lagrangian stochastic model simulations (Kljun et al., 2004; Kljun et al., 2015). Their parameterization scales footprint estimates to collapse them into similar curves across a range of stratifications and receptor heights (Kljun et al., 2004; Kljun et al., 2015).
    *   **Wang and Davis:** Developed an analytical model for the lower convective boundary layer (CBL) by adjusting analytical solutions to results from a Lagrangian stochastic model (Wang & Davis, 2008).
    *   **Kumar and Sharan:** Developed an analytical model for dispersion from a continuous source in the atmospheric boundary layer, comparing their techniques to other analytical models (Kumar & Sharan, 2010).

2. Lagrangian Particle Models / Lagrangian Stochastic (LS) Models
------------------------------------------------------------------------

*   **Description:** These models describe scalar diffusion using stochastic differential equations (Rannik et al., 2012). Particle trajectories are calculated to simulate the dispersion process. The **backward time frame approach**, initiated at the measurement point and tracked back to surface sources, is common as it focuses calculations on trajectories influencing the receptor (Kljun et al., 2002; Kljun et al., 2003; Rannik et al., 2012). The forward approach involves releasing particles at the source and tracking them past the receptor (Kljun et al., 2002; Kljun et al., 2003; Rannik et al., 2012). LS models can satisfy the well-mixed condition in inhomogeneous turbulence (Kljun et al., 2002).

*   **Characteristics:**
    *   Capable of accounting for three-dimensional turbulent diffusion and non-Gaussian inhomogeneous turbulence (Rannik et al., 2012).
    *   The backward approach is specific to a given measurement height but can consider sources at arbitrary levels or geometries with one simulation (Kljun et al., 2002).

*   **Limitations:**
    *   Computationally expensive, often not suitable for long-term observational programs (Kljun et al., 2004).
    *   Can suffer from numerical errors near the surface or violate the well-mixed condition if not using a suitable numerical scheme (Cai & Leclerc, 2007; Rannik et al., 2012).

*   **Significant Contributors and Models:**
    *   **Thomson:** Developed a Lagrangian stochastic trajectory simulation (Thomson, 1987; Rannik et al., 2012).
    *   **Leclerc and Thurtell:** Developed Lagrangian footprint models and contributed to LS model comparisons (Leclerc & Thurtell, 1990; Rannik et al., 2012).
    *   **Flesch et al. / Flesch:** Contributed to the development and application of backward Lagrangian stochastic models for footprint estimation, describing the footprint from backward models (Flesch et al., 1995; Flesch, 1996; Rannik et al., 2012).
    *   **Rotach et al.:** Developed the core three-dimensional Lagrangian stochastic particle dispersion model (LPDM) upon which models like LPDM-B are based (Rotach et al., 1996; Kljun et al., 2003).
    *   **de Haan and Rotach:** Developed the Puff-Particle Model (PPM) which has LPDM at its core, and contributed the density kernel method for evaluating touchdown locations in Lagrangian models (de Haan & Rotach, 1998; Kljun et al., 2003).
    *   **Rannik et al.:** Developed Lagrangian models, including some for forests (Rannik et al., 2000; Rannik et al., 2003; Rannik et al., 2012).
    *   **Kljun et al.:** Developed three-dimensional backward Lagrangian footprint models (e.g., LPDM-B) valid for a wide range of boundary layer stratifications and receptor heights, incorporating a spin-up procedure and density kernel method for efficiency (Kljun et al., 2002; Kljun et al., 2003; Rannik et al., 2012). Their FFP parameterization was developed and evaluated using LPDM-B simulations (Kljun et al., 2004; Kljun et al., 2015).
    *   **Kurbanmuradov and Sabelfeld:** Developed Lagrangian stochastic models (Kurbanmuradov & Sabelfeld, 2000; Rannik et al., 2012).
    *   **Cai et al.:** Used Lagrangian stochastic modeling, sometimes coupled with LES fields, to derive flux footprints, including using forward LS simulations with the inverse plume assumption (Cai et al., 2010). They also developed adjusted numerical schemes to address issues with backward simulations near the surface (Cai & Leclerc, 2007; Cai et al., 2008; Rannik et al., 2012).
    *   **Finn et al.:** Performed tracer experiments against which Lagrangian simulations were tested (Finn et al., 1996; Rannik et al., 2012).

3. Large Eddy Simulations (LES)
---------------------------------

*   **Description:** LES models numerically simulate the dispersion process and are capable of addressing spatial heterogeneity and complex topography explicitly (Rannik et al., 2012). They can be coupled with Lagrangian models (Cai et al., 2010).

*   **Characteristics:**
    *   Can simulate dispersion in heterogeneous conditions.

*   **Limitations:**
    *   Highly CPU-intensive (Rannik et al., 2012).

*   **Significant Contributors:**
    *   **Leclerc et al.:** Developed LES models for footprints (Leclerc et al., 1997; Rannik et al., 2012).
    *   **Cai et al.:** Used LES coupled with LS modeling for flux footprint calculations (Cai et al., 2010).
    *   **Steinfeld et al.:** Conducted LES studies that have been used for comparison and evaluation of footprint models (Steinfeld et al., 2008; Rannik et al., 2012).

4. Ensemble-Averaged Closure Models / Eulerian Models
-----------------------------------------------------

*   **Description:** These models use closure schemes to simulate flow fields that account for inhomogeneity (Rannik et al., 2012). They can estimate the contribution of surface areas by excluding sources/sinks in specific cells or excluding sources/sinks everywhere except the cell of interest (Rannik et al., 2012).

*   **Characteristics:**
    *   Capable of simulating flow fields over complex terrain and spatially varying vegetation (Rannik et al., 2012).
    *   Can be used for tasks like sensor placement or interpreting data over complex surfaces (Rannik et al., 2012).

*   **Limitations:**
    *   Difficult to predefine equal source strength in all grid cells, especially over complex terrain (Rannik et al., 2012).
    *   The calculated "footprint function" may represent a normalized contribution function where variations in horizontal flux distributions affect the function (Rannik et al., 2012).

*   **Significant Contributors and Models:**
    *   **Sogachev and Lloyd / Sogachev et al.:** Developed Eulerian models of higher-order turbulence closure, including the SCADIS model, and applied this approach to estimate footprints for real sites (Sogachev & Lloyd, 2005a; Sogachev et al., 2005a; Sogachev & Sedletski, 2006; Rannik et al., 2012). They introduced fractional flux functions for data interpretation (Rannik et al., 2012).
    *   **Rannik et al.:** Authors of the chapter describing this approach in detail, including its validation by comparison with other models (Rannik et al., 2012).

Validation of footprint models often involves comparing different models or evaluating them against experimental tracer release data (Foken & Leclerc, 2004; Rannik et al., 2012). While LS dispersion models have been tested against numerous dispersion experiments, fewer experimental datasets are available specifically for validating footprint *functions* (Rannik et al., 2012). The choice of an appropriate model for a given application remains a challenge (Rannik et al., 2012).
