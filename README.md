# WADNR Roads Project Distributed Model
**Updated:** 09/24/2025

## Summary
This repository contains drivers for testing and tutorials for learning the distributed roads 
model components that are being developed/are already developed in Landlab.

## Model component descriptions
### First component: `TruckPassErosion`
Calculates sediment depths for forest road cross section layers based
on traffic-induced, erosion-enhancing processes: pumping, crushing,
scattering (and by default, flow rerouting). This is a net-zero component
(i.e., the mass balance for the whole system is 0).

#### References
Alvis, A. D., Luce, C. H., & Istanbulluoglu, E. (2023). How does traffic 
affect erosion of unpaved forest roads? Environmental Reviews, 31(1), 
182–194. https://doi.org/10.1139/er-2022-0032

<p align="center" width="100%">
    <img src="./TruckPassErosion_Component.png" width="80%">
</p>

### Second component: `FlowAccumulator` or `KinwaveImplicitOverlandFlow`
Routes flow over the forest road surface. Which component is chosen depends on 
the timestep of rainfall/runoff being fed to the model. `FlowAccumulator` works
best for a coarser timestep, whereas `KinwaveImplicitOverlandFlow` works best for
a finer timestep. However, `FlowAccumulator` is a much faster flow router than
`KinwaveImplicitOverlandFlow`.

### Third component: `FastscapeEroder`
Erodes the landscape based on a stream power framework. The erosion of each node is 
based on an erosivity value, the contributing drainage area (or flow, depending on 
how the component is initialized), and the slope.
A future iteration of this spatially distributed model will use a different erosion
component based on slightly different variables (such as roughness).

### Third component (update): `OverlandFlowTransporter`
Erodes a surface with shallow overland flow using physics-based formulations for 
entrainment and transport that can directly use sediment size distribution and surface 
roughness. The sediment transport rate is calculated using Govers' equation (1992) 
with shear stress partitioning

## Repository navigation
### `Main folder`

1. `test_driver.py` is a script used to test the model components (`TruckPassErosion`, 
`FlowAccumulator`, `FastscapeEroder`) and how they work together.
   - **Input:** Model parameters.
   - **Output:** Plots of average road layer depths vs. time and erosion vs. time.

2. `test.py` is a script used to test the model components (`TruckPassErosion`, 
`FlowAccumulator`, `OverlandFlowTransporter`) and how they work together.
   - **Input:** Model parameters.
   - **Output:** Sediment load from road and ditch line, as well as a number of plots
   demonstrating different outputs from the model.

3. `group_rainfall.py` is a utilitarian script used to aggregate hourly rainfall into daily
rainfall.
   - **Input:** Hourly rainfall data.
   - **Output:** Aggregated daily rainfall data.

### `tutorials`
1. `TruckPassErosion_tutorial.ipynb` is a Jupyter notebook that demonstrates the utility
of the newly developed `TruckPassErosion` component.
together.
   - **Input:** Model parameters.
   - **Output:** Plots of road layer cross sections and road surface.
2. `FlowRouting_tutorial.ipynb` is a Jupyter notebook that demonstrates how to use either
`FlowAccumulator` or `KinwaveImplicitOverlandFlow`.
   - **Input:** A pre-rutted grid (`rutted_grid.grid`).
   - **Output:** Maps of surface water discharge.
3. `FullModel_tutorial.ipynb` is a Jupyter notebook that demonstrates how to run the full
distributed model (including `FastscapeEroder`).
   - **Input:** Necessary model parameters.
   - **Output:** Maps of surface water discharge, plots of average road layer depths vs. time,
    and erosion vs. time.