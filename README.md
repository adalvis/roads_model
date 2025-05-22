# WADNR Roads Project Model
Driver for testing and tutorials for learning the distributed roads 
model components being developed/are already developed in Landlab.

## First component: `TruckPassErosion`
Calculate sediment depths for forest road cross section layers based
on traffic-induced, erosion-enhancing processes: pumping, crushing,
scattering (and by default, flow rerouting).

### References
Alvis, A. D., Luce, C. H., & Istanbulluoglu, E. (2023). How does traffic 
affect erosion of unpaved forest roads? Environmental Reviews, 31(1), 
182–194. https://doi.org/10.1139/er-2022-0032


![Schematic describing the TruckPassErosion component.](TruckPassErosion_Component.png)

## Second component: `FlowAccumulator` or `KinwaveImplicitOverlandFlow`
Route flow over the forest road surface. Which component is chosen depends on 
the timestep of rainfall/runoff being fed to the model. `FlowAccumulator` works
best for a coarser timestep, whereas `KinwaveImplicitOverlandFlow` works best for
a finer timestep. However, `FlowAccumulator` is a much faster flow router than
`KinwaveImplicitOverlandFlow`.

## Third component: `FastscapeEroder`
