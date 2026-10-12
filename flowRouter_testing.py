#%%
# IMPORT PACKAGES
#==================================
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import landlab
from landlab.io import esri_ascii
from landlab import RasterModelGrid, imshow_grid
from landlab.grid.raster_mappers import map_sum_of_inlinks_to_node
from landlab.components import (OverlandFlowTransporter, 
                               FlowAccumulator, 
                               OverlandFlow,
                               OverlandFlowBates,
                               LinearDiffusionOverlandFlowRouter,
                               DepressionFinderAndRouter)
np.set_printoptions(threshold=np.inf)

#%%
# LOAD DEM & DEFINE ELEVATION
#==================================

with open("/home/adalvis/github/roads_model/m14_20220601_sfm.asc") as grid:
    mg = esri_ascii.load(grid, name="topographic__elevation", at="node")

z = mg.at_node["topographic__elevation"]
mg.status_at_node[z == -9999] = mg.BC_NODE_IS_CLOSED
# mg.status_at_node[z == -9999] = mg.BC_NODE_IS_FIXED_VALUE
z_init = z.copy()

#%%
# DEFINE PARAMETERS
#==================================

runoff_rate_mmhr = 7
runoff_rate = runoff_rate_mmhr / 1000 / 3600 #[m/s]
S_min = 1e-6 #minimum slope to avoid blowups
dx = mg.dx
n = 0.05
h_init = 1e-5
run_time = 5600 #secs

#%%
# INITIALIZE FLOW ACCUMULATOR
#==================================

fa = FlowAccumulator(
                    mg, 
                    surface='topographic__elevation',
                    flow_director="FlowDirectorD8", 
                    runoff_rate=runoff_rate,
                    )

#%%
# TEST FLOW ROUTING ALGORITHMS AT STEADY STATE
#==================================
#==================================
# FlowAccumulator D8 algo
#==================================
## Step 1: Discharge
fa.run_one_step()
Q_d8 = mg.at_node['surface_water__discharge'].copy()
q_d8 = Q_d8/dx

## Step 2: Flow depth from Manning's eq using Q_d8
S = np.maximum(mg.at_node['topographic__steepest_slope'], S_min)
y_d8 = ((n * q_d8) / np.sqrt(S)) ** (3/5)


#==================================
# FlowAccumulator Z_eff algo
#==================================
## Step 1: Adjust z to get Z_eff and reroute to get discharge
z_eff = z_init + y_d8
mg.at_node['topographic__elevation'][:] = z_eff

fa.run_one_step()
Q_flood = mg.at_node['surface_water__discharge'].copy()
q_flood = Q_flood/dx

## Step 2: Flow depth from Manning's eq using Q_d8
S_flood = np.maximum(mg.at_node['topographic__steepest_slope'], S_min)
y_flood = ((n * q_flood) / np.sqrt(S_flood)) ** (3/5)

## Step 3: Restore z
mg.at_node['topographic__elevation'][:] = z_init

#%%
#==================================
# OverlandFlow (de Almeida) algo
#==================================
##Step 1: Add initial water depth
mg.add_zeros("surface_water__depth", at="node", clobber=True)
mg.at_node["surface_water__depth"] += h_init

## Step 2: Initialize OverlandFlow
of = OverlandFlow(mg, mannings_n=n, rainfall_intensity=runoff_rate)

## Step 3: Run to steady state
elapsed_time = 1
while elapsed_time < run_time:
    dt = of.calc_time_step()
    of.run_one_step()

    discharge = of.discharge_mapper(mg.at_link['surface_water__discharge'], \
        convert_to_volume=True)

    elapsed_time += dt

Q_of = discharge.copy()
y_of = mg.at_node["surface_water__depth"].copy()

#%%
# #==================================
# # OverlandFlow (Bates) algo
# #==================================
# ##Step 1: Add initial water depth
# mg.add_zeros("surface_water__depth", at="node", clobber=True)
# mg.at_node["surface_water__depth"] += h_init

# ## Step 2: Initialize OverlandFlowBates
# ofb = OverlandFlowBates(mg, mannings_n=n, rainfall_intensity=runoff_rate)

# ## Step 3: Run to steady state
# elapsed_time = 1
# while elapsed_time < run_time:
#     ofb.overland_flow()
#     dt = ofb.dt
#     discharge = map_sum_of_inlinks_to_node(mg, mg.at_link['surface_water__discharge'])*dx

#     elapsed_time += dt

# Q_ofb = discharge.copy()
# y_ofb = mg.at_node["surface_water__depth"].copy()

#%%
#==================================
# LinearDiffusionOverlandFlow algo
#==================================
##Step 1: Add initial water depth
mg.add_zeros("surface_water__depth", at="node", clobber=True)
mg.at_node["surface_water__depth"] += h_init

## Step 2: Initialize OverlandFlowBates
ldof = LinearDiffusionOverlandFlowRouter(mg, roughness=n, rain_rate=runoff_rate)

## Step 3: Run to steady state
elapsed_time = 1
while elapsed_time < run_time:
    dt=50
    ldof.run_one_step(dt)
    discharge = map_sum_of_inlinks_to_node(mg, mg.at_link['water__specific_discharge'])*dx

    elapsed_time += dt

Q_ldof = discharge.copy()
y_ldof = mg.at_node["surface_water__depth"].copy()

#%%
### ----------------------------------------------------------------
### Plots for water depth
### ----------------------------------------------------------------
fig, axes = plt.subplots(1, 4, figsize=(18, 6))

vmax = 0.0005

plt.sca(axes[0])
imshow_grid(mg, y_d8, cmap='Blues', colorbar_label='Flow Depth (m)',\
    vmin=0, vmax=vmax)
axes[0].set_title('D8 Only\nFlow Depth (y_d8)')

plt.sca(axes[1])
imshow_grid(mg, y_flood, cmap='Blues', colorbar_label='Flow Depth (m)',\
    vmin=0, vmax=vmax)
axes[1].set_title('Flooding Model\nFlow Depth (y_flood)')

plt.sca(axes[2])
imshow_grid(mg, y_of, cmap='Blues', colorbar_label='Flow Depth (m)',\
    vmin=0, vmax=vmax)
axes[2].set_title('OverlandFlow\nFlow Depth (y_of)')

# plt.sca(axes[3])
# imshow_grid(mg, y_ofb, cmap='Blues', colorbar_label='Flow Depth (m)',\
#     vmin=0, vmax=vmax)
# axes[3].set_title('OverlandFlowBates\nFlow Depth (y_obf)')

plt.sca(axes[3])
imshow_grid(mg, y_ldof, cmap='Blues', colorbar_label='Flow Depth (m)',\
    vmin=0, vmax=vmax)
axes[3].set_title('LinearDiffusionOverlandFlowRouter\nFlow Depth (y_ldof)')

plt.tight_layout()
plt.show()

### ----------------------------------------------------------------
### Stats
### ----------------------------------------------------------------
print(f'Max D8 depth       : {y_d8.max():.3f} m')
print(f'Max Flood depth    : {y_flood.max():.3f} m')
print(f'Max OF depth       : {y_of.max():.3f} m')
# print(f'Max OFB depth      : {np.nanmax(y_ofb)} m')
print(f'Max LDOF depth     : {y_ldof.max():.3f} m')

#%%
### ----------------------------------------------------------------
### Plots for discharge
### ----------------------------------------------------------------
fig, axes = plt.subplots(1, 4, figsize=(18, 6))

v_max = 1e-5

plt.sca(axes[0])
imshow_grid(mg, Q_d8, cmap='Blues', colorbar_label='Discharge ($m^3/s$)',\
    vmin=0, vmax=v_max)
axes[0].set_title('D8 Only\nDischarge (y_d8)')

plt.sca(axes[1])
imshow_grid(mg, Q_flood, cmap='Blues', colorbar_label='Discharge ($m^3/s$)',\
    vmin=0, vmax=v_max)
axes[1].set_title('Flooding Model\nDischarge (y_flood)')

plt.sca(axes[2])
imshow_grid(mg, Q_of, cmap='Blues', colorbar_label='Discharge ($m^3/s$)',\
    vmin=0, vmax=v_max)
axes[2].set_title('OverlandFlow\nDischarge (y_of)')

# plt.sca(axes[3])
# imshow_grid(mg, Q_ofb, cmap='Blues', colorbar_label='Discharge ($m^3/s$)',\
#     vmin=0, vmax=v_max)
# axes[3].set_title('OverlandFlowBates\nDischarge (y_obf)')

plt.sca(axes[3])
imshow_grid(mg, Q_ldof, cmap='Blues', colorbar_label='Discharge ($m^3/s$)',\
    vmin=0, vmax=v_max)
axes[3].set_title('LinearDiffusionOverlandFlowRouter\nDischarge (y_ldof)')

plt.tight_layout()
plt.show()

### ----------------------------------------------------------------
### Stats
### ----------------------------------------------------------------
print(f'Max D8 Discharge       : {Q_d8.max():.3f} m^3/s')
print(f'Max Flood Discharge    : {Q_flood.max():.3f} m^3/s')
print(f'Max OF Discharge       : {Q_of.max():.3f} m^3/s')
# print(f'Max OFB Discharge      : {np.nanmax(Q_ofb)} m^3/s')
print(f'Max LDOF Discharge     : {Q_ldof.max():.3f} m^3/s')
# %%
