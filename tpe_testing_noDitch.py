#%%
import time

import matplotlib.pyplot as plt
import random as rnd
import numpy as np
import pandas as pd

from landlab import RasterModelGrid, imshow_grid
from landlab.components import TruckPassErosionOLD, TruckPassErosion
np.set_printoptions(threshold=np.inf)
from utilities.erodible_grid import Erodible_Grid

#%% Define parameters
run_duration = 100

#Physical constants
rho_w=1000
rho_s=2650
g=9.81

#Define site
site_name = "MEL12"

#Call parameters from .csv file
parameters = pd.read_csv("input/parameters_WY2024.csv")
site_params = parameters.loc[parameters["Site Name"] == site_name].iloc[0]
S = site_params["Road Gradient"]/100
porosity = 0.35

seed = 1 
np.random.seed(seed)

# initialize road layer depths
Sa_ini = 0.019 # active depth in m
Ss_ini = 0.23 # surfacing depth in m
Sb_ini = 2    # ballast depth in m

#Initialize average number of truck passes per day for truck pass erosion
truck_num_ini = 4

#Define roughness values for fine and coarse material
n_c = 0.05   
n_f = 0.015

#Define d50 and tau_c
d50_road = 0.0001 # [m] 
tau_c_road = 0.146

#%% Create the grid, add random noise, add fields

#Parameters for grid creation
cell_spacing = 0.1475 # cell width or length dimension in meters
cell_area = cell_spacing**2
nrows = 540 # number of rows in the grid
ncols = 64  # number of columns in the grid (NO DITCH)

#We're using half tire width for node spacing
center = 32
half_width = 7 
full_tire = False

eg = Erodible_Grid(nrows=nrows, ncols=ncols,\
    spacing=cell_spacing, full_tire=full_tire, long_slope=S,road_peak=center,ditch=False)

mg, z, road_flag, n = eg() 

noise_amplitude=0.005
road = road_flag==1


# random = [rnd.random() for x in range(len(z[road]))]
random=np.random.rand(
    len(z[road])
)
z[road] += noise_amplitude * random #z is the road elevation

#Add depth fields that will update in the components; these are the initial conditions
active_depth = mg.add_ones('active__depth', at='node')
active_depth *= Sa_ini
surf_depth = mg.add_ones('surfacing__depth', at='node')
surf_depth *= Ss_ini
ball_depth = mg.add_ones('ballast__depth', at='node')
ball_depth *= Sb_ini

#%% Pre-define location indices
ruts = [mg.nodes[:, 26-9:40-9], mg.nodes[:, 41-9:55-9]]
half_road = mg.nodes[:, 0:33]
full_road = mg.nodes[:, 0:64]

#%% DEM plot
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(3, 6))
imshow_grid(mg, z, plot_name='Synthetic road', var_name='Elevation', var_units='m',\
    grid_units=('m', 'm'), cmap='terrain', color_for_closed='black', vmin=0, vmax=5)
plt.xlabel('Road width (m)')
plt.ylabel('Road length (m)')
plt.tight_layout()
plt.show()

#%% Save some pre-run variables for comparison later
X = mg.node_x.reshape(mg.shape)
Y = mg.node_y.reshape(mg.shape)
Z = z.reshape(mg.shape)

xsec_pre = mg.at_node['topographic__elevation'][mg.nodes[100,:].flatten()].copy()

#%% Prep arrays/lists
dz_arr=[]
dz_arr_cum = []

#Depths for each layer
sa_arr = np.zeros(run_duration)
ss_arr = np.zeros(run_duration)
sb_arr = np.zeros(run_duration)

# sediment load in the ruts due to TPE
tpe_load_ruts = []

truck_num = 0     
#%% Initialize Landlab components
# tpe = TruckPassErosionOLD(mg, center, half_width, full_tire, truck_num=truck_num_ini, \
#     scat_loss=8e-5) #initialize component, 

tpe = TruckPassErosion(mg, center, half_width, full_tire, truck_num=truck_num_ini, \
    scat_loss=8e-5) #initialize component, 

z_ini_cum = mg.at_node['topographic__elevation'].copy()
active_init = mg.at_node['active__depth'].copy()
surfacing_init = mg.at_node['surfacing__depth'].copy()
ballast_init = mg.at_node['ballast__depth'].copy()

#%% Main loop
start = time.time()
for i in range(0, run_duration): # daily time step
    z_ini = mg.at_node['topographic__elevation'].copy()

    tpe.run_one_step()

    truck_num += tpe.truck_num
    
    dz = z-z_ini # calculate elevation change at each daily time step
    dz_arr.append(sum(dz))

    dz_cum = z-z_ini_cum # calculate cumulative elevation change
    dz_arr_cum.append(sum(dz_cum)) 
    
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(3, 6))
    im = imshow_grid(mg,'active__depth', var_name='Active Depth', 
                    plot_name='Active Depth, t = %i days' %i,
                    var_units='$m$', grid_units=('m','m'), 
                    cmap='pink', shrink=0.9)
    plt.xlabel('Road width (m)')
    plt.ylabel('Road length (m)')
    plt.tight_layout()
    plt.savefig('output/f_%i_days.png' %i)
    plt.close()
    # plt.show()

    mg.add_field('dz_cum', dz_cum, at='node', units='m', clobber=True)
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(3, 6))
    plt.xlabel('Road width (m)')
    plt.ylabel('Road length (m)')
    im = imshow_grid(mg,'dz_cum', var_name='Cumulative dz', var_units='m', 
                    plot_name='Elevation change, t = %i days' %i,
                    grid_units=('m','m'), cmap='RdBu', vmin=-0.0009, 
                    vmax=0.0009, shrink=0.9)
    plt.xlabel('Road width (m)')
    plt.ylabel('Road length (m)')
    plt.tight_layout()
    plt.savefig('output/dz_cum_%i_days.png' %i)
    plt.close()
    # plt.show()

    #Append TPE loading
    tpe_load_ruts.append((tpe.sed_added).sum())

    #For plotting layer depths
    sa_arr[i] = np.sum(mg.at_node['active__depth'])
    ss_arr[i] = np.sum(mg.at_node['surfacing__depth'])
    sb_arr[i] = np.sum(mg.at_node['ballast__depth'])
wall_time = time.time() - start
print("Wall time for run:", wall_time, "s")

#%% Calculations for plots
road_mass_change_dz = np.multiply(dz_arr, (cell_area*rho_s*(1-porosity)))/2
cum_road_mass_change_dz = np.multiply(dz_arr_cum, cell_area*rho_s*(1-porosity))/2

#%% Cross section plot
xsec_active = mg.at_node['topographic__elevation'][mg.nodes[100,:].flatten()]
xsec_surf =  mg.at_node['surfacing__elevation'][mg.nodes[100,:].flatten()] 
xsec_ball = mg.at_node['ballast__elevation'][mg.nodes[100,:].flatten()]

plt.figure(figsize=(8,3), layout='tight')
plt.plot(X[36], xsec_pre, color='gray', linestyle='-.', label='Before')
plt.plot(X[36], xsec_active, color ='black', linestyle='-', label = 'After - Active elevation')
plt.plot(X[36], xsec_surf, color ='magenta', linestyle='-', label = 'After - Surfacing elevation')
plt.plot(X[36], xsec_ball, color ='cyan', linestyle='-', label = 'After - Ballast elevation ')
plt.xlim(0,9.5)
plt.xlabel('Road width (m)')
plt.ylabel('Elevation (m)')
plt.legend()
plt.show()

#%% TPE loading
# plot sediment load to the active layer in the ruts from truck passes
plt.plot(range(0,run_duration), tpe_load_ruts)
plt.xlabel('Day')
plt.ylabel('Cumulative sediment load to the active layer of the ruts \nfrom tpe [$kg$]')
plt.xlim(0,run_duration)
plt.show()

#%% Depths over the road surface 
fig, ax = plt.subplots(3,1, figsize=(4,7))

ax[0].plot(range(0,run_duration), (sa_arr)/(nrows*ncols))
ax[0].set_xlabel('Day')
ax[0].set_ylabel('Active Depth\naverage [$m$]')
ax[0].set_xlim(0,run_duration)
ax[0].set_title(r'%s ($n_{f_{road}} = %0.3f$)' %(site_name, n_f))

ax[1].plot(range(0,run_duration), (ss_arr)/(nrows*ncols))
ax[1].set_xlabel('Day')
ax[1].set_ylabel('Surfacing Depth\naverage [$m$]')
ax[1].set_xlim(0,run_duration)

ax[2].plot(range(0,run_duration), (sb_arr)/(nrows*ncols))
ax[2].set_xlabel('Day')
ax[2].set_ylabel('Ballast Depth\naverage [$m$]')
ax[2].set_xlim(0,run_duration)
plt.tight_layout()
plt.show()
# %%
