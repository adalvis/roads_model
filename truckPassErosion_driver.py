"""
Purpose: Basic stochastic truck pass erosion model driver
Original creation: 03/12/2018
Latest update: 05/09/2025
Author: Amanda Alvis
"""
#%% Load python packages and set some defaults

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from landlab import RasterModelGrid 
from landlab.components import TruckPassErosion
from landlab.components import KinwaveImplicitOverlandFlow
from landlab.plot.imshow import imshow_grid

mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.weight'] = 'medium'
mpl.rcParams['axes.labelweight'] = 'normal'

np.set_printoptions(threshold=np.inf)
#np.set_printoptions(threshold=1000)

#%% Where are the truck tires on the elevation map?

# From centerline (road_peak), the truck will extend 4 nodes on either side. The tires 
# themselves are the 3rd node from road_peak. This model assumes a perfect world in 
# which the truck drives symmetrically about the road's crown. For this model, I assumed that
# the truck is 2.655m wide, with the tires being 1.475m apart.

#%% Create erodible grid method
def ErodibleGrid(nrows,ncols,spacing, full_tire):    
    mg = RasterModelGrid((nrows,ncols),spacing) #produces an 80m x 10.62m grid w/ cell size of 0.225m (approx. tire width)
    z = mg.add_zeros('topographic__elevation', at='node') #create the topographic__elevation field
    road_flag = mg.add_zeros('flag', at='node') #create a road_flag field

    mg.set_closed_boundaries_at_grid_edges(False, False, False, False) 
    
    if full_tire == False:
        road_peak = 40 #peak crowning height occurs at this x-location
        up = 0.0067 #rise of slope from ditchline to crown
        down = 0.0067 #rise of slope from crown to fillslope
        
        for g in range(nrows): #loop through road length
            elev = 0 #initialize elevation placeholder
            flag = False #initialize road_flag placeholder

            for h in range(ncols): #loop through road width
                z[g*ncols + h] = elev #update elevation based on x & y locations
                road_flag[g*ncols+h] = flag #update road_flag based on x & y locations

                if h == 0 or h == 8:
                    elev = 0
                    flag = False
                elif h == 1 or h == 7:
                    elev = -0.333375
                    flag = False
                elif h == 2 or h == 6:
                    elev = -0.5715
                    flag = False
                elif h == 3 or h == 5:
                    elev = -0.714375
                    flag = False
                elif h == 4:
                    elev = -0.762
                    flag = False
                elif h < road_peak and h > 7: #update latitudinal slopes based on location related to road_peak
                    elev += up
                    flag = True
                else:
                    elev -= down
                    flag = True
    elif full_tire == True:
        road_peak = 20 #peak crowning height occurs at this x-location
        up = 0.0134 #rise of slope from ditchline to crown
        down = 0.0134 #rise of slope from crown to fillslope
        
        for g in range(nrows): #loop through road length
            elev = 0 #initialize elevation placeholder
            flag = False #initialize road_flag placeholder

            for h in range(ncols): #loop through road width
                z[g*ncols + h] = elev #update elevation based on x & y locations
                road_flag[g*ncols+h] = flag #update road_flag based on x & y locations

                if h == 0 or h == 4:
                    elev = 0
                    flag = False
                elif h == 1 or h == 3:
                    elev = -0.5715
                    flag = False
                elif h == 2:
                    elev = -0.762
                    flag = False
                elif h < road_peak and h > 3: #update latitudinal slopes based on location related to road_peak
                    elev += up
                    flag = True
                else:
                    elev -= down
                    flag = True
        
    z += mg.node_y*0.05 #add longitudinal slope to road segment
    road_flag = road_flag.astype(bool) #Make sure road_flag is a boolean array

    n = mg.add_zeros('roughness', at='node') #create roughness field
    
    roughness = 0.1 #initialize roughness placeholder            
    
    for g in range(nrows): #loop through road length
        for h in range(ncols): #loop through road width
            n[g*ncols + h] = roughness #update roughness values based on x & y locations
            
            if h >= 0 and h <= 8: #ditchline Manning's n value is higher than OF
                roughness = 0.1
            else:
                roughness = 0.02
                
    return(mg, z, road_flag, n)           

#%% Run method to create grid; add new fields
mg, z, road_flag, n = ErodibleGrid(540,72,0.1475,False) #half tire width
# mg, z, road_flag, n = ErodibleGrid(270,36,0.295,True) #full tire width
noise_amplitude=0.001

z[mg.core_nodes] = z[mg.core_nodes] + noise_amplitude * np.random.rand(
    mg.number_of_core_nodes
)

#add absolute elevation fields that will update based on z updates
mg.at_node['active__elev'] = z
mg.at_node['surfacing__elev'] = z - 0.0275
mg.at_node['ballast__elev'] = z - 0.0275 - 0.23

#add depth fields that will update in the component
mg.at_node['active__depth'] = np.ones(540*72)*0.0275
mg.at_node['surfacing__depth'] = np.ones(540*72)*0.23
mg.at_node['ballast__depth'] = np.ones(540*72)*2.0

# mg.at_node['active__depth'] = np.ones(270*36)*0.0275
# mg.at_node['surfacing__depth'] = np.ones(270*36)*0.23
# mg.at_node['ballast__depth'] = np.ones(270*36)*2.0

#%% Plot initial grid
plt.figure(figsize = (3,6), layout='tight')
im = imshow_grid(mg, z, allow_colorbar=False, grid_units = ('m','m'), cmap = 'gist_earth', vmin = 0, vmax = 4)
cb = plt.colorbar()
cb.set_label('Elevation (m)')
plt.xlabel('Road width (m)')
plt.ylabel('Road length (m)')
plt.tight_layout()
plt.show()

#%% Prep some variables for later
xsec_pre = mg.at_node['topographic__elevation'][4392*2:4428*2].copy()
# xsec_pre = mg.at_node['topographic__elevation'][2196:2232].copy()
mg_pre = mg.at_node['topographic__elevation'].copy()

X = mg.node_x.reshape(mg.shape)
Y = mg.node_y.reshape(mg.shape)
Z = z.reshape(mg.shape)

#%% Run the component
#define how long to run the model
model_end = 10 #days

tpe = TruckPassErosion(mg) #initialize component
center = 40
# center = 20 #center node
half_width = 7 #how far each tire extends from center
# half_width = 4

import time

start = time.time()
knwv = KinwaveImplicitOverlandFlow(mg, runoff_rate=2, depth_exp=5/3) #Feed initial component a runoff rate of 2 mm/hr

# Initialize model run information
hydrograph_time = [0]
discharge_at_outlet = [0]
dt = 3600 #time step in seconds

elapsed_time = 1 #Set an initial time to avoid any 0 errors
model_run_time= 86400*10 #total model run time, in seconds; 36 hours
storm_duration = 86400 #length of storm in seconds; 24 hours

for i in range(0, model_end): #loop through model days
    tpe.run_one_step(center, half_width, False)
    print(tpe._truck_num) #this is just to ensure the truck_num was changing
    print(tpe._hiding_frac)
    # Run the model; note that this will take a bit of time!
    # while elapsed_time <= i*86400:
    #     if elapsed_time < storm_duration:
    #         knwv.run_one_step(dt)
    #     else:
    #         knwv.runoff_rate = 1e-30 #Reset runoff_rate to be ~0; post-storm runoff
    #         knwv.run_one_step(dt)

    #     # q_at_outlet = mg.at_node['surface_water_inflow__discharge'][oid].item() #get discharge at the outlet

    #     # hydrograph_time.append(elapsed_time/3600.)
    #     # discharge_at_outlet.append(q_at_outlet)
                            
    #     elapsed_time += dt #increase model time

end = time.time()
print(f"Time taken to run the code was {end-start} seconds")


#%% Cross section plot
xsec_active = mg.at_node['active__elev'][4392*2:4428*2]
xsec_surf =  mg.at_node['surfacing__elev'][4392*2:4428*2] 
xsec_ball = mg.at_node['ballast__elev'][4392*2:4428*2]

# xsec_active = mg.at_node['active__elev'][2196:2232]
# xsec_surf =  mg.at_node['surfacing__elev'][2196:2232] 
# xsec_ball = mg.at_node['ballast__elev'][2196:2232]

plt.figure(figsize=(8,3), layout='tight')
plt.plot(X[36], xsec_pre, color='gray', linestyle='-.', label='Before')
plt.plot(X[36], xsec_active, color ='black', linestyle='-', label = 'After - Active elevation')
plt.plot(X[36], xsec_surf, color ='magenta', linestyle='-', label = 'After - Surfacing elevation')
plt.plot(X[36], xsec_ball, color ='cyan', linestyle='-', label = 'After - Ballast elevation ')
plt.xlim(0,10)
plt.xlabel('Road width (m)')
plt.ylabel('Elevation (m)')
plt.legend()
plt.show()

#%% 3D plot 
X = mg.node_x.reshape(mg.shape)
Y = mg.node_y.reshape(mg.shape)
Z = z.reshape(mg.shape)

fig = plt.figure(figsize = (14,7))
ax = fig.add_subplot(111, projection='3d')
ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), np.diag([1, 2, 1, 1]))
ax.plot_surface(X, Y, Z, color='gray')
ax.view_init(elev=12, azim=-110)
ax.dist = 17
ax.set_xlim(0, 10)
ax.set_ylim(0, 80)
ax.set_zlim(0, 4)
ax.set_zticks(np.arange(0, 5, 1))
ax.set_xlabel('Road width (m)', labelpad=10)
ax.set_ylabel('Road length (m)', labelpad=17)
ax.set_zlabel('Elevation (m)', labelpad=10)
ax.set_zticklabels(labels=('0','1','2','3','4'))
plt.show()

#%% 2D rill plot
plt.figure(figsize = (3,6), layout='tight')
im = imshow_grid(mg, z, allow_colorbar=False, grid_units = ('m','m'), cmap = 'gist_earth', vmin = 0, vmax = 4)
cb = plt.colorbar()
cb.ax.tick_params(labelsize=12)
cb.set_label('Elevation (m)')
plt.xlabel('Road width (m)')
plt.ylabel('Road length (m)')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()
plt.show()

#%% Prepping for a difference map
diff = mg_pre - mg.at_node['topographic__elevation']
