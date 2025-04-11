"""
Purpose: Basic stochastic truck pass erosion model
Original creation: 03/12/2018
Update: Added deposition, ditchline (03/27/2018)
Update: Create field for roughness values (04/23/2018)
Update: Deleting unnecessary code; streamlining; adding layers (03/19/2025)
Update: Adding comments to the code (04/04/2025)
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

# From centerline (road_peak), the truck will extend 3 cells on either side. The tires 
# themselves are the 4th cell from road_peak. This model assumes a perfect world in 
# which the truck drives symmetrically about the road's crown. For this model, I assumed that
# the truck is 1.8m wide, with the tires being 1.35m apart.

tire_1 = 20 #x-position of one tire
tire_2 = 28 #x-position of other tire

out_1 = [19,21] #x-positions of the side cells of the first tire
out_2 = [27,29] #x-positions of the side cells of the other tire

back_tire_1 = [] #initialize the back of tire recovery for first tire
back_tire_2 = [] #initialize the back of tire recovery for other tire

#%% Create erodible grid method
def ErodibleGrid(nrows,ncols,spacing):    
    mg = RasterModelGrid((nrows,ncols),spacing) #produces an 80m x 10.67m grid w/ cell size of 0.225m (approx. tire width)
    z = mg.add_zeros('topographic__elevation', at='node') #create the topographic__elevation field
    road_flag = mg.add_zeros('flag', at='node') #create a road_flag field

    mg.set_closed_boundaries_at_grid_edges(False, False, False, False) 
    
    road_peak = 24 #peak crowning height occurs at this x-location
    up = 0.0067 #rise of slope from ditchline to crown
    down = 0.0067 #rise of slope from crown to fillslope
    
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
            
            if h >= 0 and h <= 4: #ditchline Manning's n value is higher than OF
                roughness = 0.1
            else:
                roughness = 0.02
                
    return(mg, z, road_flag, n)           

#%% Run method to create grid; add new fields
mg, z, road_flag, n = ErodibleGrid(355,44,0.225)

#add absolute elevation fields that will update based on z updates
mg.at_node['active__elev'] = z
mg.at_node['surfacing__elev'] = z - 0.0275
mg.at_node['ballast__elev'] = z - 0.0275 - 0.23

#add depth fields that will update in the component
mg.at_node['active__depth'] = np.ones(355*44)*0.0275
mg.at_node['surfacing__depth'] = np.ones(355*44)*0.23
mg.at_node['ballast__depth'] = np.ones(355*44)*2.0

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
xsec_pre = mg.at_node['topographic__elevation'][4400:4444].copy()
mg_pre = mg.at_node['topographic__elevation'].copy()

X = mg.node_x.reshape(mg.shape)
Y = mg.node_y.reshape(mg.shape)
Z = z.reshape(mg.shape)

#%% Determine location of tire tracks---this will be superseded by a method in the component
#get node IDs for the important nodes
tire_track_1 = mg.nodes[:, tire_1]
tire_track_2 = mg.nodes[:, tire_2]
out_tire_1 = mg.nodes[:, out_1]
out_tire_2 = mg.nodes[:, out_2]

back_tire_1.append(mg.nodes[0, tire_1])
back_tire_2.append(mg.nodes[0, tire_2])

for k in range(0,354):
    back_tire_1.append(mg.nodes[k+1, tire_1])
    back_tire_2.append(mg.nodes[k+1, tire_2])
    
back_tire_1_new = np.array(back_tire_1)    
back_tire_2_new = np.array(back_tire_2)


tire_tracks = np.array([tire_track_1, tire_track_2, out_tire_1[:,0], \
                        out_tire_1[:,1], out_tire_2[:,0], out_tire_2[:,1], \
                        back_tire_1_new, back_tire_2_new])

#%% Run the component
#define how long to run the model
model_end = 10 #days

tpe = TruckPassErosion(mg) #initialize component

for i in range(0, model_end): #loop through model days
    tpe.run_one_step(24,3)
    print(tpe.truck_num) #this is just to ensure the truck_num was changing

#%% Cross section plot
xsec_active = mg.at_node['active__elev'][4400:4444]
xsec_surf =  mg.at_node['surfacing__elev'][4400:4444] 
xsec_ball = mg.at_node['ballast__elev'][4400:4444]

plt.figure(figsize=(8,3), layout='tight')
plt.plot(X[44], xsec_pre, color='gray', linestyle='-.', label='Before')
plt.plot(X[44], xsec_active, color ='black', linestyle='-', label = 'After - Active elevation')
plt.plot(X[44], xsec_surf, color ='magenta', linestyle='-', label = 'After - Surfacing elevation')
plt.plot(X[44], xsec_ball, color ='cyan', linestyle='-', label = 'After - Ballast elevation ')
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
