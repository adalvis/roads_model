"""
Purpose: Basic stochastic truck pass erosion model driver
Original creation: 03/12/2018
Latest update: 05/14/2025
Author: Amanda Alvis
"""
#%% Load python packages and set some defaults

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from landlab import RasterModelGrid 
from landlab.components import TruckPassErosion
from landlab.components import KinwaveImplicitOverlandFlow, FlowAccumulator
from landlab.plot.imshow import imshow_grid
from landlab.plot.drainage_plot import drainage_plot

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

    mg.set_fixed_value_boundaries_at_grid_edges(False, False, False, True) 
    mg.set_closed_boundaries_at_grid_edges(True, False, True, False) 
    

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
                elif h < road_peak and h > 8: #update latitudinal slopes based on location related to road_peak
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
noise_amplitude=0.0075

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

ditch_id = np.argmin(z[0:8]) 
rut_left_id = 31
rut_right_id = 49
road_right_id = 70
#%% Plot initial grid
# Set up the figure.
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(3, 6))
imshow_grid(mg, z, plot_name='Synthetic road', var_name='Elevation', var_units='m',\
    grid_units=('m', 'm'), cmap='terrain', color_for_closed='black', vmin=0, vmax=4)
plt.xlabel('Road width (m)')
plt.ylabel('Road length (m)')

# Plot the sample nodes.
plt.plot(mg.node_x[ditch_id], mg.node_y[ditch_id], 's', zorder=10, ms=3.5, \
    clip_on=False, color='#44FFD1', markeredgecolor='k', label='Ditch')
plt.plot(mg.node_x[rut_left_id], mg.node_y[rut_left_id], '^', zorder=10, ms=3.5, \
    clip_on=False, color='#6153CC', markeredgecolor='k', label='Left rut')
plt.plot(mg.node_x[rut_right_id], mg.node_y[rut_right_id], 'o', zorder=10, ms=3.75, \
    clip_on=False, color='#A60067', markeredgecolor='k',label='Right rut')
plt.plot(mg.node_x[road_right_id], mg.node_y[road_right_id], '*', zorder=10, ms=5, \
    clip_on=False, color='#F7D08A', markeredgecolor='k',label='Right road')

_ = ax.legend(loc='center right', bbox_to_anchor=(1.25,0.5), \
    bbox_transform=plt.gcf().transFigure)
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

for i in range(0, model_end): #loop through model days
    tpe.run_one_step(center, half_width, False)

#%% FlowAccumulator
# Instantiate Landlab FlowAccumulator using 'MFD' as the flow director, spatially distributed runoff field in m/s, and
# use the partition method 'square_root_of_slope' to match instantiation of FlowAccumulator in Kinwave component
fa = FlowAccumulator(mg,
                     surface='topographic__elevation',
                     flow_director='MFD', #multiple flow directions
                     runoff_rate=1.66667e-6, #6 mm/hr converted to m/s
                     partition_method='square_root_of_slope')

# Run method to get drainage area and discharge at each node
(drainage_area, discharge) = fa.accumulate_flow()

# Check to see how many of the core nodes are sinks
sinks = mg.at_node['flow__sink_flag'][mg.core_nodes].sum()

if sinks < 0.01*mg.core_nodes.sum():
    print(sinks, r'of the core nodes are sinks. This is less than 1% of the core nodes. Code can continue.')
else:
    print(sinks, r'of the core nodes are sinks. This is more than 1% of the core nodes. \
        Consider using a DEM that has been pre-processed for sinks.')

# Obtain discharge at the outlet, midstream, and upstream nodes in mm
discharge_ditch = mg.at_node['surface_water__discharge'][ditch_id]
discharge_rut_left= mg.at_node['surface_water__discharge'][rut_left_id]
discharge_rut_right = mg.at_node['surface_water__discharge'][rut_right_id]
discharge_road_right = mg.at_node['surface_water__discharge'][road_right_id]

# Calculate runoff at the outlet, midstream, and upstream nodes in mm
runoff_ditch=1000.*2700.*((mg.at_node['surface_water__discharge'][ditch_id]))/(drainage_area[ditch_id])
runoff_rut_left=1000.*2700.*((mg.at_node['surface_water__discharge'][rut_left_id]))/(drainage_area[rut_left_id])
runoff_rut_right=1000.*2700.*((mg.at_node['surface_water__discharge'][rut_right_id]))/(drainage_area[rut_right_id])
runoff_road_right=1000.*2700.*((mg.at_node['surface_water__discharge'][road_right_id]))/(drainage_area[road_right_id])

print('Ditch discharge (m^3/s) =', np.round(discharge_ditch,5))
# print('Ditch runoff (mm) =', np.round(runoff_ditch,5))
print('Left rut discharge (m^3/s) =', np.round(discharge_rut_left,5))
# print('Left rut runoff (mm) =', np.round(runoff_rut_left,5))
print('Right rut discharge (m^3/s) =', np.round(discharge_rut_right,5))
# print('Right rut runoff (mm) =', np.round(runoff_rut_right,5))
print('Right road discharge (m^3/s) =', np.round(discharge_road_right,5))
# print('Right road runoff (mm) =', np.round(runoff_road_right,5))

# Map surface water discharge when outlet is at its maximum
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(3, 6))
plt.xlabel('Road width (m)')
plt.ylabel('Road length (m)')
imshow_grid(mg,'surface_water__discharge', plot_name='Steady state Q', 
            var_name='Q', var_units='$m^3/s$', grid_units=('m','m'), 
            cmap='Blues')

# Plot the sample nodes.
plt.plot(mg.node_x[ditch_id], mg.node_y[ditch_id], 's', zorder=10, ms=3.5, \
    clip_on=False, color='#44FFD1', markeredgecolor='k', label='Ditch')
plt.plot(mg.node_x[rut_left_id], mg.node_y[rut_left_id], '^', zorder=10, ms=3.5, \
    clip_on=False, color='#6153CC', markeredgecolor='k', label='Left rut')
plt.plot(mg.node_x[rut_right_id], mg.node_y[rut_right_id], 'o', zorder=10, ms=3.75, \
    clip_on=False, color='#A60067', markeredgecolor='k',label='Right rut')
plt.plot(mg.node_x[road_right_id], mg.node_y[road_right_id], '*', zorder=10, ms=5, \
    clip_on=False, color='#F7D08A', markeredgecolor='k',label='Right road')

_ = ax.legend(loc='center left', bbox_to_anchor=(1.25,0.5), \
    bbox_transform=plt.gcf().transFigure)

#%%
import time

knwv = KinwaveImplicitOverlandFlow(mg, runoff_rate=6, roughness=n, depth_exp=5/3) #Feed initial component a runoff rate of 2 mm/hr

# Initialize model run information
hydrograph_time = [0]
discharge_ditch = [0]
discharge_rut_left = [0]
discharge_rut_right = [0]
discharge_road_right = [0]
dt = 60 #time step in seconds

run_time_slices = (1,61,601,2401,3601)
elapsed_time = 1 #Set an initial time to avoid any 0 errors
storm_duration = 2700 #length of storm in seconds; 24 hours
model_run_time = 3601
    
# Run the model; note that this will take a bit of time!
start = time.time()

while elapsed_time <= model_run_time:
    if elapsed_time < storm_duration:
        knwv.run_one_step(dt)
    else:
        knwv.runoff_rate = 1e-30 #Reset runoff_rate to be ~0; post-storm runoff
        knwv.run_one_step(dt)

    for t in run_time_slices:
        if elapsed_time == t:
            time_model = t/60 
            imshow_grid(mg, 'surface_water__depth', plot_name='Surface water depth, t = %i min' % time_model, 
                var_name='Water depth', var_units='m', grid_units=('m','m'), vmin=0, vmax=0.005,
                cmap='pink')
            plt.show()
            fig, ax = plt.subplots(figsize=(15,10))
            drainage_plot(mg)
            plt.axis([0,10.5, 0, 0.5])
            plt.tight_layout()
            plt.show()

    q_ditch = mg.at_node['surface_water_inflow__discharge'][ditch_id].item() 
    q_rut_left = mg.at_node['surface_water_inflow__discharge'][rut_left_id].item() 
    q_rut_right = mg.at_node['surface_water_inflow__discharge'][rut_right_id].item() 
    q_road_right = mg.at_node['surface_water_inflow__discharge'][road_right_id].item() 

    hydrograph_time.append(elapsed_time/3600.)

    discharge_ditch.append(q_ditch)
    discharge_rut_left.append(q_rut_left)
    discharge_rut_right.append(q_rut_right) 
    discharge_road_right.append(q_road_right)
                        
    elapsed_time += dt #increase model time
            

end = time.time()
print(f"Time taken to run the code was {end-start} seconds")

#%%
#Plot the hydrograph
ax = plt.gca()
ax.tick_params(axis='both', which='both', direction='in', bottom='on', 
               left='on', top='on', right='on')
ax.minorticks_on()

ax.plot(hydrograph_time, discharge_ditch, '-', color='#44FFD1', markeredgecolor='k', label='Ditch')
ax.plot(hydrograph_time, discharge_rut_left, '-', color='#6153CC', markeredgecolor='k', label='Left rut')
ax.plot(hydrograph_time, discharge_rut_right, '-', color='#A60067', markeredgecolor='k',label='Right rut')
ax.plot(hydrograph_time, discharge_road_right, '-', color='#F7D08A', markeredgecolor='k',label='Right road')
ax.set(xlabel='Time (hr)', ylabel='Q ($m^3/s$)',
        title='Hydrograph')
ax.annotate('Max ditch Q = ' + str(np.max(np.round(discharge_ditch,5))) + ' $m^3/s$',(0.25,0.000525))
ax.annotate('Max left rut Q = ' + str(np.max(np.round(discharge_rut_left,5))) + ' $m^3/s$',(0.175,0.000185))
ax.annotate('Max right rut Q = ' + str(np.max(np.round(discharge_rut_right,5))) + ' $m^3/s$',(0.175,0.00016))
ax.annotate('Max right road Q = ' + str(np.max(np.round(discharge_road_right,5))) + ' $m^3/s$',(0.15,0.00035))
_=ax.legend()
plt.show()

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
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(3, 6))
imshow_grid(mg, z, plot_name='Synthetic road with ruts', var_name='Elevation', var_units='m',\
    grid_units=('m', 'm'), cmap='terrain', color_for_closed='black', vmin=0, vmax=4)
plt.xlabel('Road width (m)')
plt.ylabel('Road length (m)')

# Plot the sample nodes.
plt.plot(mg.node_x[ditch_id], mg.node_y[ditch_id], 's', zorder=10, ms=3.5, \
    clip_on=False, color='#44FFD1', markeredgecolor='k', label='Ditch')
plt.plot(mg.node_x[rut_left_id], mg.node_y[rut_left_id], '^', zorder=10, ms=3.5, \
    clip_on=False, color='#6153CC', markeredgecolor='k', label='Left rut')
plt.plot(mg.node_x[rut_right_id], mg.node_y[rut_right_id], 'o', zorder=10, ms=3.75, \
    clip_on=False, color='#A60067', markeredgecolor='k',label='Right rut')
plt.plot(mg.node_x[road_right_id], mg.node_y[road_right_id], '*', zorder=10, ms=5, \
    clip_on=False, color='#F7D08A', markeredgecolor='k',label='Right road')

_ = ax.legend(loc='center right', bbox_to_anchor=(1.25,0.5), \
    bbox_transform=plt.gcf().transFigure)
plt.tight_layout()
plt.show()

#%% Prepping for a difference map
diff = mg_pre - mg.at_node['topographic__elevation']
