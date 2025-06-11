"""
Purpose: Full road erosion model driver - testing
Original creation: 03/12/2018
Latest update: 06/05/2025
Author: Amanda Alvis
"""
#%% Load python packages and set some defaults

import numpy as np
import time
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from landlab import RasterModelGrid 
from landlab.io import native_landlab
from landlab.components import TruckPassErosion
from landlab.components import KinwaveImplicitOverlandFlow, FlowAccumulator, FastscapeEroder
from landlab.plot.imshow import imshow_grid
from landlab.plot.drainage_plot import drainage_plot

mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.weight'] = 'medium'
mpl.rcParams['axes.labelweight'] = 'normal'

np.set_printoptions(threshold=np.inf)

#%% Create erodible grid method
def ErodibleGrid(nrows, ncols, spacing, full_tire):
    mg = RasterModelGrid((nrows,ncols),spacing)
    z = mg.add_zeros('topographic__elevation', at='node') #create the topographic__elevation field
    road_flag = mg.add_zeros('flag', at='node') #create a road_flag field for determining whether a 
                                                #node is part of the road or the ditch line
    n = mg.add_zeros('roughness', at='node') #create roughness field
    

    mg.set_fixed_value_boundaries_at_grid_edges(False, False, False, True)     
    mg.set_closed_boundaries_at_grid_edges(False, False, False, False) 
    
    if full_tire == False: #When node spacing is half-tire-width
        road_peak = 40 #peak crowning height occurs at this x-location
        up = 0.0067 #rise of slope from ditchline to crown
        down = 0.0067 #rise of slope from crown to fillslope
        
        for g in range(nrows): #loop through road length
            elev = 0 #initialize elevation placeholder
            flag = False #initialize road_flag placeholder
            roughness = 0.1 #initialize roughness placeholder   

            for h in range(ncols): #loop through road width
                if h == 0 or h == 8:
                    elev = 0
                    flag = False
                    roughness = 0.1
                elif h == 1 or h == 7:
                    elev = -0.333375
                    flag = False
                    roughness = 0.1
                elif h == 2 or h == 6:
                    elev = -0.5715
                    flag = False
                    roughness = 0.1
                elif h == 3 or h == 5:
                    elev = -0.714375
                    flag = False
                    roughness = 0.1
                elif h == 4:
                    elev = -0.762
                    flag = False
                    roughness = 0.1
                elif h < road_peak and h > 8: #update latitudinal slopes based on location related to road_peak
                    elev += up
                    flag = True
                    roughness = 0.05
                else:
                    elev -= down
                    flag = True
                    roughness = 0.05

                z[g*ncols + h] = elev #update elevation based on x & y locations
                road_flag[g*ncols+h] = flag #update road_flag based on x & y locations
                n[g*ncols + h] = roughness #update roughness values based on x & y locations
    elif full_tire == True: #When node spacing is full-tire-width
        road_peak = 20 #peak crowning height occurs at this x-location
        up = 0.0134 #rise of slope from ditchline to crown
        down = 0.0134 #rise of slope from crown to fillslope
        
        for g in range(nrows): #loop through road length
            elev = 0 #initialize elevation placeholder
            flag = False #initialize road_flag placeholder
            roughness = 0.1 #initialize roughness placeholder

            for h in range(ncols): #loop through road width
                if h == 0 or h == 4:
                    elev = 0
                    flag = False
                    roughness = 0.1
                elif h == 1 or h == 3:
                    elev = -0.5715
                    flag = False
                    roughness = 0.1
                elif h == 2:
                    elev = -0.762
                    flag = False
                    roughness = 0.1
                elif h < road_peak and h > 4: #update latitudinal slopes based on location related to road_peak
                    elev += up
                    flag = True
                    roughness = 0.05
                else:
                    elev -= down
                    flag = True
                    roughness = 0.05

                z[g*ncols + h] = elev #update elevation based on x & y locations
                road_flag[g*ncols+h] = flag #update road_flag based on x & y locations
                n[g*ncols + h] = roughness #update roughness values based on x & y locations
        
    z += mg.node_y*0.05 #add longitudinal slope to road segment
    road_flag = road_flag.astype(bool) #Make sure road_flag is a boolean array
                
    return(mg, z, road_flag, n)          

#%% Run method to create grid; add new fields
mg, z, road_flag, n = ErodibleGrid(540,72,0.1475,False) #half tire width
# mg, z, road_flag, n = ErodibleGrid(270,36,0.295,True) #full tire width
noise_amplitude=0.007

# np.random.seed(0)

z[road_flag==1] += noise_amplitude * np.random.rand(
    len(z[road_flag==1])
)

#add depth fields that will update in the component
mg.at_node['active__depth'] = np.ones(540*72)*0.02
mg.at_node['surfacing__depth'] = np.ones(540*72)*0.23
mg.at_node['ballast__depth'] = np.ones(540*72)*2.0

#add absolute elevation fields that will update based on z updates
mg.at_node['active__elev'] = z
mg.at_node['surfacing__elev'] = z - mg.at_node['active__depth']
mg.at_node['ballast__elev'] = z - mg.at_node['active__depth']\
     - mg.at_node['surfacing__depth']

#for full tire width
# mg.at_node['active__depth'] = np.ones(270*36)*0.0275
# mg.at_node['surfacing__depth'] = np.ones(270*36)*0.23
# mg.at_node['ballast__depth'] = np.ones(270*36)*2.0

#%% Plot initial grid
# Set up the figure.
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(3, 6))
imshow_grid(mg, z, plot_name='Synthetic road', var_name='Elevation', var_units='m',\
    grid_units=('m', 'm'), cmap='terrain', color_for_closed='black', vmin=0, vmax=4)
plt.xlabel('Road width (m)')
plt.ylabel('Road length (m)')

# Plot the sample nodes.
# plt.plot(mg.node_x[ditch_id], mg.node_y[ditch_id], 's', zorder=10, ms=3.5, \
#     clip_on=False, color='#44FFD1', markeredgecolor='k', label='Ditch')
# plt.plot(mg.node_x[rut_left_id], mg.node_y[rut_left_id], '^', zorder=10, ms=3.5, \
#     clip_on=False, color='#6153CC', markeredgecolor='k', label='Left rut')
# plt.plot(mg.node_x[rut_right_id], mg.node_y[rut_right_id], 'o', zorder=10, ms=3.75, \
#     clip_on=False, color='#A60067', markeredgecolor='k',label='Right rut')

# _ = ax.legend(loc='center right', bbox_to_anchor=(1.25,0.5), \
#     bbox_transform=plt.gcf().transFigure)
plt.tight_layout()
plt.show()

#%% Prep some variables for later
xsec_pre = mg.at_node['topographic__elevation'][4392*2:4428*2].copy() #half tire width
xsec_surf_pre = mg.at_node['surfacing__elev'][4392*2:4428*2].copy()
# xsec_pre = mg.at_node['topographic__elevation'][2196:2232].copy() #full tire width
mg_pre = mg.at_node['topographic__elevation'].copy()
active_pre = mg.at_node['active__depth'].copy()

X = mg.node_x.reshape(mg.shape)
Y = mg.node_y.reshape(mg.shape)
Z = z.reshape(mg.shape)

#%% Run the component
#half tire width
center = 39
half_width = 7 
full_tire = False

#full tire width
# center = 20 
# half_width = 4
# full_tire=True

tpe = TruckPassErosion(mg, center, half_width, full_tire, truck_num=5, \
    scat_loss=8e-5) #initialize component
fa = FlowAccumulator(mg,
                     surface='topographic__elevation',
                     runoff_rate=1.38889e-6, #5 mm/hr converted to m/s
                     flow_director='D8',
                     )

# # Choose parameter values for the stream power (SP) equation
# # and instantiate an object of the FastscapeEroder
K_sp=0.275# erodibility in SP eqtn; this is a guess
sp = FastscapeEroder(mg, 
                     K_sp=K_sp,
                     threshold_sp=0.0,
                     discharge_field='surface_water__discharge',
                     erode_flooded_nodes=True)


mask = road_flag

z_limit = mg.at_node['topographic__elevation'] - mg.at_node['active__depth']
intensity_arr=[]
dt_arr = []
dz_arr_masked=[]
dz_arr_cum_masked = []
sa_arr=[]
ss_arr=[]
sb_arr=[]
tracks=[]

z_ini_cum = mg.at_node['topographic__elevation'].copy()

#define how long to run the model
model_end = int(365) #days
for i in range(0, model_end): #loop through model days
    z_ini = mg.at_node['topographic__elevation'].copy()
    tpe.run_one_step()
    tracks.append(tpe.tire_tracks) #cam I concatenate the values rather than having separate arrays?
    print(tpe._truck_num)
    p_storm = 0.25
    
    chance = np.random.uniform()

    if chance > p_storm:
        intensity_arr.append(0)
        dt_arr.append(0)

        dz = z-z_ini
        dz_masked = z[mask]-z_ini[mask]
        dz_arr_masked.append(sum(dz_masked))

        dz_cum = z-z_ini_cum
        dz_cum_masked = z[mask]-z_ini_cum[mask]
        dz_arr_cum_masked.append(sum(dz_cum_masked))

    elif chance <= p_storm:
        dt = np.random.exponential(scale=1/6)
        print(dt)
        dt_arr.append(dt)
        intensity = np.random.exponential(scale=5)
        print(intensity)
        intensity_arr.append(intensity)

        mg.at_node['water__unit_flux_in'] = np.ones(540*72)*intensity*2.77778e-7
        fa.accumulate_flow()
        sp.run_one_step(dt)
        if any(z[tracks[i][0]] <= z_limit[tracks[i][0]]) or\
            any(z[tracks[i][1]] <= z_limit[tracks[i][1]]):
            z[tracks[i][0:2]] = z_limit[tracks[i][0:2]]
        
        dz = z-z_ini
        dz_masked = z[mask]-z_ini[mask]
        dz_arr_masked.append(sum(dz_masked))

        dz_cum = z-z_ini_cum
        dz_cum_masked = z[mask]-z_ini_cum[mask]
        dz_arr_cum_masked.append(sum(dz_cum_masked))
        
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(3, 6))
        im = imshow_grid(mg,'surface_water__discharge', var_name='Q', 
                     plot_name='Steady state Q, t = %i days' %i,
                     var_units='$m^3/s$', grid_units=('m','m'), 
                     cmap='Blues', vmin=0, vmax=5e-6, shrink=0.9)
        plt.xlabel('Road width (m)')
        plt.ylabel('Road length (m)')
        plt.tight_layout()
        # plt.savefig('output/Q_%i_days.png' %i)
        plt.show()

        dz[mask==0] = 0
        mg.add_field('dz', dz, at='node', units='m', clobber=True)
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(3, 6))
        plt.xlabel('Road width (m)')
        plt.ylabel('Road length (m)')
        im = imshow_grid(mg,'dz', var_name='dz', var_units='m', 
                     plot_name='Elevation change, t = %i days' %i,
                     grid_units=('m','m'), cmap='RdBu', vmin=-1e-6, 
                     vmax=1e-6, shrink=0.9)
        plt.xlabel('Road width (m)')
        plt.ylabel('Road length (m)')
        plt.tight_layout()
        # plt.savefig('output/dz_%i_days.png' %i)
        plt.show()

        dz_cum[mask==0] = 0
        mg.add_field('dz_cum', dz_cum, at='node', units='m', clobber=True)
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(3, 6))
        plt.xlabel('Road width (m)')
        plt.ylabel('Road length (m)')
        im = imshow_grid(mg,'dz_cum', var_name='Cumulative dz', var_units='m', 
                     plot_name='Elevation change, t = %i days' %i,
                     grid_units=('m','m'), cmap='RdBu', vmin=-0.001, 
                     vmax=0.001, shrink=0.9)
        plt.xlabel('Road width (m)')
        plt.ylabel('Road length (m)')
        plt.tight_layout()
        # plt.savefig('output/dz_cum_%i_days.png' %i)
        plt.show()

    sa = mg.at_node['topographic__elevation'][tracks[i][0:6]].mean()\
            - mg.at_node['surfacing__elev'][tracks[i][0:6]].mean() #active depth change?
    sa_arr.append(sa)
    ss = mg.at_node['surfacing__depth'][tracks[i][0:2]].mean() #-\
    ss_arr.append(ss)
    sb = mg.at_node['ballast__depth'][tracks[i][0:2]].mean() #-\
    sb_arr.append(sb)
#%%
plt.bar(range(0,model_end), np.multiply(intensity_arr,np.multiply(dt_arr,24)))
plt.xlabel('Day')
plt.ylabel('Rainfall [mm]')
plt.xlim(0,model_end)
plt.show()

plt.plot(range(0,model_end), intensity_arr)
plt.xlabel('Day')
plt.ylabel('Rainfall intensity [mm/hr]')
plt.xlim(0,model_end)
plt.show()
#%%
plt.plot(range(0,model_end), dz_arr_masked)
plt.plot(range(0,model_end), np.zeros(len(range(0,model_end))), '--', color='gray')
plt.xlabel('Day')
plt.ylabel('Total elevation change between time steps [m]')
plt.xlim(0,model_end)
plt.ylim(-0.05,0.05)

#%%
plt.plot(range(0,model_end), dz_arr_cum_masked)
plt.plot(range(0,model_end), np.zeros(len(range(0,model_end))), '--', color='gray')
plt.xlabel('Day')
plt.ylabel('Cumulative elevation change [m]')
plt.xlim(0,model_end)
# plt.ylim(-0.05,0.05)

total_dz = np.abs(min(dz_arr_cum_masked))
total_dV = total_dz*0.1475*0.1475
total_load = total_dV*2650
total_load_div = total_load/2
sed_load = total_load_div/(540*0.1475)

print('Sediment load per meter of road: ', sed_load)


#%%
plt.plot(range(0,model_end), sa_arr)
plt.xlabel('Day')
plt.ylabel('Average active depth [m]')
plt.xlim(0,model_end)
plt.ylim(0.019,0.02)
plt.show()

plt.plot(range(0,model_end), ss_arr)
plt.xlabel('Day')
plt.ylabel('Average surfacing depth [m]')
plt.xlim(0,model_end)
plt.ylim(0.228,0.23)
plt.show()

plt.plot(range(0,model_end), sb_arr)
plt.xlabel('Day')
plt.ylabel('Average ballast depth [m]')
plt.xlim(0,model_end)
plt.ylim(1.998,2.0)
plt.show()

#%%
# 
# Initialize model run information
# hydrograph_time = [0]
# discharge_ditch = [0]
# discharge_rut_left = [0]
# discharge_rut_right = [0]

# dt_knwv = 3600 #time step in seconds
# run_time_slices = range(0,40)
# elapsed_time = 1 #Set an initial time to avoid any 0 errors
# storm_duration = 86400 #length of storm in seconds; 24 hours
# # model_run_time = 3601        
# # Run the model; note that this will take a bit of time!
#     while elapsed_time <= storm_duration*i:
#         if elapsed_time < storm_duration:
#             knwv.run_one_step(dt_knwv)
#             dle.run_one_step(dt)
#         else:
#             knwv.runoff_rate = 1e-30 #Reset runoff_rate to be ~0; post-storm runoff
#             knwv.run_one_step(dt_knwv)
#             dle.run_one_step(dt)

#         for t in run_time_slices:
#             if elapsed_time == t*86400+1:
#                 time_model = t 
#                 imshow_grid(mg, 'surface_water_inflow__discharge', plot_name='Discharge, t = %i days' % time_model, 
#                     var_name='Q', var_units='$m^3/s$', grid_units=('m','m'), vmin=0, vmax=5e-6,
#                     cmap='Blues')
#                 # Plot the sample nodes.
#                 plt.plot(mg.node_x[ditch_id], mg.node_y[ditch_id], 's', zorder=10, ms=3.5, \
#                     clip_on=False, color='#44FFD1', markeredgecolor='k', label='Ditch')
#                 plt.plot(mg.node_x[rut_left_id], mg.node_y[rut_left_id], '^', zorder=10, ms=3.5, \
#                     clip_on=False, color='#6153CC', markeredgecolor='k', label='Left rut')
#                 plt.plot(mg.node_x[rut_right_id], mg.node_y[rut_right_id], 'o', zorder=10, ms=3.75, \
#                     clip_on=False, color='#A60067', markeredgecolor='k',label='Right rut')

#                 _ = ax.legend(loc='center right', bbox_to_anchor=(1.25,0.5), \
#                     bbox_transform=plt.gcf().transFigure)
#                 plt.show()
#                 fig, ax = plt.subplots(figsize=(15,10))
#                 drainage_plot(mg)
#                 plt.axis([0,10.5, 0, 0.5])
#                 plt.tight_layout()
#                 plt.show()

#         q_ditch = mg.at_node['surface_water_inflow__discharge'][ditch_id].item() 
#         q_rut_left = mg.at_node['surface_water_inflow__discharge'][rut_left_id].item() 
#         q_rut_right = mg.at_node['surface_water_inflow__discharge'][rut_right_id].item() 

#         hydrograph_time.append(elapsed_time/3600.)

#         discharge_ditch.append(q_ditch)
#         discharge_rut_left.append(q_rut_left)
#         discharge_rut_right.append(q_rut_right) 
                            
#         elapsed_time += dt_knwv #increase model time
#         print(elapsed_time)      

#     end = time.time()
#     print(f"Time taken to run the code was {end-start} seconds")

#%%
# #Plot the hydrograph
# ax = plt.gca()
# ax.tick_params(axis='both', which='both', direction='in', bottom='on', 
#                 left='on', top='on', right='on')
# ax.minorticks_on()

# ax.plot(hydrograph_time, discharge_ditch, '-', color='#44FFD1', markeredgecolor='k', label='Ditch')
# ax.plot(hydrograph_time, discharge_rut_left, '-', color='#6153CC', markeredgecolor='k', label='Left rut')
# ax.plot(hydrograph_time, discharge_rut_right, '-', color='#A60067', markeredgecolor='k',label='Right rut')
# ax.set(xlabel='Time (hr)', ylabel='Q ($m^3/s$)',
#         title='Hydrograph')
# ax.annotate('Max ditch Q = ' + str(np.max(np.round(discharge_ditch,5))) + ' $m^3/s$',(0.25,0.000525))
# ax.annotate('Max left rut Q = ' + str(np.max(np.round(discharge_rut_left,5))) + ' $m^3/s$',(0.175,0.000185))
# ax.annotate('Max right rut Q = ' + str(np.max(np.round(discharge_rut_right,5))) + ' $m^3/s$',(0.175,0.00016))
# _=ax.legend()
# plt.show()

#%%
#save rutted grid
# native_landlab.save_grid(mg, 'rutted_grid.grid', clobber=True)

#%% FlowAccumulator
# Instantiate Landlab FlowAccumulator using 'MFD' as the flow director, spatially distributed runoff field in m/s, and
# use the partition method 'square_root_of_slope' to match instantiation of FlowAccumulator in Kinwave component
# fa = FlowAccumulator(mg,
#                      surface='topographic__elevation',
#                      flow_director='MFD', #multiple flow directions
#                      runoff_rate=1.66667e-6, #6 mm/hr converted to m/s
#                      partition_method='square_root_of_slope')

# Run method to get drainage area and discharge at each node
# (drainage_area, discharge) = fa.accumulate_flow()

# Check to see how many of the core nodes are sinks
# sinks = mg.at_node['flow__sink_flag'][mg.core_nodes].sum()

# if sinks < 0.01*mg.core_nodes.sum():
#     print(sinks, r'of the core nodes are sinks. This is less than 1% of the core nodes. Code can continue.')
# else:
#     print(sinks, r'of the core nodes are sinks. This is more than 1% of the core nodes. \
#         Consider using a DEM that has been pre-processed for sinks.')

# # Obtain discharge at each outlet in m^3/s
# discharge_ditch = mg.at_node['surface_water__discharge'][ditch_id]
# discharge_rut_left= mg.at_node['surface_water__discharge'][rut_left_id]
# discharge_rut_right = mg.at_node['surface_water__discharge'][rut_right_id]

# print('Ditch discharge (m^3/s) =', np.round(discharge_ditch,5))
# print('Left rut discharge (m^3/s) =', np.round(discharge_rut_left,5))
# print('Right rut discharge (m^3/s) =', np.round(discharge_rut_right,5))

# # Map surface water discharge when outlet is at its maximum
# fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(3, 6))
# plt.xlabel('Road width (m)')
# plt.ylabel('Road length (m)')
# imshow_grid(mg,'surface_water__discharge', plot_name='Steady state Q', 
#             var_name='Q', var_units='$m^3/s$', grid_units=('m','m'), 
#             cmap='Blues', vmin=0, vmax=5)

# # Plot the sample nodes.
# plt.plot(mg.node_x[ditch_id], mg.node_y[ditch_id], 's', zorder=10, ms=3.5, \
#     clip_on=False, color='#44FFD1', markeredgecolor='k', label='Ditch')
# plt.plot(mg.node_x[rut_left_id], mg.node_y[rut_left_id], '^', zorder=10, ms=3.5, \
#     clip_on=False, color='#6153CC', markeredgecolor='k', label='Left rut')
# plt.plot(mg.node_x[rut_right_id], mg.node_y[rut_right_id], 'o', zorder=10, ms=3.75, \
#     clip_on=False, color='#A60067', markeredgecolor='k',label='Right rut')

# plt.tight_layout()

# _ = ax.legend(loc='center left', bbox_to_anchor=(1.25,0.5), \
#     bbox_transform=plt.gcf().transFigure)
# plt.show()

# #%%
# import time

# knwv = KinwaveImplicitOverlandFlow(mg, runoff_rate=6, roughness=n, depth_exp=5/3) #Feed initial component a runoff rate of 2 mm/hr

# # Initialize model run information
# hydrograph_time = [0]
# discharge_ditch = [0]
# discharge_rut_left = [0]
# discharge_rut_right = [0]

# dt = 60 #time step in seconds
# run_time_slices = (1,61,601,2401,3601)
# elapsed_time = 1 #Set an initial time to avoid any 0 errors
# storm_duration = 2700 #length of storm in seconds; 24 hours
# model_run_time = 3601
    
# # Run the model; note that this will take a bit of time!
# start = time.time()

# while elapsed_time <= model_run_time:
#     if elapsed_time < storm_duration:
#         knwv.run_one_step(dt)
#     else:
#         knwv.runoff_rate = 1e-30 #Reset runoff_rate to be ~0; post-storm runoff
#         knwv.run_one_step(dt)

#     for t in run_time_slices:
#         if elapsed_time == t:
#             time_model = t/60 
#             imshow_grid(mg, 'surface_water_inflow__discharge', plot_name='Discharge, t = %i min' % time_model, 
#                 var_name='Q', var_units='$m^3/s$', grid_units=('m','m'), vmin=0, vmax=0.00055,
#                 cmap='Blues')
#             # Plot the sample nodes.
#             plt.plot(mg.node_x[ditch_id], mg.node_y[ditch_id], 's', zorder=10, ms=3.5, \
#                 clip_on=False, color='#44FFD1', markeredgecolor='k', label='Ditch')
#             plt.plot(mg.node_x[rut_left_id], mg.node_y[rut_left_id], '^', zorder=10, ms=3.5, \
#                 clip_on=False, color='#6153CC', markeredgecolor='k', label='Left rut')
#             plt.plot(mg.node_x[rut_right_id], mg.node_y[rut_right_id], 'o', zorder=10, ms=3.75, \
#                 clip_on=False, color='#A60067', markeredgecolor='k',label='Right rut')

#             _ = ax.legend(loc='center right', bbox_to_anchor=(1.25,0.5), \
#                 bbox_transform=plt.gcf().transFigure)
#             plt.show()
#             # fig, ax = plt.subplots(figsize=(15,10))
#             # drainage_plot(mg)
#             # plt.axis([0,10.5, 0, 0.5])
#             # plt.tight_layout()
#             # plt.show()

#     q_ditch = mg.at_node['surface_water_inflow__discharge'][ditch_id].item() 
#     q_rut_left = mg.at_node['surface_water_inflow__discharge'][rut_left_id].item() 
#     q_rut_right = mg.at_node['surface_water_inflow__discharge'][rut_right_id].item() 

#     hydrograph_time.append(elapsed_time/3600.)

#     discharge_ditch.append(q_ditch)
#     discharge_rut_left.append(q_rut_left)
#     discharge_rut_right.append(q_rut_right) 
                        
#     elapsed_time += dt #increase model time
            

# end = time.time()
# print(f"Time taken to run the code was {end-start} seconds")

# #%%
# #Plot the hydrograph
# ax = plt.gca()
# ax.tick_params(axis='both', which='both', direction='in', bottom='on', 
#                left='on', top='on', right='on')
# ax.minorticks_on()

# ax.plot(hydrograph_time, discharge_ditch, '-', color='#44FFD1', markeredgecolor='k', label='Ditch')
# ax.plot(hydrograph_time, discharge_rut_left, '-', color='#6153CC', markeredgecolor='k', label='Left rut')
# ax.plot(hydrograph_time, discharge_rut_right, '-', color='#A60067', markeredgecolor='k',label='Right rut')
# ax.set(xlabel='Time (hr)', ylabel='Q ($m^3/s$)',
#         title='Hydrograph')
# ax.annotate('Max ditch Q = ' + str(np.max(np.round(discharge_ditch,5))) + ' $m^3/s$',(0.25,0.000525))
# ax.annotate('Max left rut Q = ' + str(np.max(np.round(discharge_rut_left,5))) + ' $m^3/s$',(0.175,0.000185))
# ax.annotate('Max right rut Q = ' + str(np.max(np.round(discharge_rut_right,5))) + ' $m^3/s$',(0.175,0.00016))
# _=ax.legend()
# plt.show()

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

# # Plot the sample nodes.
# plt.plot(mg.node_x[ditch_id], mg.node_y[ditch_id], 's', zorder=10, ms=3.5, \
#     clip_on=False, color='#44FFD1', markeredgecolor='k', label='Ditch')
# plt.plot(mg.node_x[rut_left_id], mg.node_y[rut_left_id], '^', zorder=10, ms=3.5, \
#     clip_on=False, color='#6153CC', markeredgecolor='k', label='Left rut')
# plt.plot(mg.node_x[rut_right_id], mg.node_y[rut_right_id], 'o', zorder=10, ms=3.75, \
#     clip_on=False, color='#A60067', markeredgecolor='k',label='Right rut')

# _ = ax.legend(loc='center right', bbox_to_anchor=(1.25,0.5), \
#     bbox_transform=plt.gcf().transFigure)
plt.tight_layout()
plt.show()

#%% Prepping for a difference map
diff = mg_pre - mg.at_node['topographic__elevation']
