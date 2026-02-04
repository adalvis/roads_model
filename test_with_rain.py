#%%
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.widgets import Slider

from landlab import RasterModelGrid, imshow_grid
from landlab.components import OverlandFlowTransporter, FlowAccumulator, TruckPassErosion, DepressionFinderAndRouter
np.set_printoptions(threshold=np.inf)
from erodible_grid import Erodible_Grid

#%%
# Parameters
run_duration = 20   # run duration = days + 1

# call parameters from .csv file
parameters = pd.read_csv("parameters_WY2024.csv")
# site=parameters.loc[parameters["Site Name"] == "KID13"].iloc[0]   # was also doing runs for this site
site=parameters.loc[parameters["Site Name"] == "BISH05"].iloc[0]

# constants
rho_w=1000
rho_s=2650
g=9.81

# grid creation
cell_spacing = 0.1475 # cell width or length dimension in meters
cell_area = cell_spacing**2
nrows = 540 # number of rows in the grid
ncols = 72  # number of columns in the grid

# initialize road layer depths
Sa_ini = 0.01 # active depth in m
Ss_ini = 0.23 # surfacing depth in m
Sb_ini = 2    # ballast depth in m

# longitudinal_slope = site["Road Gradient"]/100
longitudinal_slope = 0.125
porosity = 0.35

# initialize average number of truck passes per day for truck pass erosion
truck_num_ini = 4

#We're using half tire width for node spacing
center = 40
half_width = 7 
full_tire = False

ditch_n = 0.1   # manning's roughness of the ditch, based on ditch BMP
ditch_grain_n = 0.05 # manning's roughness for the grains in the ditch
n_c = 0.05   
n_f = 0.015

# fractions of fine and coarse grains in the active layer
# initially 50/50
f_af = 0.25
f_ac = 0.75

# d50 = site["d50 (mm)"]/1000

d50_arr = [0.000018, 0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008, 0.0009, 0.001, 0.005]
tauc_arr = [0.05244426, 0.1456785, 0.1780515, 0.19909395, 0.226611, 0.258984, 0.291357, 0.3625776, 0.453222, 0.5244426, 0.5989005, 3.6419625]

# index the desired d50 and tau_c values, or find d50 for the site and find tau_c from chart and input here
index = 3
d50 = d50_arr[index] # [m] 
tau_c = tauc_arr[index] # tau_c dependent on d_50

#%%
# Data load and rainfall site selection

# to call high intensity
high_intensity_index = 144  # day index determined to be the beginning of a high intensity rainfall period for this site. 

# intensity data in mm/hr with dt from the dt document
intensity_2024 = pd.read_csv("WY2024_RG_daily_intensity.csv")
# change the site name and index values to get different sites and dates
# intensity_90 = intensity_2024["RG_KID1316_mm"].iloc[:90].values  # first 90 days for site "RG_BISH05"
intensity_90 = intensity_2024["RG_BISH05_mm"].iloc[high_intensity_index:].values 

# daily dt in hours
dt_2024_hours = pd.read_csv("WY2024_RG_daily_dt.csv")
# dt_2024_hours_90 = dt_2024_hours["RG_KID1316_mm"].iloc[:90].values # first 90 days for site "RG_BISH05"
dt_2024_hours_90 = dt_2024_hours["RG_BISH05_mm"].iloc[high_intensity_index:].values  
# convert to dt to days
dt_2024 = np.array(dt_2024_hours_90)/24

#%% Run method to create grid; add new fields
eg = Erodible_Grid(nrows=nrows, ncols=ncols,\
    spacing=cell_spacing, full_tire=full_tire, long_slope=longitudinal_slope) # update long slope to change DEM

mg, z, road_flag, n = eg() 

noise_amplitude=0.005   # init = 0.005
road = road_flag==1

seed = 1        # this isn't used to calculate rainfall, it ensures that anything else randomly generated isn't 
                # going to be changing between runs. I changed TPE to not have a randomly generated
                # truck number in that file.
np.random.seed(seed)
random=np.random.rand(
    len(z[road])
)
z[road] += noise_amplitude * random #z is the road elevation

#Add depth fields that will update in the component; these are the initial conditions
mg.at_node['active__depth'] = np.ones(nrows*ncols)*Sa_ini #This is the most likely to change; the active layer is essentially 0.005 m right now.
mg.at_node['surfacing__depth'] = np.ones(nrows*ncols)*Ss_ini
mg.at_node['ballast__depth'] = np.ones(nrows*ncols)*Sb_ini #This depth can technically be anything; it just needs to be much larger than the active layer and the surfacing layer depths.

#Add absolute elevation fields that will update based on z updates
mg.at_node['active__elev'] = z
mg.at_node['surfacing__elev'] = z - mg.at_node['active__depth']
mg.at_node['ballast__elev'] = z - mg.at_node['active__depth']\
     - mg.at_node['surfacing__depth']

#%% Define locations of ruts, half-road, full-road, and ditch
ruts = [mg.nodes[1:, 26:40], mg.nodes[1:, 41:55]]
half_road = mg.nodes[1:, 9:41]
full_road = mg.nodes[1:, 9:72]
ditch = mg.nodes[1:, 1:9]

#%% Plot initial grid
# Set up the figure.
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(3, 6))
imshow_grid(mg, z, plot_name='Synthetic road', var_name='Elevation', var_units='m',\
    grid_units=('m', 'm'), cmap='terrain', color_for_closed='black', vmin=0, vmax=9)
plt.xlabel('Road width (m)')
plt.ylabel('Road length (m)')
plt.tight_layout()
plt.show()

#%% Instantiate components
tpe = TruckPassErosion(mg, center, half_width, full_tire, truck_num=truck_num_ini, \
    scat_loss=8e-5, f_af=f_af, f_ac=f_ac) #initialize component, 

df_init = DepressionFinderAndRouter(mg, reroute_flow = True)
df_init.map_depressions()

fa = FlowAccumulator(mg, surface='topographic__elevation', \
    flow_director="FlowDirectorD8", runoff_rate=1.538889e-6, \
    depression_finder=None)  # runoff rate = 1.538889e-6

oft = OverlandFlowTransporter(mg, porosity=porosity, d50=d50, \
    longitudinal_slope = longitudinal_slope, tau_c=tau_c, n_c=n_c, n_f=n_f)        # trying to run component with values we can change in this file

#%% Run the model!

# prep fields

mask = road_flag
z_limit = mg.at_node['topographic__elevation'] - mg.at_node['active__depth'] #You may need to use this to create a layer limit
intensity_arr=[]
dt_arr = []
dz_arr=[]
dz_arr_cum = []

sa_arr=[]
ss_arr=[]
sb_arr=[]

# mask to exclude rut values from full road
mask1 = np.ones(mg.number_of_nodes, dtype=bool)
mask1[ruts] = False

# depth averages for each layer in the ruts
active_depth_ruts = []
surfacing_depth_ruts = []
ballast_depth_ruts = []

# outflux and discharge arrays
sediment_outflux_channel = []
sediment_influx_channel = []
sediment_outflux_ruts = []
channel_discharge_arr = []

# shear stresses (averages)
avg_shear_stress_ruts = []
avg_shear_stress_channel = []
avg_shear_stress_road = []

# manning's roughness averages
avg_n_ruts = []
avg_n_channel = []
avg_n_road = []

# sediment load in the ruts due to TPE
tpe_load_ruts = []
tpe_load_ruts_cum = []

# for futures slider plots
water_depth_frames = []
dz_cum_frames = []
active_fines_frames = []
active_depth_frames = []

# shear stress partitioning coefficient averages
fs_avg_ruts = []
fs_avg_channel = []
fs_avg_road = []

# look at sediment rate of change for road vs ruts
dzdt_avg_ruts = []
dzdt_avg_road = []

truck_num=0     
z_ini_cum = mg.at_node['topographic__elevation'].copy()

#%% Run the model!
# Main loop

start = time.time()
for i in range(0, run_duration):        # daily time step, every time step there are a certain number of truck passes = truck_num
    active_init = mg.at_node['active__depth'].copy()
    surfacing_init = mg.at_node['surfacing__depth'].copy()
    ballast_init = mg.at_node['ballast__depth'].copy()
    z_ini = mg.at_node['topographic__elevation'].copy()

    tpe.run_one_step()

    truck_num += tpe._truck_num
    print(tpe._truck_num)
    
    # use only 90 day chunks of the datasets

    # if rainfall intensity is zero that day, treat it like a no-storm
    intensity = intensity_90[i]  # use the i-th day's intensity
    dt_day = dt_2024[i] # use the i-th day's time step

    rain_m_per_s = intensity * 2.77778e-7 # conversion to m/s

    mg.at_node['water__unit_flux_in'] = np.ones(mg.number_of_nodes) * rain_m_per_s

    fa.accumulate_flow()

    if intensity <= 0:
        # mg.at_node['water__unit_flux_in'] = np.zeros(mg.number_of_nodes)

        intensity_arr.append(0)
        dt_arr.append(0)

        dz = z - z_ini
        dz_arr.append(sum(dz[mask]))

        dz_cum = z - z_ini_cum
        dz_arr_cum.append(sum(dz_cum[mask]))

        # append flux vectors
        sediment_outflux_channel.append(0)
        sediment_outflux_ruts.append(0)
        sediment_influx_channel.append(0)
        
        # append shear stress vectors
        avg_shear_stress_ruts.append(0)
        avg_shear_stress_channel.append(0)
        avg_shear_stress_road.append(0)

        # append manning's roughness vectors
        avg_n_ruts.append(np.nanmean(mg.at_node['total__roughness'][ruts]))
        avg_n_channel.append(ditch_n)
        avg_n_road.append(np.nanmean(mg.at_node['total__roughness'][full_road]))

        # appedn shear stress partitioning coefficient
        fs_avg_ruts.append(np.nanmean(mg.at_node['shear_stress__partitioning'][ruts]))
        fs_avg_channel.append(np.nanmean(mg.at_node['shear_stress__partitioning'][ditch]))
        fs_avg_road.append(np.nanmean(mg.at_node['shear_stress__partitioning'][full_road]))

        # append sediment rate of change
        dzdt_avg_ruts.append(np.nanmean(mg.at_node['sediment__rate_of_change'][ruts]))
        dzdt_avg_road.append(np.nanmean(mg.at_node['sediment__rate_of_change'][mask1]))

        # append tpe loading
            # multiply change in layer thickness in the ruts by cell area*sediment density*1-porosity
        tpe_load_ruts.append((np.multiply(tpe._active_dz[ruts], cell_area*rho_s*(1-porosity))).sum()) # gives increase in mass in the active layer due to truck passes per time step
        
        # for plotting layer depths
        active_depth = np.average(mg.at_node['active__depth'][ruts])
        active_depth_ruts.append(active_depth)
        surfacing_depth = np.average(mg.at_node['surfacing__depth'][ruts])
        surfacing_depth_ruts.append(surfacing_depth)
        ballast_depth = np.average(mg.at_node['ballast__depth'][ruts])
        ballast_depth_ruts.append(ballast_depth)

    else:
        dt = dt_day  
        dt_arr.append(dt)
                  
        intensity_arr.append(intensity)
        print(f"Day {i}: Intensity = {intensity:.2f} mm/hr")

        #=================================Calculate overland flow transport=================================
        oft.run_one_step(dt)
    
        dz = z-z_ini # calculate elevation change at each time step
        dz_arr.append(sum(dz[mask]))

        dz_cum = z-z_ini_cum # calculate cumulative elevation change
        dz_arr_cum.append(sum(dz_cum[mask])) 

        # for plotting layer depths
        active_depth = np.average(mg.at_node['active__depth'][ruts])
        active_depth_ruts.append(active_depth)
        surfacing_depth = np.average(mg.at_node['surfacing__depth'][ruts])
        surfacing_depth_ruts.append(surfacing_depth)
        ballast_depth = np.average(mg.at_node['ballast__depth'][ruts])
        ballast_depth_ruts.append(ballast_depth)
        
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(3, 6))
        im = imshow_grid(mg,'water__depth', var_name='Depth', 
                     plot_name='Water depth, t = %i days' %i,
                     var_units='$m$', grid_units=('m','m'), 
                     cmap='Blues', vmin=0, vmax=0.0005, shrink=0.9)
        plt.xlabel('Road width (m)')
        plt.ylabel('Road length (m)')
        plt.tight_layout()
        plt.savefig('output/w_%i_days.png' %i)
        plt.close()
        # plt.show()

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(3, 6))
        im = imshow_grid(mg,'surface_water__discharge', var_name='Discharge', 
                     plot_name='Discharge, t = %i days' %i,
                     var_units='$m/s^3$', grid_units=('m','m'), 
                     cmap='Blues', vmin=0, vmax=0.00001, shrink=0.9)
        plt.xlabel('Road width (m)')
        plt.ylabel('Road length (m)')
        plt.tight_layout()
        plt.savefig('output/Q_%i_days.png' %i)
        plt.close()
        # plt.show()

        # fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(3, 6))
        # im = imshow_grid(mg,'shear_stress__partitioning', var_name='Shear stress partitioning ratio', 
        #              plot_name='Shear Stress Partitioning, t = %i days' %i,
        #              var_units='$[-]]$', grid_units=('m','m'), 
        #              cmap='Blues', vmin=0, vmax=1, shrink=0.9)
        # plt.xlabel('Road width (m)')
        # plt.ylabel('Road length (m)')
        # plt.tight_layout()
        # # plt.savefig('output/Q_%i_days.png' %i)
        # plt.show()

        # mg.add_field('dz_cum', dz_cum, at='node', units='m', clobber=True)
        # fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(3, 6))
        # plt.xlabel('Road width (m)')
        # plt.ylabel('Road length (m)')
        # im = imshow_grid(mg,'dz_cum', var_name='Cumulative dz', var_units='m', 
        #              plot_name='Elevation change, t = %i days' %i,
        #              grid_units=('m','m'), cmap='RdBu', vmin=-0.001, 
        #              vmax=0.001, shrink=0.9)
        # plt.xlabel('Road width (m)')
        # plt.ylabel('Road length (m)')
        # plt.tight_layout()
        # # plt.savefig('output/dz_cum_%i_days.png' %i)
        # plt.show()

        # fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(3, 6))
        # plt.xlabel('Road width (m)')
        # plt.ylabel('Road length (m)')
        # im = imshow_grid(mg,'active__fines', var_name='Active fines', var_units='m', 
        #              plot_name='Fines depth, t = %i days' %i,
        #              grid_units=('m','m'), cmap='pink', vmin=0.002, 
        #              vmax=0.003)
        # plt.xlabel('Road width (m)')
        # plt.ylabel('Road length (m)')
        # plt.tight_layout()
        # # plt.savefig('output/dz_cum_%i_days.png' %i)
        # plt.show()

        # fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(3, 6))
        # plt.xlabel('Road width (m)')
        # plt.ylabel('Road length (m)')
        # im = imshow_grid(mg,'active__depth', var_name='Active depth', var_units='m', 
        #              plot_name='Active depth, t = %i days' %i,
        #              grid_units=('m','m'), cmap='pink', vmin=0.0025, vmax=0.02)
        # plt.xlabel('Road width (m)')
        # plt.ylabel('Road length (m)')
        # plt.tight_layout()
        # # plt.savefig('output/dz_cum_%i_days.png' %i)
        # plt.show()

        # # for future slider plots
        # water_depth_frames.append(
        #     mg.at_node["water__depth"].copy())
        # dz_cum_frames.append(
        #     dz_cum.copy())
        # active_fines_frames.append(
        #     mg.at_node["active__fines"].copy())
        # active_depth_frames.append(
        #     mg.at_node["active__depth"].copy())

        # diagnostic prints
        slope1 = mg.at_node['topographic__steepest_slope']
        water_depth = mg.at_node['water__depth']
        print("Slope range:", slope1.min(), slope1.max())
        print("Water Depth:", water_depth.min(), water_depth.max())
        print("NaN Manning values:", np.isnan(oft._n_f).sum(), np.isnan(oft._n_t).sum())
        print("Zero Manning values:", np.sum(oft._n_f == 0), np.sum(oft._n_t == 0))
        print("Negative Manning values:", np.sum(oft._n_f < 0), np.sum(oft._n_t < 0))
        elev = mg.at_node['topographic__elevation']
        slope = mg.at_node['topographic__steepest_slope']

        print("ELEV min/max/any_nan/any_inf:", elev.min(), elev.max(), np.isnan(elev).any(), np.isinf(elev).any())
        print("dzdt min/max:", np.nanmin(oft._dzdt), np.nanmax(oft._dzdt))
        # diagnostics end
        print("sediment outflux min/max:", np.nanmin(oft._sediment_outflux), np.nanmax(oft._sediment_outflux))

        #=================================Calculate channelized flow transport=================================
        slope=longitudinal_slope
        d50=d50
        tau_c=tau_c
        
        surface_water_discharge = mg.at_node['surface_water__discharge']
        channel_discharge = surface_water_discharge[ditch].sum(axis=0).max()
        channel_discharge_arr.append(channel_discharge)
        
        for k in range(len(surface_water_discharge)):
            if surface_water_discharge[k] > 0 and road_flag[k] == 0:
                mg.at_node['grain__roughness'][k] = ditch_grain_n   
                mg.at_node['total__roughness'][k] = ditch_n #this can change according to the treatment
                mg.at_node['shear_stress__partitioning'][k] = ((mg.at_node['grain__roughness'][k]) /\
                    (mg.at_node['total__roughness'][k]))**(24/13)
        
        
        R = ((mg.at_node['total__roughness'][ditch].max()*channel_discharge)/(np.sqrt((6*slope)/0.718)))**(6/13)

        # calculates effective shear stress
        shear_stress = rho_w*g*R*slope*mg.at_node['shear_stress__partitioning'][ditch]

        if np.nanmax(shear_stress) > tau_c:
            sediment_outflux = (((10**(-4.348))/(rho_s*((d50)**(0.811))))\
                *(np.nanmax(shear_stress)-tau_c)**(2.457))*np.sqrt((6/0.718)*R)#[m^3/s]
        else:
            sediment_outflux=0
        
        # append flux vectors
        sediment_outflux_channel.append(sediment_outflux)
        sediment_outflux_ruts.append((mg.at_node["sediment__volume_influx"][mg.nodes[1,9:41]]).sum())   # indexing the second from the top row, and the columns of the ruts
        # sediment_outflux_half_road.append
        sediment_influx_channel.append((mg.at_node["sediment__volume_influx"][mg.nodes[1:,8]]).sum())   # indexing all rows except the top and column 8, edge of road

        # append effective shear stress vectors
        mg.at_node['shear_stress'] = oft._shear_stress  # effective shear stress
        avg_shear_stress_ruts.append(np.nanmean(mg.at_node['shear_stress'][ruts])) # mean of the oft calculated effective shear stress for the ruts
        # need to add calculation for channel
        avg_shear_stress_channel.append(np.nanmean(shear_stress))    # appending channel effective shear stress with above effective shear stress calc for mg.nodes[1:, 1:9]
        avg_shear_stress_road.append(np.nanmean(mg.at_node['shear_stress'][full_road]))      # full road average of effective shear stress

        # append manning's n vectors
        avg_n_ruts.append(np.nanmean(mg.at_node['total__roughness'][ruts]))
        avg_n_channel.append(ditch_n)
        avg_n_road.append(np.nanmean(mg.at_node['total__roughness'][full_road]))

        # append shear stress partitioning coefficient vectors        
        fs_avg_ruts.append(np.nanmean(mg.at_node['shear_stress__partitioning'][ruts]))
        fs_avg_channel.append(np.nanmean(mg.at_node['shear_stress__partitioning'][ditch]))
        fs_avg_road.append(np.nanmean(mg.at_node['shear_stress__partitioning'][full_road]))
        
        # append tpe loading
        tpe_load_ruts.append((np.multiply(tpe._active_dz[ruts], cell_area*rho_s*(1-porosity))).sum())
        
        # append sediment rate of change
        dzdt_avg_ruts.append(np.nanmean(mg.at_node['sediment__rate_of_change'][ruts]))
        dzdt_avg_road.append(np.nanmean(mg.at_node['sediment__rate_of_change'][mask1]))
    # cumulative tpe load in the ruts over time
    tpe_load_ruts_cum = np.cumsum(tpe_load_ruts)

    sa = (mg.at_node['active__depth'][mask]).mean()-active_init[mask].mean()
    sa_arr.append(sa)
    ss = mg.at_node['surfacing__depth'][mask].mean()-surfacing_init[mask].mean()
    ss_arr.append(ss)
    sb = mg.at_node['ballast__depth'][mask].mean()-ballast_init[mask].mean()
    sb_arr.append(sb)

wall_time = time.time() - start
print("Wall time for run:", wall_time, "s")

# %% loop plots - needs more work to make slider plots
from PIL import Image
import glob

path = "output/"
images_w=[]
images_Q=[]
for file in glob.glob(path+'w*.png'):
    images_w.append(file)
for file in glob.glob(path+'Q*.png'):
    images_Q.append(file)

def number(filename):
    return int(filename[9:-9])

images_w = sorted(images_w, key=number)
images_Q = sorted(images_Q, key=number)

# Create a list of image objects
image_list_w = [Image.open(file) for file in images_w]
image_list_Q = [Image.open(file) for file in images_Q]

# Save the first image as a GIF file
image_list_w[0].save(
            'water_depth.gif',
            save_all=True,
            append_images=image_list_w[1:], # append rest of the images
            duration=1000, # in milliseconds
            loop=0)

image_list_Q[0].save(
            'discharge.gif',
            save_all=True,
            append_images=image_list_Q[1:], # append rest of the images
            duration=1000, # in milliseconds
            loop=0)
#%% Rainfall plots
plt.bar(range(0,run_duration), np.multiply(intensity_arr,np.multiply(dt_arr,24)))
plt.xlabel('Day')
plt.ylabel('Rainfall [mm]')
plt.xlim(0,run_duration)
plt.show()

plt.plot(range(0,run_duration), intensity_arr)
plt.xlabel('Day')
plt.ylabel('Rainfall intensity [mm/hr]')
plt.xlim(0,run_duration)
plt.show()

#%% Mass change plots

# convert dt_arr to seconds from days
dt_arr_secs = np.array(dt_arr)*86400
road_mass_change = np.multiply(dz_arr, (cell_area*rho_s*(1-porosity)))/2
ditch_mass_change = (np.array(sediment_influx_channel) - np.array(sediment_outflux_channel))*rho_s*dt_arr_secs

fig, ax = plt.subplots(1,2, figsize=(9,4))

# plot total mass change between time steps on the road
ax[0].plot(range(0,run_duration), road_mass_change) # added porosity consideration
ax[0].plot(range(0,run_duration), np.zeros(len(range(0,run_duration))), '--', color='gray')
ax[0].set_xlabel('Day')
ax[0].set_ylabel('$\Delta$ mass between time steps [$kg$]')
ax[0].set_xlim(0,run_duration)
ax[0].set_title('(a) Half Road')

# plot total mass change between time steps along the ditch line
ax[1].plot(range(0,run_duration), ditch_mass_change)
ax[1].plot(range(0,run_duration), np.zeros(len(range(0,run_duration))), '--', color='gray')
ax[1].set_xlabel('Day')
ax[1].set_ylabel('$\Delta$ mass between time steps [$kg$]')
ax[1].set_xlim(0,run_duration)
ax[1].set_title('(b) Ditch Line')
plt.tight_layout()
plt.show()


#%%

fig, ax = plt.subplots(1,2, figsize=(9,4))

cum_road_mass_change_dz = np.multiply(dz_arr_cum,cell_area*rho_s*(1-porosity))/2
cum_road_mass_change_oft = ((np.array(sediment_influx_channel)*rho_s*dt_arr_secs).cumsum()\
        +(np.array(sediment_outflux_ruts)*rho_s*dt_arr_secs).cumsum())

ax[0].plot(range(0,run_duration), cum_road_mass_change_dz) # added porosity consideration
ax[0].plot(range(0,run_duration), np.zeros(len(range(0,run_duration))), '--', color='gray')
ax[0].set_xlabel('Day')
ax[0].set_ylabel('Cumulative mass change (from dz) - \nhalf road [$kg$]')
ax[0].set_xlim(0,run_duration)

ax[1].plot(range(0,run_duration), -cum_road_mass_change_oft)
ax[1].plot(range(0,run_duration), np.zeros(len(range(0,run_duration))), '--', color='gray')
ax[1].set_xlabel('Day')
ax[1].set_ylabel('Cumulative mass change (from OFT)- \nhalf road [$kg$]')
ax[1].set_xlim(0,run_duration)
plt.tight_layout()
plt.show()

#%%

mass_ditch_inflow = (np.multiply(sediment_influx_channel, dt_arr_secs)*rho_s)
mass_ditch_outflow = -(np.multiply(sediment_outflux_channel,dt_arr_secs)*rho_s)
cum_mass_ditch_inflow = (np.multiply(sediment_influx_channel, dt_arr_secs)*rho_s).cumsum()
cum_mass_ditch_outflow = -(np.multiply(sediment_outflux_channel,dt_arr_secs)*rho_s).cumsum()

# plot mass inflow into ditch - cumulative and between time steps
plt.plot(range(0,run_duration), mass_ditch_inflow)
plt.plot(range(0,run_duration), np.zeros(len(range(0,run_duration))), '--', color='gray')
plt.xlabel('Day')
plt.ylabel('Mass inflow between time steps- \nrouted into ditch [$kg$]')
plt.xlim(0,run_duration)
plt.show()

# plot mass outflow out of ditch - cumulative and between time steps
plt.plot(range(0,run_duration), mass_ditch_outflow)
plt.plot(range(0,run_duration), np.zeros(len(range(0,run_duration))), '--', color='gray')
plt.xlabel('Day')
plt.ylabel('Mass outflow between time steps - \nrouted out of ditch [$kg$]')
plt.xlim(0,run_duration)
plt.show()


plt.plot(range(0,run_duration), cum_mass_ditch_inflow)
plt.plot(range(0,run_duration), np.zeros(len(range(0,run_duration))), '--', color='gray')
plt.xlabel('Day')
plt.ylabel('Cumulative mass inflow - \nrouted into ditch [$kg$]')
plt.xlim(0,run_duration)
plt.show()


plt.plot(range(0,run_duration), cum_mass_ditch_outflow)
plt.plot(range(0,run_duration), np.zeros(len(range(0,run_duration))), '--', color='gray')
plt.xlabel('Day')
plt.ylabel('Cumulative mass outflow - \nrouted out of ditch [$kg$]')
plt.xlim(0,run_duration)
plt.show()

#%%
# additional plots

# scattering, pumping, crushing plots
# plot cumulative fluxes
# this plot needs to be looked at!
# q_cs = tpe._q_cs    # [m/truck] crushing
# sum_q_cs = q_cs[mg.nodes[1,9:41]].sum() # summed across the rut for half road
# q_ps = tpe._q_ps    # [m/truck] pumping
# sum_q_ps = q_ps[mg.nodes[1,9:41]].sum() # summed across the rut for half road
# # q are in m/truck not m/s so shouldn't be multiplied by dt_arr, should be multiplied by truck number?
# plt.plot(range(0,run_duration), (np.multiply(sum_q_cs, dt_arr)*rho_s).cumsum(), color='pink', label='crushing')
# plt.plot(range(0,run_duration), (np.multiply(sum_q_ps, dt_arr)*rho_s).cumsum(), color='green', label='pumping')
# plt.plot(range(0,run_duration), np.zeros(len(range(0,run_duration))), '--', color='gray')
# plt.xlabel('Day')
# plt.ylabel('Pumping and Crushing Fluxes [kg]')
# plt.xlim(0,run_duration)
# plt.legend()
# plt.show()

# plot sediment load to the active layer in the ruts from truck passes per time step
    # this will vary when we allow truck number to vary
plt.plot(range(0,run_duration), tpe_load_ruts)
plt.xlabel('Day')
plt.ylabel('Sediment load to the active layer of the ruts \nfrom tpe per time step [$kg$]')
plt.xlim(0,run_duration)
plt.show()

# plot cumulative sediment load to the active layer in the ruts from truck passes over time
    # won't be near linear when truck number is allowed to vary 
plt.plot(range(0,run_duration), tpe_load_ruts_cum)
plt.xlabel('Day')
plt.ylabel('Cumulative sediment load to the active layer \nof the ruts from tpe [$kg$]')
plt.xlim(0,run_duration)
plt.show()

#%% shear stress plots

# plot shear stress partitioning coefficient over time
plt.plot(range(0,run_duration), fs_avg_road, label='Full Road')
plt.plot(range(0,run_duration), fs_avg_ruts, label='Ruts')
plt.xlabel('Day')
plt.ylabel('Average Shear Stress Partitioning \nCoefficient in the Ruts: f_s')
plt.xlim(0,run_duration)
plt.legend()
plt.show()

# plot average shear stresses on the road
# Perhaps do max shear stresses? Is this helpful?
plt.plot(range(0,run_duration), avg_shear_stress_road, label = 'Full Road')
plt.plot(range(0,run_duration), avg_shear_stress_ruts, label = 'Ruts')
plt.axhline(tau_c, linestyle='--', color='gray', label='Critical Shear Stress Threshold')
plt.xlabel('Day')
plt.ylabel('Road - Average Effective Shear Stress [$Pa$]')
plt.xlim(0,run_duration)
plt.legend()
plt.show()

# plot average shear stress in the ditch
plt.plot(range(0,run_duration), avg_shear_stress_channel, label = 'Ditch')
plt.axhline(tau_c, linestyle='--', color='gray', label='Critical Shear Stress Threshold')
plt.xlabel('Day')
plt.ylabel('Ditch - Average Effective Shear Stress [$Pa$]')
plt.xlim(0,run_duration)
plt.legend()
plt.show()

#%% average manning's roughness plots

# Note: Days with no rainfall will present as n = 0 due to the OFT setup, we could change this but it would increase run time slightly.

# plot the average manning's roughness on the road
plt.plot(range(0,run_duration), avg_n_road, label = 'Full Road')
plt.plot(range(0,run_duration), avg_n_ruts, label = 'Ruts')
plt.xlabel('Day')
plt.ylabel("Road - Average Manning's Roughness")
plt.xlim(0,run_duration)
plt.legend()
plt.show()

# plot the average manning's roughness in the ditch
plt.plot(range(0,run_duration), avg_n_channel, label = 'Ditch')
plt.xlabel('Day')
plt.ylabel("Ditch - Average Manning's Roughness")
plt.xlim(0,run_duration)
plt.legend()
plt.show()

#%% sediment rate of change check
plt.plot(range(0,run_duration), dzdt_avg_road)
plt.xlabel('Day')
plt.ylabel('average dzdt per time step in the road excluding ruts')
plt.xlim(0,run_duration)
plt.show()

plt.plot(range(0,run_duration), dzdt_avg_ruts)
plt.xlabel('Day')
plt.ylabel('average dzdt per time step in the ruts only')
plt.xlim(0,run_duration)
plt.show()
#%% layer thickness plots

# plot average active layer thickness over time in the ruts
plt.plot(range(0,run_duration), active_depth_ruts)
plt.xlabel('Day')
plt.ylabel('Active Depth in ruts [$m$]')
plt.xlim(0,run_duration)
plt.show()

# these have been a bit unnecessary to plot because they are relatively unchanging.
# they lose on the order of 1e-7 m per day of thickness due to TPE processes only.
# if truck_num is varied then these plots will provide more meaning.

# # plot average surfacing layer thickness over time in the ruts
# plt.plot(range(0,run_duration), surfacing_depth_ruts)
# plt.xlabel('Day')
# plt.ylabel('Surfacing Depth in ruts [$m$]')
# plt.xlim(0,run_duration)
# plt.show()

# # plot average ballast layer thickness over time in the ruts
# plt.plot(range(0,run_duration), ballast_depth_ruts)
# plt.xlabel('Day')
# plt.ylabel('Ballast Depth in ruts [$m$]')
# plt.xlim(0,run_duration)
# plt.show()

#%% prints

total_dz = np.abs(min(dz_arr_cum)) 
total_dV = total_dz*cell_area # create area of cell variable and multiply by that instead, use cell_spacing at the top and convert to area
total_load = total_dV*rho_s*(1-porosity)
total_load_div = total_load/2
duration_print = run_duration - 1

print(
    "Total rainfall (",duration_print, "days):", sum(np.multiply(intensity_arr,np.multiply(dt_arr,24))), 'mm'
    )

# kind of understand this sediment__added, but why would this be different than what I calculated for the sediment load from TPE
print(
    'Sediment pumped:', mg.at_node['sediment__added'].sum()*cell_area*rho_s*(1-porosity), 'kg'
    )

print('Comparison between sediment load from road elevation change calculation and channel influx from OFT:',\
    total_load_div - ((((np.array(sediment_influx_channel)*rho_s*dt_arr_secs).sum()))\
        +(np.array(sediment_outflux_ruts)*rho_s*dt_arr_secs).sum())
    )

print(
    'Cumulative sediment load from road (half-road dz estimate):', total_load_div, 'kg'
    )

print(
    'Cumulative sediment load from road (half-road OFT calculation):', ((((np.array(sediment_influx_channel)*rho_s*dt_arr_secs).sum()))\
        +(np.array(sediment_outflux_ruts)*rho_s*dt_arr_secs).sum()), 'kg' 
    )

print(
    'Cumulative sediment flux from road & ditch:', (np.array(sediment_outflux_channel)*rho_s*dt_arr_secs).sum()\
    +(np.array(sediment_outflux_ruts)*rho_s*dt_arr_secs).sum(), 'kg'
    )


#%%
# plt.plot(range(0,run_duration), np.multiply(sa_arr, 1000).cumsum())
# plt.xlabel('Day')
# plt.ylabel('Road active layer depth [mm]')
# plt.xlim(0,run_duration)
# # plt.ylim(19.95,20)
# plt.show()

# plt.plot(range(0,run_duration), np.array(ss_arr).cumsum())
# plt.xlabel('Day')
# plt.ylabel('Road surfacing layer depth [m]')
# plt.xlim(0,run_duration)
# plt.show()

# plt.plot(range(0,run_duration), np.array(sb_arr).cumsum())
# plt.xlabel('Day')
# plt.ylabel('Road ballast layer depth [m]')
# plt.xlim(0,run_duration)
# plt.show()
# %%
