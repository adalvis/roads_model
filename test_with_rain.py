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
run_duration = 21   # run duration = days + 1
seed = 1        # this isn't used to calculate rainfall, it ensures that anything else randomly generated isn't 
                # going to be changing between runs. I changed TPE to not have a randomly generated
                # truck number in that file.
np.random.seed(seed)

# constants
rho_w=1000
rho_s=2650
g=9.81

# grid creation
cell_spacing = 0.1475 # cell width or length dimension in meters
cell_area = cell_spacing**2
nrows = 540 # number of rows in the grid
ncols = 72  # number of columns in the grid

parameters = pd.read_csv("parameters_WY2024.csv")
# site=parameters.loc[parameters["Site Name"] == "KID13"].iloc[0]
site=parameters.loc[parameters["Site Name"] == "BISH05"].iloc[0]

# initialize average number of truck passes per day for truck pass erosion
truck_num_ini = 4

Sa_ini = 0.01 # active depth in m
# longitudinal_slope = site["Road Gradient"]/100
longitudinal_slope = 0.125
porosity = 0.35

# List of n values for ditch treatments:
    # 
ditch_n = 0.7   # manning's roughness of the ditch, based on ditch BMP
ditch_grain_n = 0.05 # manning's roughness for the grains in the ditch
n_c = 0.03   
n_f = 0.015

# fractions of fine and coarse grains in the active layer
# initially 50/50
f_af = 0.25
f_ac = 0.75

# this d50 will replace the below indexing method once I make a column for corresponding tau_c in the parameters file
# d50 = site["d50 (mm)"]/1000

d50_arr = [0.000018, 0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008, 0.0009, 0.001, 0.005]
tauc_arr = [0.05244426, 0.1456785, 0.1780515, 0.19909395, 0.226611, 0.258984, 0.291357, 0.3625776, 0.453222, 0.5244426, 0.5989005, 3.6419625]

# index the desired d50 and tau_c values, or find d50 for the site and find tau_c from chart and input here
# 0-11
index = 3
d50 = d50_arr[index] # [m] 
tau_c = tauc_arr[index] # tau_c dependent on d_50

#%%
# Data load and rainfall site selection

# to call high intensity
high_intensity_index = 144

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
    spacing=cell_spacing, full_tire=False, long_slope=longitudinal_slope) # update long slope to change DEM

mg, z, road_flag, n = eg() 

noise_amplitude=0.005   # init = 0.005
road = road_flag==1
z[road] += noise_amplitude * np.random.rand(
    len(z[road])
) #z is the road elevation

#Add depth fields that will update in the component; these are the initial conditions
mg.at_node['active__depth'] = np.ones(nrows*ncols)*Sa_ini #This is the most likely to change; the active layer is essentially 0.005 m right now.
mg.at_node['surfacing__depth'] = np.ones(nrows*ncols)*0.23
mg.at_node['ballast__depth'] = np.ones(nrows*ncols)*2.0 #This depth can technically be anything; it just needs to be much larger than the active layer and the surfacing layer depths.

#Add absolute elevation fields that will update based on z updates
mg.at_node['active__elev'] = z
mg.at_node['surfacing__elev'] = z - mg.at_node['active__depth']
mg.at_node['ballast__elev'] = z - mg.at_node['active__depth']\
     - mg.at_node['surfacing__depth']

#%% Plot initial grid
# Set up the figure.
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(3, 6))
imshow_grid(mg, z, plot_name='Synthetic road', var_name='Elevation', var_units='m',\
    grid_units=('m', 'm'), cmap='terrain', color_for_closed='black', vmin=0, vmax=9)
plt.xlabel('Road width (m)')
plt.ylabel('Road length (m)')
plt.tight_layout()
plt.show()

#%% Prep some variables for later
# leftover from Amanda
# xsec_pre = mg.at_node['topographic__elevation'][4392*2:4428*2].copy() #half tire width
# xsec_surf_pre = mg.at_node['surfacing__elev'][4392*2:4428*2].copy()
# mg_pre = mg.at_node['topographic__elevation'].copy()
# active_pre = mg.at_node['active__depth'].copy()

# X = mg.node_x.reshape(mg.shape)
# Y = mg.node_y.reshape(mg.shape)
# Z = z.reshape(mg.shape)

#%% Prep variables for component run
#We're using half tire width for node spacing
center = 40
half_width = 7 
full_tire = False

#%% Instantiate components
tpe = TruckPassErosion(mg, center, half_width, full_tire, truck_num=truck_num_ini, \
    scat_loss=8e-5, f_af=f_af, f_ac=f_ac) #initialize component, 

df_init = DepressionFinderAndRouter(mg, reroute_flow = True)
df_init.map_depressions()

fa = FlowAccumulator(mg, surface='topographic__elevation', \
    flow_director="FlowDirectorD8", runoff_rate=1.538889e-6, \
    depression_finder=None)  # runoff rate = 1.538889e-6

oft = OverlandFlowTransporter(mg, porosity=porosity, d50=d50, tau_c=tau_c, n_c=n_c, n_f=n_f,Sa_ini=Sa_ini)        # trying to run component with values we can change in this file

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

active_depth_ruts = []
surfacing_depth_ruts = []
ballast_depth_ruts = []

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

truck_num=0     
z_ini_cum = mg.at_node['topographic__elevation'].copy()

#%% Run the model!
# Main loop

start = time.time()
for i in range(0, run_duration):        # daily time step, every time step there are a certain number of truck passes = truck_num
    # should this be run duration - 1?
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
        avg_n_ruts.append(np.nanmean(mg.at_node['total__roughness'][mg.nodes[1:, 9:41]]))
        avg_n_channel.append(ditch_n)
        avg_n_road.append(np.nanmean(mg.at_node['total__roughness'][mg.nodes[1:, :]]))

        # append tpe loading
            # multiply change in layer thickness in the ruts by cell area*sediment density*1-porosity
        tpe_load_ruts.append((np.multiply(tpe._active_dz[mg.nodes[1:, 9:41]], cell_area*rho_s*(1-porosity))).sum()) # gives increase in mass in the active layer due to truck passes per time step
        
        # for plotting layer depths
        active_depth = np.average(mg.at_node['active__depth'][mg.nodes[1:,9:41]])
        active_depth_ruts.append(active_depth)
        surfacing_depth = np.average(mg.at_node['surfacing__depth'][mg.nodes[1:,9:41]])
        surfacing_depth_ruts.append(surfacing_depth)
        ballast_depth = np.average(mg.at_node['ballast__depth'][mg.nodes[1:,9:41]])
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
        active_depth = np.average(mg.at_node['active__depth'][mg.nodes[1:,9:41]])
        active_depth_ruts.append(active_depth)
        surfacing_depth = np.average(mg.at_node['surfacing__depth'][mg.nodes[1:,9:41]])
        surfacing_depth_ruts.append(surfacing_depth)
        ballast_depth = np.average(mg.at_node['ballast__depth'][mg.nodes[1:,9:41]])
        ballast_depth_ruts.append(ballast_depth)
        
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(3, 6))
        im = imshow_grid(mg,'water__depth', var_name='Depth', 
                     plot_name='Water depth, t = %i days' %i,
                     var_units='$m$', grid_units=('m','m'), 
                     cmap='Blues', vmin=0, vmax=0.001, shrink=0.9)
        plt.xlabel('Road width (m)')
        plt.ylabel('Road length (m)')
        plt.tight_layout()
        # plt.savefig('output/Q_%i_days.png' %i)
        plt.show()

        mg.add_field('dz_cum', dz_cum, at='node', units='m', clobber=True)
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(3, 6))
        plt.xlabel('Road width (m)')
        plt.ylabel('Road length (m)')
        im = imshow_grid(mg,'dz_cum', var_name='Cumulative dz', var_units='m', 
                     plot_name='Elevation change, t = %i days' %i,
                     grid_units=('m','m'), cmap='RdBu', vmin=-0.0001, 
                     vmax=0.0001, shrink=0.9)
        plt.xlabel('Road width (m)')
        plt.ylabel('Road length (m)')
        plt.tight_layout()
        # plt.savefig('output/dz_cum_%i_days.png' %i)
        plt.show()

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(3, 6))
        plt.xlabel('Road width (m)')
        plt.ylabel('Road length (m)')
        im = imshow_grid(mg,'active__fines', var_name='Active fines', var_units='m', 
                     plot_name='Fines depth, t = %i days' %i,
                     grid_units=('m','m'), cmap='pink', vmin=0.002, 
                     vmax=0.003)
        plt.xlabel('Road width (m)')
        plt.ylabel('Road length (m)')
        plt.tight_layout()
        # plt.savefig('output/dz_cum_%i_days.png' %i)
        plt.show()

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(3, 6))
        plt.xlabel('Road width (m)')
        plt.ylabel('Road length (m)')
        im = imshow_grid(mg,'active__depth', var_name='Active depth', var_units='m', 
                     plot_name='Active depth, t = %i days' %i,
                     grid_units=('m','m'), cmap='pink', vmin=0.0025, vmax=0.02)
        plt.xlabel('Road width (m)')
        plt.ylabel('Road length (m)')
        plt.tight_layout()
        # plt.savefig('output/dz_cum_%i_days.png' %i)
        plt.show()

        # for future slider plots
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

        print(
            "dzdt min/max:",
            np.nanmin(oft._dzdt),
            np.nanmax(oft._dzdt)
        # diagnostics end
)

        #=================================Calculate channelized flow transport=================================
        slope=longitudinal_slope
        d50=d50
        tau_c=tau_c
        
        surface_water_discharge = mg.at_node['surface_water__discharge']
        channel_discharge = surface_water_discharge[mg.nodes[1:,1:9]].sum(axis=1).max()
        channel_discharge_arr.append(channel_discharge)
        
        for k in range(len(surface_water_discharge)):
            if surface_water_discharge[k] > 0 and road_flag[k] == 0:
                mg.at_node['grain__roughness'][k] = ditch_grain_n    # should this be altered? Grain roughness in the ditch is probably different
                mg.at_node['total__roughness'][k] = ditch_n #this can change according to the treatment
                mg.at_node['shear_stress__partitioning'][k] = ((mg.at_node['grain__roughness'][k]) /\
                    (mg.at_node['total__roughness'][k]))**(24/13)
        
        
        R = ((mg.at_node['total__roughness'][mg.nodes[1:,1:9]].max()*channel_discharge)/\
            (np.sqrt(6*slope/0.718)))**(6/13)       # indexing all rows except the top and columns 1:9
                
        shear_stress = rho_w*g*R*slope*mg.at_node['shear_stress__partitioning'][mg.nodes[1:,1:9]]

        if shear_stress.max() > tau_c:
            sediment_outflux = (((10**(-4.348))/(rho_s*((d50)**(0.811))))\
                *(shear_stress.max()-tau_c)**(2.457))*np.sqrt(6*R/0.718) #[m^3/s]
        else:
            sediment_outflux=0
        
        # append flux vectors
        sediment_outflux_channel.append(sediment_outflux)
        sediment_outflux_ruts.append((mg.at_node["sediment__volume_influx"][mg.nodes[1,9:41]]).sum())   # indexing the second from the top row, and the columns of the ruts
        sediment_influx_channel.append((mg.at_node["sediment__volume_influx"][mg.nodes[1:,8]]).sum())   # indexing all rows except the top and column 8, edge of road

        # append shear stress vectors
        mg.at_node['shear_stress'] = oft._shear_stress
        avg_shear_stress_ruts.append(np.nanmean(mg.at_node['shear_stress'][mg.nodes[1:,9:41]])) # mean of the oft calculated shear stress for the ruts
        # need to add calculation for channel
        avg_shear_stress_channel.append(np.nanmean(shear_stress))    # appending channel shear stress with above shear stress calc for mg.nodes[1:, 1:9]
        avg_shear_stress_road.append(np.nanmean(mg.at_node['shear_stress'][mg.nodes[1:, :]]))      # full road average of shear stress

        # append manning's n vectors
        avg_n_ruts.append(np.nanmean(mg.at_node['total__roughness'][mg.nodes[1:, 9:41]]))
        avg_n_channel.append(ditch_n)
        avg_n_road.append(np.nanmean(mg.at_node['total__roughness'][mg.nodes[1:, :]]))

        # append tpe loading
        tpe_load_ruts.append((np.multiply(tpe._active_dz[mg.nodes[1:, 9:41]], cell_area*rho_s*(1-porosity))).sum())

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

#%% loop plots - needs more work to make slider plots
# def slider_plot(
#     mg,
#     frames,
#     title,
#     cmap,
#     vmin=None,
#     vmax=None,
#     units=""
# ):
#     plt.figure(figsize=(3, 6))
#     plt.subplots_adjust(bottom=0.25)

#     im = imshow_grid(
#         mg,
#         frames[0],
#         cmap=cmap,
#         vmin=vmin,
#         vmax=vmax,
#         var_units=units,
#         plot_name=f"{title} – day 0"
#     )

#     ax = plt.gca() 

#     ax_slider = plt.axes([0.15, 0.1, 0.7, 0.03])
#     slider = Slider(
#         ax=ax_slider,
#         label="Day",
#         valmin=0,
#         valmax=len(frames) - 1,
#         valinit=0,
#         valstep=1
#     )

#     def update(val):
#         i = int(slider.val)
#         im.set_array(frames[i])
#         ax.set_title(f"{title} – day {i}")
#         plt.draw()

#     slider.on_changed(update)
#     plt.show()

# # water depth plot
# slider_plot(
#     mg,
#     water_depth_frames,
#     title="Water depth",
#     cmap="Blues",
#     vmin=0,
#     vmax=0.001,
#     units="m"
# )

# # cumulative dz plot
# slider_plot(
#     mg,
#     dz_cum_frames,
#     title="Cumulative elevation change",
#     cmap="RdBu",
#     vmin=-1e-4,
#     vmax=1e-4,
#     units="m"
# )

# # active fines plot
# slider_plot(
#     mg,
#     active_fines_frames,
#     title="Active fines depth",
#     cmap="pink",
#     vmin=0.002,
#     vmax=0.003,
#     units="m"
# )

# # active layer depth
# slider_plot(
#     mg,
#     active_depth_frames,
#     title="Active layer depth",
#     cmap="pink",
#     vmin=0.0025,
#     vmax=0.02,
#     units="m"
# )

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

plt.plot(range(0,run_duration), np.multiply(dz_arr,cell_area*rho_s*(1-porosity))) # added porosity consideration
plt.plot(range(0,run_duration), np.zeros(len(range(0,run_duration))), '--', color='gray')
plt.xlabel('Day')
plt.ylabel('Total mass change between time steps - \nroad [$kg$]')
plt.xlim(0,run_duration)
plt.show()

plt.plot(range(0,run_duration), np.multiply(dz_arr_cum,cell_area*rho_s*(1-porosity))/2) # added porosity consideration
plt.plot(range(0,run_duration), np.zeros(len(range(0,run_duration))), '--', color='gray')
plt.xlabel('Day')
plt.ylabel('Cumulative mass change (from dz) - \nhalf road [$kg$]')
plt.xlim(0,run_duration)
plt.show()

plt.plot(range(0,run_duration), -((((np.array(sediment_influx_channel)*rho_s*dt_arr_secs).cumsum()))\
        +(np.array(sediment_outflux_ruts)*rho_s*dt_arr_secs).cumsum()))
plt.plot(range(0,run_duration), np.zeros(len(range(0,run_duration))), '--', color='gray')
plt.xlabel('Day')
plt.ylabel('Cumulative mass change (from OFT)- \nhalf road [$kg$]')
plt.xlim(0,run_duration)
plt.show()

#%%

sediment_mass = (np.array(sediment_influx_channel)*rho_s*dt_arr_secs - (np.array(sediment_outflux_ruts)*rho_s*dt_arr_secs) - \
    np.array(sediment_outflux_channel)*rho_s*dt_arr_secs)

# plot total mass change between time steps along the ditch line
plt.plot(range(0,run_duration), sediment_mass)
plt.plot(range(0,run_duration), np.zeros(len(range(0,run_duration))), '--', color='gray')
plt.xlabel('Day')
plt.ylabel('Total mass change between time steps - \nditch line [$kg$]')
plt.xlim(0,run_duration)
plt.show()

# plot mass inflow into ditch - cumulative and between time steps
plt.plot(range(0,run_duration), (np.multiply(sediment_influx_channel,dt_arr_secs)*rho_s))
plt.plot(range(0,run_duration), np.zeros(len(range(0,run_duration))), '--', color='gray')
plt.xlabel('Day')
plt.ylabel('Mass inflow between time steps- \nrouted into ditch [$kg$]')
plt.xlim(0,run_duration)
plt.show()

plt.plot(range(0,run_duration), (np.multiply(sediment_influx_channel,dt_arr_secs)*rho_s).cumsum())
plt.plot(range(0,run_duration), np.zeros(len(range(0,run_duration))), '--', color='gray')
plt.xlabel('Day')
plt.ylabel('Cumulative mass inflow - \nrouted into ditch [$kg$]')
plt.xlim(0,run_duration)
plt.show()

# plot mass outflow out of ditch - cumulative and between time steps
plt.plot(range(0,run_duration), -(np.multiply(sediment_outflux_channel,dt_arr_secs)*rho_s)\
    -(np.array(sediment_outflux_ruts)*rho_s*dt_arr_secs))
plt.plot(range(0,run_duration), np.zeros(len(range(0,run_duration))), '--', color='gray')
plt.xlabel('Day')
plt.ylabel('Mass outflow between time steps - \nrouted out of ditch [$kg$]')
plt.xlim(0,run_duration)
plt.show()

plt.plot(range(0,run_duration), -(np.multiply(sediment_outflux_channel,dt_arr_secs)*rho_s).cumsum()\
    -(np.array(sediment_outflux_ruts)*rho_s*dt_arr_secs).cumsum())
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

# plot average shear stresses on the road
plt.plot(range(0,run_duration), avg_shear_stress_road, label = 'Full Road')
plt.plot(range(0,run_duration), avg_shear_stress_ruts, label = 'Ruts')
plt.axhline(tau_c, linestyle='--', color='gray', label='Critical Shear Stress Threshold')
plt.xlabel('Day')
plt.ylabel('Road - Average Shear Stress [$Pa$]')
plt.xlim(0,run_duration)
plt.legend()
plt.show()

# plot average shear stress in the ditch
plt.plot(range(0,run_duration), avg_shear_stress_channel, label = 'Ditch')
plt.axhline(tau_c, linestyle='--', color='gray', label='Critical Shear Stress Threshold')
plt.xlabel('Day')
plt.ylabel('Ditch - Average Shear Stress [$Pa$]')
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
#%% layer thickness plots

# plot average active layer thickness over time in the ruts
plt.plot(range(0,run_duration), active_depth_ruts)
plt.xlabel('Day')
plt.ylabel('Active Depth in ruts [$m$]')
plt.xlim(0,run_duration)
plt.show()

# plot average surfacing layer thickness over time in the ruts
plt.plot(range(0,run_duration), surfacing_depth_ruts)
plt.xlabel('Day')
plt.ylabel('Surfacing Depth in ruts [$m$]')
plt.xlim(0,run_duration)
plt.show()

# plot average ballast layer thickness over time in the ruts
plt.plot(range(0,run_duration), ballast_depth_ruts)
plt.xlabel('Day')
plt.ylabel('Ballast Depth in ruts [$m$]')
plt.xlim(0,run_duration)
plt.show()

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
