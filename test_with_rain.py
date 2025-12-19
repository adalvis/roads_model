#%%
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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

parameters = pd.read_csv("parameters_WY2024.csv")
# site=parameters.loc[parameters["Site Name"] == "KID13"].iloc[0]
site=parameters.loc[parameters["Site Name"] == "BISH05"].iloc[0]

# initialize average number of truck passes per day for truck pass erosion
truck_num_ini = 4

Sa_ini = 0.010 # active depth in m
longitudinal_slope = site["Road_Gradient"]/100
porosity = 0.35

# List of n values for ditch treatments:
    # 
ditch_n = 0.7   # manning's roughness of the ditch, based on ditch BMP
n_c = 0.03   # n_c and n_f for OFT initialization
n_f = 0.015

# fractions of fine and coarse grains in the active layer
# initially 50/50
# based off of hydrometer report for the site
f_af = 0.25
f_ac = 0.75

# this d50 will replace the below indexing method once I make a column for corresponding tau_c in the parameters file
# d50 = site["d50 (mm)"]/1000

d50_arr = [0.000018, 0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008, 0.0009, 0.001, 0.005]
tauc_arr = [0.05244426, 0.1456785, 0.1780515, 0.19909395, 0.226611, 0.258984, 0.291357, 0.3625776, 0.453222, 0.5244426, 0.5989005, 3.6419625]

# index the desired d50 and tau_c values, or find d50 for the site and find tau_c from chart and input here
# 0-11
index = 3
d50 = d50_arr[index] # [m] ensure this value is changed in function files as well
tau_c = tauc_arr[index] # tau_c dependent on d_50

#%%
# Data load and rainfall site selection

# intensity data in mm/hr with dt from the dt document
intensity_2024 = pd.read_csv("WY2024_RG_daily_intensity.csv")
# change the site name and index values to get different sites and dates
# intensity_90 = intensity_2024["RG_KID1316_mm"].iloc[:90].values  # first 90 days for site "RG_BISH05"
intensity_90 = intensity_2024["RG_BISH05_mm"].iloc[:90].values

# daily dt in hours
dt_2024_hours = pd.read_csv("WY2024_RG_daily_dt.csv")
# dt_2024_hours_90 = dt_2024_hours["RG_KID1316_mm"].iloc[:90].values # first 90 days for site "RG_BISH05"
dt_2024_hours_90 = dt_2024_hours["RG_BISH05_mm"].iloc[:90].values
# convert to dt to days
dt_2024 = np.array(dt_2024_hours_90)/24

#%% Run method to create grid; add new fields
eg = Erodible_Grid(nrows=540, ncols=72,\
    spacing=0.1475, full_tire=False, long_slope=longitudinal_slope) # update long slope to change DEM

mg, z, road_flag, n = eg() 

noise_amplitude=0.005   # init = 0.005
road = road_flag==1
z[road] += noise_amplitude * np.random.rand(
    len(z[road])
) #z is the road elevation

#Add depth fields that will update in the component; these are the initial conditions
mg.at_node['active__depth'] = np.ones(540*72)*Sa_ini #This is the most likely to change; the active layer is essentially 0.005 m right now.
mg.at_node['surfacing__depth'] = np.ones(540*72)*0.23
mg.at_node['ballast__depth'] = np.ones(540*72)*2.0 #This depth can technically be anything; it just needs to be much larger than the active layer and the surfacing layer depths.

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
truck_num=0     # why is this truck number 0, this is initializing, but does it overwrite the initial truck number?
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

        sediment_outflux_channel.append(0)
        sediment_outflux_ruts.append(0)
        sediment_influx_channel.append(0)

        # for plotting layer depths
        active_depth = np.average(mg.at_node['active__depth'][mg.nodes[1,9:41]])
        active_depth_ruts.append(active_depth)
        surfacing_depth = np.average(mg.at_node['surfacing__depth'][mg.nodes[1,9:41]])
        surfacing_depth_ruts.append(surfacing_depth)
        ballast_depth = np.average(mg.at_node['ballast__depth'][mg.nodes[1,9:41]])
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
        active_depth = np.average(mg.at_node['active__depth'][mg.nodes[1,9:41]])
        active_depth_ruts.append(active_depth)
        surfacing_depth = np.average(mg.at_node['surfacing__depth'][mg.nodes[1,9:41]])
        surfacing_depth_ruts.append(surfacing_depth)
        ballast_depth = np.average(mg.at_node['ballast__depth'][mg.nodes[1,9:41]])
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

        # diagnostics
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
        print("SLOPE min/max/any_nan/any_inf:", np.nanmin(slope), np.nanmax(slope), np.isnan(slope).any(), np.isinf(slope).any())

        print(
            "Mass from dz:",
            np.sum(oft._dzdt * dt * 86400 * 0.1475*0.1475 * 2650),
            "Mass from flux:",
            np.sum(oft._sediment_outflux * 2650 * dt * 86400)
        )
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
        
        # work on ditch roughness here?
        for k in range(len(surface_water_discharge)):
            if surface_water_discharge[k] > 0 and road_flag[k] == 0:
                mg.at_node['grain__roughness'][k] = 0.05    # should this be altered? Grain roughness in the ditch is probably different
                mg.at_node['total__roughness'][k] = ditch_n #this can change according to the treatment
                mg.at_node['shear_stress__partitioning'][k] = ((mg.at_node['grain__roughness'][k]) /\
                    (mg.at_node['total__roughness'][k]))**(24/13)
        
        
        R = ((mg.at_node['total__roughness'][mg.nodes[1:,1:9]].max()*channel_discharge)/\
            (np.sqrt(6*slope/0.718)))**(6/13)
        shear_stress = rho_w*g*R*slope*mg.at_node['shear_stress__partitioning'][mg.nodes[1:,1:9]].max()

        if shear_stress > tau_c:
            sediment_outflux = (((10**(-4.348))/(rho_s*((d50)**(0.811))))\
                *(shear_stress-tau_c)**(2.457))*np.sqrt(6*R/0.718) #[m^3/s]
        else:
            sediment_outflux=0
        sediment_outflux_channel.append(sediment_outflux)
        sediment_outflux_ruts.append((mg.at_node["sediment__volume_influx"][mg.nodes[1,9:41]]).sum())
        sediment_influx_channel.append((mg.at_node["sediment__volume_influx"][mg.nodes[1:,8]]).sum())

    sa = (mg.at_node['active__depth'][mask]).mean()-active_init[mask].mean()
    sa_arr.append(sa)
    ss = mg.at_node['surfacing__depth'][mask].mean()-surfacing_init[mask].mean()
    ss_arr.append(ss)
    sb = mg.at_node['ballast__depth'][mask].mean()-ballast_init[mask].mean()
    sb_arr.append(sb)

wall_time = time.time() - start
print("Wall time for run:", wall_time, "s")

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

plt.plot(range(0,run_duration), np.multiply(dz_arr,0.1475**2*2650*(1-porosity))) # added porosity consideration
plt.plot(range(0,run_duration), np.zeros(len(range(0,run_duration))), '--', color='gray')
plt.xlabel('Day')
plt.ylabel('Total mass change between time steps - \nroad [$kg$]')
plt.xlim(0,run_duration)
plt.show()

plt.plot(range(0,run_duration), np.multiply(dz_arr_cum,0.1475**2*2650*(1-porosity))/2) # added porosity consideration
plt.plot(range(0,run_duration), np.zeros(len(range(0,run_duration))), '--', color='gray')
plt.xlabel('Day')
plt.ylabel('Cumulative mass change (from dz) - \nhalf road [$kg$]')
plt.xlim(0,run_duration)
plt.show()

plt.plot(range(0,run_duration), -((((np.array(sediment_influx_channel)*rho_s*dt_arr*86400).cumsum()))\
        +(np.array(sediment_outflux_ruts)*rho_s*dt_arr*86400).cumsum()))
plt.plot(range(0,run_duration), np.zeros(len(range(0,run_duration))), '--', color='gray')
plt.xlabel('Day')
plt.ylabel('Cumulative mass change (from OFT)- \nhalf road [$kg$]')
plt.xlim(0,run_duration)
plt.show()

#%%
sediment_mass = (np.array(sediment_influx_channel)*rho_s*dt_arr*86400 - (np.array(sediment_outflux_ruts)*rho_s*dt_arr*86400) - \
    np.array(sediment_outflux_channel)*rho_s*dt_arr*86400)

plt.plot(range(0,run_duration), sediment_mass)
plt.plot(range(0,run_duration), np.zeros(len(range(0,run_duration))), '--', color='gray')
plt.xlabel('Day')
plt.ylabel('Total mass change between time steps - \nditch line [$kg$]')
plt.xlim(0,run_duration)
plt.show()

plt.plot(range(0,run_duration), (np.multiply(sediment_influx_channel,dt_arr)*rho_s*86400).cumsum())
plt.plot(range(0,run_duration), np.zeros(len(range(0,run_duration))), '--', color='gray')
plt.xlabel('Day')
plt.ylabel('Cumulative mass inflow - \nrouted into ditch [$kg$]')
plt.xlim(0,run_duration)
plt.show()

plt.plot(range(0,run_duration), -(np.multiply(sediment_outflux_channel,dt_arr)*rho_s*86400).cumsum()\
    -(np.array(sediment_outflux_ruts)*rho_s*dt_arr*86400).cumsum())
plt.plot(range(0,run_duration), np.zeros(len(range(0,run_duration))), '--', color='gray')
plt.xlabel('Day')
plt.ylabel('Cumulative mass outflow - \nrouted out of ditch [$kg$]')
plt.xlim(0,run_duration)
plt.show()

#%%
# plots to see what is happening

# scattering, pumping, crushing plots
# plot cumulative fluxes
# this plot needs to be looked at!
q_cs = tpe._q_cs    # [m/truck] crushing
sum_q_cs = q_cs[mg.nodes[1,9:41]].sum() # summed across the rut for half road
q_ps = tpe._q_ps    # [m/truck] pumping
sum_q_ps = q_ps[mg.nodes[1,9:41]].sum() # summed across the rut for half road
# q are in m/truck not m/s so shouldn't be multiplied by dt_arr, should be multiplied by truck number?
plt.plot(range(0,run_duration), (np.multiply(sum_q_cs, dt_arr)*rho_s).cumsum(), color='pink', label='crushing')
plt.plot(range(0,run_duration), (np.multiply(sum_q_ps, dt_arr)*rho_s).cumsum(), color='green', label='pumping')
plt.plot(range(0,run_duration), np.zeros(len(range(0,run_duration))), '--', color='gray')
plt.xlabel('Day')
plt.ylabel('Pumping and Crushing Fluxes [kg]')
plt.xlim(0,run_duration)
plt.legend()
plt.show()

# # plot average active layer thickness over time in the ruts
# plt.plot(range(0,run_duration), sa_arr)
# plt.xlabel('Day')
# plt.ylabel('Active Depth [m]')
# plt.xlim(0,run_duration)
# plt.show()

# plot average active layer thickness over time in the ruts
plt.plot(range(0,run_duration), active_depth_ruts)
plt.xlabel('Day')
plt.ylabel('Active Depth in ruts [m]')
plt.xlim(0,run_duration)
plt.show()

# plot average surfacing layer thickness over time in the ruts
plt.plot(range(0,run_duration), surfacing_depth_ruts)
plt.xlabel('Day')
plt.ylabel('Surfacing Depth in ruts [m]')
plt.xlim(0,run_duration)
plt.show()

# plot average ballast layer thickness over time in the ruts
plt.plot(range(0,run_duration), ballast_depth_ruts)
plt.xlabel('Day')
plt.ylabel('Ballast Depth in ruts [m]')
plt.xlim(0,run_duration)
plt.show()

#%%

total_dz = np.abs(min(dz_arr_cum)) 
total_dV = total_dz*0.1475*0.1475
total_load = total_dV*2650*(1-porosity)
total_load_div = total_load/2
duration_print = run_duration - 1

print(
    "Total rainfall (",duration_print, "days):", sum(np.multiply(intensity_arr,np.multiply(dt_arr,24))), 'mm'
    )

print(
    'Sediment pumped:', mg.at_node['sediment__added'].sum()*0.1475*0.1475*2650*(1-porosity), 'kg'
    )

print('Comparison between sediment load from road elevation change calculation and channel influx from OFT:',\
    total_load_div - ((((np.array(sediment_influx_channel)*rho_s*dt_arr*86400).sum()))\
        +(np.array(sediment_outflux_ruts)*rho_s*dt_arr*86400).sum())
    )

print(
    'Sediment load from road (half-road dz estimate):', total_load_div, 'kg'
    )

print(
    'Sediment load from road (half-road OFT calculation):', ((((np.array(sediment_influx_channel)*rho_s*dt_arr*86400).sum()))\
        +(np.array(sediment_outflux_ruts)*rho_s*dt_arr*86400).sum()), 'kg' 
    )

print(
    'Sediment load from road & ditch:', (np.array(sediment_outflux_channel)*rho_s*dt_arr*86400).sum()\
    +(np.array(sediment_outflux_ruts)*rho_s*dt_arr*86400).sum(), 'kg'
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
