#%%
import time

import matplotlib.pyplot as plt
import numpy as np

from landlab import RasterModelGrid, imshow_grid
from landlab.components import OverlandFlowTransporter, FlowAccumulator, TruckPassErosion
np.set_printoptions(threshold=np.inf)
from erodible_grid import Erodible_Grid

#%%
# Parameters
run_duration = 30  # run duration, days - shouldn't this be 90 days??
seed = 4
np.random.seed(seed)

Sa_ini = 0.005 # active depth in m
longitudinal_slope = 0.125 # 12.5% for site BISH05
ditch_n = 0.3

# options of d50 and corresponding tau_c
d50_arr = [0.000018, 0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008, 0.0009, 0.001, 0.005]
tauc_arr = [0.05244426, 0.1456785, 0.1780515, 0.19909395, 0.226611, 0.258984, 0.291357, 0.3625776, 0.453222, 0.5244426, 0.5989005, 3.6419625]

# index the desired d50 and tau_c values, or find d50 for the site and find tau_c from chart and input here
# 0-11
index = 2
d_50 = d50_arr[index] # [m] ensure this value is changed in function files as well
tau_c = tauc_arr[index] # tau_c dependent on d_50

#%% Run method to create grid; add new fields
eg = Erodible_Grid(nrows=540, ncols=72,\
    spacing=0.1475, full_tire=False, long_slope=longitudinal_slope) # Update long_slope to change slope of DEM

mg, z, road_flag, n =eg()

noise_amplitude=0.005
road = road_flag==1
z[road] += noise_amplitude * np.random.rand(
    len(z[road])
) #z is the road elevation

#Add depth fields that will update in the component; these are the initial conditions
mg.at_node['active__depth'] = np.ones(540*72)*0.005 #This is the most likely to change; the active layer is essentially 0.005 m right now.
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
    grid_units=('m', 'm'), cmap='terrain', color_for_closed='black', vmin=0, vmax=4)
plt.xlabel('Road width (m)')
plt.ylabel('Road length (m)')
plt.tight_layout()
plt.show()

#%% Prep some variables for later
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
tpe = TruckPassErosion(mg, center, half_width, full_tire, truck_num=5, \
    scat_loss=8e-5) #initialize component
fa = FlowAccumulator(mg, surface='topographic__elevation', \
    flow_director="FlowDirectorD8", runoff_rate=1.538889e-6, \
    depression_finder="DepressionFinderAndRouter")
oft = OverlandFlowTransporter(mg, d50=d_50, tau_c=tau_c)

#%% Run the model!
mask = road_flag
z_limit = mg.at_node['topographic__elevation'] - mg.at_node['active__depth'] #You may need to use this to create a layer limit
intensity_arr=[]
dt_arr = []
dz_arr=[]
dz_arr_cum = []
# sa_arr=[]
# ss_arr=[]
# sb_arr=[]
sediment_outflux_channel = []
sediment_influx_channel = []
sediment_outflux_ruts = []
channel_discharge_arr = []
truck_num=0 # what is this?
z_ini_cum = mg.at_node['topographic__elevation'].copy()

#%% Run the model!
# Main loop
start = time.time()
for i in range(0, run_duration):
    # active_init = mg.at_node['active__depth'].copy()
    # surfacing_init = mg.at_node['surfacing__depth'].copy()
    # ballast_init = mg.at_node['ballast__depth'].copy()
    z_ini = mg.at_node['topographic__elevation'].copy()
    
    tpe.run_one_step()

    truck_num += tpe._truck_num
    print(tpe._truck_num)
    
    #==========Update this section (intensity & dt) if you want to include experiment rainfall data==========
    # You should not need an if/elif statement; you should be able to just loop through the intensity and dt datasets;
    # use only 90 day chunks of the datasets
    p_storm = 0.25
    no_chance = np.random.uniform()

    if no_chance > p_storm:
        intensity_arr.append(0)
        dt_arr.append(0)

        dz = z-z_ini #calculate elevation change at each time step
        dz_arr.append(sum(dz[mask]))

        dz_cum = z-z_ini_cum #calculate cumulative elevation change
        dz_arr_cum.append(sum(dz_cum[mask]))

        sediment_outflux_channel.append(0)
        sediment_outflux_ruts.append(0)
        sediment_influx_channel.append(0)

    elif no_chance <= p_storm:
        dt = np.random.exponential(scale=1/6)
        dt_arr.append(dt)
        print(dt)

        intensity = np.random.exponential(scale=5)
        intensity_arr.append(intensity)
        print(intensity)

        mg.at_node['water__unit_flux_in'] = np.ones(540*72)*intensity*2.77778e-7 # intensity converted to m/s
        fa.accumulate_flow()
        
        #=================================Calculate overland flow transport=================================
        oft.run_one_step(dt)

        dz = z-z_ini #calculate elevation change at each time step
        dz_arr.append(sum(dz[mask]))

        dz_cum = z-z_ini_cum #calculate cumulative elevation change
        dz_arr_cum.append(sum(dz_cum[mask])) 
        
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
        # fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(3, 6))
        # plt.xlabel('Road width (m)')
        # plt.ylabel('Road length (m)')
        # im = imshow_grid(mg,'dz_cum', var_name='Cumulative dz', var_units='m', 
        #              plot_name='Elevation change, t = %i days' %i,
        #              grid_units=('m','m'), cmap='RdBu', vmin=-0.0001, 
        #              vmax=0.0001, shrink=0.9)
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

        #=================================Calculate channelized flow transport=================================
        rho_w=1000
        rho_s=2650
        g=9.81
        slope=longitudinal_slope
        d50=d_50 # originally 1.8e-5
        tau_c=tau_c
        
        surface_water_discharge = mg.at_node['surface_water__discharge']
        channel_discharge = surface_water_discharge[mg.nodes[1:,1:9]].sum(axis=1).max()
        channel_discharge_arr.append(channel_discharge)
        
        for k in range(len(surface_water_discharge)):
            if surface_water_discharge[k] > 0 and road_flag[k] == 0:
                mg.at_node['grain__roughness'][k] = 0.05
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

    # sa = (mg.at_node['active__depth'][mask]).mean()-active_init[mask].mean()
    # sa_arr.append(sa)
    # ss = mg.at_node['surfacing__depth'][mask].mean()-surfacing_init[mask].mean()
    # ss_arr.append(ss)
    # sb = mg.at_node['ballast__depth'][mask].mean()-ballast_init[mask].mean()
    # sb_arr.append(sb)

        # these are just tests to see what is happening after running the for loop
        print("dt (days):", dt)
        print("Sed flux (mean):", np.mean(mg.at_node['sediment__volume_outflux']))
        print("Mass flux per day (kg):",
          np.mean(mg.at_node['sediment__volume_outflux']) * rho_s * dt * 86400)
        print("Slope range:", np.nanmin(mg.at_node['topographic__steepest_slope']),
          np.nanmax(mg.at_node['topographic__steepest_slope']))
        print("Receiver stats:", np.unique(mg.at_node['flow__receiver_node']).size)
        print("Water depth range:", np.nanmin(mg.at_node['water__depth']),
          np.nanmax(mg.at_node['water__depth']))
        print("Discharge range:", np.nanmin(mg.at_node['surface_water__discharge']),
          np.nanmax(mg.at_node['surface_water__discharge']))

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
plt.plot(range(0,run_duration), np.multiply(dz_arr,0.1475**2*2650))
plt.plot(range(0,run_duration), np.zeros(len(range(0,run_duration))), '--', color='gray')
plt.xlabel('Day')
plt.ylabel('Total mass change between time steps - \nroad [$kg$]')
plt.xlim(0,run_duration)
plt.show()

plt.plot(range(0,run_duration), np.multiply(dz_arr_cum,0.1475**2*2650)/2)
plt.plot(range(0,run_duration), np.zeros(len(range(0,run_duration))), '--', color='gray')
plt.xlabel('Day')
plt.ylabel('Cumulative mass change (from dz) - \nhalf road [$kg$]')
plt.xlim(0,run_duration)
plt.show()

plt.plot(range(0,run_duration), -((((np.array(sediment_influx_channel)*rho_s*dt_arr).cumsum()))\
        +(np.array(sediment_outflux_ruts)*rho_s*dt_arr).cumsum()))
plt.plot(range(0,run_duration), np.zeros(len(range(0,run_duration))), '--', color='gray')
plt.xlabel('Day')
plt.ylabel('Cumulative mass change (from OFT)- \nhalf road [$kg$]')
plt.xlim(0,run_duration)
plt.show()

#%%


# we are using m3/s, dt in days, and kg/m3, so we should multiply by 86400, but that just makes it way higher! Seems to be addressed in OFT?
sediment_mass = (np.array(sediment_influx_channel)*rho_s*dt_arr - (np.array(sediment_outflux_ruts)*rho_s*dt_arr) - \
    np.array(sediment_outflux_channel)*rho_s*dt_arr)

plt.plot(range(0,run_duration), sediment_mass)
plt.plot(range(0,run_duration), np.zeros(len(range(0,run_duration))), '--', color='gray')
plt.xlabel('Day')
plt.ylabel('Total mass change between time steps - \nditch line [$kg$]')
plt.xlim(0,run_duration)
plt.show()

plt.plot(range(0,run_duration), (np.multiply(sediment_influx_channel,dt_arr)*rho_s).cumsum())
plt.plot(range(0,run_duration), np.zeros(len(range(0,run_duration))), '--', color='gray')
plt.xlabel('Day')
plt.ylabel('Cumulative mass inflow - \nrouted into ditch [$kg$]')
plt.xlim(0,run_duration)
plt.show()

plt.plot(range(0,run_duration), -(np.multiply(sediment_outflux_channel,dt_arr)*rho_s).cumsum()\
    -(np.array(sediment_outflux_ruts)*rho_s*dt_arr).cumsum())
plt.plot(range(0,run_duration), np.zeros(len(range(0,run_duration))), '--', color='gray')
plt.xlabel('Day')
plt.ylabel('Cumulative mass outflow - \nrouted out of ditch [$kg$]')
plt.xlim(0,run_duration)
plt.show()

#%%
total_dz = np.abs(min(dz_arr_cum))
total_dV = total_dz*0.1475*0.1475 # change cell size, set automatically based on grid
total_load = total_dV*2650
total_load_div = total_load/2

print(
    'Total rainfall (90 days):', sum(np.multiply(intensity_arr,np.multiply(dt_arr,24))), 'mm'
    )

print(
    'Sediment pumped:', mg.at_node['sediment__added'].sum()*0.1475*0.1475*2650*0.6, 'kg'
    )

print('Comparison between sediment load from road elevation change calculation and channel influx from OFT:',\
    total_load_div - ((((np.array(sediment_influx_channel)*rho_s*dt_arr).sum()))\
        +(np.array(sediment_outflux_ruts)*rho_s*dt_arr).sum())
    )

print(
    'Sediment load from road (half-road dz estimate):', total_load_div, 'kg'
    )

print(
    'Sediment load from road (half-road OFT calculation):', ((((np.array(sediment_influx_channel)*rho_s*dt_arr).sum()))\
        +(np.array(sediment_outflux_ruts)*rho_s*dt_arr).sum()), 'kg'
    )

print(
    'Sediment load from road & ditch:', (np.array(sediment_outflux_channel)*rho_s*dt_arr).sum()\
    +(np.array(sediment_outflux_ruts)*rho_s*dt_arr).sum(), 'kg'
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
