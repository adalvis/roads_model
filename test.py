import time

import matplotlib.pyplot as plt
import numpy as np

from landlab import RasterModelGrid, imshow_grid
from landlab.components import OverlandFlowTransporter, FlowAccumulator, TruckPassErosion
np.set_printoptions(threshold=np.inf)

#%%
# Parameters
run_duration = 90  # run duration, days
dt = 1  # time-step duration, days
seed = 1 #56

#%%
# Control and derived parameters, and other setup
elapsed_time = 0.0
np.random.seed(seed)

#%%
def ErodibleGrid(nrows, ncols, spacing, full_tire):
    mg = RasterModelGrid((nrows,ncols),spacing)
    z = mg.add_zeros('topographic__elevation', at='node') #create the topographic__elevation field
    road_flag = mg.add_zeros('flag', at='node') #create a road_flag field for determining whether a 
                                                #node is part of the road or the ditch line
    n = mg.add_zeros('roughness', at='node') #create roughness field
    
    mg.set_fixed_value_boundaries_at_grid_edges(True, True, True, True)     
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
                    elev = -0.109
                    flag = False
                    roughness = 0.1
                elif h == 2 or h == 6:
                    elev = -0.1875
                    flag = False
                    roughness = 0.1
                elif h == 3 or h == 5:
                    elev = -0.2344
                    flag = False
                    roughness = 0.1
                elif h == 4:
                    elev = -0.25
                    flag = False
                    roughness = 0.1
                elif h <= road_peak and h > 8: #update latitudinal slopes based on location related to road_peak
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
                    elev = -0.1875
                    flag = False
                    roughness = 0.1
                elif h == 2:
                    elev = -0.25
                    flag = False
                    roughness = 0.1
                elif h <= road_peak and h > 4: #update latitudinal slopes based on location related to road_peak
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
noise_amplitude=0.005

np.random.seed(1)

# cores = mg.core_nodes
road = road_flag==1

z[road] += noise_amplitude * np.random.rand(
    len(z[road])
)

#add depth fields that will update in the component
mg.at_node['active__depth'] = np.ones(540*72)*0.005
mg.at_node['surfacing__depth'] = np.ones(540*72)*0.23
mg.at_node['ballast__depth'] = np.ones(540*72)*2.0

#add absolute elevation fields that will update based on z updates
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
xsec_pre = mg.at_node['topographic__elevation'][4392*2:4428*2].copy() #half tire width
xsec_surf_pre = mg.at_node['surfacing__elev'][4392*2:4428*2].copy()
mg_pre = mg.at_node['topographic__elevation'].copy()
active_pre = mg.at_node['active__depth'].copy()

X = mg.node_x.reshape(mg.shape)
Y = mg.node_y.reshape(mg.shape)
Z = z.reshape(mg.shape)

#%% Run the component
#half tire width
center = 40
half_width = 7 
full_tire = False

#%%
# Instantiate components
tpe = TruckPassErosion(mg, center, half_width, full_tire, truck_num=5, \
    scat_loss=8e-5) #initialize component
fa = FlowAccumulator(mg, surface='topographic__elevation', \
    flow_director="FlowDirectorD8", runoff_rate=1.538889e-6, \
    depression_finder="DepressionFinderAndRouter")
oft = OverlandFlowTransporter(mg)

#%%
mask = road_flag
z_limit = mg.at_node['topographic__elevation'] - mg.at_node['active__depth']
intensity_arr=[]
dt_arr = []
dz_arr=[]
dz_arr_cum = []
sa_arr=[]
ss_arr=[]
sb_arr=[]
tracks=[]
sediment_outflux_channel = []
sediment_influx_channel = []
sediment_outflux_ruts = []
channel_discharge_arr = []
truck_num=0

z_ini_cum = mg.at_node['topographic__elevation'].copy()

# Main loop
start = time.time()
for i in range(0, run_duration):
    active_init = mg.at_node['active__depth'].copy()
    surfacing_init = mg.at_node['surfacing__depth'].copy()
    ballast_init = mg.at_node['ballast__depth'].copy()
    z_ini = mg.at_node['topographic__elevation'].copy()
    tpe.run_one_step()
    tracks.append(tpe.tire_tracks)
    truck_num += tpe._truck_num
    print(tpe._truck_num)
    p_storm = 0.25

    chance = np.random.uniform()

    if chance > p_storm:
        intensity_arr.append(0)
        dt_arr.append(0)

        dz = z-z_ini #calculate elevation change at each time step
        dz_arr.append(sum(dz[mask]))

        dz_cum = z-z_ini_cum #calculate cumulative elevation change
        dz_arr_cum.append(sum(dz_cum[mask]))

        sediment_outflux_channel.append(0)
        sediment_outflux_ruts.append(0)
        sediment_influx_channel.append(0)

    elif chance <= p_storm:
        dt = np.random.exponential(scale=1/6)
        print(dt)
        dt_arr.append(dt)
        intensity = np.random.exponential(scale=5)
        print(intensity)
        intensity_arr.append(intensity)

        mg.at_node['water__unit_flux_in'] = np.ones(540*72)*intensity*2.77778e-7

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

        #=================================Calculate channelized flow transport=================================
        rho_w=1000
        rho_s=2650
        g=9.81
        slope=0.05
        d50=1.8e-5
        tau_c=0.052
        
        surface_water_discharge = mg.at_node['surface_water__discharge']
        channel_discharge = surface_water_discharge[mg.nodes[1:,1:9]].sum(axis=1).max()
        channel_discharge_arr.append(channel_discharge)
        
        for k in range(len(surface_water_discharge)):
            if surface_water_discharge[k] > 0 and road_flag[k] == 0:
                mg.at_node['grain__roughness'][k] = 0.05
                mg.at_node['total__roughness'][k] = 0.1 #this can change according to the treatment
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

wall_time = time.time() - start
print("Wall time for run:", wall_time, "s")

#%%
# # Display the final topography in map view
# imshow_grid(mg, z)

# dx=0.1475
# nrows=540
# ncols=72

# # Display the final topography in 3d surface view
# X = np.arange(0, dx * ncols, dx)
# Y = np.arange(0, dx * nrows, dx)
# X, Y = np.meshgrid(X, Y)

# fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
# ax.plot_surface(X, Y, z.reshape((nrows, ncols)))
#%%
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
#%%
plt.plot(range(0,run_duration), np.multiply(dz_arr,0.1475**2*2650))
plt.plot(range(0,run_duration), np.zeros(len(range(0,run_duration))), '--', color='gray')
plt.xlabel('Day')
plt.ylabel('Total mass change between time steps - \nroad [$kg$]')
plt.xlim(0,run_duration)
# plt.ylim(-0.05,0.05)
plt.show()

plt.plot(range(0,run_duration), np.multiply(dz_arr_cum,0.1475**2*2650)/2)
plt.plot(range(0,run_duration), np.zeros(len(range(0,run_duration))), '--', color='gray')
plt.xlabel('Day')
plt.ylabel('Cumulative mass change (from dz) - \nhalf road [$kg$]')
plt.xlim(0,run_duration)
# plt.ylim(-0.05,0.05)
plt.show()

total_dz = np.abs(min(dz_arr_cum))
total_dV = total_dz*0.1475*0.1475
total_load = total_dV*2650
total_load_div = total_load/2

plt.plot(range(0,run_duration), -((((np.array(sediment_influx_channel)*rho_s*dt_arr).cumsum()))\
        +(np.array(sediment_outflux_ruts)*rho_s*dt_arr).cumsum()))
plt.plot(range(0,run_duration), np.zeros(len(range(0,run_duration))), '--', color='gray')
plt.xlabel('Day')
plt.ylabel('Cumulative mass change (from OFT)- \nhalf road [$kg$]')
plt.xlim(0,run_duration)
# plt.ylim(-0.05,0.05)
plt.show()

#%%
sediment_mass = (np.array(sediment_influx_channel)*rho_s*dt_arr - (np.array(sediment_outflux_ruts)*rho_s*dt_arr) - \
    np.array(sediment_outflux_channel)*rho_s*dt_arr)# - np.array(sediment_outflux_ruts)*rho_s*dt_arr)

plt.plot(range(0,run_duration), sediment_mass)
plt.plot(range(0,run_duration), np.zeros(len(range(0,run_duration))), '--', color='gray')
plt.xlabel('Day')
plt.ylabel('Total mass change between time steps - \nditch line [$kg$]')
plt.xlim(0,run_duration)
# plt.ylim(-0.05,0.05)
plt.show()

plt.plot(range(0,run_duration), (np.multiply(sediment_influx_channel,dt_arr)*rho_s).cumsum())
plt.plot(range(0,run_duration), np.zeros(len(range(0,run_duration))), '--', color='gray')
plt.xlabel('Day')
plt.ylabel('Cumulative mass inflow - \nrouted into ditch [$kg$]')
plt.xlim(0,run_duration)
# plt.ylim(-0.05,0.05)
plt.show()

plt.plot(range(0,run_duration), -(np.multiply(sediment_outflux_channel,dt_arr)*rho_s).cumsum()\
    -(np.array(sediment_outflux_ruts)*rho_s*dt_arr).cumsum())
plt.plot(range(0,run_duration), np.zeros(len(range(0,run_duration))), '--', color='gray')
plt.xlabel('Day')
plt.ylabel('Cumulative mass outflow - \nrouted out of ditch [$kg$]')
plt.xlim(0,run_duration)
# plt.ylim(-0.05,0.05)
plt.show()

# plt.plot(range(0,run_duration), sediment_mass.cumsum())
# plt.plot(range(0,run_duration), np.zeros(len(range(0,run_duration))), '--', color='gray')
# plt.xlabel('Day')
# plt.ylabel('Cumulative mass change - \nditch line [$kg$]')
# plt.xlim(0,run_duration)
# # plt.ylim(-0.05,0.05)
# plt.show()

print('Total rainfall (90 days):', sum(np.multiply(intensity_arr,np.multiply(dt_arr,24))), 'mm')
print('Comparison between sediment load from road elevation change calculation and channel influx from OFT:',\
    total_load_div - ((((np.array(sediment_influx_channel)*rho_s*dt_arr).sum()))\
        +(np.array(sediment_outflux_ruts)*rho_s*dt_arr).sum()))

print('Sediment load from road (half-road estimate):', total_load_div, 'kg')
print('Sediment load from road (actual):', ((((np.array(sediment_influx_channel)*rho_s*dt_arr).sum()))\
        +(np.array(sediment_outflux_ruts)*rho_s*dt_arr).sum()), 'kg')

print('Sediment load from road & ditch:', (np.array(sediment_outflux_channel)*rho_s*dt_arr).sum()\
    +(np.array(sediment_outflux_ruts)*rho_s*dt_arr).sum(), 'kg')

print('Sediment pumped:', mg.at_node['sediment__added'].sum()*0.1475*0.1475*2650*0.6, 'kg')

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
