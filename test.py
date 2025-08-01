import time

import matplotlib.pyplot as plt
import numpy as np

from landlab import RasterModelGrid, imshow_grid
from landlab.components import OverlandFlowTransporter, FlowAccumulator, TruckPassErosion
np.set_printoptions(threshold=np.inf)

#%%
# Parameters
run_duration = 90  # run duration, y
dt = 1  # time-step duration, y
seed = 1

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
noise_amplitude=0.007

np.random.seed(42)

# cores = mg.core_nodes
road = road_flag==1

z[road] += noise_amplitude * np.random.rand(
    len(z[road])
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
fa = FlowAccumulator(mg, surface='topographic__elevation', flow_director="FlowDirectorD8", runoff_rate=1.38889e-6)
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

z_ini_cum = mg.at_node['topographic__elevation'].copy()


# Main loop
start = time.time()
for i in range(0, run_duration):
    z_ini = mg.at_node['topographic__elevation'].copy()
    tpe.run_one_step()
    tracks.append(tpe.tire_tracks) #can I concatenate the values rather than having separate arrays?
    print(tpe._truck_num)
    p_storm = 0.25

    chance = np.random.uniform()

    if chance > p_storm:
        intensity_arr.append(0)
        dt_arr.append(0)

        dz = z-z_ini #calculate elevation change at each time step
        dz_arr.append(sum(dz))

        dz_cum = z-z_ini_cum #calculate cumulative elevation change
        dz_arr_cum.append(sum(dz_cum))

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
        if z[i] <= z_limit[i]:
            z[i] = z_limit[i]

        dz = z-z_ini #calculate elevation change at each time step
        dz_arr.append(sum(dz))

        dz_cum = z-z_ini_cum #calculate cumulative elevation change
        dz_arr_cum.append(sum(dz_cum)) 
        
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

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 9))
        im = imshow_grid(mg,'surface_water__discharge', var_name='Q', 
                     plot_name='Steady state Q, t = %i days' %i,
                     var_units='$m^3/2$', grid_units=('m','m'), 
                     cmap='Blues', vmin=0, vmax=1e-6)
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
                     grid_units=('m','m'), cmap='RdBu', vmin=-0.001, 
                     vmax=0.001, shrink=0.9)
        plt.xlabel('Road width (m)')
        plt.ylabel('Road length (m)')
        plt.tight_layout()
        # plt.savefig('output/dz_cum_%i_days.png' %i)
        plt.show()

        #=================================Calculate channelized flow transport=================================
        surface_water_discharge = mg.at_node['surface_water__discharge']
        unit_discharge = surface_water_discharge/mg.dx
        for k in range(len(unit_discharge)):
            if unit_discharge[k] > 0 and road_flag[k] == 0:
                mg.at_node['grain__roughness'][k] = 0.05
                mg.at_node['total__roughness'][k] = 0.1
                mg.at_node['shear_stress__partitioning'][k] = \
                    (mg.at_node['grain__roughness'][k]\
                        /mg.at_node['total__roughness'][k])^(24/13)
   #     for i in self.grid.nodes.reshape(np.size(self.grid.nodes))[self._road_flag==0]:
    #         if self._unit_discharge[i]
                        
    #         self._water_depth[i] = ((self._n_t[i]*self._unit_discharge[i])/\
    #             (np.sqrt(6*self._slope[i]/0.718)))**(6/13) #use overall channel S

    sa = mg.at_node['active__elev'].mean()\
            - mg.at_node['surfacing__elev'].mean() 
    sa_arr.append(sa)
    ss = mg.at_node['surfacing__depth'].mean()
    ss_arr.append(ss)
    sb = mg.at_node['ballast__depth'].mean()
    sb_arr.append(sb)

wall_time = time.time() - start
print("Wall time for run:", wall_time, "s")

#%%
# Display the final topography in map view
imshow_grid(mg, z)

dx=0.1475
nrows=540
ncols=72

# Display the final topography in 3d surface view
X = np.arange(0, dx * ncols, dx)
Y = np.arange(0, dx * nrows, dx)
X, Y = np.meshgrid(X, Y)

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
ax.plot_surface(X, Y, z.reshape((nrows, ncols)))
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
plt.plot(range(0,run_duration), dz_arr)
plt.plot(range(0,run_duration), np.zeros(len(range(0,run_duration))), '--', color='gray')
plt.xlabel('Day')
plt.ylabel('Total elevation change between time steps - road [m]')
plt.xlim(0,run_duration)
# plt.ylim(-0.05,0.05)

#%%
plt.plot(range(0,run_duration), dz_arr_cum)
plt.plot(range(0,run_duration), np.zeros(len(range(0,run_duration))), '--', color='gray')
plt.xlabel('Day')
plt.ylabel('Cumulative elevation change - road [m]')
plt.xlim(0,run_duration)
# plt.ylim(-0.05,0.05)
plt.show()

total_dz = np.abs(min(dz_arr_cum))
total_dV = total_dz*0.1475*0.1475
total_load = total_dV*2650
total_load_div = total_load/2
sed_load = total_load_div/(540*0.1475)

print('Sediment load per meter of road: ', sed_load)


#%%
plt.plot(range(0,run_duration), sa_arr)
plt.xlabel('Day')
plt.ylabel('Individual layer elevation changes [m]')
plt.xlim(0,run_duration)
# plt.ylim(0.019,0.02)
plt.show()

plt.plot(range(0,run_duration), ss_arr)
plt.xlabel('Day')
plt.ylabel('Individual layer elevation changes [m]')
plt.xlim(0,run_duration)
plt.show()

plt.plot(range(0,run_duration), sb_arr)
plt.xlabel('Day')
plt.ylabel('Individual layer elevation changes [m]')
plt.xlim(0,run_duration)
plt.show()
# %%
