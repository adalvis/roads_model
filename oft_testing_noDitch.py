# %%
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from landlab import RasterModelGrid, imshow_grid
from landlab.components import OverlandFlowTransporter, FlowAccumulator, TruckPassErosion, DepressionFinderAndRouter
np.set_printoptions(threshold=np.inf)
from utilities.erodible_grid import Erodible_Grid

# %% Define parameters
run_duration = 122

#Physical constants
rho_w=1000
rho_s=2650
g=9.81

#Define site
site_name = "MEL12"
rain_gauge_list = ["RG_BISH05_mm","RG_BISH12_mm","RG_DEL0103_mm",\
    "RG_KID1316_mm","RG_KID46_mm","RG_MEL05_mm","RG_MEL14_mm",\
    "RG_NASE0104_mm","RG_NASE05i_mm","RG_NEWS1920_mm"]
rain_gauge = rain_gauge_list[6]

#Call parameters from .csv file
parameters = pd.read_csv("input/parameters_WY2024.csv")
site_params = parameters.loc[parameters["Site Name"] == site_name].iloc[0]
S = site_params["Road Gradient"]/100
porosity = 0.35

#Define index to start model run from
intensity_index = 62

seed = 1 
np.random.seed(seed)

#Intensity data in mm/hr
intensity = pd.read_csv("input/WY2024_RG_daily_intensity.csv")
# intensity = np.random.choice([0,4],size=(run_duration),p=[0.4,0.6])

#Change the site name and index values to get different sites and dates
intensity_90 = intensity[rain_gauge].iloc[intensity_index:].values
# intensity_90 = intensity.copy()

#Daily storm duration in hours
dt_hours = pd.read_csv("input/WY2024_RG_daily_dt.csv")
dt_hours_90 = dt_hours[rain_gauge].iloc[intensity_index:].values 
# dt_hours = np.array([4 if x==4 else 0 for x in intensity])
# dt_hours_90 = dt_hours.copy()
#Convert to dt to days
dt = np.array(dt_hours_90)/24

# initialize road layer depths
Sa_ini = 0.01 # active depth in m
Ss_ini = 0.23 # surfacing depth in m
Sb_ini = 2    # ballast depth in m

#Initialize average number of truck passes per day for truck pass erosion
truck_num_ini = 4

#Define roughness values for fine and coarse material
n_c = 0.05   
n_f = 0.015

# fractions of fine and coarse grains in the active layer
# FIX THIS FORMULATION
f_af = 0.75
f_ac = 0.25

#Define d50 and tau_c
d50_road = 0.0001 # [m] 
tau_c_road = 0.146

# %% Create the grid, add random noise, add fields

#Parameters for grid creation
cell_spacing = 0.1475 # cell width or length dimension in meters
cell_area = cell_spacing**2
nrows = 540 # number of rows in the grid
ncols = 64  # number of columns in the grid (NO DITCH)

#We're using half tire width for node spacing
center = 32
half_width = 7 
full_tire = False

eg = Erodible_Grid(nrows=nrows, ncols=ncols,\
    spacing=cell_spacing, full_tire=full_tire, long_slope=S,road_peak=center,ditch=False)

mg, z, road_flag, n = eg() 

noise_amplitude=0.005
road = road_flag==1

# seed = 42 
# np.random.seed(seed)
random=np.random.rand(
    len(z[road])
)
z[road] += noise_amplitude * random #z is the road elevation

#Add depth fields that will update in the components; these are the initial conditions
active_depth = mg.add_ones('active__depth', at='node')
active_depth *= Sa_ini
surf_depth = mg.add_ones('surfacing__depth', at='node')
surf_depth *= Ss_ini
ball_depth = mg.add_ones('ballast__depth', at='node')
ball_depth *= Sb_ini

# %% Pre-define location indices
ruts = [mg.nodes[:, 26-9:40-9], mg.nodes[:, 41-9:55-9]]
half_road = mg.nodes[:, 0:33]
full_road = mg.nodes[:, 0:64]

# %% DEM plot
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(3, 6))
imshow_grid(mg, z, plot_name='Synthetic road', var_name='Elevation', var_units='m',\
    grid_units=('m', 'm'), cmap='terrain', color_for_closed='black', vmin=0, vmax=5)
plt.xlabel('Road width (m)')
plt.ylabel('Road length (m)')
plt.tight_layout()
plt.show()

# %% Save some pre-run variables for comparison later
X = mg.node_x.reshape(mg.shape)
Y = mg.node_y.reshape(mg.shape)
Z = z.reshape(mg.shape)

xsec_pre = mg.at_node['topographic__elevation'][mg.nodes[100,:].flatten()].copy()
# %% Prep arrays/lists
mask = road_flag
intensity_arr=[]
dt_arr = []
dz_arr=[]
dz_arr_cum = []

#Depths for each layer
sa_arr = np.zeros(run_duration)
ss_arr = np.zeros(run_duration)
sb_arr = np.zeros(run_duration)

#Flux/mass arrays
mass_fillslope_inflow = np.zeros(run_duration)
mass_fillslope_rut_outflow = np.zeros(run_duration)
mass_ditch_inflow = np.zeros(run_duration)
mass_ditch_rut_outflow = np.zeros(run_duration)
total_road_mass = np.zeros(run_duration)
road_mass_change_oft = np.zeros(run_duration)

#
road_shear_frac_arr = np.zeros(run_duration)
road_shear_cum_arr = np.zeros(run_duration)
road_shear_i = np.zeros(mg.number_of_nodes, dtype=bool)
road_shear_cum = np.zeros(mg.number_of_nodes, dtype=bool)

# shear stresses (averages)
avg_shear_stress_ruts = []
avg_shear_stress_road = []

# manning's roughness averages
avg_n_ruts = []
avg_n_road = []

# sediment load in the ruts due to TPE
tpe_load_ruts = []

# shear stress partitioning coefficient averages
fs_avg_ruts = []
fs_avg_road = []

truck_num = 0     
z_ini_cum = mg.at_node['topographic__elevation'].copy()
active_init = mg.at_node['active__depth'][full_road].copy()
surfacing_init = mg.at_node['surfacing__depth'][full_road].copy()
ballast_init = mg.at_node['ballast__depth'][full_road].copy()

# %% Initialize Landlab components
tpe = TruckPassErosion(mg, center, half_width, full_tire, truck_num=truck_num_ini, \
    scat_loss=8e-5, f_af=f_af, f_ac=f_ac) #initialize component, 

df_init = DepressionFinderAndRouter(mg, reroute_flow = True)
df_init.map_depressions()

fa = FlowAccumulator(mg, surface='topographic__elevation', \
    flow_director="FlowDirectorD8", runoff_rate=1.538889e-6,)

oft = OverlandFlowTransporter(mg, porosity=porosity, d50=d50_road, \
    longitudinal_slope=S, tau_c=tau_c_road, n_c=n_c, n_f=n_f)

# %% Intensity distribution per storm
p = np.linspace(0.0001, 35, 1000)
def pdf(p, p_mean):
    p_prime = p/p_mean
    pdf = 1/p_mean*np.exp(-p/p_mean)
    return pdf, p_prime

def cdf(p, p_mean):
    p_prime = p/p_mean
    cdf = 1-np.exp(-p_prime)
    return cdf, p_prime

def p_prime_arr(prob):
    p_prime_arr = -np.log(1-prob)
    return p_prime_arr

prob_arr = np.concatenate((np.array([0.25, 0.5]), np.linspace(0.6, 0.99, 8)))
frac_arr = np.abs(np.concatenate((np.array([0.25]), np.diff(1-prob_arr))))

pdf, p_prime = pdf(p, 5)
cdf, p_prime = cdf(p, 5)
p_prime_arr = p_prime_arr(prob_arr)

# Plots are helpful, save for later
# fig, ax = plt.subplots()
# ax.plot(p, pdf)
# ax.annotate(r'$f(p)=\frac{1}{p_{mean}}exp(-\frac{p}{p_{mean}})$', xy=(10,0.1), xycoords='data',
#             xytext=(0.5, 0.5), textcoords='axes fraction',
#             va='top', ha='center',size=12)
# ax.set_xlim(0,35)
# ax.set_ylim(0,0.2)
# ax.set_xlabel('$p$ [mm/hr]')
# ax.set_ylabel('$f(p)$')
# ax.set_title('PDF w.r.t. precipitation')
# plt.show()

# fig, ax = plt.subplots(1,2, figsize=(14,4.5))
# ax[0].plot(p, cdf)
# ax[0].annotate(r'$F(p)=1-exp{(-\frac{p}{p_{mean}})}$', xy=(10,0.5), xycoords='data',
#                xytext=(0.5, 0.5), textcoords='axes fraction',
#                va='top', ha='center',size=12)
# ax[0].set_xlim(0,35)
# ax[0].set_ylim(0,1)
# ax[0].set_xlabel('$p$ [mm/hr]')
# ax[0].set_ylabel('$F(p)$')
# ax[0].set_title('CDF w.r.t. precipitation')

# ax[1].plot(p_prime, cdf)
# ax[1].annotate("$F(p\')=1-exp{(-p\')}$", xy=(2,0.5), xycoords='data',
#                xytext=(0.5, 0.5), textcoords='axes fraction',
#                va='top', ha='center',size=12)
# ax[1].set_xlim(0,7)
# ax[1].set_ylim(0,1)
# ax[1].set_xlabel('$p\' =\\frac{p}{p_{mean}}$')
# ax[1].set_ylabel('$F(p\')$')
# ax[1].set_title('CDF w.r.t. nondimensionalized precipitation')
# plt.show()

# fig, ax = plt.subplots()
# for i in range(len(prob_arr)):
#     plt.plot([0, max(p_prime)],[prob_arr[i],prob_arr[i]], '--', color='gray')
# ax.plot(p_prime, cdf)
# ax.annotate('$F(p\')=%0.2f$'%prob_arr[0], xy=(5, prob_arr[0]-0.035), size=8)
# ax.annotate('$F(p\')=%0.2f$'%prob_arr[1], xy=(5, prob_arr[1]-0.035), size=8)
# ax.annotate('$F(p\')=%0.2f$'%prob_arr[2], xy=(5, prob_arr[2]-0.035), size=8)
# ax.annotate('$F(p\')=%0.2f$'%prob_arr[3], xy=(5, prob_arr[3]-0.035), size=8)
# ax.annotate('$F(p\')=%0.2f$'%prob_arr[4], xy=(5, prob_arr[4]-0.035), size=8)
# ax.annotate('$F(p\')=%0.2f$'%prob_arr[5], xy=(5, prob_arr[5]-0.035), size=8)
# ax.annotate('$F(p\')=%0.2f$'%prob_arr[6], xy=(5, prob_arr[6]-0.035), size=8)
# ax.annotate('$F(p\')=%0.2f$'%prob_arr[7], xy=(5, prob_arr[7]-0.035), size=8)
# ax.annotate('$F(p\')=%0.2f$'%prob_arr[8], xy=(5, prob_arr[8]-0.035), size=8)
# ax.annotate('$F(p\')=%0.2f$'%prob_arr[9], xy=(5, prob_arr[9]-0.035), size=8)
# ax.set_xlim(0,7)
# ax.set_ylim(0,1)
# ax.set_xlabel('$p\'$')
# ax.set_ylabel('$F(p\')$')
# ax.set_title('CDF w.r.t. nondimensionalized precipitation')
# plt.show()

# %% Main loop
start = time.time()
for i in range(0, run_duration): # daily time step
    z_ini = mg.at_node['topographic__elevation'].copy()

    tpe.run_one_step()

    truck_num += tpe.truck_num
    
    intensity = intensity_90[i]  # use the i-th day's intensity
    dt_day = dt[i] # use the i-th day's time step

    if intensity <= 0:
        print(f"Day {i}: No rainfall")
        rain_m_per_s = intensity * 2.77778e-7 # conversion to m/s
        mg.at_node['water__unit_flux_in'] = np.ones(mg.number_of_nodes) * rain_m_per_s
        fa.accumulate_flow()

        intensity_arr.append(0)
        dt_arr.append(0)

        dz = z - z_ini
        dz_arr.append(sum(dz[full_road.flatten()]))

        dz_cum = z - z_ini_cum
        dz_arr_cum.append(sum(dz_cum[full_road.flatten()]))

        road_shear_cum_arr[i] = road_shear_cum_arr[i-1]

        #Append shear stress vectors
        avg_shear_stress_ruts.append(0)
        avg_shear_stress_road.append(0)

        #Append manning's roughness vectors
        avg_n_ruts.append(np.nanmean(mg.at_node['total__roughness'][ruts]))
        avg_n_road.append(np.nanmean(mg.at_node['total__roughness'][full_road]))

        #Append shear stress partitioning coefficient
        fs_avg_ruts.append(np.nanmean(mg.at_node['shear_stress__partitioning'][ruts]))
        fs_avg_road.append(np.nanmean(mg.at_node['shear_stress__partitioning'][full_road]))

        #Append TPE loading
        tpe_load_ruts.append((tpe.sed_added).sum()\
            *cell_area*rho_s*(1-porosity)) # gives increase in mass in the active layer due to truck passes
        
        #For plotting layer depths
        sa_arr[i] = np.sum(mg.at_node['active__depth'][full_road])
        ss_arr[i] = np.sum(mg.at_node['surfacing__depth'][full_road])
        sb_arr[i] = np.sum(mg.at_node['ballast__depth'][full_road])

    else:
        intensity_arr.append(intensity)
        print(f"Day {i}: Average Intensity = {intensity:.2f} mm/hr")
        dt_arr.append(dt_day)
        print(f"Day {i}: Length of Storm = {dt_day:.2f} day")
        
        road_shear_i[:] = 0

        for j, storm_frac in enumerate(frac_arr):
            intensity_dist = p_prime_arr[j]*intensity
            dt_frac = storm_frac*dt_day
            #=================================Calculate overland flow transport=================================
            rain_m_per_s = intensity_dist * 2.77778e-7 # conversion to m/s
            mg.at_node['water__unit_flux_in'] = np.ones(mg.number_of_nodes) * rain_m_per_s
            fa.accumulate_flow()
            oft.run_one_step(dt_frac)

            mass_ditch_rut_outflow_i = (mg.at_node["sediment__volume_influx"][mg.nodes[0,1:33]]).sum()*rho_s*dt_frac*86400
            mass_ditch_inflow_i = (mg.at_node["sediment__volume_influx"][mg.nodes[:,0]]).sum()*rho_s*dt_frac*86400
            mass_fillslope_inflow_i = (mg.at_node["sediment__volume_influx"][mg.nodes[:,63]]).sum()*rho_s*dt_frac*86400
            mass_fillslope_rut_outflow_i = (mg.at_node["sediment__volume_influx"][mg.nodes[0,33:63]]).sum()*rho_s*dt_frac*86400

            mass_ditch_rut_outflow[i] += mass_ditch_rut_outflow_i
            mass_ditch_inflow[i] += mass_ditch_inflow_i
            mass_fillslope_inflow[i] += mass_fillslope_inflow_i
            mass_fillslope_rut_outflow[i] += mass_fillslope_rut_outflow_i
            
            # mg.at_node['shear_stress'] = oft.shear_stress
            road_shear = oft.shear_stress[full_road].flatten()

            road_shear_i = np.array([True if shear_stress>=tau_c_road or road_shear_i[x]==True \
                 else False for x, shear_stress in enumerate(road_shear)])
            road_shear_cum = np.array([True if shear_stress>=tau_c_road or road_shear_cum[x]==True\
                 else False for x, shear_stress in enumerate(road_shear)])

        road_shear_frac_arr[i] = road_shear_i[road_shear_i==True].sum()/len(road_shear_i)
        road_shear_cum_arr[i] = road_shear_cum[road_shear_cum==True].sum()/len(road_shear_cum)

        mg.at_node['shear_stress'] = oft.shear_stress
        avg_shear_stress_ruts.append(np.nanmean(mg.at_node['shear_stress'][ruts]))
        avg_shear_stress_road.append(np.nanmean(mg.at_node['shear_stress'][full_road]))

        road_mass_change_oft[i] = mass_ditch_inflow[i] + mass_ditch_rut_outflow[i]
        total_road_mass[i] = mass_ditch_inflow[i] + mass_ditch_rut_outflow[i] + mass_fillslope_inflow[i] + mass_fillslope_rut_outflow[i]
        
        dz = z-z_ini # calculate elevation change at each daily time step
        dz_arr.append(sum(dz[full_road.flatten()]))

        dz_cum = z-z_ini_cum # calculate cumulative elevation change
        dz_arr_cum.append(sum(dz_cum[full_road.flatten()])) 
        
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(3, 6))
        im = imshow_grid(mg,'active__fines', var_name='Active Fines Depth', 
                     plot_name='Active Fines Depth, t = %i days' %i,
                     var_units='$m$', grid_units=('m','m'), 
                     cmap='pink', vmin=0, vmax=0.00005, shrink=0.9)
        plt.xlabel('Road width (m)')
        plt.ylabel('Road length (m)')
        plt.tight_layout()
        plt.savefig('output/f_%i_days.png' %i)
        plt.close()
        # plt.show()

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(3, 6))
        im = imshow_grid(mg,'topographic__steepest_slope', var_name='Slope', 
                     plot_name='Slope, t = %i days' %i,
                     var_units='$m/m$', grid_units=('m','m'), 
                     cmap='jet',  shrink=0.9, vmin=0, vmax=1.0,)
        plt.xlabel('Road width (m)')
        plt.ylabel('Road length (m)')
        plt.tight_layout()
        plt.savefig('output/s_%i_days.png' %i)
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

        mg.add_field('dz_cum', dz_cum, at='node', units='m', clobber=True)
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(3, 6))
        plt.xlabel('Road width (m)')
        plt.ylabel('Road length (m)')
        im = imshow_grid(mg,'dz_cum', var_name='Cumulative dz', var_units='m', 
                     plot_name='Elevation change, t = %i days' %i,
                     grid_units=('m','m'), cmap='RdBu', vmin=-0.0009, 
                     vmax=0.0009, shrink=0.9)
        plt.xlabel('Road width (m)')
        plt.ylabel('Road length (m)')
        plt.tight_layout()
        plt.savefig('output/dz_cum_%i_days.png' %i)
        plt.close()
        # plt.show()

        #Append manning's n vectors
        avg_n_ruts.append(np.nanmean(mg.at_node['total__roughness'][ruts]))
        avg_n_road.append(np.nanmean(mg.at_node['total__roughness'][full_road]))

        #Append shear stress partitioning coefficient vectors        
        fs_avg_ruts.append(np.nanmean(mg.at_node['shear_stress__partitioning'][ruts]))
        fs_avg_road.append(np.nanmean(mg.at_node['shear_stress__partitioning'][full_road]))
        
        #Append TPE loading
        tpe_load_ruts.append((tpe.sed_added).sum()*cell_area*rho_s*(1-porosity))
    
        #For plotting layer depths
        sa_arr[i] = np.sum(mg.at_node['active__depth'][full_road])
        ss_arr[i] = np.sum(mg.at_node['surfacing__depth'][full_road])
        sb_arr[i] = np.sum(mg.at_node['ballast__depth'][full_road])
    if i == 92-62 or i == 123-62 or i == 152-62 or i == 183-62:
        print("End of month total:", np.round((mass_ditch_inflow + mass_ditch_rut_outflow).sum(),2), "kg" )
    print("Change in sum of depths:", np.round((active_init.sum()+surfacing_init.sum()+ballast_init.sum())\
         - (sa_arr[i]+ss_arr[i]+sb_arr[i]),2))
    print("Change in z from OFT:", np.round(np.divide(total_road_mass.cumsum()[i],\
         cell_area*rho_s*(1-porosity)),2))
wall_time = time.time() - start
print("Wall time for run:", wall_time, "s")

# %% Calculations for plots
road_mass_change_dz = np.multiply(dz_arr, (cell_area*rho_s*(1-porosity)))/2
cum_road_mass_change_dz = np.multiply(dz_arr_cum, cell_area*rho_s*(1-porosity))/2

cum_road_mass_change_oft = road_mass_change_oft.cumsum()
# %% GIF creation
from PIL import Image
import glob

path = "output/"
images_w=[]
images_Q=[]
images_dz=[]
images_s=[]
for file in glob.glob(path+'f*.png'):
    images_w.append(file)
for file in glob.glob(path+'s*.png'):
    images_s.append(file)
for file in glob.glob(path+'Q*.png'):
    images_Q.append(file)
for file in glob.glob(path+'dz_cum*.png'):
    images_dz.append(file)

def number(filename):
    return int(filename[9:-9])

def num(filename):
    return int(filename[14:-9])

images_w = sorted(images_w, key=number)
images_s = sorted(images_s, key=number)
images_Q = sorted(images_Q, key=number)
images_dz= sorted(images_dz, key=num)

# Create a list of image objects
image_list_s = [Image.open(file) for file in images_s]
image_list_w = [Image.open(file) for file in images_w]
image_list_Q = [Image.open(file) for file in images_Q]
image_list_dz = [Image.open(file) for file in images_dz]

# Save the first image as a GIF file
image_list_w[0].save(
            'water_depth.gif',
            save_all=True,
            append_images=image_list_w[1:], # append rest of the images
            duration=1000, # in milliseconds
            loop=0)

image_list_s[0].save(
            'slope.gif',
            save_all=True,
            append_images=image_list_s[1:], # append rest of the images
            duration=1000, # in milliseconds
            loop=0)

image_list_Q[0].save(
            'discharge.gif',
            save_all=True,
            append_images=image_list_Q[1:], # append rest of the images
            duration=1000, # in milliseconds
            loop=0)

image_list_dz[0].save(
            'dz_cum.gif',
            save_all=True,
            append_images=image_list_dz[1:], # append rest of the images
            duration=1000, # in milliseconds
            loop=0)

# %% Cross section plot
xsec_active = mg.at_node['topographic__elevation'][mg.nodes[100,:].flatten()]
xsec_surf =  mg.at_node['surfacing__elevation'][mg.nodes[100,:].flatten()] 
xsec_ball = mg.at_node['ballast__elevation'][mg.nodes[100,:].flatten()]

plt.figure(figsize=(8,3), layout='tight')
plt.plot(X[36], xsec_pre, color='gray', linestyle='-.', label='Before')
plt.plot(X[36], xsec_active, color ='black', linestyle='-', label = 'After - Active elevation')
plt.plot(X[36], xsec_surf, color ='magenta', linestyle='-', label = 'After - Surfacing elevation')
plt.plot(X[36], xsec_ball, color ='cyan', linestyle='-', label = 'After - Ballast elevation ')
plt.xlim(0,9.5)
plt.xlabel('Road width (m)')
plt.ylabel('Elevation (m)')
plt.legend()
plt.show()

# %% Rainfall plots
fig, ax = plt.subplots(2,1)
ax[0].bar(range(0,run_duration), np.multiply(intensity_arr,np.multiply(dt_arr,24)))
ax[0].set_xlabel('Day')
ax[0].set_ylabel('Rainfall [mm]')
ax[0].set_xlim(0,run_duration)

ax[1].plot(range(0,run_duration), intensity_arr)
ax[1].set_xlabel('Day')
ax[1].set_ylabel('Rainfall intensity [mm/hr]')
ax[1].set_xlim(0,run_duration)
plt.suptitle(r'%s' %site_name)
plt.tight_layout()
plt.show()

# %% Important model output
total_dz = np.abs(min(dz_arr_cum)) 
total_dV = total_dz*cell_area 
total_load = total_dV*rho_s*(1-porosity)
total_load_div = total_load/2

print(
    "Total rainfall,",
    run_duration, "days:", 
    np.round(sum(np.multiply(intensity_arr,np.multiply(dt_arr,24)),2)), 'mm'
    )
print(
    'Sediment pumped:', 
    np.round(mg.at_node['sediment__added'].sum()*cell_area*rho_s*(1-porosity),2), 'kg'
    )
print(
    'Cumulative sediment load from road (half-road OFT calculation):', 
    np.round((mass_ditch_inflow + mass_ditch_rut_outflow).sum(),2), 'kg' 
    )
print(
    'Cumulative sediment load from road (half-road OFT calculation - fillslope side):', 
    np.round((mass_fillslope_inflow + mass_fillslope_rut_outflow).sum(),2), 'kg' 
    )
print(
    'Cumulative sediment load from road (full-road OFT calculation):', 
    np.round((total_road_mass).sum(),2), 'kg' 
    )
print(
    'Cumulative sediment load from road (half-road dz estimate):', 
    np.round(total_load_div,2), 'kg'
    )
print(
    'Cumulative sediment load from road (full-road dz estimate):', 
    np.round(total_load, 2), 'kg'
    )
print('Comparison between sediment load from road elevation change calculation and channel influx from OFT:',
    np.round(total_load - (total_road_mass).sum(),2)
    )

# %% Delta mass between time steps
fig, ax = plt.subplots(1,2, figsize=(9,4))

# plot total mass change between time steps on the road
ax[0].plot(range(0,run_duration), -road_mass_change_oft) # added porosity consideration
ax[0].plot(range(0,run_duration), np.zeros(len(range(0,run_duration))), '--', color='gray')
ax[0].set_xlabel('Day')
ax[0].set_ylabel(r'$\Delta$ mass between time steps [$kg$]')
ax[0].set_xlim(0,run_duration)
ax[0].set_title('(a) Half Road (OFT)')

# plot total mass change between time steps along the ditch line
ax[1].plot(range(0,run_duration), road_mass_change_dz)
ax[1].plot(range(0,run_duration), np.zeros(len(range(0,run_duration))), '--', color='gray')
ax[1].set_xlabel('Day')
ax[1].set_ylabel(r'$\Delta$ mass between time steps [$kg$]')
ax[1].set_xlim(0,run_duration)
ax[1].set_title('(b) Half Road (dz)')
plt.tight_layout()
plt.show()

# %% Cumulative mass change
fig, ax = plt.subplots(1,2, figsize=(9,4))

ax[0].plot(range(0,run_duration), -cum_road_mass_change_oft)
ax[0].plot(range(0,run_duration), np.zeros(len(range(0,run_duration))), '--', color='gray')
ax[0].set_xlabel('Day')
ax[0].set_ylabel('Cumulative mass change - \nhalf road [$kg$]')
ax[0].set_xlim(0,run_duration)
ax[0].set_title('(a) Half Road (OFT)')

ax[1].plot(range(0,run_duration), cum_road_mass_change_dz)
ax[1].plot(range(0,run_duration), np.zeros(len(range(0,run_duration))), '--', color='gray')
ax[1].set_xlabel('Day')
ax[1].set_ylabel('Cumulative mass change - \nhalf road [$kg$]')
ax[1].set_xlim(0,run_duration)
ax[1].set_title('(b) Half Road (dz)')

plt.tight_layout()
plt.show()

# %% Testing?
# plt.plot(np.multiply(sa_arr,(cell_area*rho_s*(1-porosity)))/2, -cum_road_mass_change_oft)
# plt.xlabel("Active depth mass\nhalf road [$kg$]")
# plt.ylabel('Cumulative mass change (from OFT)\nhalf road [$kg$]')
# plt.show()
# %% TPE loading
# plot sediment load to the active layer in the ruts from truck passes
plt.plot(range(0,run_duration), tpe_load_ruts)
plt.xlabel('Day')
plt.ylabel('Cumulative sediment load to the active layer of the ruts \nfrom tpe [$kg$]')
plt.xlim(0,run_duration)
plt.show()

# %% Total depths over the road surface 
fig, ax = plt.subplots(4,1, figsize=(4,9))

active_init.sum()+surfacing_init.sum()+ballast_init.sum()

ax[0].plot(range(0,run_duration), (-active_init.sum()-surfacing_init.sum() + (sa_arr+ss_arr))/(nrows*ncols))
ax[0].set_xlabel('Day')
ax[0].set_ylabel('Active + Surfacing Depth\nchange [$m$]')
ax[0].set_xlim(0,run_duration)
ax[0].set_title(r'%s ($n_{f_{road}} = %0.3f$)' %(site_name, n_f))

ax[1].plot(range(0,run_duration), (-active_init.sum()+sa_arr)/(nrows*ncols))
ax[1].set_xlabel('Day')
ax[1].set_ylabel('Active Depth\nchange [$m$]')
ax[1].set_xlim(0,run_duration)

ax[2].plot(range(0,run_duration), (-surfacing_init.sum()+ss_arr)/(nrows*ncols))
ax[2].set_xlabel('Day')
ax[2].set_ylabel('Surfacing Depth\nchange [$m$]')
ax[2].set_xlim(0,run_duration)

ax[3].plot(range(0,run_duration), (-ballast_init.sum()+sb_arr)/(nrows*ncols))
ax[3].set_xlabel('Day')
ax[3].set_ylabel('Ballast Depth\nchange [$m$]')
ax[3].set_xlim(0,run_duration)
plt.tight_layout()
plt.show()

# %%
# plt.bar(range(0,run_duration), np.multiply(road_shear_frac_arr, 100))
# plt.xlabel('Day')
# plt.ylabel('Percentage of Road $\\geq$ Critical \nShear Stress')
# plt.xlim(0,run_duration)
# plt.title(r'%s' %site_name)
# plt.show()

# plt.plot(range(0,run_duration), np.multiply(road_shear_cum_arr, 100))
# plt.xlabel('Day')
# plt.ylabel('Cumulative Percentage of Road $\\geq$ Critical \nShear Stress')
# plt.xlim(0,run_duration)
# plt.title(r'%s' %site_name)
# plt.show()

# # plot shear stress partitioning coefficient over time
# plt.plot(range(0,run_duration), fs_avg_road, label='Full Road')
# plt.plot(range(0,run_duration), fs_avg_ruts, label='Ruts')
# plt.xlabel('Day')
# plt.ylabel('Average Shear Stress Partitioning \nCoefficient: $f_s$')
# plt.xlim(0,run_duration)
# plt.legend()
# plt.show()

# %%
