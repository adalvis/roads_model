#%%
# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import numpy as np
from scipy.stats import expon

np.set_printoptions(threshold=np.inf)

#%%
data = pd.read_csv('/home/adalvis/github/roads_model/'+\
	'WY2025_RG_hourly.csv', index_col='date') #Change path to dataset you want to aggregate

data.index = pd.to_datetime(data.index)
data = data.asfreq('h') #Ensure data is actually hourly.
data[data<0]=0 #Make sure there are no negative values.
data.fillna(0, inplace=True) #Make sure there are no NaN values.
df = data.copy()

#%%
fig1, ax1 = plt.subplots()
df.plot(ax=ax1, linewidth=0.75) 
plt.xlabel('Date')
plt.ylabel('Intensity (mm/hr)')
plt.title('Raw WY 2025 data')
plt.tight_layout()
plt.show()

#%%
df['dates'] = df.index.date
df_daily = df.groupby('dates').agg('sum') #total daily rainfall
df_count = df.groupby('dates').agg(lambda x: (x != 0).sum()) #total duration of rainfall in a day given in hours
df_intensity = df_daily/df_count #intensity in mm/hr
df_intensity.fillna(0, inplace=True) #no NaN values

fig2, ax2 = plt.subplots()
df_daily.plot(ax=ax2, linewidth=0.75) 
plt.xlabel('Date')
plt.ylabel('Rainfall depth (mm)')
plt.title('Total daily depth WY 2025')
plt.tight_layout()
plt.show()

fig3, ax3 = plt.subplots()
df_intensity.plot(ax=ax3, linewidth=0.75) 
plt.xlabel('Date')
plt.ylabel('Rainfall intensity (mm/hr)')
plt.title('Daily rainfall intensity WY 2025')
plt.tight_layout()
plt.show()

fig4, ax4 = plt.subplots()
df_count.plot(ax=ax4, linewidth=0.75) 
plt.xlabel('Date')
plt.ylabel('Duration of rainfall [hr]')
plt.title('Daily rainfall duration WY 2025')
plt.tight_layout()
plt.show()

#%%
# Save output
# df_daily.to_csv('/home/adalvis/github/roads_model/'+\
# 	'WY2025_RG_daily_depth.csv')
# df_count.to_csv('/home/adalvis/github/roads_model/'+\
# 	'WY2025_RG_daily_dt.csv')
# df_intensity.to_csv('/home/adalvis/github/roads_model/'+\
# 	'WY2025_RG_daily_intensity.csv')
# %%
