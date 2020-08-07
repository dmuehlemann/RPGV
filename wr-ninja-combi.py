# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 14:50:44 2020

@author: Dirk
"""


from netCDF4 import Dataset, num2date
import numpy as np
from pathlib import Path
#import cartopy.crs as ccrs
#import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime, timedelta




######################Load Datasets#################
#Load weather regime dataset
data_folder = Path("../data/")
# filename = data_folder / 'wr-gph-djf-daily-mean.nc'
filename = data_folder / 'wr-gph-djf-daily-mean-c4-xarray.nc'
f_out = data_folder / 'results/mean_wr_country_djf_c4.csv'



ncin = Dataset(filename, 'r')
z_djf = ncin.variables['z'][:]
lons = ncin.variables['longitude'][:]
lats = ncin.variables['latitude'][:]
wr = ncin.variables['WR'][:]
time = ncin.variables['time'][:]
dates = num2date(time, ncin.variables['time'].units)
ncin.close()

#Load renewable ninja dataset

filename = data_folder / 'ninja/ninja_europe_pv_v1.1/ninja_pv_europe_v1.1_merra2.csv'
ninja = pd.read_csv(filename)






######################Calculate ninja CF mean for every weather regime#######

#Format time of wr data
day = [int(d.strftime('%Y%m%d')) for d in dates]
day = list(dict.fromkeys(day))
#d = [datetime.strptime(str(d), '%Y%m%d') for d in day]


#Format time of ninja data
ninja_day = [i.replace('-','') for i in ninja.time[:]]
for i in range(0, len(ninja_day)): 
    ninja_day[i] = int(ninja_day[i][:8])
    


#get ninja time index for every weather regime
wr_day = [[day[x] for x in np.where(wr == 0)[0][:]]]
temp = []

for a in wr_day[0]:
   temp.extend([i for i, e in enumerate(ninja_day) if e == a])
   
ninja_wr_index =[temp]

for b in range(1,max(wr)+1):
    wr_day.append([day[x] for x in np.where(wr == b)[0][:]])
    temp = []
    for a in wr_day[b]:
        temp.extend([i for i, e in enumerate(ninja_day) if e == a])
    ninja_wr_index.append(temp)

#all ninja days found in all weather regimes
flattened_nina_wr_index = [y for x in ninja_wr_index for y in x]



#Calculate mean per weather regime and country
mean_wr_country = pd.DataFrame()

for a in range(0, max(wr)+1):
    temp2 = pd.DataFrame()
    for i in ninja.drop(['time'], axis=1):
        temp = pd.DataFrame([ninja[i][ninja_wr_index[a]].mean(axis=0)], index=[i], columns=['WR'+str(a)])
        temp2 = temp2.append(temp)
    mean_wr_country = pd.concat([mean_wr_country, temp2], axis=1)
 
    
#calculate mean per country
mean_country = pd.DataFrame()
for i in ninja.drop(['time'], axis=1):
    mean_country_temp = pd.DataFrame([ninja[i][flattened_nina_wr_index].mean(axis=0)], index=[i], columns=['all WR'])
    mean_country = mean_country.append((mean_country_temp))                                                                                        


#calculate anomaly per weather region relative to mean per country
relative_mean_wr_country = pd.DataFrame()

# for a in NOT RANGE range(0, max(wr)+1):
#     relative_mean_wr_country_temp2 = pd.DataFrame()
#     for i in ninja.drop(['time'], axis=1):
#         relative_mean_wr_country_temp = pd.DataFrame(mean_wr_country[a].loc[i])
#         relative_mean_wr_country_temp2 = relative_mean_wr_country_temp2.append(relative_mean_wr_country_temp)
#     relative_mean_wr_country = pd.concat([relative_mean_wr_country, relative_mean_wr_country_temp2], axis=1)


    
#save results as csv
#♦mean_wr_country.to_csv(f_out)


#########MAP data#############
import geopandas as gpd
shapefile = data_folder / 'map/NUTS_RG_01M_2021_4326.shp/NUTS_RG_01M_2021_4326.shp'
#Read shapefile using Geopandas
eu = gpd.read_file(shapefile)[['NAME_LATN', 'CNTR_CODE', 'geometry']]
#Rename columns.
eu.columns = ['country', 'country_code', 'geometry']
eu.head()


for_plotting = eu.merge(mean_wr_country, left_on = 'country_code', right_index=True)
for_plotting.info()

ax = for_plotting.dropna().plot(column='WR0', cmap =    
                                'YlGnBu', figsize=(15,9),   
                                 scheme='quantiles', k=8, legend =  
                                  True,);
#add title to the map
ax.set_title('Capacity factor per country for weather regime 1 in winter', fontdict= 
            {'fontsize':25})
#remove axes
ax.set_axis_off()
#move legend to an empty space
ax.get_legend().set_bbox_to_anchor((.12,.12))
ax.set_xlim(left=-20, right=40)
ax.set_ylim(bottom=30, top=80)
ax.get_figure()




