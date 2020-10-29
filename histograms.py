# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 12:00:19 2020

@author: Dirk
"""



import numpy as np
from pathlib import Path
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import xarray as xr
# import calendar
# import pandas as pd
import matplotlib as mpl
import geopandas as gpd
# from mapclassify import Quantiles, UserDefined


######################Load Datasets#################
data_folder = Path("../data/")

filename_std_ano = data_folder / 'z_all_std_ano_30days_lowpass_2_0-1.nc'
z_all_std_ano = xr.open_dataset(filename_std_ano)['z']

file_wr = data_folder / 'wr_time-c7_std_30days_lowpass_2_0-1_short10.nc'
wr = xr.open_dataset(file_wr)

file_ninja = data_folder / 'ninja_and_wr_30days_lowpass_2_0-1_short10.nc'
ninja = xr.open_dataset(file_ninja)

###here no Brexit ;-)
ninja = ninja.rename_vars({'GB':'UK'})



#####CF calculations##########
season = {0: 'DJF', 1: 'MAM', 2: 'JJA', 3: 'SON'}





ninja_wr = []
for i in range(0, int(wr.wr.max())+1):
    ninja_wr.append((ninja.drop('wr').where(ninja.wr==i, drop=True)).resample(time='1D').mean().dropna(dim='time'))
    # ninja_wr[i] = ninja_wr[i].expand_dims('CF')

mean_country = ninja.drop('wr').mean()

mean_country_season= ninja.drop('wr').groupby('time.season').mean()

# for i in range(0, len(ninja_wr)): 
#     ninja_wr[i] = ninja_wr[i].resample(time='1D').mean()

##Histogram


#Loop torugh all countries
for c in ninja_wr[0]:
    
    #Rows and colums
    r = 5
    col = wr.wr.max().values+1
    
    #Create subplots and colorbar for wr
    plt.close("all")
    f, ax = plt.subplots(
        ncols=col,
        nrows=r,
        figsize=(30, 20),
    )
    
    #Loop torugh all wr
    for i in range(0, len(ninja_wr)):
        # print(i)
        #Loop torugh all seasons
        for s in season:
            #All seasons
            data = ninja_wr[i][c].where(ninja_wr[i]['time.season']==season[s]).dropna(dim='time')
            xr.plot.hist(data, \
                ax=ax[s,i], range=[0, 0.27], ylim=(None,160))
            ax[s,i].axvline(data.mean(), color='k', linestyle='dashed', linewidth=1)
            ax[s,i].axvline(mean_country_season.sel(season=season[s])[c], color='r', linestyle='dashed', linewidth=1)
            ax[s,i].set_title(str(season[s]+' WR'+str(i)))
            ax[s,i].set_xlabel(None)
        
       
        #Tot
        data = ninja_wr[i][c]
        xr.plot.hist(data, \
            ax=ax[4,i], range=[0, 0.27], ylim=(None,250))
        ax[4,i].axvline(data.mean(), color='k', linestyle='dashed', linewidth=1)
        ax[4,i].axvline(mean_country[c], color='r', linestyle='dashed', linewidth=1)
        ax[4,i].set_title('TOT WR'+str(i))
        ax[4,i].set_xlabel('CF')

    
    f.suptitle('Histogram of capacity factors in '+str(c), fontsize=26)
    f.savefig('../data/fig/histogram/'+str(c)+'-histogram.png')
            





