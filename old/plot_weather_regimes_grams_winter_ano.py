# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 11:33:56 2020

@author: Dirk
"""


import numpy as np
from pathlib import Path
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import xarray as xr
# import calendar
import pandas as pd
import matplotlib as mpl
import geopandas as gpd
# from mapclassify import Quantiles, UserDefined


######################Load Datasets#################

data_folder = Path("../data/")

filename_std_ano = data_folder / 'z_all_std_ano_grams_lowpass_2_0-1.nc'
filename = data_folder / 'gph-daily-mean-lowpass_2_0-1.nc'
z_all_std_ano = xr.open_dataset(filename_std_ano)['z']
z_all = xr.open_dataset(filename)['z']

file_wr = data_folder / 'wr_time-c7_std_grams_lowpass_2_0-1.nc'
wr = xr.open_dataset(file_wr)




######################Calculate Anomalies#################
days_clima=90
days_std=30
climatology_mean = z_all.rolling(time=days_clima, center=True).mean().ffill(dim='time').bfill(dim='time').groupby("time.dayofyear").mean("time")

stand_anomalies = xr.apply_ufunc(
    lambda x, m: (x - m),
    z_all.groupby("time.dayofyear"),
    climatology_mean,
)

z_all_std_ano = stand_anomalies
######################Plot results#################

#Rows and colums
r = 1
c = wr.wr.max().values+1

#Create subplots and colorbar for wr
cmap = mpl.cm.get_cmap("RdBu_r")
plt.close("all")
f, ax = plt.subplots(
    ncols=c,
    nrows=r,
    subplot_kw={"projection": ccrs.Orthographic(central_longitude=-20, central_latitude=60)},
    figsize=(20, 6),
)
cbar_ax = f.add_axes([0.3, .85, 0.4, 0.02])


vmax_std_ano = 1000
vmin_std_ano = -1000

for i in range(0,wr.wr.max().values+1):
    mean_wr_std_ano_all = z_all_std_ano[np.where(wr.wr==i)[0][:]]
    mean_wr_std_ano = mean_wr_std_ano_all.groupby('time.season').mean().sel(season='DJF')
    frequency = len(np.where(wr.wr == i)[0]) / len(wr.wr)
 
    if i != 0:

        #standard anomalie height plot

        title= 'WR' + str(i) + ' ' +  str(np.round(frequency * 100, decimals=1)) + "%"
        ax[i].coastlines()
        ax[i].set_global()
        mean_wr_std_ano.plot.contourf(ax=ax[i], cmap=cmap, vmin=vmin_std_ano, vmax=vmax_std_ano,
                                  transform=ccrs.PlateCarree(), add_colorbar=False)
        ax[i].set_title(title, fontsize=16)
        
        
       
        
        
    else:
  
        #standard anomalie height plot
        # vmax_std_ano = mean_wr_std_ano.max()
        # vmin_std_ano = mean_wr_std_ano.min()
        title= 'WR' + str(i) + ' ' +  str(np.round(frequency * 100, decimals=1)) + "%"
        ax[i].coastlines()
        ax[i].set_global()
        mean_wr_std_ano.plot.contourf(ax=ax[i], vmin=vmin_std_ano, vmax=vmax_std_ano, cmap=cmap,
                                  transform=ccrs.PlateCarree(), add_colorbar=True, 
                                  cbar_kwargs={'label': "Standardized of geoptential height at 500hPa","orientation": "horizontal"}, 
                                  cbar_ax=cbar_ax)
        ax[0].set_title(title, fontsize=16) 
        
               
        


plt.suptitle("Mean weather regime fields winter (anomalies 90 days / 30 days grams lowpassfilter)", fontsize=20)
plt.savefig("../data/fig/wr_plot_grams_winter_lowpass_2_0-1.png")


