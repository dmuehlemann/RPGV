# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 16:44:38 2020

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
# import geopandas as gpd
# from mapclassify import Quantiles, UserDefined


######################Load Datasets#################

data_folder = Path("../data/")

#######Data1 

filename_std_ano = data_folder / 'z_all_std_ano_14days_lowpass_2_0-25.nc'
z_all_std_ano_1 = xr.open_dataset(filename_std_ano)['z']

file_wr = data_folder / 'wr_time-c7_std_14days_lowpass_2_0-25.nc'
wr_1 = xr.open_dataset(file_wr)

#######Data2
filename_std_ano = data_folder / 'z_all_std_ano_grams_lowpass_2_0-25.nc'
z_all_std_ano_2 = xr.open_dataset(filename_std_ano)['z']

file_wr = data_folder / 'wr_time-c7_std_grams_lowpass_2_0-25.nc'
wr_2 = xr.open_dataset(file_wr)


#########Correlate###########

corr = []
for a in range(0,wr_1.wr.max().values+1):
    mean_wr_std_ano = z_all_std_ano_1[np.where(wr_1.wr==a)[0][:]].mean(axis=0)
    frequency = len(np.where(wr_1.wr == a)[0]) / len(wr_1.wr)
    corr2 = []
    for i in range(0,wr_2.wr.max().values+1):
        mean_wr_std_ano_filter = z_all_std_ano_2[np.where(wr_2.wr==i)[0][:]].mean(axis=0)
        corr2.append(xr.corr(mean_wr_std_ano, mean_wr_std_ano_filter))
    corr.append(corr2)
        
for i in range(0, len(corr)):
    corr[i] = xr.concat(corr[i], dim='z')


 
        
######################Plot results#################

#Rows and colums
r = 2
c = wr_1.wr.max().values+1

#Create subplots and colorbar for wr
cmap = mpl.cm.get_cmap("RdBu_r")
plt.close("all")
f, ax = plt.subplots(
    ncols=c,
    nrows=r,
    subplot_kw={"projection": ccrs.Orthographic(central_longitude=-20, central_latitude=60)},
    figsize=(20, 8),
)
cbar_ax = f.add_axes([0.3, 0.1, 0.4, 0.02])


vmax_std_ano = 2
vmin_std_ano = -2

for i in range(0,wr_1.wr.max().values+1):
    mean_wr_std_ano = z_all_std_ano_1[np.where(wr_1.wr==i)[0][:]].mean(axis=0)
    mean_wr_std_ano_filter = z_all_std_ano_2[np.where(wr_2.wr==int(corr[i].argmax().values))[0][:]].mean(axis=0)
    frequency = len(np.where(wr_1.wr == i)[0]) / len(wr_1.wr)
    frequency_filter = len(np.where(wr_2.wr == int(corr[i].argmax().values))[0]) / len(wr_2.wr)
 
    if i != 0:

        #standard anomalie height plot

        title= 'WR' + str(i) + ' ' +  str(np.round(frequency * 100, decimals=1)) + "%"
        ax[0,i].coastlines()
        ax[0,i].set_global()
        mean_wr_std_ano.plot.contourf(ax=ax[0,i], cmap=cmap, vmin=vmin_std_ano, vmax=vmax_std_ano,
                                  transform=ccrs.PlateCarree(), add_colorbar=False)
        ax[0,i].set_title(title, fontsize=16)
        
        
        title= 'WR' + str(corr[i].argmax().values) + ' ' +  str(np.round(frequency_filter * 100, decimals=1)) + "% \n corr: " + str(round(corr[i].values.max(),4))
        ax[1,i].coastlines()
        ax[1,i].set_global()
        mean_wr_std_ano_filter.plot.contourf(ax=ax[1,i], cmap=cmap, vmin=vmin_std_ano, vmax=vmax_std_ano,
                                  transform=ccrs.PlateCarree(), add_colorbar=False)
        ax[1,i].set_title(title, fontsize=16)       
        
        
    else:
  
        #standard anomalie height plot
        # vmax_std_ano = mean_wr_std_ano.max()
        # vmin_std_ano = mean_wr_std_ano.min()
        title= 'WR' + str(i) + ' ' +  str(np.round(frequency * 100, decimals=1)) + "%"
        ax[0,i].coastlines()
        ax[0,i].set_global()
        mean_wr_std_ano.plot.contourf(ax=ax[0,i], vmin=vmin_std_ano, vmax=vmax_std_ano, cmap=cmap,
                                  transform=ccrs.PlateCarree(), add_colorbar=True, 
                                  cbar_kwargs={'label': "Standardized anomalies of geoptential height at 500hPa","orientation": "horizontal"}, 
                                  cbar_ax=cbar_ax)
        ax[0,i].set_title(title, fontsize=16)
        
        
        
        title= 'WR' + str(corr[i].argmax().values) + ' ' +  str(np.round(frequency_filter * 100, decimals=1)) + "% \n corr: " + str(round(corr[i].values.max(),4))
        ax[1,i].coastlines()
        ax[1,i].set_global()
        mean_wr_std_ano_filter.plot.contourf(ax=ax[1,i], vmin=vmin_std_ano, vmax=vmax_std_ano, cmap=cmap,
                                  transform=ccrs.PlateCarree(), add_colorbar=False)
        ax[1,i].set_title(title, fontsize=16) 
        
          


plt.suptitle("Mean weather regime fields (standardized anomalies 14days lowpass filter (2/0.25) vs. 90days (grams))", fontsize=20)
plt.savefig("../data/fig/wr_plot_14days_lowpass_2_0-25_corr_grams.png")


