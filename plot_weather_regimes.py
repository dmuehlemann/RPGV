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
import matplotlib as mpl


######################Load Datasets#################

data_folder = Path("../data/")

filename_std_ano = data_folder / 'z_all_std_ano_30days_lowpass_2_0-1.nc'
z_all_std_ano = xr.open_dataset(filename_std_ano)['z']

file_wr = data_folder / 'wr_time-c7_std_30days_lowpass_2_0-1_short3.nc'
wr = xr.open_dataset(file_wr)

fig_out = data_folder / 'fig/wr_plot_30days_lowpass_2_0-1_short3.png'

######################Plot results#################

#Rows and colums
r = 1
c = wr.wr.max().values+1

#Create subplots and colorbar for wr
cmap = mpl.cm.get_cmap("RdBu_r")
csfont = {'fontname':'Times New Roman'}
plt.close("all")
f, ax = plt.subplots(
    ncols=c,
    nrows=r,
    subplot_kw={"projection": ccrs.Orthographic(central_longitude=-20, central_latitude=60)},
    figsize=(20, 4),
)
cbar_ax = f.add_axes([0.3, .2, 0.4, 0.02])


vmax_std_ano = 2.1
vmin_std_ano = -2.1

for i in range(0,wr.wr.max().values+1):
    mean_wr_std_ano = z_all_std_ano[np.where(wr.wr==i)[0][:]].mean(axis=0)
    frequency = len(np.where(wr.wr == i)[0]) / len(wr.wr)
 
    if i != 0:

        #standard anomalie height plot
        if i==wr.wr.max().values:
            title= 'no regime ' +  str(np.round(frequency * 100, decimals=1)) + "%"
        else:
            title= 'WR' + str(i) + ' ' +  str(np.round(frequency * 100, decimals=1)) + "%"
        ax[i].coastlines()
        ax[i].set_global()
        mean_wr_std_ano.plot.imshow(ax=ax[i], cmap=cmap, 
                                  transform=ccrs.PlateCarree(), add_colorbar=False)
        ax[i].set_title(title, fontsize=20, **csfont)
        
        
       
        
        
    else:
  
        #standard anomalie height plot
        # vmax_std_ano = mean_wr_std_ano.max()
        # vmin_std_ano = mean_wr_std_ano.min()
        title= 'WR' + str(i) + ' ' +  str(np.round(frequency * 100, decimals=1)) + "%"
        ax[i].coastlines()
        ax[i].set_global()
        con =  mean_wr_std_ano.plot.imshow(ax=ax[i], cmap=cmap,
                                  transform=ccrs.PlateCarree(), add_colorbar=False) 
                                  # cbar_kwargs={'label': "Standardized anomalies of geoptential height at 500hPa","orientation": "horizontal"}, 
                                  # cbar_ax=cbar_ax)
        cb = plt.colorbar(con, cax=cbar_ax, orientation='horizontal')
        cb.set_label(label='Standardized anomalies of geopotential height at 500 hPa [unitless]',size=16,fontfamily='times new roman')
        ax[0].set_title(title, fontsize=20, **csfont)
        
               
        


plt.suptitle("Mean weather regime anomalies (geopotential height)", fontsize=20, **csfont)
plt.savefig(fig_out)


