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

filename = data_folder / 'gph-daily-mean.nc'
z_all = xr.open_dataset(filename)

fig_out = data_folder / 'fig/gph_plot.tiff'



######################Plot results#################

#Rows and colums
r = 1
c = wr.wr.max().values+1

#Create subplots and colorbar for wr
cmap = mpl.cm.get_cmap("RdYlBu_r")
plt.close("all")
f, ax = plt.subplots(
    ncols=c,
    nrows=r,
    subplot_kw={"projection": ccrs.Orthographic(central_longitude=-20, central_latitude=60)},
    figsize=(15, 3),
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
            title= 'WR' + str(i+1) + ' ' +  str(np.round(frequency * 100, decimals=1)) + "%"
        ax[i].coastlines()
        ax[i].set_global()
        # mean_wr_std_ano.plot.contourf(ax=ax[i], cmap=cmap, vmin=vmin_std_ano, vmax=vmax_std_ano,
        #                           transform=ccrs.PlateCarree(), add_colorbar=False)
        
        mean_wr_std_ano.plot.imshow(ax=ax[i], cmap=cmap, transform=ccrs.PlateCarree(), add_colorbar=False)
        ax[i].set_title(title, fontsize=12)
        
        
       
        
        
    else:
  
        #standard anomalie height plot
        # vmax_std_ano = mean_wr_std_ano.max()
        # vmin_std_ano = mean_wr_std_ano.min()
        title= 'WR' + str(i+1) + ' ' +  str(np.round(frequency * 100, decimals=1)) + "%"
        ax[i].coastlines()
        ax[i].set_global()
        # mean_wr_std_ano.plot.contourf(ax=ax[i], vmin=vmin_std_ano, vmax=vmax_std_ano, cmap=cmap,
        #                           transform=ccrs.PlateCarree(), add_colorbar=True, 
        #                           cbar_kwargs={'label': "Standardized anomalies of geoptential height at 500hPa","orientation": "horizontal"}, 
        #                           cbar_ax=cbar_ax)
        con = mean_wr_std_ano.plot.imshow(ax=ax[i],  cmap=cmap, transform=ccrs.PlateCarree(), add_colorbar=False) 
        cb = plt.colorbar(con, cax=cbar_ax, orientation='horizontal',)
        cb.ax.tick_params(labelsize=12)
        cb.set_label(label='Standardized anomalies of geoptential height at 500hPa [unitless]',size=12,)
        
        
        ax[0].set_title(title, fontsize=12) 
        
               
        


# plt.suptitle("Mean weather regime fields (geopotential height)", fontsize=20)
plt.savefig(fig_out, dpi=300)


