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


filename_std_ano_ssrd = data_folder / 'radiation/ssrd_all_std_ano.nc'
ssrd_std_ano = xr.open_dataset(filename_std_ano_ssrd)['ssrd']



file_wr = data_folder / 'wr_time-c7_std_30days_lowpass_2_0-1_short3.nc'
wr = xr.open_dataset(file_wr)
wr = wr.sel(time=slice("1981-01-01", "2020-05-31"))

fig_out = data_folder / 'fig/ssrd_plot.png'

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




vmax_std_ano = 0.4
vmin_std_ano = -0.4



for i in range(0,wr.wr.max().values+1):
    mean_wr_std_ano = ssrd_std_ano[np.where(wr.wr==i)[0][:]].mean(axis=0)
    frequency = len(np.where(wr.wr == i)[0]) / len(wr.wr)
 
    if i != 0:

        #standard anomalie height plot
        if i==wr.wr.max().values:
            title= 'no regime ' +  str(np.round(frequency * 100, decimals=1)) + "%"
        else:
            title= 'WR' + str(i) + ' ' +  str(np.round(frequency * 100, decimals=1)) + "%"
        ax[i].coastlines()
        ax[i].set_global()
        mean_wr_std_ano.plot.imshow(ax=ax[i], cmap=cmap, transform=ccrs.PlateCarree(), add_colorbar=False, vmin=vmin_std_ano, vmax=vmax_std_ano)
        ax[i].set_title(title, fontsize=20, **csfont)
        
        
       
        
        
    else:
  
        #standard anomalie height plot
        # vmax_std_ano = mean_wr_std_ano.max()
        # vmin_std_ano = mean_wr_std_ano.min()
        title= 'WR' + str(i) + ' ' +  str(np.round(frequency * 100, decimals=1)) + "%"
        ax[i].coastlines()
        ax[i].set_global()
        con = mean_wr_std_ano.plot.imshow(ax=ax[i], cmap=cmap, transform=ccrs.PlateCarree(), add_colorbar=False, vmin=vmin_std_ano, vmax=vmax_std_ano) 
                                  
        cb = plt.colorbar(con, cax=cbar_ax, orientation='horizontal')
        cb.set_label(label='Standardized anomalies of surface solar radiation [unitless]',size=16,fontfamily='times new roman')
        ax[0].set_title(title, fontsize=20, **csfont)
        
        # cb.set_label(label='Temperature ($^{\circ}$C)', size='large', weight='bold')
        
        
extent = [-10, 35, 34, 72]
for i in ax:
    i.set_extent(extent)
    # i.gridlines()
    # i.coastlines(resolution='10m')               
  


plt.suptitle("Mean weather regime anomalies (surface solar radiation)", fontsize=20, **csfont)
plt.savefig(fig_out)


