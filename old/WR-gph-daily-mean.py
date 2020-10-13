# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 14:50:44 2020

@author: Dirk
"""


# from netCDF4 import Dataset
import numpy as np
from pathlib import Path
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import xarray as xr
import calendar
import pandas as pd
import matplotlib as mpl



# def savefig(title,lons,lats,data):
#     proj = ccrs.Orthographic(central_longitude=-20, central_latitude=60)
#     ax = plt.axes(projection=proj)
#     ax.set_global()
#     ax.coastlines()
#     ax.contourf(lons, lats, data[:,:], 
#                 cmap=plt.cm.RdBu_r, transform=ccrs.PlateCarree())
#     plt.title(title, fontsize=16)
    
#     plt.savefig('../data/fig/%s.png' % title)
#     plt.show()

######################Load Datasets#################

data_folder = Path("../data/")
filename_std_ano = data_folder / 'z_all_std_ano.nc'
filename = data_folder / 'gph-daily-mean.nc'


z_all_std_ano = xr.open_dataset(filename_std_ano)['z']
z_all = xr.open_dataset(filename)['z']

file_wr = data_folder / 'wr_time-c7_std.nc'
wr = xr.open_dataset(file_wr)


#########Mean 500 hPa geopotential fields for every different weather regimes###########
####Save every single figure --> OLD!!!#####################
# lats = z_all.latitude
# lons = z_all.longitude
# mean_wr = np.array(z_all[np.where(wr.wr==0)[0][:]].mean(axis=0))
# mean_wr = mean_wr[np.newaxis, :]
# title = 'Mean WR0 500 hPa geopotential'
# savefig(title,lons, lats,mean_wr[0,:,:])
# print("percentage frequency WR0: " + str(len(np.where(wr.wr == 0)[0][:]) / len(wr.wr)))

# for i in range(1,wr.wr.max().values+1):
#     temp = np.array(z_all[np.where(wr.wr==i)[0][:]].mean(axis=0))
#     temp = temp[np.newaxis, :]
#     mean_wr = np.append(mean_wr, temp, axis=0)
#     title = 'Mean WR'+str(i) + ' 500 hPa geopotential'
#     savefig(title,lons, lats,mean_wr[i,:,:])
#     print("percentage frequency WR" + str(i) + ": " + str(len(np.where(wr.wr == i)[0]) / len(wr.wr)))




#########Mean 500 hPa geopotential and standard anomaly fields# ####
#########for every different weather regimes in one plot###########

#Rows and colums
r = 2
c = wr.wr.max().values+1


cmap = mpl.cm.get_cmap("RdBu_r")
plt.close("all")
f, ax = plt.subplots(
    ncols=c,
    nrows=r,
    subplot_kw={"projection": ccrs.Orthographic(central_longitude=-20, central_latitude=60)},
    figsize=(30, 10),
)
cbar_ax = f.add_axes([0.3, 0.1, 0.4, 0.02])
for i in range(0,wr.wr.max().values+1):
    mean_wr_gph = z_all[np.where(wr.wr==i)[0][:]].mean(axis=0)
    mean_wr_std_ano = z_all_std_ano[np.where(wr.wr==i)[0][:]].mean(axis=0)
    frequency = len(np.where(wr.wr == i)[0]) / len(wr.wr)
 
    if i != 0:
        #Geopotential height plot
        # vmax_gph = mean_wr_gph.max()
        # vmin_gph = mean_wr_gph.min()
        # title= 'gph WR' + str(i) + ' ' +  str(np.round(frequency * 100)) + "%"
        # ax[0, i].coastlines()
        # ax[0, i].set_global()
        # mean_wr_gph.plot.contourf(ax=ax[0, i], cmap=cmap, vmin=vmin_gph, vmax=vmax_gph,
        #                           transform=ccrs.PlateCarree(), add_colorbar=False)
        # ax[0, i].set_title(title, fontsize=16)
        
        

        #standard anomalie height plot
        vmax_std_ano = mean_wr_std_ano.max()
        vmin_std_ano = mean_wr_std_ano.min()
        title= 'std ano WR' + str(i) + ' ' +  str(np.round(frequency * 100)) + "%"
        ax[0, i].coastlines()
        ax[0, i].set_global()
        fig = mean_wr_std_ano.plot.contourf(ax=ax[0, i], cmap=cmap, vmin=vmin_std_ano, vmax=vmax_std_ano,
                                  transform=ccrs.PlateCarree(), add_colorbar=False)
        ax[0, i].set_title(title, fontsize=16)
        
        
    else:
                 
        #Geopotential height plot
        # vmax_gph = mean_wr_gph.max()
        # vmin_gph = mean_wr_gph.min()
        # title= 'gph WR' + str(i) + ' ' +  str(np.round(frequency * 100)) + "%"
        # ax[0, i].coastlines()
        # ax[0, i].set_global()
        # mean_wr_gph.plot.contourf(ax=ax[0,i], vmin=vmin_gph, vmax=vmax_gph, cmap=cmap,
        #                           transform=ccrs.PlateCarree(), add_colorbar=True, cbar_kwargs={"orientation": "horizontal"}, cbar_ax=cbar_ax)
        # ax[0, i].set_title(title, fontsize=16)
        
                
        
        #standard anomalie height plot
        vmax_std_ano = mean_wr_std_ano.max()
        vmin_std_ano = mean_wr_std_ano.min()
        title= 'std ano WR' + str(i) + ' ' +  str(np.round(frequency * 100)) + "%"
        ax[0, i].coastlines()
        ax[0, i].set_global()
        fig = mean_wr_std_ano.plot.contourf(ax=ax[0, i], vmin=vmin_std_ano, vmax=vmax_std_ano, cmap=cmap,
                                  transform=ccrs.PlateCarree(), add_colorbar=True, cbar_kwargs={"orientation": "horizontal"}, cbar_ax=cbar_ax)
        ax[0, i].set_title(title, fontsize=16) 
               
        
        
        
        
    monthly_frequency = wr.where(wr==i).dropna(dim='time').groupby('time.month').count()
    ax[1, i] = plt.subplot(2, c, c + 1 + i)  # override the GeoAxes object
    monthly_frequency = pd.Series(data=monthly_frequency.wr, index=calendar.month_abbr[1:13])
    
    monthly_frequency.plot.bar(ax=ax[1,i])
    # pc.rolling(window=5, center=True).mean().plot(
    #     ax=ax[1, i], ls="--", color="black", lw=2
    # )
    


plt.subplots_adjust(left=0.05, right=0.92, bottom=0.25)
# plt.suptitle("Mean weather regime fields (geopotential height)", fontsize=20)
# plt.savefig("../data/fig/WR_mean_gph.png")
plt.suptitle("Mean weather regime fields (standardized anomalies)", fontsize=20)
plt.savefig("../data/fig/WR_mean_std_ano.png")

