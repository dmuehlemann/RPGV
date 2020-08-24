# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 14:50:44 2020

@author: Dirk
"""


from netCDF4 import Dataset
import numpy as np
from pathlib import Path
import cartopy.crs as ccrs
import matplotlib.pyplot as plt



def savefig(title,lons,lats,data):
    proj = ccrs.Orthographic(central_longitude=-20, central_latitude=60)
    ax = plt.axes(projection=proj)
    ax.set_global()
    ax.coastlines()
    ax.contourf(lons, lats, data[:,:], 
                cmap=plt.cm.RdBu_r, transform=ccrs.PlateCarree())
    plt.title(title, fontsize=16)
    plt.savefig('fig/%s .png' % title)
    plt.show()
    


######################Load Dataset#################
data_folder = Path("../data/")
# filename = data_folder / 'wr-gph-djf-daily-mean.nc'
filename = data_folder / 'wr_time-c7_std.nc'
ncin = Dataset(filename, 'r')
z_djf = ncin.variables['z'][:]
lons = ncin.variables['longitude'][:]
lats = ncin.variables['latitude'][:]
wr = ncin.variables['WR'][:]
ncin.close()



######################Save figure#################

# for i in np.where(wr == 0)[0][:]:
#     title = 'Weather regime 0 index - %i' % i 
#     savefig(title,lons,lats,z_djf[i,:,:])
    
    
######################Calculate mean per cluster#################

z_djf_mean = z_djf.mean(axis=0)
z_djf = z_djf - z_djf_mean



#########Mean fields of different weather regimes###########

mean_wr = z_djf[np.where(wr == 0)[0][:],:,:].mean(axis=(0))
mean_wr = mean_wr[np.newaxis, :]
title = 'Mean WR0'
savefig(title,lons, lats,mean_wr[0,:,:])
print("percentage frequency WR0: " + str(len(np.where(wr == 1)[0][:]) / len(wr)))

for i in range(1,max(wr)+1):
    temp = z_djf[np.where(wr == i)[0][:],:,:].mean(axis=(0))
    temp = temp[np.newaxis, :]
    mean_wr = np.append(mean_wr, temp, axis=0)
    title = 'Mean WR'+str(i)
    savefig(title,lons, lats,mean_wr[i,:,:])
    print("percentage frequency WR" + str(i) + ": " + str(len(np.where(wr == i)[0][:]) / len(wr)))


