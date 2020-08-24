# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 10:23:32 2020

@author: Dirk
"""


import cartopy.crs as ccrs
# import time
# from netCDF4 import Dataset
import matplotlib.pyplot as plt
import numpy as np
# from datetime import datetime
from eofs.xarray import Eof
from pathlib import Path
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
import xarray as xr 
import matplotlib as mpl





######################Functions######################
######################elbow test for cluster size####
def elbow(pcs):
    ks = range(1, 10)
    inertias = []
    for k in ks:
        # Create a KMeans instance with k clusters: model
        model = KMeans(n_clusters=k)
    
        # Fit model to samples
        model.fit(pcs[:,:30])
    
        # Append the inertia to the list of inertias
        inertias.append(model.inertia_)
    
    plt.plot(ks, inertias, '-o', color='black')
    plt.xlabel('number of clusters, k')
    plt.ylabel('inertia')
    plt.xticks(ks)
    plt.show()



######################Plot like Jan##############
def plot(solver):
    N = 5
    eofs = solver.eofs(neofs=N)
    #eofs = solver.eofsAsCovariance(neofs=N)
    pcs = solver.pcs(npcs=N)
    variance_fraction = solver.varianceFraction()

    cmap = mpl.cm.get_cmap("RdBu_r")
    plt.close("all")
    f, ax = plt.subplots(
        ncols=N,
        nrows=2,
        subplot_kw={"projection": ccrs.Orthographic(central_longitude=-20, central_latitude=60)},
        figsize=(24, 10),
    )
    cbar_ax = f.add_axes([0.3, 0.1, 0.4, 0.02])
    for i in range(5):
 
        if i != 0:
            vmax = eofs.max()
            vmin = eofs.min()
            title=str(np.round(variance_fraction.sel({"mode": i}).values * 100, 1)) + "% variance "
    
            ax[0, i].coastlines()
            ax[0, i].set_global()
            eofs[i].plot.contourf(ax=ax[0, i], cmap=cmap, vmin=vmin, vmax=vmax,
                                     transform=ccrs.PlateCarree(), add_colorbar=False)
            ax[0, i].set_title(title, fontsize=16)
        else:
            vmax = eofs.max()
            vmin = eofs.min()
            title=str(np.round(variance_fraction.sel({"mode": i}).values * 100, 1)) + "% variance "
    
            ax[0, i].coastlines()
            ax[0, i].set_global()
            eofs[i].plot.contourf(ax=ax[0, i], vmin=vmin, vmax=vmax, cmap=cmap,
                                     transform=ccrs.PlateCarree(), add_colorbar=True, cbar_kwargs={"orientation": "horizontal"}, cbar_ax=cbar_ax)
            ax[0, i].set_title(title, fontsize=16)

        ax[1, i] = plt.subplot(2, N, N + 1 + i)  # override the GeoAxes object
        pc = pcs.sel({"mode": i}, drop=True)
        pc = pd.Series(data=pc.values[13514:13880], index=pc.time.values[13514:13880])
        pc.plot(ax=ax[1, i])
        pc.rolling(window=5, center=True).mean().plot(
            ax=ax[1, i], ls="--", color="black", lw=2
        )
    plt.subplots_adjust(left=0.05, right=0.92, bottom=0.25)
    plt.suptitle("EOF")
    plt.savefig("../data/fig/testdata-EOF.png")



######################Dataset#################
data_folder = Path("../data/")
filename = data_folder / 'z_all_std_ano.nc'
f_out = data_folder / 'wr_time-c7_std.nc'

z_all_ano_std = xr.open_dataset(filename)['z']




######################Create testdata######################
z_all_ano_std = xr.open_dataset(filename)['z'][13514:13880]
z_all_ano_std[0:40, 0:120, 0:240] = 1.1
z_all_ano_std[0:40, 120:, 0:240] = -1.1
z_all_ano_std[0:40, 0:120, 240:] = -0.1
z_all_ano_std[0:40, 120:, 240:] = 0.1

z_all_ano_std[200:240, 0:120, 0:240] = -0.6
z_all_ano_std[200:240, 120:, 0:240] = 0.4
z_all_ano_std[200:240, 0:120, 240:] = 0.6
z_all_ano_std[200:240, 120:, 240:] = -0.4


z_all_ano_std[100:160, 0:120, 0:240] = -0.75
z_all_ano_std[100:160, 120:, 0:240] = 0.25
z_all_ano_std[100:160, 0:120, 240:] = 0.75
z_all_ano_std[100:160, 120:, 240:] = -0.25

z_all_ano_std[270:330, 0:120, 0:240] = -1.5
z_all_ano_std[270:330, 120:, 0:240] = 0.1
z_all_ano_std[270:330, 0:120, 240:] = 1.5
z_all_ano_std[270:330, 120:, 240:] = 0.1

z_all_ano_std[340:342, 0:120, 0:240] = -0.3
z_all_ano_std[340:342, 120:, 0:240] = 0.6
z_all_ano_std[340:342, 0:120, 240:] = 0.3
z_all_ano_std[340:342, 120:, 240:] = -0.6



#cmap = mpl.cm.get_cmap("RdBu_r")
plt.close("all")
f1, ax1 = plt.subplots(
    ncols=2,
    nrows=1,
    subplot_kw={"projection": ccrs.Orthographic(central_longitude=-20, central_latitude=60)},
    figsize=(24, 10),
)

z_all_ano_std[0].plot.contourf(ax=ax1[0], transform=ccrs.PlateCarree(), add_colorbar=True)
z_all_ano_std[101].plot.contourf(ax=ax1[1], transform=ccrs.PlateCarree(), add_colorbar=False)
plt.show()


######################EOF analysis######################

# Create an EOF solver to do the EOF analysis. Square-root of cosine of
# latitude weights are applied before the computation of EOFs.
coslat = np.cos(np.deg2rad(z_all_ano_std.coords['latitude'].values)).clip(0., 1.)
wgts = np.sqrt(coslat)[..., np.newaxis]
solver = Eof(z_all_ano_std, weights=wgts)

# Retrieve the leading EOFs, expressed as the covariance between the leading PC
# time series and the input geopotential height anomalies at each grid point.
eofs = solver.eofsAsCovariance(neofs=5)



plot(solver)



######################K_MEANS CLUSTERING#################
elbow(solver.pcs())


model = KMeans(n_clusters=7)
# Fit model to samples
model.fit(solver.pcs()[:,:15])

#Plot clusters on the first two PCA
sns.scatterplot(solver.pcs()[:,1], solver.pcs()[:,2], alpha=.1, hue = model.labels_, palette="Paired")
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')


#### Create Dataset weathter regime / time############
wr_time = xr.DataArray(model.labels_, dims=("time"), coords={"time": z_all_ano_std.time}, name='wr')
#wr_time.to_netcdf(f_out)

#print wr of testdata
print(wr_time[0:40])
print(wr_time[200:240])
print(wr_time[100:160])
print(wr_time[270:330])
print(wr_time[340:342])


