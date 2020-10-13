"""
# -*- coding: utf-8 -*-

Created on Fri Aug  7 14:34:13 2020

@author: Dirk


Compute and plot the leading EOF of geopotential height on the 500 hPa
pressure surface over the European/Atlantic sector.

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
# import seaborn as sns
from sklearn.cluster import KMeans
import xarray as xr 
import matplotlib as mpl



######################Dataset#################
data_folder = Path("../data/")
filename = data_folder / 'z_all_std_ano_30days_lowpass_2_0-25.nc'
f_out = data_folder / 'wr_time-c7_std_30days_lowpass_2_0-25.nc'
fig_out = data_folder / "fig/EOF7_30days_lowpass_2_0-1.png"
fig_out2 = data_folder / "fig/clusters-3PCs_30days_lowpass_2_0-1.png"
z_all_ano_std = xr.open_dataset(filename)['z']




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
    N = 7
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
    for i in range(N):
 
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
    plt.savefig(fig_out)


######################EOF analysis######################

# Create an EOF solver to do the EOF analysis. Square-root of cosine of
# latitude weights are applied before the computation of EOFs.
coslat = np.cos(np.deg2rad(z_all_ano_std.coords['latitude'].values)).clip(0., 1.)
wgts = np.sqrt(coslat)[..., np.newaxis]
solver = Eof(z_all_ano_std, weights=wgts)

plot(solver)



######################K_MEANS CLUSTERING#################
elbow(solver.pcs())

model = KMeans(n_clusters=7)
# Fit model to samples
model.fit(solver.pcs()[:,:16])

#Plot clusters on the first two PCA
# sns.scatterplot(solver.pcs()[:,0], solver.pcs()[:,1], alpha=.1, hue = model.labels_, palette="Paired")
# plt.xlabel('PCA 1')
# plt.ylabel('PCA 2')


#Plot k-mean 3D
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
xs = solver.pcs()[:,0]
ys = solver.pcs()[:,1]
zs = solver.pcs()[:,2]
ax.scatter(xs, ys, zs, alpha=0.1, c=model.labels_)
ax.set_xlabel('PC0')
ax.set_ylabel('PC1')
ax.set_zlabel('PC2')
fig.savefig(fig_out2)


#### Create Dataset weathter regime / time
wr_time = xr.DataArray(model.labels_, dims=("time"), coords={"time": z_all_ano_std.time}, name='wr')
wr_time.to_netcdf(f_out)
