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
    #eofs = solver.eofs(neofs=N)
    eofs = solver.eofsAsCovariance(neofs=N)
    pcs = solver.pcs(npcs=N)
    variance_fraction = solver.varianceFraction()

    cmap = mpl.cm.get_cmap("RdBu_r")
    plt.close("all")
    f, ax = plt.subplots(
        ncols=N,
        nrows=2,
        subplot_kw={"projection": ccrs.Orthographic(central_longitude=-20, central_latitude=60)},
        figsize=(12, 5),
    )
    cbar_ax = f.add_axes([0.3, 0.1, 0.4, 0.02])
    for i in range(5):
 
        if i != 0:
            #vmax = np.round(eofs.max())
            #vmin = np.round(eofs.min())
            title=str(np.round(variance_fraction.sel({"mode": i}).values * 100, 1)) + "% variance "
    
            ax[0, i].coastlines()
            ax[0, i].set_global()
            eofs[i].plot.contourf(ax=ax[0, i], cmap=cmap,
                                     transform=ccrs.PlateCarree(), add_colorbar=False)
            ax[0, i].set_title(title, fontsize=16)
        else:
            #vmax = np.round(eofs.max())
            #vmin = np.round(eofs.min())
            title=str(np.round(variance_fraction.sel({"mode": i}).values * 100, 1)) + "% variance "
    
            ax[0, i].coastlines()
            ax[0, i].set_global()
            eofs[i].plot.contourf(ax=ax[0, i], cmap=cmap,
                                     transform=ccrs.PlateCarree(), add_colorbar=True, cbar_kwargs={"orientation": "horizontal"}, cbar_ax=cbar_ax)
            ax[0, i].set_title(title, fontsize=16)

        ax[1, i] = plt.subplot(2, N, N + 1 + i)  # override the GeoAxes object
        pc = pcs.sel({"mode": i}, drop=True)
        pc = pd.Series(data=pc.values, index=pc.time.values)
        pc.plot(ax=ax[1, i])
        pc.rolling(window=5, center=True).mean().plot(
            ax=ax[1, i], ls="--", color="black", lw=2
        )
    plt.subplots_adjust(left=0.05, right=0.92, bottom=0.25)
    plt.suptitle("test")
    plt.savefig("../data/fig/test.png")
    #     base_path
    #     + plot_path
    #     + panel_name
    #     + "/solarpower_eofs_"
    #     + str(int(number))
    #     + ".png"
    # )
    # add mean timeseries
    # mpl.rcParams["axes.spines.left"] = True
    # mpl.rcParams["axes.spines.bottom"] = True
    # f, ax = plt.subplots()
    # all_power.mean(dim=["lat", "lon", "number"]).PV.plot(ax=ax)
    # ax.set_title(
    #     "PV Generation (mean over Europe, " + time_scale + " y, ensemble)"
    # )
    # plt.tight_layout()
    # plt.savefig(base_path + plot_path + panel_name + "/mean_timeseries.png")







######################Dataset#################
data_folder = Path("../data/")
filename = data_folder / 'z_all_std_ano.nc'
f_out = data_folder / 'wr_time-c7_std.nc'

z_all_ano_std = xr.open_dataset(filename)['z']


######################EOF analysis######################

# Create an EOF solver to do the EOF analysis. Square-root of cosine of
# latitude weights are applied before the computation of EOFs.
coslat = np.cos(np.deg2rad(z_all_ano_std.coords['latitude'].values)).clip(0., 1.)
wgts = np.sqrt(coslat)[..., np.newaxis]
solver = Eof(z_all_ano_std, weights=wgts)

# Retrieve the leading EOFs, expressed as the covariance between the leading PC
# time series and the input geopotential height anomalies at each grid point.
eofs = solver.eofsAsCovariance(neofs=5)

# Plot the leading EOF expressed as covariance in the European/Atlantic domain.
#clevs = np.linspace(-75, 75, 11)
# proj = ccrs.Orthographic(central_longitude=-20, central_latitude=60)
# ax = plt.axes(projection=proj)
# ax.coastlines()
# ax.set_global()
# eofs[0].plot.contourf(ax=ax, cmap=plt.cm.RdBu_r,
#                          transform=ccrs.PlateCarree(), add_colorbar=False,)
# ax.set_title('EOF0 expressed as covariance', fontsize=16)
# plt.show()


plot(solver)



######################K_MEANS CLUSTERING#################
elbow(solver.pcs())

#z_all_org = z_all + z_all.mean(dim='time')
model = KMeans(n_clusters=7)
# Fit model to samples
model.fit(solver.pcs()[:,:15])
#z_djf_pca_kmeans = np.concatenate([z_djf_org, pd.DataFrame(solver.pcs())], axis = 1)
#Plot clusters on the first two PCA
sns.scatterplot(solver.pcs()[:,1], solver.pcs()[:,2], alpha=.1, hue = model.labels_, palette = ['g', 'r', 'c', 'm', 'b', 'w', 'y'])
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')


#### Create Dataset weathter regime / time############

wr_time = xr.DataArray(model.labels_, dims=("time"), coords={"time": z_all_ano_std.time}, name='wr')
wr_time.to_netcdf(f_out)
#z_all_ano.expand_dims(dim='WR', axis=None)




#createdata(filename, f_out, solver, model)


