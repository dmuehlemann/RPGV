"""
# -*- coding: utf-8 -*-

Created on Fri Aug  7 14:34:13 2020

@author: Dirk


Compute and plot the leading EOF of geopotential height on the 500 hPa
pressure surface over the European/Atlantic sector.

"""
import cartopy.crs as ccrs
import time
from netCDF4 import Dataset
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
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



###################### create Dataset ##############
def createdata(f_in, f_out, solver, model):
   with Dataset(f_in) as ds_src:
       var_gph = ds_src.variables['z']
       var_time = ds_src.variables['time']
       with Dataset(f_out, mode = 'w', format = 'NETCDF3_64BIT_OFFSET') as ds_dest:
            # Dimensions
            for name in ['latitude', 'longitude']:
                dim_src = ds_src.dimensions[name]
                ds_dest.createDimension(name, dim_src.size)
                var_src = ds_src.variables[name]
                var_dest = ds_dest.createVariable(name, var_src.datatype, (name,))
                var_dest[:] = var_src[:]
                var_dest.setncattr('units', var_src.units)
                var_dest.setncattr('long_name', var_src.long_name)
     
            ds_dest.createDimension('time', None)
            var = ds_dest.createVariable('time', np.int32, ('time',))
            time_units = 'hours since 1900-01-01 00:00:00'
            time_cal = 'gregorian'
            var[:] = var_time[:]
            var.setncattr('units', time_units)
            var.setncattr('long_name', 'time')
            var.setncattr('calendar', time_cal)
     
            # Variables
            var = ds_dest.createVariable(var_gph.name, np.double, var_gph.dimensions)
            var[:, :, :] = ds_src.variables['z'][:,:,:]
            var.setncattr('units', 'm**2 s**-2')
            var.setncattr('long_name', "Geopotential")

     
            # Attributes
            ds_dest.setncattr('Conventions', 'CF-1.6')
            ds_dest.setncattr('history', '%s %s'
                    % (datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    ' '.join(time.tzname)))
            
            
            #Weahter Regimes
            dim_src = len(model.labels_)
            ds_dest.createDimension('WR', dim_src)
            var_src = model.labels_
            var_dest = ds_dest.createVariable('WR', np.int32, ('WR',))
            var_dest[:] = var_src[:]
            var_dest.setncattr('units', 'number of weather Regime')
            var_dest.setncattr('long_name', 'Weather regime created with EOF and k-mean clustering')
                
                
                
                
                
                
            print('Done! Data saved in %s' % f_out)


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
filename = data_folder / 'test-data3.nc'
#f_out = data_folder / 'wr_z_all_ano_c4.nc'

z_all_ano = xr.open_dataset(filename)['z']


######################EOF analysis######################

# Create an EOF solver to do the EOF analysis. Square-root of cosine of
# latitude weights are applied before the computation of EOFs.
coslat = np.cos(np.deg2rad(z_all_ano.coords['latitude'].values)).clip(0., 1.)
wgts = np.sqrt(coslat)[..., np.newaxis]
solver = Eof(z_all_ano, weights=wgts)

# Retrieve the leading EOFs, expressed as the covariance between the leading PC
# time series and the input geopotential height anomalies at each grid point.
eofs = solver.eofsAsCovariance(neofs=5)

# Plot the leading EOF expressed as covariance in the European/Atlantic domain.
#clevs = np.linspace(-75, 75, 11)
proj = ccrs.Orthographic(central_longitude=-20, central_latitude=60)
ax = plt.axes(projection=proj)
ax.coastlines()
ax.set_global()
eofs[1].plot.contourf(ax=ax, cmap=plt.cm.RdBu_r,
                         transform=ccrs.PlateCarree(), add_colorbar=False,)
ax.set_title('EOF0 expressed as covariance', fontsize=16)
plt.show()




######################K_MEANS CLUSTERING#################
elbow(solver.pcs())

#z_all_org = z_all + z_all.mean(dim='time')
model = KMeans(n_clusters=3)
# Fit model to samples
model.fit(solver.pcs()[:,:15])
#z_djf_pca_kmeans = np.concatenate([z_djf_org, pd.DataFrame(solver.pcs())], axis = 1)
#Plot clusters on the first two PCA
sns.scatterplot(solver.pcs()[:,0], solver.pcs()[:,1], alpha=.1, hue = model.labels_, palette = ['g', 'r', 'b'])
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')


#### Create Dataset############

createdata(filename, f_out, solver, model)


