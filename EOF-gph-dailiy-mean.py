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
# import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
import xarray as xr 





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
            var.setncattr('units', var_gph.units)
            var.setncattr('long_name', var_gph.long_name)
     
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




######################Dataset#################
data_folder = Path("../data/")
filename = data_folder / 'gph-daily-mean.nc'
f_out = data_folder / 'wr-gph-daily-mean-c4.nc'



z_all = xr.open_dataset(filename)['z']
z_djf =z_all.sel(time=z_all['time.season']=='DJF')
z_mam =z_all.sel(time=z_all['time.season']=='MAM')
z_jja =z_all.sel(time=z_all['time.season']=='JJA')
z_son =z_all.sel(time=z_all['time.season']=='SON')






######################EOF analysis######################
# Compute anomalies by removing the time-mean.
z_djf_ano = z_djf - z_djf.mean(dim='time')
z_mam_ano = z_mam - z_mam.mean(dim='time')
z_jja_ano = z_jja - z_jja.mean(dim='time')
z_son_ano = z_son - z_son.mean(dim='time')


# z_djf_ano.to_netcdf('z_djf_ano.nc')
# z_mam_ano.to_netcdf('z_mam_ano.nc')
# z_jja_ano.to_netcdf('z_jja_ano.nc')
# z_son_ano.to_netcdf('z_son_ano.nc')



#not good for memory usage ;-)
#z_all_ano = xr.merge([z_djf_ano, z_mam_ano, z_jja_ano, z_son_ano])

# files = ["z_djf_ano.nc", "z_jja_ano.nc", "z_mam_ano.nc", "z_son_ano.nc"]
# z_all_ano = xr.open_mfdataset(files)





# climatology = z_all.groupby("time.month").mean("time")
# anomalies = z_all.groupby("time.month") - climatology





# Create an EOF solver to do the EOF analysis. Square-root of cosine of
# latitude weights are applied before the computation of EOFs.
coslat = np.cos(np.deg2rad(z_all_ano['z'].coords['latitude'].values)).clip(0., 1.)
wgts = np.sqrt(coslat)[..., np.newaxis]
solver = Eof(z_all_ano['z'], weights=wgts)

# Retrieve the leading EOFs, expressed as the covariance between the leading PC
# time series and the input geopotential height anomalies at each grid point.
eof1 = solver.eofsAsCovariance(neofs=1)

# Plot the leading EOF expressed as covariance in the European/Atlantic domain.
#clevs = np.linspace(-75, 75, 11)
proj = ccrs.Orthographic(central_longitude=-20, central_latitude=60)
ax = plt.axes(projection=proj)
ax.coastlines()
ax.set_global()
eof1[0].plot.contourf(ax=ax, cmap=plt.cm.RdBu_r,
                         transform=ccrs.PlateCarree(), add_colorbar=False)
ax.set_title('EOF1 expressed as covariance', fontsize=16)
plt.show()




######################K_MEANS CLUSTERING#################
elbow(solver.pcs())

#z_all_org = z_all + z_all.mean(dim='time')
model = KMeans(n_clusters=4)
# Fit model to samples
model.fit(solver.pcs()[:,:15])
#z_djf_pca_kmeans = np.concatenate([z_djf_org, pd.DataFrame(solver.pcs())], axis = 1)
#Plot clusters on the first two PCA
sns.scatterplot(solver.pcs()[:,1], solver.pcs()[:,2], alpha=.1, hue = model.labels_, palette = ['g', 'r', 'c', 'm'])
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')


#### Create Dataset############

#createdata(filename, f_out, solver, model)


