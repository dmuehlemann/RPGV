# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 10:53:51 2020

@author: Dirk
"""

from pathlib import Path
import xarray as xr




##############MERGE DAILY MEANS#########################

data_folder = Path("../data/")
file1 = data_folder / 'gph-djf-daily-mean-v2.nc'
file2 = data_folder / 'gph-mam-daily-mean-v2.nc'
file3 = data_folder / 'gph-jja-daily-mean-v2.nc'
file4 = data_folder / 'gph-son-daily-mean-v2.nc'

f_out = data_folder / 'gph-daily-mean.nc'

files = [file1, file2, file3, file4]

data = xr.open_mfdataset(files)
data = data.sortby('time', ascending=True)

data.to_netcdf(f_out)


##############MERGE ALL DATA#########################
data_folder = Path("../data/")
file1 = data_folder / 'gph-djf-all.nc'
file2 = data_folder / 'gph-mam-all.nc'
file3 = data_folder / 'gph-jja-all.nc'
file4 = data_folder / 'gph-son-all.nc'
files = [file1, file2, file3, file4]
f_out = data_folder / 'gph-all.nc'
data = xr.open_mfdataset(files)
data = data.sortby('time', ascending=True)

data.to_netcdf(f_out)


