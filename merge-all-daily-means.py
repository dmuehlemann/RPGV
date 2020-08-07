# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 10:53:51 2020

@author: Dirk
"""

from pathlib import Path
import xarray as xr

data_folder = Path("../data/")
file1 = data_folder / 'gph-djf-daily-mean.nc'
file2 = data_folder / 'gph-mam-daily-mean.nc'
#file3 = data_folder / 'gph-jja-daily-mean.nc'
file4 = data_folder / 'gph-son-daily-mean.nc'



files = [file1, file2, file4]

data = xr.open_mfdataset(files)

data.to_netcdf('gph-daily-mean.nc')

