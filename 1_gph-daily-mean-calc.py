# -*- coding: utf-8 -*-
"""
Created on Sam Aug  7 11:50:05 2020

@author: Dirk Mühlemann

This script calculates daily means of the 
seasonal ERA5 500 hPa gepotential height data 


"""


from pathlib import Path
import xarray as xr


data_folder = Path("../data/")
file1 = data_folder / 'gph-djf-all.nc'
file2 = data_folder / 'gph-mam-all.nc'
file3 = data_folder / 'gph-jja-all.nc'
file4 = data_folder / 'gph-son-all.nc'
files = [file1, file2, file3, file4]

gph_djf_all = xr.open_dataset(file1)
gph_djf_daily_mean = gph_djf_all.resample(time='1D').mean().dropna(dim='time')
gph_djf_daily_mean.to_netcdf('../data/gph-djf-daily-mean.nc')

gph_mam_all = xr.open_dataset(file2)
gph_mam_daily_mean =gph_mam_all.resample(time='1D').mean().dropna(dim='time')
gph_mam_daily_mean.to_netcdf('../data/gph-mam-daily-mean.nc')

gph_jja_all = xr.open_dataset(file3)
gph_jja_daily_mean = gph_jja_all.resample(time='1D').mean().dropna(dim='time')
gph_jja_daily_mean.to_netcdf('../data/gph-jja-daily-mean.nc')

gph_son_all = xr.open_dataset(file4)
gph_son_daily_mean = gph_son_all.resample(time='1D').mean().dropna(dim='time')
gph_son_daily_mean.to_netcdf('../data/gph-son-daily-mean.nc')
