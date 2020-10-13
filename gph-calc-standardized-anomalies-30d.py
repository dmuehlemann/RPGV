# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 12:02:34 2020

@author: Dirk
"""

import xarray as xr
from pathlib import Path


#Input and output data
data_folder = Path("../data/")
filename = data_folder / 'gph-daily-mean-lowpass_2_0-25.nc'
f_out = data_folder / 'z_all_std_ano_30days_lowpass_2_0-25.nc'

z_all = xr.open_dataset(filename)


#calculate anomalies and save it in netCDF file
days_clima=30
days_std=30
climatology_mean = z_all.rolling(time=days_clima, center=True).mean().ffill(dim='time').bfill(dim='time').groupby("time.dayofyear").mean("time")
climatology_std = z_all.rolling(time=days_std, center=True).std().ffill(dim='time').bfill(dim='time').groupby("time.dayofyear").mean("time")

std_ano = xr.apply_ufunc(
    lambda x, m, s: (x - m) / s,
    z_all.groupby("time.dayofyear"),
    climatology_mean,
    climatology_std,
)


std_ano.to_netcdf(f_out)
