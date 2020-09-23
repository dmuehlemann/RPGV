# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 12:02:34 2020

@author: Dirk
"""

import xarray as xr
from pathlib import Path
# import numpy as np
#import statistics as stat

#calculate anomalies and save it in netCDF files --> allready done
data_folder = Path("../data/")
filename = data_folder / 'gph-daily-mean.nc'




z_all = xr.open_dataset(filename)

#std = z_all_ano.sel(time=z_all_ano['time.month']==1).std('time')

days_clima=90
days_std=30
climatology_mean = z_all.rolling(time=days_clima, center=True).mean().ffill(dim='time').bfill(dim='time').groupby("time.dayofyear").mean("time")
climatology_std = z_all.rolling(time=days_std, center=True).std().ffill(dim='time').bfill(dim='time').groupby("time.dayofyear").mean("time")

# climatology_mean = z_all.groupby("time.month").mean("time")
# climatology_std = z_all.groupby("time.month").std("time")


stand_anomalies = xr.apply_ufunc(
    lambda x, m, s: (x - m) / s,
    z_all.groupby("time.dayofyear"),
    climatology_mean,
    climatology_std,
)

#stand_anomalies.mean("location").to_dataframe()[["tmin", "tmax"]].plot()

stand_anomalies.to_netcdf("../data/z_all_std_ano_grams.nc")
