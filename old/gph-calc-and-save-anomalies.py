# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 10:51:11 2020

@author: Dirk
"""

import xarray as xr
from pathlib import Path

#calculate anomalies and save it in netCDF files --> allready done
data_folder = Path("../data/")
filename = data_folder / 'gph-daily-mean.nc'




z_all = xr.open_dataset(filename)['z']

z_djf =z_all.sel(time=z_all['time.season']=='DJF')
z_mam =z_all.sel(time=z_all['time.season']=='MAM')
z_jja =z_all.sel(time=z_all['time.season']=='JJA')
z_son =z_all.sel(time=z_all['time.season']=='SON')
# Compute anomalies by removing the time-mean.
z_djf_ano = z_djf - z_djf.mean(dim='time')
z_mam_ano = z_mam - z_mam.mean(dim='time')
z_jja_ano = z_jja - z_jja.mean(dim='time')
z_son_ano = z_son - z_son.mean(dim='time')


files = [data_folder / "z_djf_ano.nc", data_folder / "z_jja_ano.nc", data_folder / "z_mam_ano.nc", data_folder / "z_son_ano.nc"]
z_djf_ano.to_netcdf(files[0])
z_mam_ano.to_netcdf(files[1])
z_jja_ano.to_netcdf(files[2])
z_son_ano.to_netcdf(files[3])


z_all_ano = xr.open_mfdataset(files)
z_all_ano.to_netcdf("../data/z_all_ano.nc")



# climatology = z_all.groupby("time.month").mean("time")
# anomalies = z_all.groupby("time.month") - climatology