# -*- coding: utf-8 -*-
"""
Created on Fri Oct  2 12:52:46 2020

@author: Dirk
"""


# from netCDF4 import Dataset, num2date
# import numpy as np
from pathlib import Path
#import cartopy.crs as ccrs
# import matplotlib.pyplot as plt
import pandas as pd
# from datetime import datetime, timedelta
import xarray as xr



######################Load Datasets#################
data_folder = Path("../data/")
f_in = data_folder / 'wr_time-c7_std_30days_lowpass_2_0-1_short3.nc'
f_out = data_folder / 'ninja_and_wr_30days_lowpass_2_0-1_short3.nc'

#Load weather regime dataset
wr_time = xr.open_dataset(f_in)


#Load renewable ninja dataset and convert it to xarray
filename = data_folder / 'ninja/ninja_europe_pv_v1.1/ninja_pv_europe_v1.1_merra2.csv'
ninja = pd.read_csv(filename, header=0, parse_dates=[0], index_col=0, squeeze=True)
ninja = ninja.to_xarray()




######################Find same date of ninja and ERA5 data and add weather regime to ninja data
ninja = ninja.assign(wr=lambda ninja: ninja.AL * 0)
for i in range(0,int(wr_time.wr.max())+1):
    mask = (
        # ninja.time.dt.strftime("%Y%m%d")==wr_time.where(wr_time==i).time.dt.strftime("%Y%m%d")
        ninja.time.dt.strftime("%Y%m%d").isin(wr_time.where(wr_time==i, drop=True).time.dt.strftime("%Y%m%d"))
    )
        
    ninja['wr'] = xr.where(mask, i, ninja['wr'])


ninja.to_netcdf(f_out)


# ninja2 = ninja.expand_dims({'wrd': ninja.wr})
# ninja = ninja.drop('wrd')

