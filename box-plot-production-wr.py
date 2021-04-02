# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 13:30:25 2021

@author: Dirk
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 13:33:08 2020

@author: Dirk
"""


# import numpy as np
from pathlib import Path
# import cartopy.crs as ccrs
# import matplotlib.pyplot as plt
import xarray as xr
# import calendar
# import pandas as pd
# import matplotlib as mpl
# import geopandas as gpd
# from mapclassify import Quantiles, UserDefined
# from scipy.optimize import lsq_linear


######################LOAD DATASET#################


data_folder = Path("../data/")


file_ninja = data_folder / 'ninja_and_wr_30days_lowpass_2_0-1_short3.nc'
ninja = xr.open_dataset(file_ninja)


#####################END LOAD DATASET#####################################


######################CREATE NEEDED DATASETS#########################
###here no Brexit ;-)
ninja = ninja.rename_vars({'GB':'UK'})

# ninja_tot = []
ninja_season= []
ninja_season_std= []
##calculate delata CF for each WR

# mean_day = ninja.drop('wr').resample(time='1D').mean()
# mean_day_std = ninja.drop('wr').resample(time='1D').std()

# mean_season_d = mean_day.groupby('time.season').mean()
# mean_season_d_std = mean_day.groupby('time.season').std()

mean_season = ninja.drop('wr').groupby('time.season').mean()
for i in range(0, int(ninja.wr.max()+1)):
    # ninja_tot.append(ninja.drop('wr').where(ninja.wr==i, drop=True).mean())# - ninja.drop('wr').mean())/ninja.drop('wr').mean())
    ninja_season.append(ninja.drop('wr').where(ninja.wr==i, drop=True).groupby('time.day').mean() \
                          - mean_season \
                          #/ ninja.drop('wr').groupby('time.season').mean()
                          )
for i in range(0, int(ninja.wr.max()+1)):
    # ninja_tot.append(ninja.drop('wr').where(ninja.wr==i, drop=True).mean())# - ninja.drop('wr').mean())/ninja.drop('wr').mean())
    ninja_season_std.append((ninja.drop('wr').resample(time='1D').mean().where(ninja.wr==i, drop=True).groupby('time.season') \
                          - mean_season).groupby('time.season').std()
                          #/ ninja.drop('wr').groupby('time.season').mean()
                          )        
        
a=0        
for i in ninja_season_std:
    file = data_folder / str('ninja_season_std_wr'+str(a)+'.nc')
    i.to_netcdf(file)
    a = a +1

# a=0        
# for i in ninja_season:
#     file = data_folder / str('ninja_season_wr'+str(a)+'.nc')
#     i.to_netcdf(file)
#     a = a +1
# file_mean = data_folder / str('ninja_season_mean.nc')    
# mean_season.to_netcdf(file_mean)