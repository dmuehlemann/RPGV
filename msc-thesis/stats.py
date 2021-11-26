# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 10:48:33 2020

@author: Dirk
"""


import numpy as np
from pathlib import Path
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import xarray as xr
# import calendar
import pandas as pd
import matplotlib as mpl
import geopandas as gpd
# from mapclassify import Quantiles, UserDefined


from scipy import stats

######################Load Datasets#################

data_folder = Path("../data/")

file_ninja = data_folder / 'ninja_and_wr_30days_lowpass_2_0-1_short10.nc'
ninja = xr.open_dataset(file_ninja)
DJF = ninja.groupby('time.season').groups['DJF']
ninja_DJF = ninja.drop('wr').where(ninja.time[DJF], drop=True)

wr0 = ninja.drop('wr').where(ninja.wr==0, drop=True)
DJF = wr0.groupby('time.season').groups['DJF']
wr0_DJF = ninja.drop('wr').where(ninja.time[DJF], drop=True)



wr1 = ninja.drop('wr').where(ninja.wr==1, drop=True)

stats.ttest_ind(wr0.CH, wr0.CH, equal_var = False)
res = []
for i in wr1:
    stat = stats.ttest_ind(ninja.drop('wr')[i], wr0[i], equal_var = False)
    print(stat.pvalue)
    res.append(stat.pvalue)
    
matches = (x for x in res if x > 0.1)




#kei aniig chunt en p-value für jeded land maht das sinn??
stats.ttest_ind(ninja_DJF.to_array(), wr0_DJF.to_array(), equal_var = False, axis=1)
