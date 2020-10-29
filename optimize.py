# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 13:33:08 2020

@author: Dirk
"""


import numpy as np
from pathlib import Path
# import cartopy.crs as ccrs
# import matplotlib.pyplot as plt
import xarray as xr
# import calendar
import pandas as pd
# import matplotlib as mpl
# import geopandas as gpd
# from mapclassify import Quantiles, UserDefined
from scipy.optimize import minimize


######################Load Datasets#################
######################Load Datasets#################

data_folder = Path("../data/")


file_ninja = data_folder / 'ninja_and_wr_30days_lowpass_2_0-1_short10.nc'
ninja = xr.open_dataset(file_ninja)

ic_file = data_folder / 'source/installed_capacities_IRENA.csv'

ic = pd.read_csv(ic_file, header=0, parse_dates=[0], index_col=0, squeeze=True)




###here no Brexit ;-)
ninja = ninja.rename_vars({'GB':'UK'})
ic = ic.rename(columns={'GB':'UK'})
ic = ic.to_xarray()

####################################################


def func(ic):
    #asdf
   print(ic)
   
   #########SPLIT mal ICs auf
   var = (delta_cf[2].sel(season='DJF')*ic).to_array().sum() - (delta_cf[0].sel(season='DJF')*ic).to_array().sum()
   # print(var)
   return(var)



#########Create Testdata##########################
delta_cf = []
#testdata delta_cf per weather regime and season
for i in range(0, int(ninja.wr.max())):
    DE = (ninja.DE.where(ninja.wr==i, drop=True).groupby('time.season').mean()- ninja.DE.groupby('time.season').mean())/ninja.DE.groupby('time.season').mean()
    ES = (ninja.ES.where(ninja.wr==i, drop=True).groupby('time.season').mean()- ninja.ES.groupby('time.season').mean())/ninja.ES.groupby('time.season').mean()
    delta_cf.append(DE.to_dataset().merge(ES.to_dataset()))

    
#create Testdata IC
ic = ic.DE[9].to_dataset().merge(ic.ES[9]).to_array()


# x0 = np.array([1.3, 0.7, 0.8, 1.9, 1.2])
res = minimize(func, ic, method='nelder-mead',
               options={'xatol': 8000, 'disp': True})








