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
from scipy.optimize import minimize_scalar
import itertools as it


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


def func(ic_reduced):
   
   print(ic_reduced)
   
   
   
    c = 0
    for i in range(0, len(ninja_season)):
        if c ==0:
            tot_P = (ninja_season[i].sel(season='DJF').to_array().values*ic_reduced).sum()
        else:
            tot_P = np.append(tot_P, (ninja_season[i].sel(season='DJF').to_array().values*ic_reduced).sum())
        c = c+1
 
    diff_P = np.array([abs(y - x) for x, y in it.combinations(tot_P, 2)])
    sum_diff_P = diff_P.sum()
    print(sum_diff_P)
    return(sum_diff_P) 
       
   
   
   # var = abs((ninja_season[2].sel(season='DJF').to_array().values*ic_reduced).sum() - (ninja_season[0].sel(season='DJF').to_array().values*ic_reduced).sum())
   # print(var)
   # return(var) 
   




#########Create Testdata##########################

ninja_season= []
#testdata delta_cf per weather regime and season
for i in range(0, int(ninja.wr.max()+1)):
    DE = ninja.DE.where(ninja.wr==i, drop=True).groupby('time.season').mean()#- ninja.DE.groupby('time.season').mean())/ninja.DE.groupby('time.season').mean()
    ES = ninja.ES.where(ninja.wr==i, drop=True).groupby('time.season').mean()#- ninja.ES.groupby('time.season').mean())/ninja.ES.groupby('time.season').mean()
    ninja_season.append(DE.to_dataset().merge(ES.to_dataset()))


tot = ninja.DE.groupby('time.season').mean().to_dataset().merge(ninja.ES.groupby('time.season').mean())
    
#create Testdata IC
ic_temp = ic.DE[9].to_dataset().merge(ic.ES[9])
ic_arr = ic_temp.to_array()


# x0 = np.array([1.3, 0.7, 0.8, 1.9, 1.2])
# bounds=[(48960, 100000),(8761,100000)],
res = minimize(func, ic_arr, method='L-BFGS-B',bounds=[(48960, 100000),(8761,100000)],
                options={'maxiter': 1000000})


# minimize_scalar(lambda ic_arr: func(ic_arr), bounds=(0,500), method='bounded', options={'maxiter': 10})


# c = [ninja_season[2].sel(season='DJF').to_array().values, ninja_season[0].sel(season='DJF').to_array().values]
# A = [[-3, 1], [1, 2]]
# b = [6, 4]
# x0_bounds = (0, None)
# x1_bounds = (0, None)
# from scipy.optimize import linprog
# res = linprog(c, bounds=[x0_bounds, x1_bounds])





