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
from scipy.optimize import lsq_linear


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
   
   print(ic_reduced[8])
   print(ic_reduced[11])
   
   
   # c = 0
   # for i in range(0, len(ninja_season)):
   #     if c ==0:
   #         tot_P = (ninja_season[i].sel(season='DJF').to_array().values*ic_reduced).sum()
   #     else:
   #         tot_P = np.append(tot_P, (ninja_season[i].sel(season='DJF').to_array().values*ic_reduced).sum())
   #     c = c+1
 
   # diff_P = np.array([abs(y - x) for x, y in it.combinations(tot_P, 2)])
   # sum_diff_P = diff_P.sum()
   
   # print(sum_diff_P)
   # return(sum_diff_P)
   

   var = abs((ninja_season[2].sel(season='DJF').to_array().values*ic_reduced).sum() - (ninja_season[0].sel(season='DJF').to_array().values*ic_reduced).sum())
   print(var)
   return(var) 



#########Create Testdata##########################
ninja_tot = []
ninja_season= []


##percentage deviation
for i in range(0, int(ninja.wr.max()+1)):
    # ninja_tot.append(ninja.drop('wr').where(ninja.wr==i, drop=True).mean())# - ninja.drop('wr').mean())/ninja.drop('wr').mean())
    ninja_season.append(ninja.drop('wr').where(ninja.wr==i, drop=True).groupby('time.season').mean() \
                          - ninja.drop('wr').groupby('time.season').mean() \
                          #/ ninja.drop('wr').groupby('time.season').mean()
                          )



#Create array for DJF with all WR and countries
c = np.append([ninja_season[0].sel(season='DJF').to_array().values],[ninja_season[1].sel(season='DJF').to_array().values], axis=0)
c_all = np.append([ninja_season[0].sel(season='DJF').to_array().values],[ninja_season[1].sel(season='DJF').to_array().values], axis=0)
for i in range(2,len(ninja_season)):
    c_all = np.append(c_all,[ninja_season[i].sel(season='DJF').to_array().values], axis=0)


#create array for all available IC which will be used for bounds
b = 0
for i in ninja:
    for a in ic:
        if i==a:
            if b==0:
                ic_reduced = xr.DataArray(ic[a][-1])
                b = b +1
            else:
            # for b in range(0, len(ninja_season)):
                # ninja_season[b][a] = ninja_season[b][a] * ic[a][-1]
                # ninja_tot[b][a] = ninja_tot[b][a] * ic[a][-1]
                # ic_new = np.append(ic_new, ic[a][-1])
                ic_reduced = xr.merge([ic_reduced, ic[a][-1]])
                # print(ic[a][-1])
            
    
bounds = []
for i in ic_reduced:
    print(ic_reduced[i].values)
    bounds.append((ic_reduced[i].values,ic_reduced[i].values*100))







lb = ic_reduced.to_array().values
ub = lb *1000
b = np.zeros(c.shape[0])
b_all = np.zeros(c_all.shape[0])

res = lsq_linear(c, b, bounds=(lb, ub))
# res_np_lsq1 = np.linalg.lstsq(c, b)


res2 = lsq_linear(c_all, b_all, bounds=(lb, ub))





















# # x0 = np.array([1.3, 0.7, 0.8, 1.9, 1.2])

# ic_reduced = ic_reduced.to_array()
# res = minimize(func, ic_reduced, method='L-BFGS-B',bounds=bounds,
#                 #options={'maxiter': 50}
#                 )

# (res.x - ic_reduced).to_dataset(dim='variable')
# # minimize_scalar(lambda ic_arr: func(ic_arr), bounds=(0,500), method='bounded', options={'maxiter': 10})





